"""
Staff Timing Validation Block - Unified staff timing and coverage validation.

Validates:
1. Actual arrivals match scheduled role assignments
2. Business rules like: 1 recovery nurse per shift, 1 gastro nurse per concurrent doctor
3. Adequate coverage during all procedure times
4. Early arrival detection (arrived before needed)
5. Early departure detection (left before coverage ended)

Configurable timing rules per role:
- arrival_buffer_minutes: How early should they arrive before first procedure
- departure_buffer_minutes: How long after last procedure can they leave
- match_to: Which procedures to match (all_procedures, gastro_procedures, own_procedures)

Uses DuckDB SQL for efficient interval overlap detection and staff counting.
"""

import logging
import re
from datetime import datetime, time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from csv_analyzer.workflows.block import BlockRegistry
from csv_analyzer.workflows.ontology import DataType
from csv_analyzer.workflows.base_block import BaseBlock, BlockContext

logger = logging.getLogger(__name__)


def parse_time(time_str: str) -> Optional[time]:
    """Parse a time string to time object."""
    if pd.isna(time_str) or not time_str:
        return None
    
    time_str = str(time_str).strip()
    
    # Remove common prefixes like "* "
    time_str = re.sub(r'^\*\s*', '', time_str)
    
    # Try various formats
    for fmt in ["%H:%M", "%H:%M:%S", "%I:%M %p", "%I:%M:%S %p"]:
        try:
            dt = datetime.strptime(time_str, fmt)
            return dt.time()
        except ValueError:
            continue
    
    # Handle datetime strings (extract time part)
    if " " in time_str:
        time_part = time_str.split(" ")[-1]
        return parse_time(time_part)
    
    return None


def parse_time_range(time_range_str: str) -> Tuple[Optional[time], Optional[time]]:
    """
    Parse a time range string like "19:00 - 19:15" to (start_time, end_time).
    
    Returns:
        Tuple of (start_time, end_time) or (None, None) if parsing fails
    """
    if pd.isna(time_range_str) or not time_range_str:
        return None, None
    
    time_range_str = str(time_range_str).strip()
    
    # Try splitting on " - " or "-"
    for delimiter in [" - ", "-"]:
        if delimiter in time_range_str:
            parts = time_range_str.split(delimiter)
            if len(parts) == 2:
                start = parse_time(parts[0].strip())
                end = parse_time(parts[1].strip())
                return start, end
    
    # If no delimiter, try parsing as single time (start = end)
    single_time = parse_time(time_range_str)
    if single_time:
        return single_time, single_time
    
    return None, None


def normalize_date(date_val, prefer_dmy: bool = False) -> Optional[str]:
    """
    Normalize date to YYYY-MM-DD format.
    
    Handles multiple formats:
    - MM/DD/YYYY (American) - e.g., 12/1/2025 = December 1, 2025
    - DD/MM/YYYY (European) - e.g., 25/12/2025 = December 25, 2025
    - DD.MM.YYYY (European with dots) - e.g., 01.12.2025 = December 1, 2025
    - YYYY-MM-DD (ISO) - already normalized
    
    Args:
        date_val: The date value to normalize
        prefer_dmy: If True, prefer DD/MM/YYYY when ambiguous. Default False (MM/DD/YYYY).
    
    Disambiguation: if first number > 12, it must be a day (European format).
    Otherwise, use prefer_dmy parameter to decide.
    """
    if pd.isna(date_val):
        return None
    date_str = str(date_val).strip()
    
    # Try YYYY-MM-DD format (already normalized)
    match = re.match(r'(\d{4})-(\d{1,2})-(\d{1,2})', date_str)
    if match:
        year, month, day = match.groups()
        return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
    
    # Try DD.MM.YYYY format (European with dots) - always DD/MM
    match = re.match(r'(\d{1,2})\.(\d{1,2})\.(\d{4})', date_str)
    if match:
        day, month, year = match.groups()
        return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
    
    # Try X/Y/YYYY format - need to disambiguate MM/DD vs DD/MM
    match = re.match(r'(\d{1,2})/(\d{1,2})/(\d{4})', date_str)
    if match:
        first, second, year = match.groups()
        first_num = int(first)
        second_num = int(second)
        
        # If first number > 12, it must be a day (European DD/MM/YYYY)
        if first_num > 12:
            day, month = first, second
        # If second number > 12, it must be a day (American MM/DD/YYYY)
        elif second_num > 12:
            month, day = first, second
        # Both <= 12: use prefer_dmy parameter
        elif prefer_dmy:
            day, month = first, second
        else:
            month, day = first, second
        
        return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
    
    return date_str


def normalize_employee_name(name: str) -> str:
    """
    Normalize employee name for matching across different data sources.
    
    Handles variations like:
    - "גוטמן דניאלה" vs "דניאלה גוטמן"
    - Extra spaces, parentheses
    """
    if pd.isna(name) or not name:
        return ""
    
    name = str(name).strip()
    
    # Remove parenthetical content like "(מרק)"
    name = re.sub(r'\([^)]*\)', '', name)
    
    # Remove extra whitespace
    name = ' '.join(name.split())
    
    # Sort name parts to handle first/last name order differences
    parts = sorted(name.split())
    return ' '.join(parts).lower()


def fuzzy_name_similarity(name1: str, name2: str) -> float:
    """
    Calculate similarity ratio between two names using SequenceMatcher.
    Returns a value between 0.0 (completely different) and 1.0 (identical).
    """
    from difflib import SequenceMatcher
    if not name1 or not name2:
        return 0.0
    return SequenceMatcher(None, name1, name2).ratio()


def build_name_mapping(names_list1: List[str], names_list2: List[str], threshold: float = 0.85) -> Dict[str, str]:
    """
    Build a mapping of similar names across two lists.
    
    For each name in list1, finds the best matching name in list2 if similarity >= threshold.
    Returns a dict mapping normalized names to their canonical (matched) form.
    
    This handles:
    - Exact matches after normalization
    - Typos/misspellings (e.g., "אנה פרדנקו" vs "אנה פדרנקו")
    - Name order variations (already handled by normalize_employee_name)
    """
    mapping = {}
    
    # Normalize all names first
    norm1 = {normalize_employee_name(n): n for n in names_list1 if n}
    norm2 = {normalize_employee_name(n): n for n in names_list2 if n}
    
    # First pass: exact matches
    for norm_name in norm1:
        if norm_name in norm2:
            mapping[norm_name] = norm_name
    
    # Second pass: fuzzy matches for unmatched names
    unmatched1 = [n for n in norm1 if n not in mapping]
    unmatched2 = [n for n in norm2 if n not in mapping.values()]
    
    for name1 in unmatched1:
        best_match = None
        best_score = 0.0
        
        for name2 in unmatched2:
            score = fuzzy_name_similarity(name1, name2)
            if score >= threshold and score > best_score:
                best_score = score
                best_match = name2
        
        if best_match:
            # Map both names to the same canonical form (use the one from list2/schedule)
            mapping[name1] = best_match
            # Also ensure the matched name maps to itself
            mapping[best_match] = best_match
    
    return mapping


# Default timing rules configuration
# Each role can have:
# - min_count, max_count: Role count limits per shift
# - scope: "per_shift" or "per_concurrent_doctor"
# - arrival_buffer_minutes: Can arrive up to X minutes before first relevant procedure
# - departure_buffer_minutes: Minimum time to stay after last procedure (0 = can leave when procedure ends)
# - departure_max_minutes: Maximum time allowed to stay after last procedure (null = no limit)
# - match_to: "all_procedures", "gastro_procedures", or "own_procedures"
# - notify_double_shift: If true, flag when employee works this role in both morning and evening shifts
DEFAULT_TIMING_RULES = {
    "אח התאוששות": {
        "min_count": 1,
        "max_count": 1,
        "scope": "per_shift",
        "arrival_buffer_minutes": 30,
        "departure_buffer_minutes": 0,
        "departure_max_minutes": 60,
        "match_to": "all_procedures",
        "notify_double_shift": True
    },
    "אח התאוששות ערב": {
        "min_count": 1,
        "max_count": 1,
        "scope": "per_shift",
        "arrival_buffer_minutes": 30,
        "departure_buffer_minutes": 0,
        "departure_max_minutes": 60,
        "match_to": "all_procedures",
        "notify_double_shift": True
    },
    "אח גסטרו": {
        "min_count": 1,
        "max_count": None,
        "scope": "per_concurrent_doctor",
        "arrival_buffer_minutes": 30,
        "departure_buffer_minutes": 0,
        "match_to": "gastro_procedures",
        "notify_double_shift": True
    },
    "כע עזר": {
        "min_count": 1,
        "max_count": None,
        "scope": "per_shift",
        "arrival_buffer_minutes": 30,
        "departure_buffer_minutes": 0,
        "match_to": "all_procedures",
        "notify_double_shift": True
    },
    "כע עזר ערב": {
        "min_count": 1,
        "max_count": None,
        "scope": "per_shift",
        "arrival_buffer_minutes": 30,
        "departure_buffer_minutes": 0,
        "match_to": "all_procedures",
        "notify_double_shift": True
    },
    "כח עזר": {
        "min_count": 1,
        "max_count": None,
        "scope": "per_shift",
        "arrival_buffer_minutes": 30,
        "departure_buffer_minutes": 0,
        "match_to": "all_procedures",
        "notify_double_shift": True
    },
}

# Keep backward compatibility alias
DEFAULT_ROLE_LIMITS = DEFAULT_TIMING_RULES


class StaffTimingValidationBlock(BaseBlock):
    """
    Unified staff timing and coverage validation block.
    
    Validates:
    1. Actual staff arrivals match scheduled role assignments
    2. Business rules (configurable role counts per shift/doctor)
    3. Gastro nurse coverage for concurrent doctors
    4. Early arrival detection (arrived too early before procedures)
    5. Early departure detection (left before coverage period ended)
    
    Uses DuckDB SQL for efficient time interval analysis.
    
    Configurable timing rules per role via 'timing_rules' parameter.
    """
    
    # Required document types
    REQUIRED_DOC_TYPES = ["shift_schedule", "employee_shifts"]
    OPTIONAL_DOC_TYPES = ["medical_actions"]
    
    # Column candidates for shift_schedule
    # Includes both original Hebrew names AND canonized schema field names
    SCHED_EMPLOYEE_CANDIDATES = ['employee_name', 'עובד', 'employee', 'name', 'staff_name', 'Unnamed: 2']
    SCHED_SHIFT_TYPE_CANDIDATES = ['shift_code', 'ק.משמרת', 'shift_type', 'type', 'Unnamed: 3']
    SCHED_ROLE_CANDIDATES = ['role_name', 'שם משמרת', 'role', 'position', 'shift_name', 'Unnamed: 4']
    SCHED_DATE_CANDIDATES = ['schedule_date', 'יום בחודש', 'date', 'shift_date', 'day', 'Unnamed: 5']
    
    # Column candidates for employee_shifts (actual arrivals)
    # Note: Employee name often comes from _meta columns added during classification
    SHIFT_TIME_CANDIDATES = ['shift_start', 'כניסה', 'clock_in', 'entry_time', 'start_time']
    SHIFT_END_CANDIDATES = ['shift_end', 'יציאה', 'clock_out', 'exit_time', 'end_time']
    SHIFT_DATE_CANDIDATES = ['shift_date', '_meta_date_range_start', 'תאריך', 'date']
    SHIFT_NAME_CANDIDATES = ['_meta_employee_name', 'employee_name', 'שם_עובד', 'שם', 'name']
    
    # Column candidates for medical_actions (procedures)
    # Includes both original Hebrew names AND canonized schema field names
    PROC_TIME_CANDIDATES = ['treatment_time_range', 'שעות טיפול בפועל', 'treatment_time', 'procedure_time', 'time_range']
    PROC_START_CANDIDATES = ['treatment_start_time', 'שעות טיפול בפועל_start', 'start_time', 'procedure_start']
    PROC_END_CANDIDATES = ['treatment_end_time', 'שעות טיפול בפועל_end', 'end_time', 'procedure_end']
    PROC_DATE_CANDIDATES = ['treatment_date', 'תאריך טיפול', 'תאריך', 'date']
    PROC_STAFF_CANDIDATES = ['treating_staff', 'צוות מטפל', 'staff_name', 'provider', 'doctor']
    PROC_CAT_CANDIDATES = ['treatment_category', 'קטגורית טיפול', 'category']
    
    def run(self) -> Dict[str, str]:
        """
        Validate staff timing and coverage against schedule and business rules.
        
        Returns:
            Dict with 'result' key containing S3 URI of validation results
        """
        # Get configuration parameters
        timing_rules = self.get_param("timing_rules", DEFAULT_TIMING_RULES)
        shift_time_boundary = self.get_param("shift_time_boundary", "13:00")
        morning_shift_start = self.get_param("morning_shift_start", "06:00")
        evening_shift_end = self.get_param("evening_shift_end", "21:00")
        
        self.logger.info("Running staff timing validation")
        self.logger.info(f"Timing rules configured for {len(timing_rules)} roles")
        
        # Load classified data
        classified_data = self.load_classified_data("data")
        
        # Check for required document types
        missing_types = [dt for dt in self.REQUIRED_DOC_TYPES if dt not in classified_data]
        if missing_types:
            self.logger.warning(
                f"⚠️  Skipping staff_timing: missing required doc types: {missing_types}. "
                f"Available: {list(classified_data.keys())}"
            )
            empty_df = self._get_empty_result_df()
            result_uri = self.save_to_s3("result", empty_df)
            return {"result": result_uri, "skipped": True, "reason": f"Missing: {missing_types}"}
        
        # Load data sources
        schedule_df = classified_data.get("shift_schedule")
        shifts_df = classified_data.get("employee_shifts")
        procedures_df = classified_data.get("medical_actions")  # Optional
        
        self.logger.info(
            f"Loaded: {len(schedule_df)} scheduled roles, "
            f"{len(shifts_df)} actual shifts"
            + (f", {len(procedures_df)} procedures" if procedures_df is not None else "")
        )
        
        # Prepare data
        schedule_prepared = self._prepare_schedule(schedule_df)
        shifts_prepared = self._prepare_shifts(shifts_df, shift_time_boundary)
        procedures_prepared = None
        all_procedures_prepared = None
        if procedures_df is not None and len(procedures_df) > 0:
            procedures_prepared = self._prepare_procedures_gastro(procedures_df)
            all_procedures_prepared = self._prepare_all_procedures(procedures_df)
        
        # Run validation
        results = self._validate_all(
            schedule_prepared,
            shifts_prepared,
            procedures_prepared,
            all_procedures_prepared,
            timing_rules,
            shift_time_boundary,
        )
        
        # Save results
        result_uri = self.save_to_s3("result", results)
        
        # Log summary
        issues = results[results['status'] != 'OK']
        self.logger.info(
            f"Staff timing validation complete: "
            f"{len(results)} checks, {len(issues)} issues found"
        )
        
        return {"result": result_uri}
    
    def _prepare_schedule(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare schedule DataFrame with normalized columns."""
        result = pd.DataFrame()
        
        # Check if first row contains actual headers (common in Hebrew CSV files)
        # If so, use those as column names and skip that row
        df = self._fix_header_row(df)
        
        # Find columns
        emp_col = self._find_column(df, self.SCHED_EMPLOYEE_CANDIDATES)
        type_col = self._find_column(df, self.SCHED_SHIFT_TYPE_CANDIDATES)
        role_col = self._find_column(df, self.SCHED_ROLE_CANDIDATES)
        date_col = self._find_column(df, self.SCHED_DATE_CANDIDATES)
        
        if not emp_col or not role_col or not date_col:
            self.logger.warning(
                f"Could not find required schedule columns. "
                f"Found: emp={emp_col}, role={role_col}, date={date_col}. "
                f"Available columns: {list(df.columns)}"
            )
            return pd.DataFrame()
        
        self.logger.info(f"Schedule columns found: emp={emp_col}, type={type_col}, role={role_col}, date={date_col}")
        
        # Extract and normalize
        result['employee_name'] = df[emp_col].astype(str)
        result['employee_name_normalized'] = result['employee_name'].apply(normalize_employee_name)
        result['shift_type'] = df[type_col].astype(str) if type_col else 'unknown'
        result['role'] = df[role_col].astype(str)
        result['shift_date'] = df[date_col].apply(
            lambda x: normalize_date(x) if pd.notna(x) else None
        )
        
        # Filter out invalid rows (including header rows that became data)
        result = result[result['shift_date'].notna()]
        result = result[result['employee_name'].str.strip() != '']
        # Filter out rows where employee_name looks like a header
        result = result[~result['employee_name'].isin(['עובד', 'employee', 'name', 'nan', ''])]
        
        self.logger.info(f"Prepared {len(result)} schedule entries")
        return result
    
    def _fix_header_row(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fix DataFrame where actual headers are in row 0 instead of being column names.
        
        Common pattern in Hebrew CSV files exported from Excel.
        """
        # Check if columns are "Unnamed: X" pattern and first row has Hebrew headers
        unnamed_cols = [c for c in df.columns if str(c).startswith('Unnamed:')]
        
        if len(unnamed_cols) >= 3:
            # Check if first row contains expected Hebrew headers
            first_row = df.iloc[0].astype(str).tolist()
            hebrew_headers = ['עובד', 'ק.משמרת', 'שם משמרת', 'יום בחודש']
            
            if any(h in first_row for h in hebrew_headers):
                # Use first row as headers
                new_columns = df.iloc[0].tolist()
                df = df.iloc[1:].reset_index(drop=True)
                df.columns = new_columns
                self.logger.info("Fixed header row - extracted actual column names from row 0")
        
        return df
    
    def _prepare_shifts(self, df: pd.DataFrame, shift_time_boundary: str) -> pd.DataFrame:
        """Prepare actual shifts DataFrame with normalized columns."""
        result = pd.DataFrame()
        
        # Find columns
        time_col = self._find_column(df, self.SHIFT_TIME_CANDIDATES)
        end_col = self._find_column(df, self.SHIFT_END_CANDIDATES)
        date_col = self._find_column(df, self.SHIFT_DATE_CANDIDATES)
        name_col = self._find_column(df, self.SHIFT_NAME_CANDIDATES)
        
        if not time_col or not name_col:
            self.logger.warning(
                f"Could not find required shift columns. "
                f"Found: time={time_col}, name={name_col}"
            )
            return pd.DataFrame()
        
        # Parse arrival time
        result['arrival_time'] = df[time_col].apply(
            lambda x: parse_time(str(x)) if pd.notna(x) else None
        )
        
        # Parse departure time
        if end_col:
            result['departure_time'] = df[end_col].apply(
                lambda x: parse_time(str(x)) if pd.notna(x) else None
            )
        else:
            result['departure_time'] = None
        
        # Normalize date
        if date_col and date_col in df.columns:
            result['shift_date'] = df[date_col].apply(
                lambda x: normalize_date(x) if pd.notna(x) else None
            )
        else:
            result['shift_date'] = None
        
        # Employee name
        result['employee_name'] = df[name_col].astype(str)
        result['employee_name_normalized'] = result['employee_name'].apply(normalize_employee_name)
        
        # Determine shift type based on arrival time
        boundary_time = parse_time(shift_time_boundary)
        
        def determine_shift_type(arrival: Optional[time]) -> str:
            if arrival is None:
                return 'unknown'
            if boundary_time and arrival < boundary_time:
                return 'בוקר'
            return 'ערב'
        
        result['inferred_shift_type'] = result['arrival_time'].apply(determine_shift_type)
        
        # Filter out invalid rows
        result = result[result['arrival_time'].notna()]
        result = result[result['shift_date'].notna()]
        
        self.logger.info(f"Prepared {len(result)} actual shift arrivals")
        return result
    
    def _prepare_procedures_gastro(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare gastro procedures DataFrame for concurrent doctor detection."""
        result = self._prepare_all_procedures(df)
        
        if result.empty:
            return result
        
        # Filter to only gastro procedures (where gastro nurse is needed)
        result = result[result['category'].str.contains('גסטרו', case=False, na=False)]
        
        self.logger.info(f"Prepared {len(result)} gastro procedures for concurrent doctor detection")
        return result
    
    def _prepare_all_procedures(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare all procedures DataFrame with time intervals for timing validation."""
        result = pd.DataFrame()
        
        # Find columns - try direct start/end columns first, then time range
        start_col = self._find_column(df, self.PROC_START_CANDIDATES)
        end_col = self._find_column(df, self.PROC_END_CANDIDATES)
        time_col = self._find_column(df, self.PROC_TIME_CANDIDATES)
        date_col = self._find_column(df, self.PROC_DATE_CANDIDATES)
        staff_col = self._find_column(df, self.PROC_STAFF_CANDIDATES)
        cat_col = self._find_column(df, self.PROC_CAT_CANDIDATES)
        
        # Parse times - prefer direct columns over time range
        if start_col and end_col:
            result['start_time'] = df[start_col].apply(
                lambda x: parse_time(str(x)) if pd.notna(x) else None
            )
            result['end_time'] = df[end_col].apply(
                lambda x: parse_time(str(x)) if pd.notna(x) else None
            )
            self.logger.info(f"Using direct time columns: start={start_col}, end={end_col}")
        elif time_col:
            # Parse time ranges like "19:00 - 19:15"
            time_ranges = df[time_col].apply(parse_time_range)
            result['start_time'] = time_ranges.apply(lambda x: x[0])
            result['end_time'] = time_ranges.apply(lambda x: x[1])
            self.logger.info(f"Using time range column: {time_col}")
        else:
            self.logger.warning(
                f"Could not find procedure time columns. "
                f"Available: {list(df.columns)}"
            )
            return pd.DataFrame()
        
        # Normalize date (procedures use DD/MM/YYYY format)
        if date_col:
            result['procedure_date'] = df[date_col].apply(
                lambda x: normalize_date(x, prefer_dmy=True) if pd.notna(x) else None
            )
        else:
            result['procedure_date'] = None
        
        # Doctor/staff name (normalize for matching)
        if staff_col:
            result['doctor'] = df[staff_col].astype(str)
            result['doctor_normalized'] = result['doctor'].apply(normalize_employee_name)
        else:
            result['doctor'] = ''
            result['doctor_normalized'] = ''
        
        # Category (for filtering by procedure type)
        if cat_col:
            result['category'] = df[cat_col].astype(str)
        else:
            result['category'] = 'unknown'
        
        # Filter out invalid rows
        result = result[result['start_time'].notna()]
        result = result[result['procedure_date'].notna()]
        
        self.logger.info(f"Prepared {len(result)} total procedures for timing validation")
        return result
    
    def _validate_all(
        self,
        schedule_df: pd.DataFrame,
        shifts_df: pd.DataFrame,
        gastro_procedures_df: Optional[pd.DataFrame],
        all_procedures_df: Optional[pd.DataFrame],
        timing_rules: Dict[str, Dict],
        shift_time_boundary: str,
        fuzzy_name_threshold: float = 0.85,
    ) -> pd.DataFrame:
        """
        Validate all staff timing and coverage rules using DuckDB.
        
        Returns DataFrame with validation results.
        """
        results = []
        
        if schedule_df.empty or shifts_df.empty:
            self.logger.warning("Empty schedule or shifts data - cannot validate")
            return self._get_empty_result_df()
        
        # Build fuzzy name mapping to handle typos/misspellings
        schedule_names = schedule_df['employee_name'].unique().tolist() if 'employee_name' in schedule_df.columns else []
        shift_names = shifts_df['employee_name'].unique().tolist() if 'employee_name' in shifts_df.columns else []
        
        name_mapping = build_name_mapping(shift_names, schedule_names, threshold=fuzzy_name_threshold)
        
        # Log fuzzy matches found (non-exact matches)
        fuzzy_matches = [(k, v) for k, v in name_mapping.items() if k != v]
        if fuzzy_matches:
            self.logger.info(f"Fuzzy name matches found: {fuzzy_matches}")
        
        # Apply canonical names for matching
        def get_canonical_name(normalized_name: str) -> str:
            return name_mapping.get(normalized_name, normalized_name)
        
        # Add canonical_name column to both dataframes
        schedule_df = schedule_df.copy()
        shifts_df = shifts_df.copy()
        
        schedule_df['canonical_name'] = schedule_df['employee_name_normalized'].apply(get_canonical_name)
        shifts_df['canonical_name'] = shifts_df['employee_name_normalized'].apply(get_canonical_name)
        
        # Register tables with DuckDB
        self.duckdb.register("schedule", schedule_df)
        self.duckdb.register("arrivals", shifts_df)
        
        # Enrich shifts with role information from schedule
        enriched_shifts = self._enrich_shifts_with_roles(schedule_df, shifts_df)
        if not enriched_shifts.empty:
            self.duckdb.register("enriched_arrivals", enriched_shifts)
        
        # 1. Validate schedule vs actual arrivals (who showed up?)
        schedule_validation = self._validate_schedule_attendance(schedule_df, shifts_df)
        results.extend(schedule_validation)
        
        # 2. Validate role counts per shift
        role_count_validation = self._validate_role_counts(
            schedule_df, shifts_df, timing_rules
        )
        results.extend(role_count_validation)
        
        # 3. Validate gastro nurse coverage for concurrent doctors
        if gastro_procedures_df is not None and not gastro_procedures_df.empty:
            self.duckdb.register("gastro_procedures", gastro_procedures_df)
            gastro_validation = self._validate_gastro_coverage(
                schedule_df, shifts_df, gastro_procedures_df, timing_rules
            )
            results.extend(gastro_validation)
        
        # 4. Validate arrival timing (did they arrive too early?)
        if all_procedures_df is not None and not all_procedures_df.empty and not enriched_shifts.empty:
            self.duckdb.register("all_procedures", all_procedures_df)
            arrival_validation = self._validate_arrival_timing(
                enriched_shifts, all_procedures_df, timing_rules, shift_time_boundary
            )
            results.extend(arrival_validation)
        
        # 5. Validate departure timing (did they leave too early?)
        if all_procedures_df is not None and not all_procedures_df.empty and not enriched_shifts.empty:
            departure_validation = self._validate_departure_timing(
                enriched_shifts, all_procedures_df, timing_rules, shift_time_boundary
            )
            results.extend(departure_validation)
        
        # 6. Validate double shifts (employees working both morning and evening)
        if not enriched_shifts.empty:
            double_shift_validation = self._validate_double_shifts(
                enriched_shifts, timing_rules
            )
            results.extend(double_shift_validation)
        
        # Convert to DataFrame
        if results:
            result_df = pd.DataFrame(results)
        else:
            result_df = self._get_empty_result_df()
        
        return result_df
    
    def _validate_schedule_attendance(
        self,
        schedule_df: pd.DataFrame,
        shifts_df: pd.DataFrame,
    ) -> List[Dict]:
        """
        Check if scheduled employees actually arrived.
        
        Distinguishes between:
        - OK: Employee showed up for the scheduled shift
        - NO_SHOW: We have shift data for this employee, but they didn't show up for this date
        - NO_DATA: We don't have any shift data for this employee at all
        """
        results = []
        
        # First, get all unique employees that have ANY shift data (using canonical names)
        employees_with_data_sql = """
        SELECT DISTINCT canonical_name
        FROM arrivals
        """
        employees_with_data = set(
            self.duckdb.execute(employees_with_data_sql).fetchdf()['canonical_name'].tolist()
        )
        self.logger.info(f"Found {len(employees_with_data)} employees with shift data")
        
        # Use DuckDB for efficient matching
        sql = """
        SELECT 
            s.shift_date,
            s.shift_type,
            s.role,
            s.employee_name as scheduled_employee,
            s.canonical_name,
            a.employee_name as arrived_employee,
            a.arrival_time
        FROM schedule s
        LEFT JOIN arrivals a 
            ON s.shift_date = a.shift_date
            AND s.canonical_name = a.canonical_name
        ORDER BY s.shift_date, s.shift_type, s.role
        """
        
        df = self.duckdb.execute(sql).fetchdf()
        
        for _, row in df.iterrows():
            arrival_str = str(row['arrival_time']) if pd.notna(row['arrival_time']) else 'N/A'
            emp_canonical = row['canonical_name']
            has_any_shift_data = emp_canonical in employees_with_data
            showed_up = pd.notna(row['arrived_employee'])
            
            if showed_up:
                status = 'OK'
                evidence = (
                    f"Employee {row['scheduled_employee']} arrived at {arrival_str} "
                    f"for {row['role']} role on {row['shift_date']}."
                )
            elif has_any_shift_data:
                # We have shift data for this employee, but they didn't show for this date
                status = 'NO_SHOW'
                evidence = (
                    f"Employee {row['scheduled_employee']} was scheduled as {row['role']} "
                    f"for {row['shift_type']} shift on {row['shift_date']} but did not clock in. "
                    f"(Shift data exists for this employee on other dates)"
                )
            else:
                # No shift data at all for this employee
                status = 'NO_DATA'
                evidence = (
                    f"Employee {row['scheduled_employee']} was scheduled as {row['role']} "
                    f"for {row['shift_type']} shift on {row['shift_date']}. "
                    f"No shift file data available for this employee."
                )
            
            results.append({
                'date': row['shift_date'],
                'shift_type': row['shift_type'],
                'role': row['role'],
                'validation_type': 'schedule_attendance',
                'expected_count': 1,
                'actual_count': 1 if status == 'OK' else 0,
                'status': status,
                'employees_expected': row['scheduled_employee'],
                'employees_arrived': row['arrived_employee'] if pd.notna(row['arrived_employee']) else '',
                'doctor_name': '',
                'evidence': evidence,
            })
        
        return results
    
    def _validate_role_counts(
        self,
        schedule_df: pd.DataFrame,
        shifts_df: pd.DataFrame,
        timing_rules: Dict[str, Dict],
    ) -> List[Dict]:
        """Validate role counts per shift against limits."""
        results = []
        
        # Get unique dates and shift types from schedule
        sql = """
        SELECT DISTINCT shift_date, shift_type
        FROM schedule
        WHERE shift_date IS NOT NULL
        ORDER BY shift_date, shift_type
        """
        date_shifts = self.duckdb.execute(sql).fetchdf()
        
        for _, date_row in date_shifts.iterrows():
            shift_date = date_row['shift_date']
            shift_type = date_row['shift_type']
            
            # Count scheduled roles for this shift
            sql_scheduled = f"""
            SELECT role, COUNT(*) as count, 
                   STRING_AGG(employee_name, ', ') as employees
            FROM schedule
            WHERE shift_date = '{shift_date}' 
              AND shift_type = '{shift_type}'
            GROUP BY role
            """
            scheduled_roles = self.duckdb.execute(sql_scheduled).fetchdf()
            
            for _, role_row in scheduled_roles.iterrows():
                role = role_row['role']
                scheduled_count = role_row['count']
                scheduled_employees = role_row['employees']
                
                # Check timing rules (supports both old and new format)
                role_config = timing_rules.get(role, {})
                min_count = role_config.get('min_count') or role_config.get('min', 0) or 0
                max_count = role_config.get('max_count') or role_config.get('max')
                scope = role_config.get('scope', 'per_shift')
                
                # Skip roles with per_concurrent_doctor scope (handled separately)
                if scope == 'per_concurrent_doctor':
                    continue
                
                # Determine status
                status = 'OK'
                evidence = f"Role {role} on {shift_date} ({shift_type}): "
                
                if min_count and scheduled_count < min_count:
                    status = 'UNDERSTAFFED'
                    evidence += f"scheduled {scheduled_count}, minimum required {min_count}."
                elif max_count is not None and scheduled_count > max_count:
                    status = 'OVERSTAFFED'
                    evidence += f"scheduled {scheduled_count}, maximum allowed {max_count}."
                else:
                    evidence += f"scheduled {scheduled_count}, within limits."
                
                evidence += f" Employees: {scheduled_employees}"
                
                results.append({
                    'date': shift_date,
                    'shift_type': shift_type,
                    'role': role,
                    'validation_type': 'role_count',
                    'expected_count': min_count,
                    'actual_count': scheduled_count,
                    'status': status,
                    'employees_expected': scheduled_employees,
                    'employees_arrived': '',
                    'doctor_name': '',
                    'evidence': evidence,
                })
        
        return results
    
    def _validate_gastro_coverage(
        self,
        schedule_df: pd.DataFrame,
        shifts_df: pd.DataFrame,
        procedures_df: pd.DataFrame,
        timing_rules: Dict[str, Dict],
    ) -> List[Dict]:
        """
        Validate gastro nurse coverage for concurrent doctors.
        
        At any given time, the number of gastro nurses must be >= 
        the number of doctors performing procedures simultaneously.
        """
        results = []
        
        # Find concurrent doctors using interval overlap in DuckDB
        sql = """
        WITH 
        -- Convert times to minutes for easier comparison
        procedure_intervals AS (
            SELECT 
                procedure_date,
                doctor,
                start_time,
                end_time,
                EXTRACT(HOUR FROM start_time) * 60 + EXTRACT(MINUTE FROM start_time) as start_min,
                EXTRACT(HOUR FROM end_time) * 60 + EXTRACT(MINUTE FROM end_time) as end_min
            FROM gastro_procedures
            WHERE start_time IS NOT NULL AND end_time IS NOT NULL
        ),
        
        -- Find all unique time points where procedures start or end
        time_points AS (
            SELECT DISTINCT procedure_date, start_min as time_point FROM procedure_intervals
            UNION
            SELECT DISTINCT procedure_date, end_min as time_point FROM procedure_intervals
        ),
        
        -- Count concurrent doctors at each time point
        concurrent_at_points AS (
            SELECT 
                t.procedure_date,
                t.time_point,
                COUNT(DISTINCT p.doctor) as concurrent_doctors,
                STRING_AGG(DISTINCT p.doctor, ', ') as doctors
            FROM time_points t
            JOIN procedure_intervals p 
                ON t.procedure_date = p.procedure_date
                AND t.time_point >= p.start_min 
                AND t.time_point < p.end_min
            GROUP BY t.procedure_date, t.time_point
        ),
        
        -- Find max concurrent doctors per date
        max_concurrent AS (
            SELECT 
                procedure_date,
                MAX(concurrent_doctors) as max_concurrent_doctors,
                FIRST(doctors) as sample_doctors
            FROM concurrent_at_points
            GROUP BY procedure_date
        )
        
        SELECT * FROM max_concurrent
        ORDER BY procedure_date
        """
        
        try:
            concurrent_df = self.duckdb.execute(sql).fetchdf()
        except Exception as e:
            self.logger.warning(f"Error calculating concurrent doctors: {e}")
            return results
        
        # For each date, check if we have enough gastro nurses
        for _, row in concurrent_df.iterrows():
            proc_date = row['procedure_date']
            max_doctors = row['max_concurrent_doctors']
            sample_doctors = row['sample_doctors']
            
            # Count gastro nurses scheduled for this date
            sql_gastro = f"""
            SELECT 
                shift_type,
                COUNT(*) as gastro_count,
                STRING_AGG(employee_name, ', ') as gastro_nurses
            FROM schedule
            WHERE shift_date = '{proc_date}'
              AND role LIKE '%גסטרו%'
            GROUP BY shift_type
            """
            
            gastro_df = self.duckdb.execute(sql_gastro).fetchdf()
            
            # Check coverage for each shift type
            for shift_type in ['בוקר', 'ערב']:
                gastro_for_shift = gastro_df[gastro_df['shift_type'] == shift_type]
                
                if gastro_for_shift.empty:
                    gastro_count = 0
                    gastro_nurses = ''
                else:
                    gastro_count = gastro_for_shift.iloc[0]['gastro_count']
                    gastro_nurses = gastro_for_shift.iloc[0]['gastro_nurses']
                
                # Determine status
                if gastro_count >= max_doctors:
                    status = 'OK'
                    evidence = (
                        f"Gastro coverage OK on {proc_date} ({shift_type}): "
                        f"{gastro_count} gastro nurses for max {max_doctors} concurrent doctors. "
                        f"Nurses: {gastro_nurses}. Doctors: {sample_doctors}"
                    )
                else:
                    status = 'UNDERSTAFFED'
                    evidence = (
                        f"INSUFFICIENT gastro coverage on {proc_date} ({shift_type}): "
                        f"only {gastro_count} gastro nurses for {max_doctors} concurrent doctors! "
                        f"Nurses: {gastro_nurses}. Doctors: {sample_doctors}"
                    )
                
                results.append({
                    'date': proc_date,
                    'shift_type': shift_type,
                    'role': 'אח גסטרו',
                    'validation_type': 'gastro_concurrent_coverage',
                    'expected_count': max_doctors,
                    'actual_count': gastro_count,
                    'status': status,
                    'employees_expected': f'{max_doctors} (for {max_doctors} concurrent doctors)',
                    'employees_arrived': gastro_nurses,
                    'doctor_name': sample_doctors,
                    'evidence': evidence,
                })
        
        return results
    
    def _enrich_shifts_with_roles(
        self,
        schedule_df: pd.DataFrame,
        shifts_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Enrich shift arrivals with role information from schedule.
        
        Joins arrivals with schedule to determine what role each arriving
        employee was scheduled for. Matches on shift type (morning/evening)
        to handle employees who work double shifts.
        
        Returns:
            DataFrame with arrival + role information
        """
        if schedule_df.empty or shifts_df.empty:
            return pd.DataFrame()
        
        # Use DuckDB for efficient join
        # Match on date + employee + shift type to correctly handle double shifts
        sql = """
        SELECT 
            a.employee_name,
            a.employee_name_normalized,
            a.shift_date,
            a.arrival_time,
            a.departure_time,
            a.inferred_shift_type,
            s.role,
            s.shift_type as scheduled_shift_type
        FROM arrivals a
        LEFT JOIN schedule s 
            ON a.shift_date = s.shift_date
            AND a.canonical_name = s.canonical_name
            AND a.inferred_shift_type = s.shift_type
        WHERE a.arrival_time IS NOT NULL
        """
        
        try:
            result = self.duckdb.execute(sql).fetchdf()
            self.logger.info(f"Enriched {len(result)} arrivals with role information")
            return result
        except Exception as e:
            self.logger.warning(f"Error enriching shifts with roles: {e}")
            return pd.DataFrame()
    
    def _validate_arrival_timing(
        self,
        enriched_shifts: pd.DataFrame,
        procedures_df: pd.DataFrame,
        timing_rules: Dict[str, Dict],
        shift_time_boundary: str,
    ) -> List[Dict]:
        """
        Validate that employees arrived at appropriate times based on their role.
        
        Checks if employees arrived too early (before needed) based on:
        - Role's arrival_buffer_minutes setting
        - First relevant procedure time
        - Role's match_to setting (all_procedures, gastro_procedures, own_procedures)
        
        Returns:
            List of validation result dicts
        """
        results = []
        
        if enriched_shifts.empty or procedures_df.empty:
            return results
        
        # Parse boundary time
        boundary_time = parse_time(shift_time_boundary)
        boundary_minutes = boundary_time.hour * 60 + boundary_time.minute if boundary_time else 780  # 13:00
        
        # Group by date and shift type to find first/last procedures
        for _, arrival in enriched_shifts.iterrows():
            shift_date = arrival['shift_date']
            employee_name = arrival['employee_name']
            role = arrival.get('role')
            arrival_time = arrival['arrival_time']
            shift_type = arrival.get('scheduled_shift_type') or arrival.get('inferred_shift_type')
            
            if pd.isna(role) or not role or role == 'nan':
                continue  # Skip arrivals without known role
            
            # Get timing rules for this role
            role_config = timing_rules.get(role, {})
            arrival_buffer = role_config.get('arrival_buffer_minutes', 30)  # Can arrive up to X min before first procedure
            match_to = role_config.get('match_to', 'all_procedures')
            
            # Filter procedures for this date and shift type
            date_procedures = procedures_df[procedures_df['procedure_date'] == shift_date].copy()
            
            if date_procedures.empty:
                continue
            
            # Convert procedure times to minutes for filtering by shift type
            date_procedures['start_minutes'] = date_procedures['start_time'].apply(
                lambda t: t.hour * 60 + t.minute if pd.notna(t) else None
            )
            
            # Filter by shift type (morning < boundary, evening >= boundary)
            if shift_type == 'בוקר':
                date_procedures = date_procedures[date_procedures['start_minutes'] < boundary_minutes]
            elif shift_type == 'ערב':
                date_procedures = date_procedures[date_procedures['start_minutes'] >= boundary_minutes]
            
            if date_procedures.empty:
                continue
            
            # Apply match_to filter
            if match_to == 'gastro_procedures':
                date_procedures = date_procedures[
                    date_procedures['category'].str.contains('גסטרו', case=False, na=False)
                ]
            elif match_to == 'own_procedures':
                # Match to procedures where this employee is the treating staff
                emp_normalized = normalize_employee_name(employee_name)
                date_procedures = date_procedures[
                    date_procedures['doctor_normalized'] == emp_normalized
                ]
            # 'all_procedures' uses all procedures
            
            if date_procedures.empty:
                continue
            
            # Find first procedure for this shift
            first_procedure_time = date_procedures['start_time'].min()
            if pd.isna(first_procedure_time):
                continue
            
            # Calculate expected arrival time (first_procedure - buffer)
            first_proc_minutes = first_procedure_time.hour * 60 + first_procedure_time.minute
            expected_arrival_minutes = first_proc_minutes - arrival_buffer
            
            # Calculate actual arrival minutes
            arrival_minutes = arrival_time.hour * 60 + arrival_time.minute if pd.notna(arrival_time) else None
            
            if arrival_minutes is None:
                continue
            
            # Check if arrived too early
            # gap_minutes = how many minutes before first procedure they arrived
            gap_minutes = first_proc_minutes - arrival_minutes
            
            # Build match explanation
            match_explanation = self._get_match_explanation(match_to, len(date_procedures))
            
            if gap_minutes > arrival_buffer:
                # Arrived too early (more than allowed buffer before first procedure)
                minutes_too_early = gap_minutes - arrival_buffer
                status = 'EARLY_ARRIVAL'
                evidence = (
                    f"{employee_name} ({role}) arrived at {arrival_time} on {shift_date} ({shift_type}), "
                    f"but first procedure is at {first_procedure_time}. "
                    f"Can arrive up to {arrival_buffer} min before, but arrived {gap_minutes} min before ({minutes_too_early} min too early). "
                    f"[{match_explanation}]"
                )
            else:
                status = 'OK'
                evidence = (
                    f"{employee_name} ({role}) arrived at {arrival_time} on {shift_date} ({shift_type}), "
                    f"first procedure at {first_procedure_time}. "
                    f"Arrived {gap_minutes} min before (within {arrival_buffer} min allowed) - OK. "
                    f"[{match_explanation}]"
                )
            
            results.append({
                'date': shift_date,
                'shift_type': shift_type,
                'role': role,
                'validation_type': 'arrival_timing',
                'expected_count': arrival_buffer,
                'actual_count': gap_minutes,
                'status': status,
                'employees_expected': f'Arrive by {expected_arrival_minutes // 60:02d}:{expected_arrival_minutes % 60:02d}',
                'employees_arrived': employee_name,
                'doctor_name': '',
                'evidence': evidence,
            })
        
        return results
    
    def _validate_departure_timing(
        self,
        enriched_shifts: pd.DataFrame,
        procedures_df: pd.DataFrame,
        timing_rules: Dict[str, Dict],
        shift_time_boundary: str,
    ) -> List[Dict]:
        """
        Validate that employees didn't leave too early based on their role.
        
        Checks if employees left before coverage period ended based on:
        - Role's departure_buffer_minutes setting
        - Last relevant procedure end time
        - Role's match_to setting
        
        Returns:
            List of validation result dicts
        """
        results = []
        
        if enriched_shifts.empty or procedures_df.empty:
            return results
        
        # Parse boundary time
        boundary_time = parse_time(shift_time_boundary)
        boundary_minutes = boundary_time.hour * 60 + boundary_time.minute if boundary_time else 780
        
        # Only process arrivals that have departure times
        departures = enriched_shifts[enriched_shifts['departure_time'].notna()]
        
        for _, arrival in departures.iterrows():
            shift_date = arrival['shift_date']
            employee_name = arrival['employee_name']
            role = arrival.get('role')
            departure_time = arrival['departure_time']
            shift_type = arrival.get('scheduled_shift_type') or arrival.get('inferred_shift_type')
            
            if pd.isna(role) or not role or role == 'nan':
                continue
            
            if pd.isna(departure_time):
                continue
            
            # Get timing rules for this role
            role_config = timing_rules.get(role, {})
            departure_buffer = role_config.get('departure_buffer_minutes', 0)
            departure_max = role_config.get('departure_max_minutes')  # None = no max limit
            match_to = role_config.get('match_to', 'all_procedures')
            
            # Filter procedures for this date and shift type
            date_procedures = procedures_df[procedures_df['procedure_date'] == shift_date].copy()
            
            if date_procedures.empty:
                continue
            
            # Convert procedure times to minutes for filtering by shift type
            date_procedures['end_minutes'] = date_procedures['end_time'].apply(
                lambda t: t.hour * 60 + t.minute if pd.notna(t) else None
            )
            date_procedures['start_minutes'] = date_procedures['start_time'].apply(
                lambda t: t.hour * 60 + t.minute if pd.notna(t) else None
            )
            
            # Filter by shift type
            if shift_type == 'בוקר':
                date_procedures = date_procedures[date_procedures['start_minutes'] < boundary_minutes]
            elif shift_type == 'ערב':
                date_procedures = date_procedures[date_procedures['start_minutes'] >= boundary_minutes]
            
            if date_procedures.empty:
                continue
            
            # Apply match_to filter
            if match_to == 'gastro_procedures':
                date_procedures = date_procedures[
                    date_procedures['category'].str.contains('גסטרו', case=False, na=False)
                ]
            elif match_to == 'own_procedures':
                emp_normalized = normalize_employee_name(employee_name)
                date_procedures = date_procedures[
                    date_procedures['doctor_normalized'] == emp_normalized
                ]
            
            if date_procedures.empty:
                continue
            
            # Find last procedure end time for this shift
            last_procedure_end = date_procedures['end_time'].max()
            if pd.isna(last_procedure_end):
                continue
            
            # Calculate allowed departure time (last_procedure + buffer)
            last_proc_minutes = last_procedure_end.hour * 60 + last_procedure_end.minute
            allowed_departure_minutes = last_proc_minutes + departure_buffer
            
            # Calculate actual departure minutes
            departure_minutes = departure_time.hour * 60 + departure_time.minute
            
            # Build match explanation
            match_explanation = self._get_match_explanation(match_to, len(date_procedures))
            
            # Calculate max allowed departure time if configured
            max_departure_minutes = last_proc_minutes + departure_max if departure_max else None
            
            # Calculate how long after procedure they stayed
            minutes_after_procedure = departure_minutes - last_proc_minutes
            
            # Check if left too early (before minimum buffer)
            if departure_minutes < allowed_departure_minutes:
                minutes_too_early = allowed_departure_minutes - departure_minutes
                status = 'EARLY_DEPARTURE'
                allowed_time = f"{allowed_departure_minutes // 60:02d}:{allowed_departure_minutes % 60:02d}"
                evidence = (
                    f"{employee_name} ({role}) left at {departure_time} on {shift_date} ({shift_type}), "
                    f"but last procedure ended at {last_procedure_end}. "
                    f"With {departure_buffer} min buffer, should stay until {allowed_time}. "
                    f"Left {minutes_too_early} min too early. "
                    f"[{match_explanation}]"
                )
            # Check if stayed too late (after maximum allowed)
            elif max_departure_minutes and departure_minutes > max_departure_minutes:
                minutes_too_late = departure_minutes - max_departure_minutes
                status = 'LATE_DEPARTURE'
                max_time = f"{max_departure_minutes // 60:02d}:{max_departure_minutes % 60:02d}"
                evidence = (
                    f"{employee_name} ({role}) left at {departure_time} on {shift_date} ({shift_type}). "
                    f"Last procedure ended at {last_procedure_end}. "
                    f"Should leave within {departure_max} min (by {max_time}), but stayed {minutes_too_late} min too late. "
                    f"[{match_explanation}]"
                )
            else:
                status = 'OK'
                if departure_max:
                    evidence = (
                        f"{employee_name} ({role}) left at {departure_time} on {shift_date} ({shift_type}). "
                        f"Last procedure ended at {last_procedure_end}. "
                        f"Left {minutes_after_procedure} min after (within 0-{departure_max} min window) - OK. "
                        f"[{match_explanation}]"
                    )
                else:
                    evidence = (
                        f"{employee_name} ({role}) left at {departure_time} on {shift_date} ({shift_type}). "
                        f"Last procedure ended at {last_procedure_end}. "
                        f"Departure is {departure_buffer} min or more after last procedure - OK. "
                        f"[{match_explanation}]"
                    )
            
            results.append({
                'date': shift_date,
                'shift_type': shift_type,
                'role': role,
                'validation_type': 'departure_timing',
                'expected_count': departure_buffer,
                'actual_count': departure_minutes - last_proc_minutes,
                'status': status,
                'employees_expected': f'Stay until {allowed_departure_minutes // 60:02d}:{allowed_departure_minutes % 60:02d}',
                'employees_arrived': employee_name,
                'doctor_name': '',
                'evidence': evidence,
            })
        
        return results
    
    def _validate_double_shifts(
        self,
        enriched_shifts: pd.DataFrame,
        timing_rules: Dict[str, Dict],
    ) -> List[Dict]:
        """
        Detect employees who worked both morning and evening shifts on the same day.
        
        Only flags roles where notify_double_shift is True in timing_rules.
        
        Returns:
            List of validation result dicts
        """
        results = []
        
        if enriched_shifts.empty:
            return results
        
        # Get roles that have notify_double_shift enabled
        notifiable_roles = set()
        for role, config in timing_rules.items():
            if config.get('notify_double_shift', False):
                notifiable_roles.add(role)
        
        if not notifiable_roles:
            return results
        
        # Group by date and employee to find double shifts
        # We look for employees who appear in multiple shift types on the same day
        shifts_with_roles = enriched_shifts[enriched_shifts['role'].notna() & (enriched_shifts['role'] != 'nan')]
        
        if shifts_with_roles.empty:
            return results
        
        # Group by date and normalized employee name
        # Use canonical_name if available, fall back to employee_name_normalized
        group_col = 'canonical_name' if 'canonical_name' in shifts_with_roles.columns else 'employee_name_normalized'
        grouped = shifts_with_roles.groupby(['shift_date', group_col])
        
        for (date, emp_norm), group in grouped:
            # Get unique shift types for this employee on this date
            shift_types = group['inferred_shift_type'].dropna().unique().tolist()
            
            # Check if they worked both morning and evening
            if 'בוקר' in shift_types and 'ערב' in shift_types:
                # Get the roles they worked in each shift
                morning_data = group[group['inferred_shift_type'] == 'בוקר']
                evening_data = group[group['inferred_shift_type'] == 'ערב']
                
                morning_roles = morning_data['role'].dropna().unique().tolist()
                evening_roles = evening_data['role'].dropna().unique().tolist()
                
                # Check if any of their roles have notify_double_shift enabled
                all_roles = set(morning_roles + evening_roles)
                notifiable = all_roles & notifiable_roles
                
                if notifiable:
                    # Get employee display name (use first occurrence)
                    employee_name = group['employee_name'].iloc[0]
                    
                    # Get arrival and departure times for evidence
                    morning_arrival = morning_data['arrival_time'].min() if not morning_data.empty else None
                    morning_departure = morning_data['departure_time'].max() if not morning_data.empty else None
                    evening_arrival = evening_data['arrival_time'].min() if not evening_data.empty else None
                    evening_departure = evening_data['departure_time'].max() if not evening_data.empty else None
                    
                    # Build evidence message
                    evidence = (
                        f"{employee_name} worked DOUBLE SHIFT on {date}: "
                        f"Morning ({', '.join(morning_roles)}) {morning_arrival or '?'}-{morning_departure or '?'}, "
                        f"Evening ({', '.join(evening_roles)}) {evening_arrival or '?'}-{evening_departure or '?'}."
                    )
                    
                    # Add single result per employee per date (combine all roles)
                    results.append({
                        'date': date,
                        'shift_type': ', '.join(sorted(shift_types)),
                        'role': ', '.join(sorted(all_roles)),
                        'validation_type': 'double_shift',
                        'expected_count': 1,
                        'actual_count': 2,
                        'status': 'DOUBLE_SHIFT',
                        'employees_expected': 'Single shift',
                        'employees_arrived': employee_name,
                        'doctor_name': '',
                        'evidence': evidence,
                    })
        
        return results
    
    def _get_match_explanation(self, match_to: str, procedure_count: int) -> str:
        """
        Get human-readable explanation of why procedures were matched to this role.
        
        Args:
            match_to: The match_to setting from timing rules
            procedure_count: Number of procedures matched
            
        Returns:
            Explanation string
        """
        if match_to == 'all_procedures':
            return f"Role covers ALL {procedure_count} procedures in this shift"
        elif match_to == 'gastro_procedures':
            return f"Role covers {procedure_count} GASTRO procedures (category contains 'גסטרו')"
        elif match_to == 'own_procedures':
            return f"Matched to {procedure_count} procedures where employee is treating staff"
        else:
            return f"Matched to {procedure_count} procedures (match_to={match_to})"
    
    def _get_empty_result_df(self) -> pd.DataFrame:
        """Get an empty result DataFrame with correct columns."""
        return pd.DataFrame(columns=[
            "date", "shift_type", "role", "validation_type",
            "expected_count", "actual_count", "status",
            "employees_expected", "employees_arrived",
            "doctor_name", "evidence"
        ])
    
    def _find_column(self, df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
        """Find first matching column from candidates."""
        cols = list(df.columns)
        for candidate in candidates:
            if candidate in cols:
                return candidate
        return None


# Register the unified block with new name
@BlockRegistry.register(
    name="staff_timing_validation",
    inputs=[
        {"name": "data", "ontology": DataType.CLASSIFIED_DATA, "required": True}
    ],
    outputs=[
        {"name": "result", "ontology": DataType.INSIGHT_RESULT}
    ],
    parameters=[
        {
            "name": "timing_rules",
            "type": "object",
            "default": DEFAULT_TIMING_RULES,
            "description": "Timing rules per role. Each role can have: min_count, max_count, scope, arrival_buffer_minutes, departure_buffer_minutes, match_to"
        },
        {
            "name": "shift_time_boundary",
            "type": "string",
            "default": "13:00",
            "description": "Time boundary between morning and evening shifts"
        },
        {
            "name": "morning_shift_start",
            "type": "string",
            "default": "06:00",
            "description": "Expected morning shift start time"
        },
        {
            "name": "evening_shift_end",
            "type": "string",
            "default": "21:00",
            "description": "Expected evening shift end time"
        },
    ],
    block_class=StaffTimingValidationBlock,
    description="Unified staff timing validation: role counts, arrival/departure timing, gastro nurse coverage",
)
def staff_timing_validation(ctx: BlockContext) -> Dict[str, str]:
    """Validate staff timing and coverage against schedule and business rules."""
    return StaffTimingValidationBlock(ctx).run()


# Keep backward-compatible alias
@BlockRegistry.register(
    name="staff_coverage_validation",
    inputs=[
        {"name": "data", "ontology": DataType.CLASSIFIED_DATA, "required": True}
    ],
    outputs=[
        {"name": "result", "ontology": DataType.INSIGHT_RESULT}
    ],
    parameters=[
        {
            "name": "timing_rules",
            "type": "object",
            "default": DEFAULT_TIMING_RULES,
            "description": "Timing rules per role (backward compatible alias)"
        },
        {
            "name": "shift_time_boundary",
            "type": "string",
            "default": "13:00",
            "description": "Time boundary between morning and evening shifts"
        },
    ],
    block_class=StaffTimingValidationBlock,
    description="[DEPRECATED] Use staff_timing_validation instead",
)
def staff_coverage_validation(ctx: BlockContext) -> Dict[str, str]:
    """Backward compatible alias for staff_timing_validation."""
    return StaffTimingValidationBlock(ctx).run()

