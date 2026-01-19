"""
Staff Coverage Validation Block - Validate staff roles against schedule and business rules.

Validates:
1. Actual arrivals match scheduled role assignments
2. Business rules like: 1 recovery nurse per shift, 1 gastro nurse per concurrent doctor
3. Adequate coverage during all procedure times

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


def normalize_date(date_val) -> Optional[str]:
    """
    Normalize date to YYYY-MM-DD format.
    
    Handles multiple formats:
    - MM/DD/YYYY (American) - e.g., 12/1/2025 = December 1, 2025
    - DD/MM/YYYY (European) - e.g., 25/12/2025 = December 25, 2025
    - YYYY-MM-DD (ISO) - already normalized
    
    Disambiguation: if first number > 12, it must be a day (European format).
    Otherwise, assume American format (month/day/year).
    """
    if pd.isna(date_val):
        return None
    date_str = str(date_val).strip()
    
    # Try YYYY-MM-DD format (already normalized)
    match = re.match(r'(\d{4})-(\d{1,2})-(\d{1,2})', date_str)
    if match:
        year, month, day = match.groups()
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
        # Both <= 12: assume American format (MM/DD/YYYY) as default
        # This is common in Israeli business software exports
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


# Default role limits configuration
DEFAULT_ROLE_LIMITS = {
    "אח התאוששות": {"min": 1, "max": 1, "scope": "per_shift"},
    "אח התאוששות ערב": {"min": 1, "max": 1, "scope": "per_shift"},
    "אח גסטרו": {"min": 1, "max": None, "scope": "per_concurrent_doctor"},
    "כע עזר": {"min": 1, "max": None, "scope": "per_shift"},
    "כע עזר ערב": {"min": 1, "max": None, "scope": "per_shift"},
    "כח עזר": {"min": 1, "max": None, "scope": "per_shift"},
}


class StaffCoverageValidationBlock(BaseBlock):
    """
    Staff coverage validation block.
    
    Validates:
    1. Actual staff arrivals match scheduled role assignments
    2. Business rules (configurable role counts per shift/doctor)
    3. Gastro nurse coverage for concurrent doctors
    
    Uses DuckDB SQL for efficient time interval analysis.
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
    PROC_TIME_CANDIDATES = ['treatment_hours', 'שעות טיפול בפועל', 'treatment_time', 'procedure_time', 'time_range']
    PROC_DATE_CANDIDATES = ['treatment_date', 'תאריך טיפול', 'תאריך', 'date']
    PROC_STAFF_CANDIDATES = ['treating_staff', 'צוות מטפל', 'staff_name', 'provider', 'doctor']
    PROC_CAT_CANDIDATES = ['treatment_category', 'קטגורית טיפול', 'category']
    
    def run(self) -> Dict[str, str]:
        """
        Validate staff coverage against schedule and business rules.
        
        Returns:
            Dict with 'result' key containing S3 URI of validation results
        """
        # Get configuration parameters
        role_limits = self.get_param("role_limits", DEFAULT_ROLE_LIMITS)
        shift_time_boundary = self.get_param("shift_time_boundary", "13:00")
        morning_shift_start = self.get_param("morning_shift_start", "06:00")
        evening_shift_end = self.get_param("evening_shift_end", "21:00")
        
        self.logger.info("Running staff coverage validation")
        self.logger.info(f"Role limits: {role_limits}")
        
        # Load classified data
        classified_data = self.load_classified_data("data")
        
        # Check for required document types
        missing_types = [dt for dt in self.REQUIRED_DOC_TYPES if dt not in classified_data]
        if missing_types:
            self.logger.warning(
                f"⚠️  Skipping staff_coverage: missing required doc types: {missing_types}. "
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
        if procedures_df is not None and len(procedures_df) > 0:
            procedures_prepared = self._prepare_procedures(procedures_df)
        
        # Run validation
        results = self._validate_coverage(
            schedule_prepared,
            shifts_prepared,
            procedures_prepared,
            role_limits,
            shift_time_boundary,
        )
        
        # Save results
        result_uri = self.save_to_s3("result", results)
        
        # Log summary
        issues = results[results['status'] != 'OK']
        self.logger.info(
            f"Staff coverage validation complete: "
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
    
    def _prepare_procedures(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare procedures DataFrame with time intervals for concurrent doctor detection."""
        result = pd.DataFrame()
        
        # Find columns
        time_col = self._find_column(df, self.PROC_TIME_CANDIDATES)
        date_col = self._find_column(df, self.PROC_DATE_CANDIDATES)
        staff_col = self._find_column(df, self.PROC_STAFF_CANDIDATES)
        cat_col = self._find_column(df, self.PROC_CAT_CANDIDATES)
        
        if not time_col or not staff_col:
            self.logger.warning(
                f"Could not find required procedure columns. "
                f"Found: time={time_col}, staff={staff_col}"
            )
            return pd.DataFrame()
        
        # Parse time ranges
        time_ranges = df[time_col].apply(parse_time_range)
        result['start_time'] = time_ranges.apply(lambda x: x[0])
        result['end_time'] = time_ranges.apply(lambda x: x[1])
        
        # Normalize date
        if date_col:
            result['procedure_date'] = df[date_col].apply(
                lambda x: normalize_date(x) if pd.notna(x) else None
            )
        else:
            result['procedure_date'] = None
        
        # Doctor name
        result['doctor'] = df[staff_col].astype(str)
        
        # Category (for filtering gastro procedures)
        if cat_col:
            result['category'] = df[cat_col].astype(str)
        else:
            result['category'] = 'unknown'
        
        # Filter to only gastro procedures (where gastro nurse is needed)
        result = result[result['category'].str.contains('גסטרו', case=False, na=False)]
        
        # Filter out invalid rows
        result = result[result['start_time'].notna()]
        result = result[result['procedure_date'].notna()]
        result = result[result['doctor'].str.strip() != '']
        
        self.logger.info(f"Prepared {len(result)} gastro procedures for concurrent doctor detection")
        return result
    
    def _validate_coverage(
        self,
        schedule_df: pd.DataFrame,
        shifts_df: pd.DataFrame,
        procedures_df: Optional[pd.DataFrame],
        role_limits: Dict[str, Dict],
        shift_time_boundary: str,
    ) -> pd.DataFrame:
        """
        Validate staff coverage using DuckDB for efficient analysis.
        
        Returns DataFrame with validation results.
        """
        results = []
        
        if schedule_df.empty or shifts_df.empty:
            self.logger.warning("Empty schedule or shifts data - cannot validate")
            return self._get_empty_result_df()
        
        # Register tables with DuckDB
        self.duckdb.register("schedule", schedule_df)
        self.duckdb.register("arrivals", shifts_df)
        
        # 1. Validate schedule vs actual arrivals (who showed up?)
        schedule_validation = self._validate_schedule_attendance(schedule_df, shifts_df)
        results.extend(schedule_validation)
        
        # 2. Validate role counts per shift
        role_count_validation = self._validate_role_counts(
            schedule_df, shifts_df, role_limits
        )
        results.extend(role_count_validation)
        
        # 3. Validate gastro nurse coverage for concurrent doctors
        if procedures_df is not None and not procedures_df.empty:
            self.duckdb.register("procedures", procedures_df)
            gastro_validation = self._validate_gastro_coverage(
                schedule_df, shifts_df, procedures_df, role_limits
            )
            results.extend(gastro_validation)
        
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
        """Check if scheduled employees actually arrived."""
        results = []
        
        # Use DuckDB for efficient matching
        sql = """
        SELECT 
            s.shift_date,
            s.shift_type,
            s.role,
            s.employee_name as scheduled_employee,
            a.employee_name as arrived_employee,
            a.arrival_time,
            CASE 
                WHEN a.employee_name IS NOT NULL THEN 'OK'
                ELSE 'NO_SHOW'
            END as status
        FROM schedule s
        LEFT JOIN arrivals a 
            ON s.shift_date = a.shift_date
            AND s.employee_name_normalized = a.employee_name_normalized
        ORDER BY s.shift_date, s.shift_type, s.role
        """
        
        df = self.duckdb.execute(sql).fetchdf()
        
        for _, row in df.iterrows():
            arrival_str = str(row['arrival_time']) if pd.notna(row['arrival_time']) else 'N/A'
            
            if row['status'] == 'NO_SHOW':
                evidence = (
                    f"Employee {row['scheduled_employee']} was scheduled as {row['role']} "
                    f"for {row['shift_type']} shift on {row['shift_date']} but did not arrive."
                )
            else:
                evidence = (
                    f"Employee {row['scheduled_employee']} arrived at {arrival_str} "
                    f"for {row['role']} role on {row['shift_date']}."
                )
            
            results.append({
                'date': row['shift_date'],
                'shift_type': row['shift_type'],
                'role': row['role'],
                'validation_type': 'schedule_attendance',
                'expected_count': 1,
                'actual_count': 1 if row['status'] == 'OK' else 0,
                'status': row['status'],
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
        role_limits: Dict[str, Dict],
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
                
                # Check role limits
                role_config = role_limits.get(role, {})
                min_count = role_config.get('min', 0)
                max_count = role_config.get('max')
                scope = role_config.get('scope', 'per_shift')
                
                # Skip roles with per_concurrent_doctor scope (handled separately)
                if scope == 'per_concurrent_doctor':
                    continue
                
                # Determine status
                status = 'OK'
                evidence = f"Role {role} on {shift_date} ({shift_type}): "
                
                if scheduled_count < min_count:
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
                    'employees_arrived': '',  # Would need to cross-reference with arrivals
                    'doctor_name': '',
                    'evidence': evidence,
                })
        
        return results
    
    def _validate_gastro_coverage(
        self,
        schedule_df: pd.DataFrame,
        shifts_df: pd.DataFrame,
        procedures_df: pd.DataFrame,
        role_limits: Dict[str, Dict],
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
            FROM procedures
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


# Register the block
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
            "name": "role_limits",
            "type": "object",
            "default": DEFAULT_ROLE_LIMITS,
            "description": "Role count limits per shift/doctor. Scope can be: per_shift, per_concurrent_doctor"
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
    block_class=StaffCoverageValidationBlock,
    description="Validate staff coverage against schedule and business rules (role counts, gastro nurse per concurrent doctor)",
)
def staff_coverage_validation(ctx: BlockContext) -> Dict[str, str]:
    """Validate staff coverage against schedule and business rules."""
    return StaffCoverageValidationBlock(ctx).run()

