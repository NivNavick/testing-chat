"""
Early Arrival Block - Detect early employee arrivals.

Matches employee arrivals to medical procedures to determine
which employees arrived earlier than necessary.

DEPRECATED: This pandas-based block has been replaced by a SQL version.
Use the sql_insight block with insight_name='early_arrival' instead.

Migration path:
  OLD (this file):
    handler: early_arrival_matcher
    
  NEW (SQL):
    handler: sql_insight
    parameters:
      insight_name: early_arrival
      max_early_minutes: 30

The SQL version is defined in:
  csv_analyzer/insights/definitions/early_arrival.yaml
"""

import logging
import re
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from csv_analyzer.workflows.block import BlockRegistry
from csv_analyzer.workflows.ontology import DataType
from csv_analyzer.workflows.base_block import BaseBlock, BlockContext

logger = logging.getLogger(__name__)


def parse_time(time_str: str) -> Optional[datetime]:
    """Parse a time string to datetime (date part is arbitrary)."""
    if pd.isna(time_str) or not time_str:
        return None
    
    time_str = str(time_str).strip()
    
    # Remove common prefixes like "* "
    time_str = re.sub(r'^\*\s*', '', time_str)
    
    # Try various formats
    for fmt in ["%H:%M", "%H:%M:%S", "%I:%M %p", "%I:%M:%S %p"]:
        try:
            return datetime.strptime(time_str, fmt)
        except ValueError:
            continue
    
    # Handle datetime strings (extract time part)
    if " " in time_str:
        time_part = time_str.split(" ")[-1]
        return parse_time(time_part)
    
    return None


def normalize_date(date_val) -> Optional[str]:
    """Normalize date to YYYY-MM-DD format."""
    if pd.isna(date_val):
        return None
    date_str = str(date_val).strip()
    
    # Try DD/MM/YYYY format
    match = re.match(r'(\d{1,2})/(\d{1,2})/(\d{4})', date_str)
    if match:
        day, month, year = match.groups()
        return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
    
    # Try YYYY-MM-DD format (already normalized)
    match = re.match(r'(\d{4})-(\d{1,2})-(\d{1,2})', date_str)
    if match:
        year, month, day = match.groups()
        return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
    
    return date_str


def time_diff_minutes(t1: datetime, t2: datetime) -> int:
    """Calculate difference in minutes between two times (t2 - t1)."""
    delta = t2 - t1
    return int(delta.total_seconds() / 60)


def _requires_multi_staff(treatment_name: str, multi_staff_procedures: Optional[List[Dict]]) -> int:
    """
    Check if a procedure requires multiple staff members.
    
    Args:
        treatment_name: Name of the treatment/procedure
        multi_staff_procedures: List of {identifier: str, required_staff: int}
    
    Returns:
        Number of staff required (1 if not in multi_staff list)
    """
    if not multi_staff_procedures or not treatment_name:
        return 1
    
    for config in multi_staff_procedures:
        identifier = config.get("identifier", "")
        if identifier and identifier.lower() in treatment_name.lower():
            return config.get("required_staff", 1)
    
    return 1


def match_arrivals_to_procedures(
    arrivals: List[Dict],
    procedures: List[Dict],
    max_early_minutes: int = 30,
    multi_staff_procedures: Optional[List[Dict]] = None,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Match arrivals to procedures using greedy algorithm.
    
    For each procedure (sorted by time), find the earliest unmatched arrival
    where: 0 <= (procedure_time - arrival_time) <= max_early_minutes
    
    Args:
        arrivals: List of arrival records
        procedures: List of procedure records
        max_early_minutes: Maximum minutes early to still count as OK
        multi_staff_procedures: List of procedures requiring multiple staff
            Format: [{"identifier": "גסטרוסקופיה", "required_staff": 2}, ...]
    """
    arrivals_sorted = sorted(arrivals, key=lambda x: x["arrival_time"])
    procedures_sorted = sorted(procedures, key=lambda x: x["procedure_time"])
    
    # STEP 1: Find all eligible arrivals for each procedure
    procedure_candidates = {}  # proc_id -> [(arrival_index, arrival_dict, time_diff)]
    
    for proc in procedures_sorted:
        proc_time = proc["procedure_time"]
        proc_name = proc.get("treatment_name", "")
        proc_id = f"{proc_time.strftime('%H:%M')}_{proc_name}"
        
        # Find all arrivals that are eligible for this procedure (0-30 min before)
        candidates = []
        for i, arr in enumerate(arrivals_sorted):
            arr_time = arr["arrival_time"]
            diff = time_diff_minutes(arr_time, proc_time)
            
            if 0 <= diff <= max_early_minutes:
                candidates.append((i, arr, diff))
        
        procedure_candidates[proc_id] = candidates
    
    # STEP 2: Determine matches vs conflicts
    matched_indices = set()
    matched_arrivals = []
    conflicted_arrivals = []  # Arrivals with multiple candidates for same procedure
    
    for proc in procedures_sorted:
        proc_time = proc["procedure_time"]
        proc_name = proc.get("treatment_name", "")
        proc_category = proc.get("category", "")
        proc_staff = proc.get("treating_staff", "")
        proc_id = f"{proc_time.strftime('%H:%M')}_{proc_name}"
        
        candidates = procedure_candidates.get(proc_id, [])
        
        # Skip already matched arrivals
        available_candidates = [(i, arr, diff) for (i, arr, diff) in candidates if i not in matched_indices]
        
        if not available_candidates:
            continue
        
        # Determine how many staff this procedure requires
        required_staff = _requires_multi_staff(proc_name, multi_staff_procedures)
        
        # CONFLICT DETECTION: If more arrivals than required staff -> mark all as UNCERTAIN
        if len(available_candidates) > required_staff:
            # Multiple employees for same procedure - can't determine who worked it
            for i, arr, diff in available_candidates:
                matched_indices.add(i)
                arr_copy = arr.copy()
                arr_copy["matched_procedure_time"] = proc_time.strftime("%H:%M")
                arr_copy["matched_treatment"] = proc_name
                arr_copy["treating_staff"] = proc_staff
                arr_copy["minutes_before_procedure"] = diff
                arr_copy["minutes_early"] = 0
                arr_copy["status"] = "UNCERTAIN"
                
                loc = arr.get('location')
                location_str = f" at {loc}" if loc and loc != 'UNKNOWN' and loc != 'None' and str(loc).lower() != 'nan' else ""
                
                procedure_detail = f"{proc_name}" if proc_name else "Unknown procedure"
                if proc_category:
                    procedure_detail = f"{proc_category} - {procedure_detail}"
                
                arr_copy["evidence"] = (
                    f"Arrived at {arr['arrival_time'].strftime('%H:%M')}{location_str}, "
                    f"{diff} min before procedure at {proc_time.strftime('%H:%M')}. "
                    f"Procedure: '{procedure_detail}'. "
                    f"CONFLICT: {len(available_candidates)} employees arrived for this procedure "
                    f"(requires {required_staff} staff). Cannot determine who actually worked it. "
                    f"Other candidates: {', '.join([c[1].get('employee_name', 'Unknown') for c in available_candidates if c[0] != i][:3])}."
                )
                conflicted_arrivals.append(arr_copy)
        else:
            # Exact match or under-staffed: Match up to required_staff (earliest first)
            for idx, (i, arr, diff) in enumerate(available_candidates[:required_staff]):
                matched_indices.add(i)
                arr_copy = arr.copy()
                arr_copy["matched_procedure_time"] = proc_time.strftime("%H:%M")
                arr_copy["matched_treatment"] = proc_name
                arr_copy["treating_staff"] = proc_staff
                arr_copy["minutes_before_procedure"] = diff
                arr_copy["status"] = "OK"
                
                loc = arr.get('location')
                location_str = f" at {loc}" if loc and loc != 'UNKNOWN' and loc != 'None' and str(loc).lower() != 'nan' else ""
                treatment_str = f" Treatment: {proc_name}." if proc_name else ""
                category_str = f" Category: {proc_category}." if proc_category else ""
                
                # Add staff position if multi-staff
                staff_position_str = ""
                if required_staff > 1:
                    staff_position_str = f" (Staff {idx + 1}/{required_staff})"
                
                arr_copy["evidence"] = (
                    f"Arrived at {arr['arrival_time'].strftime('%H:%M')}{location_str}, "
                    f"covered procedure at {proc_time.strftime('%H:%M')} "
                    f"({diff} min before){staff_position_str}.{treatment_str}{category_str}"
                )
                matched_arrivals.append(arr_copy)
    
    # Add conflicted arrivals to matched (they've been processed)
    matched_arrivals.extend(conflicted_arrivals)
    
    unmatched_arrivals = []
    for i, arr in enumerate(arrivals_sorted):
        if i not in matched_indices:
            arr_copy = arr.copy()
            arr_time = arr["arrival_time"]
            loc = arr.get('location')
            location_str = f" at {loc}" if loc and loc != 'UNKNOWN' and loc != 'None' and str(loc).lower() != 'nan' else ""
            
            if procedures_sorted:
                # Find the nearest FUTURE procedure (arrival must be before procedure)
                future_procs = [p for p in procedures_sorted if p["procedure_time"] > arr_time]
                
                if not future_procs:
                    # Arrived after all procedures - no work available
                    arr_copy["matched_procedure_time"] = None
                    arr_copy["matched_treatment"] = None
                    arr_copy["treating_staff"] = None
                    arr_copy["minutes_before_procedure"] = None
                    arr_copy["minutes_early"] = None
                    arr_copy["status"] = "NO_PROCEDURES"
                    arr_copy["evidence"] = (
                        f"Arrived at {arr_time.strftime('%H:%M')}{location_str}, "
                        f"after all procedures for this date/location were completed."
                    )
                    unmatched_arrivals.append(arr_copy)
                    continue
                
                nearest_proc = min(
                    future_procs,
                    key=lambda p: time_diff_minutes(arr_time, p["procedure_time"])
                )
                nearest_time = nearest_proc["procedure_time"]
                nearest_name = nearest_proc.get("treatment_name", "")
                nearest_category = nearest_proc.get("category", "")
                nearest_staff = nearest_proc.get("treating_staff", "")
                gap = time_diff_minutes(arr_time, nearest_time)
                
                treatment_str = f" Nearest treatment: {nearest_name}." if nearest_name else ""
                category_str = f" Category: {nearest_category}." if nearest_category else ""
                
                if gap > max_early_minutes:
                    minutes_early = gap - max_early_minutes
                    arr_copy["matched_procedure_time"] = nearest_time.strftime("%H:%M")
                    arr_copy["matched_treatment"] = nearest_name
                    arr_copy["treating_staff"] = nearest_staff
                    arr_copy["minutes_before_procedure"] = gap
                    arr_copy["minutes_early"] = minutes_early
                    arr_copy["status"] = "EARLY"
                    arr_copy["evidence"] = (
                        f"Arrived at {arr_time.strftime('%H:%M')}{location_str}, "
                        f"nearest procedure at {nearest_time.strftime('%H:%M')} "
                        f"({gap} min gap > {max_early_minutes} allowed). "
                        f"Arrived {minutes_early} min early.{treatment_str}{category_str}"
                    )
                else:
                    # Check if this procedure requires multiple staff
                    required_staff = _requires_multi_staff(nearest_name, multi_staff_procedures)
                    
                    if required_staff > 1:
                        # Multi-staff procedure - mark as OK (secondary staff)
                        arr_copy["matched_procedure_time"] = nearest_time.strftime("%H:%M")
                        arr_copy["matched_treatment"] = nearest_name
                        arr_copy["treating_staff"] = nearest_staff
                        arr_copy["minutes_before_procedure"] = gap
                        arr_copy["minutes_early"] = 0
                        arr_copy["status"] = "OK"
                        arr_copy["evidence"] = (
                            f"Arrived at {arr_time.strftime('%H:%M')}{location_str}, "
                            f"for procedure at {nearest_time.strftime('%H:%M')} "
                            f"({gap} min before). Secondary staff for multi-staff procedure "
                            f"(requires {required_staff} staff).{treatment_str}{category_str}"
                        )
                    else:
                        # Single-staff procedure but not matched - UNCERTAIN
                        arr_copy["matched_procedure_time"] = nearest_time.strftime("%H:%M")
                        arr_copy["matched_treatment"] = nearest_name
                        arr_copy["treating_staff"] = nearest_staff
                        arr_copy["minutes_before_procedure"] = gap
                        arr_copy["minutes_early"] = 0
                        arr_copy["status"] = "UNCERTAIN"
                        
                        # Enhanced evidence with procedure details
                        procedure_detail = f"{nearest_name}" if nearest_name else "Unknown procedure"
                        if nearest_category:
                            procedure_detail = f"{nearest_category} - {procedure_detail}"
                        
                        arr_copy["evidence"] = (
                            f"Arrived at {arr_time.strftime('%H:%M')}{location_str}, "
                            f"{gap} min before procedure at {nearest_time.strftime('%H:%M')}. "
                            f"Procedure: '{procedure_detail}'. "
                            f"Another employee was matched first (greedy algorithm). "
                            f"Cannot verify if this employee worked as secondary staff "
                            f"(nurses not recorded in treating_staff field). "
                            f"Could be: (1) Redundant (no work), or (2) Assisted primary staff."
                        )
            else:
                arr_copy["matched_procedure_time"] = None
                arr_copy["matched_treatment"] = None
                arr_copy["treating_staff"] = None
                arr_copy["minutes_before_procedure"] = None
                arr_copy["minutes_early"] = None
                arr_copy["status"] = "NO_PROCEDURES"
                arr_copy["evidence"] = (
                    f"Arrived at {arr_time.strftime('%H:%M')}{location_str}, "
                    f"but no procedures found for this date/location."
                )
            
            unmatched_arrivals.append(arr_copy)
    
    return matched_arrivals, unmatched_arrivals


class EarlyArrivalBlock(BaseBlock):
    """
    Early arrival detection block.
    
    Matches employee arrivals to medical procedures using greedy matching.
    Flags arrivals that are:
    - EARLY: Arrived too early (gap > max_early_minutes)
    - OK: Matched to procedure within threshold
    - EXCESS: All procedures already covered
    - NO_PROCEDURES: No procedures for date/location
    """
    
    # Required document types for this insight
    REQUIRED_DOC_TYPES = ["employee_shifts", "medical_actions"]
    
    def run(self) -> Dict[str, str]:
        """
        Match arrivals to procedures and detect early arrivals.
        
        Returns:
            Dict with 'result' key containing S3 URI of results DataFrame
            Returns 'skipped': True if required doc types are missing
        """
        max_early_minutes = self.get_param("max_early_minutes", 30)
        multi_staff_procedures = self.get_param("multi_staff_procedures", [])
        
        self.logger.info(f"Running early arrival matcher with max_early_minutes={max_early_minutes}")
        if multi_staff_procedures:
            self.logger.info(f"Multi-staff procedures configured: {len(multi_staff_procedures)} types")
        
        # Load classified data
        classified_data = self.load_classified_data("data")
        
        # Check for required document types
        missing_types = [dt for dt in self.REQUIRED_DOC_TYPES if dt not in classified_data]
        if missing_types:
            self.logger.warning(
                f"⚠️  Skipping early_arrival: missing required doc types: {missing_types}. "
                f"Available: {list(classified_data.keys())}"
            )
            # Return empty result with skipped flag
            empty_df = pd.DataFrame(columns=[
                "employee_name", "employee_id", "shift_date", "location",
                "arrival_time", "matched_procedure_time", "matched_treatment",
                "treating_staff", "minutes_early", "status", "evidence"
            ])
            result_uri = self.save_to_s3("result", empty_df)
            return {"result": result_uri, "skipped": True, "reason": f"Missing: {missing_types}"}
        
        # Find shifts and procedures DataFrames
        shifts_df = classified_data.get("employee_shifts")
        procedures_df = classified_data.get("medical_actions")
        
        self.logger.info(f"Loaded {len(shifts_df)} shifts and {len(procedures_df)} procedures")
        self.logger.info(f"Shifts columns: {list(shifts_df.columns)}")
        self.logger.info(f"Procedures columns: {list(procedures_df.columns)}")
        
        # Find columns using candidate lists (canonical first, then fallback)
        shift_time_col = self._find_column(shifts_df, ['shift_start', 'כניסה', 'clock_in', 'entry_time', 'start_time'])
        shift_date_col = self._find_column(shifts_df, ['shift_date', 'תאריך', '_meta_date_range_start', 'date'])
        shift_emp_name_col = self._find_column(shifts_df, ['employee_name', '_meta_employee_name', 'שם_עובד', 'שם', 'name'])
        shift_emp_id_col = self._find_column(shifts_df, ['employee_id', '_meta_employee_id', 'מספר_עובד', 'emp_id'])
        shift_loc_col = self._find_column(shifts_df, ['department_code', 'location', 'הערה', 'מחלקה', 'clinical_notes'])
        
        proc_time_col = self._find_column(procedures_df, ['treatment_start_time', 'שעות טיפול בפועל_start', 'performed_datetime', 'start_time'])
        proc_date_col = self._find_column(procedures_df, ['treatment_date', 'תאריך טיפול', 'תאריך', 'date'])
        proc_cat_col = self._find_column(procedures_df, ['treatment_category', 'קטגורית טיפול', 'category', 'department'])
        proc_name_col = self._find_column(procedures_df, ['treatment_name', 'שם טיפול', 'procedure_name', 'name', 'description'])
        proc_staff_col = self._find_column(procedures_df, ['treating_staff', 'צוות מטפל', 'staff_name', 'provider', 'doctor'])
        
        if not shift_time_col:
            raise ValueError(f"Could not find time column in shifts. Available: {list(shifts_df.columns)}")
        if not proc_time_col:
            raise ValueError(f"Could not find time column in procedures. Available: {list(procedures_df.columns)}")
        
        self.logger.info(f"Shift columns: time={shift_time_col}, date={shift_date_col}, name={shift_emp_name_col}")
        self.logger.info(f"Procedure columns: time={proc_time_col}, date={proc_date_col}, cat={proc_cat_col}")
        
        # Extract shift data
        shifts_data = self._extract_shifts(
            shifts_df, shift_time_col, shift_date_col, shift_emp_name_col,
            shift_emp_id_col, shift_loc_col
        )
        
        # Extract procedure data
        procedures_data = self._extract_procedures(
            procedures_df, proc_time_col, proc_date_col, proc_cat_col,
            proc_name_col, proc_staff_col
        )
        
        self.logger.info(f"Extracted {len(shifts_data)} shifts and {len(procedures_data)} procedures")
        
        if not shifts_data:
            result_df = pd.DataFrame(columns=[
                "employee_name", "employee_id", "shift_date", "location",
                "arrival_time", "matched_procedure_time", "matched_treatment",
                "treating_staff", "minutes_early", "status", "evidence"
            ])
            result_uri = self.save_to_s3("result", result_df)
            return {"result": result_uri}
        
        # Infer missing locations
        shifts_data = self._infer_locations(shifts_data)
        
        # Group and process
        results = self._process_groups(shifts_data, procedures_data, max_early_minutes, multi_staff_procedures)
        
        # Build result DataFrame
        result_df = pd.DataFrame(results)
        
        if not result_df.empty:
            status_order = {"EARLY": 0, "UNCERTAIN": 1, "NO_PROCEDURES": 2, "OK": 3}
            result_df["_status_order"] = result_df["status"].map(status_order)
            result_df = result_df.sort_values(["_status_order", "shift_date", "arrival_time"])
            result_df = result_df.drop(columns=["_status_order"])
            
            # Add cost calculation based on salary data
            result_df = self._add_cost_calculation(result_df, classified_data)
        
        self.logger.info(
            f"Early arrival analysis complete: "
            f"{len(result_df)} total, "
            f"{len(result_df[result_df['status'] == 'EARLY']) if not result_df.empty else 0} early"
        )
        
        # Save result
        result_uri = self.save_to_s3("result", result_df)
        
        return {"result": result_uri}
    
    def _find_column(self, df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
        """Find first matching column from candidates."""
        cols = list(df.columns)
        for candidate in candidates:
            if candidate in cols:
                return candidate
        return None
    
    def _extract_shifts(
        self,
        df: pd.DataFrame,
        time_col: str,
        date_col: Optional[str],
        name_col: Optional[str],
        id_col: Optional[str],
        loc_col: Optional[str],
    ) -> List[Dict]:
        """Extract shift data from DataFrame."""
        shifts_data = []
        for _, row in df.iterrows():
            time_val = row.get(time_col)
            if pd.isna(time_val) or not str(time_val).strip():
                continue
            
            parsed_time = parse_time(str(time_val))
            if not parsed_time:
                continue
            
            date_val = row.get(date_col) if date_col else None
            normalized_date = normalize_date(date_val) if date_val else "unknown"
            
            shifts_data.append({
                "employee_name": str(row.get(name_col, "")) if name_col else "",
                "employee_id": str(row.get(id_col, "")) if id_col else "",
                "shift_date": normalized_date,
                "location": str(row.get(loc_col, "UNKNOWN")) if loc_col else "UNKNOWN",
                "arrival_time": parsed_time,
            })
        
        return shifts_data
    
    def _extract_procedures(
        self,
        df: pd.DataFrame,
        time_col: str,
        date_col: Optional[str],
        cat_col: Optional[str],
        name_col: Optional[str],
        staff_col: Optional[str],
    ) -> List[Dict]:
        """Extract procedure data from DataFrame."""
        procedures_data = []
        for _, row in df.iterrows():
            time_val = row.get(time_col)
            if pd.isna(time_val) or not str(time_val).strip():
                continue
            
            parsed_time = parse_time(str(time_val))
            if not parsed_time:
                continue
            
            date_val = row.get(date_col) if date_col else None
            normalized_date = normalize_date(date_val) if date_val else "unknown"
            
            procedures_data.append({
                "procedure_date": normalized_date,
                "procedure_time": parsed_time,
                "category": str(row.get(cat_col, "")) if cat_col else "",
                "treatment_name": str(row.get(name_col, "")) if name_col else "",
                "treating_staff": str(row.get(staff_col, "")) if staff_col else "",
            })
        
        return procedures_data
    
    def _infer_locations(self, shifts_data: List[Dict]) -> List[Dict]:
        """Infer missing locations from employee's other shifts."""
        # Build map: employee_id -> most common location
        employee_locations = {}
        for shift in shifts_data:
            emp_id = shift.get("employee_id", "")
            loc = shift.get("location")
            if emp_id and loc and loc != "UNKNOWN" and str(loc).lower() not in ['nan', 'none', '']:
                if emp_id not in employee_locations:
                    employee_locations[emp_id] = []
                employee_locations[emp_id].append(loc)
        
        # Get most common location for each employee
        employee_primary_location = {}
        for emp_id, locs in employee_locations.items():
            if locs:
                most_common = Counter(locs).most_common(1)[0][0]
                employee_primary_location[emp_id] = most_common
        
        # Fill missing locations
        for shift in shifts_data:
            loc = shift.get("location")
            if not loc or loc == "UNKNOWN" or str(loc).lower() in ['nan', 'none', '']:
                emp_id = shift.get("employee_id", "")
                if emp_id in employee_primary_location:
                    shift["location"] = employee_primary_location[emp_id]
                    shift["location_inferred"] = True
        
        return shifts_data
    
    def _process_groups(
        self,
        shifts_data: List[Dict],
        procedures_data: List[Dict],
        max_early_minutes: int,
        multi_staff_procedures: Optional[List[Dict]] = None,
    ) -> List[Dict]:
        """Process shift and procedure groups."""
        # Group shifts by date and location
        shift_groups = defaultdict(list)
        for shift in shifts_data:
            key = (shift["shift_date"], shift["location"])
            shift_groups[key].append(shift)
        
        # Group procedures by date
        proc_by_date = defaultdict(list)
        for proc in procedures_data:
            proc_by_date[proc["procedure_date"]].append(proc)
        
        # Process each group
        results = []
        for (date_str, location), arrivals in shift_groups.items():
            day_procs = proc_by_date.get(date_str, [])
            
            # Filter by category/location match
            matching_procs = [
                p for p in day_procs
                if p["category"] == location or 
                   location.lower() in p["category"].lower() or
                   p["category"].lower() in location.lower()
            ]
            
            if not matching_procs and day_procs:
                matching_procs = day_procs
            
            # Run matching
            matched, unmatched = match_arrivals_to_procedures(
                arrivals,
                matching_procs,
                max_early_minutes,
                multi_staff_procedures
            )
            
            for arr in matched + unmatched:
                location_display = arr.get("location", "")
                # Replace 'nan' string or NaN values with empty string
                if not location_display or str(location_display).lower() in ['nan', 'none']:
                    location_display = ""
                elif arr.get("location_inferred"):
                    location_display = f"{location_display} (inferred)"
                
                results.append({
                    "employee_name": arr.get("employee_name"),
                    "employee_id": arr.get("employee_id"),
                    "shift_date": arr.get("shift_date"),
                    "location": location_display,
                    "arrival_time": arr["arrival_time"].strftime("%H:%M"),
                    "matched_procedure_time": arr.get("matched_procedure_time"),
                    "matched_treatment": arr.get("matched_treatment"),
                    "treating_staff": arr.get("treating_staff"),
                    "minutes_early": arr.get("minutes_early", 0),
                    "status": arr.get("status"),
                    "evidence": arr.get("evidence"),
                })
        
        return results
    
    def _add_cost_calculation(
        self,
        result_df: pd.DataFrame,
        classified_data: Dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """
        Add wasted_cost column based on employee salary data.
        
        Calculates cost for:
        - EARLY: minutes_early / 60 * hourly_rate
        - OK/NO_PROCEDURES/UNCERTAIN: Empty (NULL)
        
        Args:
            result_df: Results DataFrame with status column
            classified_data: Dict of classified DataFrames
            
        Returns:
            DataFrame with added wasted_cost column
        """
        # Build hourly rate lookup from salary data
        hourly_rates = self._build_hourly_rate_lookup(classified_data)
        
        if not hourly_rates:
            self.logger.info("No salary data available - wasted_cost will be empty")
            result_df["wasted_cost"] = None
            return result_df
        
        # Build shift hours lookup from employee_shifts data
        shift_hours_lookup = self._build_shift_hours_lookup(classified_data)
        
        # Calculate cost for each row
        costs = []
        total_early_cost = 0
        
        for _, row in result_df.iterrows():
            status = row["status"]
            employee_id = str(row.get("employee_id", ""))
            employee_name = str(row.get("employee_name", ""))
            shift_date = str(row.get("shift_date", ""))
            
            # Get hourly rate for this employee
            hourly_rate = hourly_rates.get(employee_id) or hourly_rates.get(employee_name)
            
            if status == "OK" or status == "NO_PROCEDURES" or status == "UNCERTAIN" or not hourly_rate:
                costs.append(None)
                continue
            
            # Calculate cost based on status
            if status == "EARLY":
                minutes_early = row.get("minutes_early", 0)
                if minutes_early and pd.notna(minutes_early):
                    cost = (float(minutes_early) / 60.0) * float(hourly_rate)
                    costs.append(round(cost, 2))
                    total_early_cost += cost
                    self.logger.debug(
                        f"EARLY cost for {employee_name}: {minutes_early} min × "
                        f"₪{hourly_rate}/hr = ₪{cost:.2f}"
                    )
                else:
                    costs.append(None)
            else:
                # Other statuses (UNCERTAIN, etc.) don't have cost
                costs.append(None)
        
        result_df["wasted_cost"] = costs
        
        # Log summary
        if total_early_cost > 0:
            self.logger.info(
                f"💰 Cost analysis: EARLY wasted ₪{total_early_cost:.2f} "
                f"(UNCERTAIN statuses excluded - cannot verify)"
            )
        
        return result_df
    
    def _build_hourly_rate_lookup(
        self,
        classified_data: Dict[str, pd.DataFrame],
    ) -> Dict[str, float]:
        """
        Build employee ID/name -> hourly rate lookup from salary data.
        
        Tries multiple sources:
        1. employee_monthly_salary: Convert monthly -> hourly (÷ 160 hours)
        2. employee_compensation: Use hourly_rate or rate_primary field
        
        Returns:
            Dict mapping employee_id or employee_name to hourly rate
        """
        hourly_rates = {}
        
        # Try employee_monthly_salary first
        if "employee_monthly_salary" in classified_data:
            salary_df = classified_data["employee_monthly_salary"]
            self.logger.info(f"Loading salary data from employee_monthly_salary ({len(salary_df)} rows)")
            
            # Find relevant columns
            name_col = self._find_column(salary_df, ["employee_name", "name", "שם", "שם עובד"])
            id_col = self._find_column(salary_df, ["employee_id", "id", "מספר עובד"])
            rate_col = self._find_column(salary_df, ["rate_primary", "hourly_rate", "rate", "תעריף"])
            
            # Also check for monthly salary columns to convert
            avg_salary_col = self._find_column(salary_df, ["avg_monthly_salary", "monthly_salary", "salary"])
            
            for _, row in salary_df.iterrows():
                employee_id = str(row.get(id_col, "")) if id_col else ""
                employee_name = str(row.get(name_col, "")) if name_col else ""
                
                # Try to get hourly rate directly
                hourly_rate = None
                if rate_col and pd.notna(row.get(rate_col)):
                    try:
                        hourly_rate = float(row[rate_col])
                    except (ValueError, TypeError):
                        # Skip invalid rate values (e.g., '73/82', non-numeric)
                        pass
                
                # If no hourly rate, try to calculate from monthly salary
                # Assume 160 work hours per month (20 days × 8 hours)
                elif avg_salary_col and pd.notna(row.get(avg_salary_col)):
                    try:
                        monthly_salary = float(row[avg_salary_col])
                        hourly_rate = monthly_salary / 160.0
                        self.logger.debug(
                            f"Calculated hourly rate for {employee_name}: "
                            f"₪{monthly_salary}/month ÷ 160 hrs = ₪{hourly_rate:.2f}/hr"
                        )
                    except (ValueError, TypeError):
                        # Skip invalid salary values
                        pass
                
                if hourly_rate and hourly_rate > 0:
                    if employee_id:
                        hourly_rates[employee_id] = hourly_rate
                    if employee_name:
                        hourly_rates[employee_name] = hourly_rate
        
        # Try employee_compensation as fallback
        elif "employee_compensation" in classified_data:
            comp_df = classified_data["employee_compensation"]
            self.logger.info(f"Loading salary data from employee_compensation ({len(comp_df)} rows)")
            
            name_col = self._find_column(comp_df, ["employee_name", "name", "שם"])
            id_col = self._find_column(comp_df, ["employee_id", "id", "מספר עובד"])
            rate_col = self._find_column(comp_df, ["hourly_rate", "rate", "תעריף"])
            
            for _, row in comp_df.iterrows():
                employee_id = str(row.get(id_col, "")) if id_col else ""
                employee_name = str(row.get(name_col, "")) if name_col else ""
                
                if rate_col and pd.notna(row.get(rate_col)):
                    try:
                        hourly_rate = float(row[rate_col])
                        
                        if hourly_rate > 0:
                            if employee_id:
                                hourly_rates[employee_id] = hourly_rate
                            if employee_name:
                                hourly_rates[employee_name] = hourly_rate
                    except (ValueError, TypeError):
                        # Skip invalid rate values
                        pass
        
        self.logger.info(f"Built hourly rate lookup for {len(hourly_rates)} employees")
        return hourly_rates
    
    def _build_shift_hours_lookup(
        self,
        classified_data: Dict[str, pd.DataFrame],
    ) -> Dict[str, float]:
        """
        Build employee+date -> actual shift hours lookup from employee_shifts data.
        
        Args:
            classified_data: Dict of classified DataFrames
            
        Returns:
            Dict mapping "employee_name_date" to actual shift hours
        """
        shift_hours = {}
        
        if "employee_shifts" not in classified_data:
            self.logger.debug("No employee_shifts data available for shift hours lookup")
            return shift_hours
        
        shifts_df = classified_data["employee_shifts"]
        self.logger.info(f"Building shift hours lookup from employee_shifts ({len(shifts_df)} rows)")
        
        # Find relevant columns
        name_col = self._find_column(shifts_df, ["employee_name", "name", "שם", "שם עובד"])
        date_col = self._find_column(shifts_df, ["shift_date", "date", "תאריך"])
        hours_col = self._find_column(shifts_df, ["actual_hours", "hours", "שעות"])
        
        if not name_col or not date_col or not hours_col:
            self.logger.warning(
                f"Could not find required columns for shift hours lookup: "
                f"name={name_col}, date={date_col}, hours={hours_col}"
            )
            return shift_hours
        
        for _, row in shifts_df.iterrows():
            employee_name = str(row.get(name_col, ""))
            shift_date = str(row.get(date_col, ""))
            actual_hours = row.get(hours_col)
            
            if not employee_name or not shift_date:
                continue
            
            # Try to parse hours as float
            try:
                hours_float = float(actual_hours) if pd.notna(actual_hours) else None
                if hours_float and hours_float > 0:
                    shift_key = f"{employee_name}_{shift_date}"
                    shift_hours[shift_key] = hours_float
            except (ValueError, TypeError):
                continue
        
        self.logger.info(f"Built shift hours lookup for {len(shift_hours)} shift records")
        return shift_hours


# Register the block
@BlockRegistry.register(
    name="early_arrival_matcher",
    inputs=[
        {"name": "data", "ontology": DataType.CLASSIFIED_DATA, "required": True}
    ],
    outputs=[
        {"name": "result", "ontology": DataType.INSIGHT_RESULT}
    ],
    parameters=[
        {"name": "max_early_minutes", "type": "integer", "default": 30, "description": "Maximum allowed early arrival minutes"}
    ],
    block_class=EarlyArrivalBlock,
    description="Match employee arrivals to medical procedures, detect early arrivals",
)
def early_arrival_matcher(ctx: BlockContext) -> Dict[str, str]:
    """Match employee arrivals to procedures and detect early arrivals."""
    return EarlyArrivalBlock(ctx).run()

