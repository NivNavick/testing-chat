"""
Early Arrival Block - Detect early employee arrivals.

Matches employee arrivals to medical procedures to determine
which employees arrived earlier than necessary.

Migrated from CodeInsightsRegistry to BlockRegistry.
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


def match_arrivals_to_procedures(
    arrivals: List[Dict],
    procedures: List[Dict],
    max_early_minutes: int = 30,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Match arrivals to procedures using greedy algorithm.
    
    For each procedure (sorted by time), find the earliest unmatched arrival
    where: 0 <= (procedure_time - arrival_time) <= max_early_minutes
    """
    arrivals_sorted = sorted(arrivals, key=lambda x: x["arrival_time"])
    procedures_sorted = sorted(procedures, key=lambda x: x["procedure_time"])
    
    matched_indices = set()
    matched_arrivals = []
    
    for proc in procedures_sorted:
        proc_time = proc["procedure_time"]
        proc_name = proc.get("treatment_name", "")
        proc_category = proc.get("category", "")
        proc_staff = proc.get("treating_staff", "")
        
        for i, arr in enumerate(arrivals_sorted):
            if i in matched_indices:
                continue
            
            arr_time = arr["arrival_time"]
            diff = time_diff_minutes(arr_time, proc_time)
            
            if 0 <= diff <= max_early_minutes:
                matched_indices.add(i)
                arr_copy = arr.copy()
                arr_copy["matched_procedure_time"] = proc_time.strftime("%H:%M")
                arr_copy["matched_treatment"] = proc_name
                arr_copy["treating_staff"] = proc_staff
                arr_copy["minutes_before_procedure"] = diff
                arr_copy["status"] = "OK"
                
                # Build evidence with location and treatment
                loc = arr.get('location')
                location_str = f" at {loc}" if loc and loc != 'UNKNOWN' and loc != 'None' and str(loc).lower() != 'nan' else ""
                treatment_str = f" Treatment: {proc_name}." if proc_name else ""
                category_str = f" Category: {proc_category}." if proc_category else ""
                
                arr_copy["evidence"] = (
                    f"Arrived at {arr_time.strftime('%H:%M')}{location_str}, "
                    f"covered procedure at {proc_time.strftime('%H:%M')} "
                    f"({diff} min before).{treatment_str}{category_str}"
                )
                matched_arrivals.append(arr_copy)
                break
    
    unmatched_arrivals = []
    for i, arr in enumerate(arrivals_sorted):
        if i not in matched_indices:
            arr_copy = arr.copy()
            arr_time = arr["arrival_time"]
            loc = arr.get('location')
            location_str = f" at {loc}" if loc and loc != 'UNKNOWN' and loc != 'None' and str(loc).lower() != 'nan' else ""
            
            if procedures_sorted:
                nearest_proc = min(
                    procedures_sorted,
                    key=lambda p: abs(time_diff_minutes(arr_time, p["procedure_time"]))
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
                    arr_copy["matched_procedure_time"] = nearest_time.strftime("%H:%M")
                    arr_copy["matched_treatment"] = nearest_name
                    arr_copy["treating_staff"] = nearest_staff
                    arr_copy["minutes_before_procedure"] = gap
                    arr_copy["minutes_early"] = 0
                    arr_copy["status"] = "EXCESS"
                    arr_copy["evidence"] = (
                        f"Arrived at {arr_time.strftime('%H:%M')}{location_str}, "
                        f"but all procedures already have assigned staff. "
                        f"Nearest procedure at {nearest_time.strftime('%H:%M')}.{treatment_str}{category_str}"
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
    
    def run(self) -> Dict[str, str]:
        """
        Match arrivals to procedures and detect early arrivals.
        
        Returns:
            Dict with 'result' key containing S3 URI of results DataFrame
        """
        max_early_minutes = self.get_param("max_early_minutes", 30)
        
        self.logger.info(f"Running early arrival matcher with max_early_minutes={max_early_minutes}")
        
        # Load classified data
        classified_data = self.load_classified_data("data")
        
        # Find shifts and procedures DataFrames
        shifts_df = classified_data.get("employee_shifts")
        procedures_df = classified_data.get("medical_actions")
        
        if shifts_df is None:
            raise ValueError(f"No shifts data found. Available: {list(classified_data.keys())}")
        if procedures_df is None:
            raise ValueError(f"No procedures data found. Available: {list(classified_data.keys())}")
        
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
        results = self._process_groups(shifts_data, procedures_data, max_early_minutes)
        
        # Build result DataFrame
        result_df = pd.DataFrame(results)
        
        if not result_df.empty:
            status_order = {"EARLY": 0, "EXCESS": 1, "NO_PROCEDURES": 2, "OK": 3}
            result_df["_status_order"] = result_df["status"].map(status_order)
            result_df = result_df.sort_values(["_status_order", "shift_date", "arrival_time"])
            result_df = result_df.drop(columns=["_status_order"])
        
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
                max_early_minutes
            )
            
            for arr in matched + unmatched:
                location_display = arr.get("location", "")
                if arr.get("location_inferred"):
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

