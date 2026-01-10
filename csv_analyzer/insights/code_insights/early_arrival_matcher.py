"""
Early Arrival Matcher - Code Insight Handler

Implements bipartite matching between employee arrivals and medical procedures
to determine which employees arrived earlier than necessary.

This handler relies on the classification/canonization pipeline to provide
properly typed tables with canonical column names. It does NOT do its own
column detection.

Algorithm:
1. Group by (date, location/category)
2. Sort arrivals by time, procedures by time
3. For each procedure, find the earliest unmatched arrival that can cover it
4. Remaining unmatched arrivals are flagged as EARLY

Rule: An employee can arrive up to `max_early_minutes` (default 30) before a procedure.
"""

import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import pandas as pd

from csv_analyzer.insights.code_insights import CodeInsightsRegistry

if TYPE_CHECKING:
    from csv_analyzer.insights.engine import InsightsEngine

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
                gap = time_diff_minutes(arr_time, nearest_time)
                
                treatment_str = f" Nearest treatment: {nearest_name}." if nearest_name else ""
                category_str = f" Category: {nearest_category}." if nearest_category else ""
                
                if gap > max_early_minutes:
                    minutes_early = gap - max_early_minutes
                    arr_copy["matched_procedure_time"] = nearest_time.strftime("%H:%M")
                    arr_copy["matched_treatment"] = nearest_name
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
                arr_copy["minutes_before_procedure"] = None
                arr_copy["minutes_early"] = None
                arr_copy["status"] = "NO_PROCEDURES"
                arr_copy["evidence"] = (
                    f"Arrived at {arr_time.strftime('%H:%M')}{location_str}, "
                    f"but no procedures found for this date/location."
                )
            
            unmatched_arrivals.append(arr_copy)
    
    return matched_arrivals, unmatched_arrivals


@CodeInsightsRegistry.register("early_arrival_matcher")
def early_arrival_matcher(
    engine: "InsightsEngine",
    params: Dict[str, Any],
) -> pd.DataFrame:
    """
    Match employee arrivals to medical procedures.
    
    EXPECTS canonized tables:
    - employee_shifts: with columns shift_date, shift_start, employee_name, etc.
    - medical_actions: with columns treatment_date, treatment_start_time, treatment_category, etc.
    
    The classification/canonization pipeline should have already mapped:
    - כניסה → shift_start
    - יציאה → shift_end  
    - תאריך → shift_date
    - תאריך טיפול → treatment_date
    - שעות טיפול בפועל_start → treatment_start_time
    - קטגורית טיפול → treatment_category
    """
    max_early_minutes = params.get("max_early_minutes", 30)
    
    logger.info(f"Running early arrival matcher with max_early_minutes={max_early_minutes}")
    
    available_tables = engine.data_store.list_tables()
    logger.info(f"Available tables: {available_tables}")
    
    # Find shifts table (employee_shifts or any table with shift columns)
    shifts_df = None
    shifts_table = None
    for table_name in available_tables:
        if "shift" in table_name.lower() or table_name == "employee_shifts":
            shifts_df = engine.execute_sql(f'SELECT * FROM "{table_name}"')
            shifts_table = table_name
            break
    
    # Find procedures table (medical_actions or any table with treatment columns)
    procedures_df = None
    procedures_table = None
    for table_name in available_tables:
        if "medical" in table_name.lower() or "action" in table_name.lower() or table_name == "medical_actions":
            procedures_df = engine.execute_sql(f'SELECT * FROM "{table_name}"')
            procedures_table = table_name
            break
    
    if shifts_df is None:
        raise ValueError(f"No shifts table found. Available: {available_tables}")
    if procedures_df is None:
        raise ValueError(f"No procedures table found. Available: {available_tables}")
    
    logger.info(f"Using shifts from '{shifts_table}', procedures from '{procedures_table}'")
    logger.info(f"Shifts columns: {list(shifts_df.columns)}")
    logger.info(f"Procedures columns: {list(procedures_df.columns)}")
    
    # Define column mappings - canonical names first, then fallback to Hebrew
    shift_time_candidates = ['shift_start', 'כניסה', 'clock_in', 'entry_time', 'start_time']
    shift_date_candidates = ['shift_date', 'תאריך', '_meta_date_range_start', 'date']
    shift_emp_name_candidates = ['employee_name', '_meta_employee_name', 'שם_עובד', 'שם', 'name']
    shift_emp_id_candidates = ['employee_id', '_meta_employee_id', 'מספר_עובד', 'emp_id']
    shift_loc_candidates = ['department_code', 'location', 'הערה', 'מחלקה', 'clinical_notes']
    
    proc_time_candidates = ['treatment_start_time', 'שעות טיפול בפועל_start', 'performed_datetime', 'start_time']
    proc_date_candidates = ['treatment_date', 'תאריך טיפול', 'תאריך', 'date']
    proc_cat_candidates = ['treatment_category', 'קטגורית טיפול', 'category', 'department']
    proc_name_candidates = ['treatment_name', 'שם טיפול', 'procedure_name', 'name', 'description']
    
    def find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
        """Find first matching column from candidates."""
        cols = list(df.columns)
        for candidate in candidates:
            if candidate in cols:
                return candidate
        return None
    
    # Find columns in shifts
    shift_time_col = find_column(shifts_df, shift_time_candidates)
    shift_date_col = find_column(shifts_df, shift_date_candidates)
    shift_emp_name_col = find_column(shifts_df, shift_emp_name_candidates)
    shift_emp_id_col = find_column(shifts_df, shift_emp_id_candidates)
    shift_loc_col = find_column(shifts_df, shift_loc_candidates)
    
    logger.info(f"Shift columns found: time={shift_time_col}, date={shift_date_col}, name={shift_emp_name_col}")
    
    if not shift_time_col:
        raise ValueError(f"Could not find time column in shifts. Available: {list(shifts_df.columns)}")
    
    # Find columns in procedures
    proc_time_col = find_column(procedures_df, proc_time_candidates)
    proc_date_col = find_column(procedures_df, proc_date_candidates)
    proc_cat_col = find_column(procedures_df, proc_cat_candidates)
    proc_name_col = find_column(procedures_df, proc_name_candidates)
    
    logger.info(f"Procedure columns found: time={proc_time_col}, date={proc_date_col}, cat={proc_cat_col}, name={proc_name_col}")
    
    if not proc_time_col:
        raise ValueError(f"Could not find time column in procedures. Available: {list(procedures_df.columns)}")
    
    # Extract shift data
    shifts_data = []
    for _, row in shifts_df.iterrows():
        time_val = row.get(shift_time_col)
        if pd.isna(time_val) or not str(time_val).strip():
            continue
        
        parsed_time = parse_time(str(time_val))
        if not parsed_time:
            continue
        
        date_val = row.get(shift_date_col) if shift_date_col else None
        normalized_date = normalize_date(date_val) if date_val else "unknown"
        
        shifts_data.append({
            "employee_name": str(row.get(shift_emp_name_col, "")) if shift_emp_name_col else "",
            "employee_id": str(row.get(shift_emp_id_col, "")) if shift_emp_id_col else "",
            "shift_date": normalized_date,
            "location": str(row.get(shift_loc_col, "UNKNOWN")) if shift_loc_col else "UNKNOWN",
            "arrival_time": parsed_time,
        })
    
    # Extract procedure data
    procedures_data = []
    for _, row in procedures_df.iterrows():
        time_val = row.get(proc_time_col)
        if pd.isna(time_val) or not str(time_val).strip():
            continue
        
        parsed_time = parse_time(str(time_val))
        if not parsed_time:
            continue
        
        date_val = row.get(proc_date_col) if proc_date_col else None
        normalized_date = normalize_date(date_val) if date_val else "unknown"
        
        category = str(row.get(proc_cat_col, "")) if proc_cat_col else ""
        treatment_name = str(row.get(proc_name_col, "")) if proc_name_col else ""
        
        procedures_data.append({
            "procedure_date": normalized_date,
            "procedure_time": parsed_time,
            "category": category,
            "treatment_name": treatment_name,
        })
    
    logger.info(f"Extracted {len(shifts_data)} shifts and {len(procedures_data)} procedures")
    
    if not shifts_data:
        return pd.DataFrame(columns=[
            "employee_name", "employee_id", "shift_date", "location",
            "arrival_time", "matched_procedure_time", "matched_treatment",
            "minutes_early", "status", "evidence"
        ])
    
    # Group shifts by date and location
    from collections import defaultdict
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
        
        # Run matching - pass full procedure data including treatment name
        matched, unmatched = match_arrivals_to_procedures(
            arrivals,
            matching_procs,  # Pass full procedure data
            max_early_minutes
        )
        
        for arr in matched + unmatched:
            results.append({
                "employee_name": arr.get("employee_name"),
                "employee_id": arr.get("employee_id"),
                "shift_date": arr.get("shift_date"),
                "location": arr.get("location"),
                "arrival_time": arr["arrival_time"].strftime("%H:%M"),
                "matched_procedure_time": arr.get("matched_procedure_time"),
                "matched_treatment": arr.get("matched_treatment"),
                "minutes_early": arr.get("minutes_early", 0),
                "status": arr.get("status"),
                "evidence": arr.get("evidence"),
            })
    
    result_df = pd.DataFrame(results)
    
    if not result_df.empty:
        status_order = {"EARLY": 0, "EXCESS": 1, "NO_PROCEDURES": 2, "OK": 3}
        result_df["_status_order"] = result_df["status"].map(status_order)
        result_df = result_df.sort_values(["_status_order", "shift_date", "arrival_time"])
        result_df = result_df.drop(columns=["_status_order"])
    
    logger.info(
        f"Early arrival analysis complete: "
        f"{len(result_df)} total, "
        f"{len(result_df[result_df['status'] == 'EARLY']) if not result_df.empty else 0} early"
    )
    
    return result_df
