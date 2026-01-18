"""
Manpower Cost Block - Calculate shift costs and average cost per procedure.

Analyzes employee shifts and compensation data to calculate:
- Total manpower cost per shift
- Average cost per procedure performed during shift

DEPRECATED: This pandas-based block has been replaced by a SQL version.
Use the sql_insight block with insight_name='manpower_cost' instead.

Migration path:
  OLD (this file):
    handler: manpower_cost_per_shift
    
  NEW (SQL):
    handler: sql_insight
    parameters:
      insight_name: manpower_cost
      default_shift_hours: 8.0
      benefits_loading: 0.25

The SQL version is defined in:
  csv_analyzer/insights/definitions/manpower_cost.yaml
"""

import logging
import re
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

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


def detect_shift_type(start_time: datetime) -> str:
    """Detect shift type based on start time."""
    hour = start_time.hour
    if 5 <= hour < 12:
        return "MORNING"
    elif 12 <= hour < 14:
        return "MID_DAY"
    elif 14 <= hour < 17:
        return "AFTERNOON"
    elif 17 <= hour < 21:
        return "EVENING"
    else:
        return "NIGHT"


class ManpowerCostBlock(BaseBlock):
    """
    Calculate manpower cost per shift and average cost per procedure.
    
    Analyzes:
    - Shift duration × hourly rate = shift cost
    - Procedures performed during shift
    - Average cost per procedure = shift_cost / procedure_count
    """
    
    # Required document types for this insight
    REQUIRED_DOC_TYPES = ["employee_shifts"]
    # Either employee_compensation OR employee_monthly_salary can provide compensation data
    COMPENSATION_DOC_TYPES = ["employee_compensation", "employee_monthly_salary"]
    OPTIONAL_DOC_TYPES = ["medical_actions"]
    
    def run(self) -> Dict[str, str]:
        """
        Calculate manpower costs per shift.
        
        Returns:
            Dict with 'result' key containing S3 URI of results DataFrame
            Returns 'skipped': True if required doc types are missing
        """
        default_shift_hours = self.get_param("default_shift_hours", 8.0)
        benefits_loading = self.get_param("benefits_loading", 0.25)  # 25% for salaried
        default_monthly_hours = self.get_param("default_monthly_hours", 186.0)  # For salary to hourly conversion
        
        self.logger.info(f"Running manpower cost analysis (default_hours={default_shift_hours})")
        
        # Load classified data
        classified_data = self.load_classified_data("data")
        
        # Check for required document types
        missing_types = [dt for dt in self.REQUIRED_DOC_TYPES if dt not in classified_data]
        if missing_types:
            self.logger.warning(
                f"⚠️  Skipping manpower_cost: missing required doc types: {missing_types}. "
                f"Available: {list(classified_data.keys())}"
            )
            empty_df = pd.DataFrame(columns=[
                "shift_date", "employee_id", "employee_name", "position",
                "shift_start", "shift_end", "shift_hours", "hourly_cost",
                "employment_type", "shift_cost", "procedure_count",
                "avg_cost_per_procedure", "detected_shift_type", "evidence"
            ])
            result_uri = self.save_to_s3("result", empty_df)
            return {"result": result_uri, "skipped": True, "reason": f"Missing: {missing_types}"}
        
        # Check for compensation data (either employee_compensation or employee_monthly_salary)
        compensation_df = None
        compensation_source = None
        for comp_type in self.COMPENSATION_DOC_TYPES:
            if comp_type in classified_data:
                compensation_df = classified_data.get(comp_type)
                compensation_source = comp_type
                break
        
        if compensation_df is None:
            self.logger.warning(
                f"⚠️  Skipping manpower_cost: no compensation data found. "
                f"Need one of: {self.COMPENSATION_DOC_TYPES}. Available: {list(classified_data.keys())}"
            )
            empty_df = pd.DataFrame(columns=[
                "shift_date", "employee_id", "employee_name", "position",
                "shift_start", "shift_end", "shift_hours", "hourly_cost",
                "employment_type", "shift_cost", "procedure_count",
                "avg_cost_per_procedure", "detected_shift_type", "evidence"
            ])
            result_uri = self.save_to_s3("result", empty_df)
            return {"result": result_uri, "skipped": True, "reason": f"Missing compensation data"}
        
        self.logger.info(f"Using compensation data from: {compensation_source}")
        
        shifts_df = classified_data.get("employee_shifts")
        procedures_df = classified_data.get("medical_actions")
        
        self.logger.info(f"Loaded {len(shifts_df)} shifts, {len(compensation_df)} compensation records")
        if procedures_df is not None:
            self.logger.info(f"Loaded {len(procedures_df)} procedures")
        
        # Build compensation lookup based on source type
        if compensation_source == "employee_monthly_salary":
            compensation_lookup = self._build_compensation_from_monthly_salary(
                compensation_df, benefits_loading, default_monthly_hours
            )
        else:
            compensation_lookup = self._build_compensation_lookup(compensation_df, benefits_loading)
        
        # Extract shift data
        shifts_data = self._extract_shifts(shifts_df, default_shift_hours)
        
        # Extract procedure data if available
        procedures_data = self._extract_procedures(procedures_df) if procedures_df is not None else []
        
        # Group procedures by date for matching
        procs_by_date = defaultdict(list)
        for proc in procedures_data:
            procs_by_date[proc["procedure_date"]].append(proc)
        
        # Calculate costs for each shift
        results = []
        for shift in shifts_data:
            emp_id = shift["employee_id"]
            emp_name = shift.get("employee_name", "")
            shift_date = shift["shift_date"]
            
            # Get compensation info - try employee_id first, then employee_name
            comp = compensation_lookup.get(emp_id, {})
            if not comp and emp_name:
                comp = compensation_lookup.get(emp_name, {})
            
            hourly_cost = comp.get("hourly_cost", 0)
            employment_type = comp.get("employment_type", "unknown")
            employee_name = comp.get("employee_name") or emp_name
            position = comp.get("position", "")
            
            # Calculate shift cost
            shift_hours = shift["shift_hours"]
            shift_cost = round(shift_hours * hourly_cost, 2)
            
            # Count procedures during this shift
            day_procs = procs_by_date.get(shift_date, [])
            
            # Match procedures to this employee's shift time window
            shift_start = shift.get("shift_start_parsed")
            shift_end = shift.get("shift_end_parsed")
            
            matching_procs = []
            for proc in day_procs:
                proc_time = proc.get("procedure_time_parsed")
                if proc_time and shift_start and shift_end:
                    # Check if procedure falls within shift window
                    if shift_start <= proc_time <= shift_end:
                        matching_procs.append(proc)
                elif proc.get("treating_staff") and employee_name:
                    # Fallback: match by treating staff name
                    if employee_name.lower() in proc.get("treating_staff", "").lower():
                        matching_procs.append(proc)
            
            procedure_count = len(matching_procs)
            avg_cost_per_procedure = round(shift_cost / procedure_count, 2) if procedure_count > 0 else None
            
            # Detect shift type
            detected_shift_type = detect_shift_type(shift_start) if shift_start else "UNKNOWN"
            
            # Build evidence
            evidence = self._build_evidence(
                shift, hourly_cost, shift_hours, shift_cost,
                employment_type, procedure_count, avg_cost_per_procedure,
                detected_shift_type
            )
            
            results.append({
                "shift_date": shift_date,
                "employee_id": emp_id,
                "employee_name": employee_name,
                "position": position,
                "shift_start": shift.get("shift_start"),
                "shift_end": shift.get("shift_end"),
                "shift_hours": round(shift_hours, 2),
                "hourly_cost": round(hourly_cost, 2),
                "employment_type": employment_type,
                "shift_cost": shift_cost,
                "procedure_count": procedure_count,
                "avg_cost_per_procedure": avg_cost_per_procedure,
                "detected_shift_type": detected_shift_type,
                "evidence": evidence,
            })
        
        result_df = pd.DataFrame(results)
        
        if not result_df.empty:
            # Sort by shift type then date
            shift_order = {"MORNING": 1, "MID_DAY": 2, "AFTERNOON": 3, "EVENING": 4, "NIGHT": 5, "UNKNOWN": 6}
            result_df["_shift_order"] = result_df["detected_shift_type"].map(shift_order)
            result_df = result_df.sort_values(["_shift_order", "shift_date", "shift_start"])
            result_df = result_df.drop(columns=["_shift_order"])
        
        self.logger.info(
            f"Manpower cost analysis complete: {len(result_df)} shifts, "
            f"total cost: {result_df['shift_cost'].sum():,.0f}"
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
    
    def _build_compensation_lookup(
        self, 
        df: pd.DataFrame, 
        benefits_loading: float
    ) -> Dict[str, Dict]:
        """Build employee_id -> compensation info lookup."""
        lookup = {}
        
        emp_id_col = self._find_column(df, ['employee_id', 'emp_id', 'מספר_עובד', 'מזהה_עובד'])
        name_col = self._find_column(df, ['employee_name', 'name', 'שם_עובד', 'שם'])
        type_col = self._find_column(df, ['employment_type', 'type', 'סוג_העסקה'])
        rate_col = self._find_column(df, ['hourly_rate', 'rate', 'תעריף_שעתי', 'תעריף'])
        effective_col = self._find_column(df, ['effective_hourly_cost', 'עלות_שעה_אפקטיבית'])
        gross_col = self._find_column(df, ['gross_salary', 'salary', 'שכר_ברוטו', 'משכורת'])
        hours_col = self._find_column(df, ['monthly_hours', 'hours', 'שעות_חודשיות'])
        position_col = self._find_column(df, ['position', 'job_title', 'תפקיד', 'role'])
        
        for _, row in df.iterrows():
            emp_id = str(row.get(emp_id_col, "")) if emp_id_col else ""
            if not emp_id:
                continue
            
            # Determine hourly cost
            if effective_col and pd.notna(row.get(effective_col)):
                hourly_cost = float(row[effective_col])
            elif rate_col and pd.notna(row.get(rate_col)):
                hourly_cost = float(row[rate_col])
            elif gross_col and hours_col:
                gross = row.get(gross_col)
                hours = row.get(hours_col)
                if pd.notna(gross) and pd.notna(hours) and float(hours) > 0:
                    # Calculate effective hourly: gross * (1 + benefits) / hours
                    hourly_cost = float(gross) * (1 + benefits_loading) / float(hours)
                else:
                    hourly_cost = 0
            else:
                hourly_cost = 0
            
            employment_type = str(row.get(type_col, "unknown")) if type_col else "unknown"
            
            lookup[emp_id] = {
                "hourly_cost": hourly_cost,
                "employment_type": employment_type,
                "employee_name": str(row.get(name_col, "")) if name_col else "",
                "position": str(row.get(position_col, "")) if position_col else "",
            }
        
        return lookup
    
    def _build_compensation_from_monthly_salary(
        self,
        df: pd.DataFrame,
        benefits_loading: float,
        default_monthly_hours: float,
    ) -> Dict[str, Dict]:
        """
        Build employee compensation lookup from employee_monthly_salary data.
        
        This handles the monthly salary format which has:
        - employee_name, position, rate_primary
        - Monthly columns like 1.25, 2.25, etc. representing monthly totals
        """
        lookup = {}
        
        name_col = self._find_column(df, ['employee_name', 'name', 'שם_עובד', 'שם'])
        position_col = self._find_column(df, ['position', 'job_title', 'תפקיד', 'role'])
        rate_col = self._find_column(df, ['rate_primary', 'rate', 'תעריף', 'hourly_rate'])
        
        # Find monthly columns (pattern: M.YY like 1.25, 2.25, etc.)
        monthly_cols = []
        for col in df.columns:
            col_str = str(col)
            if re.match(r'^\d{1,2}\.\d{2}$', col_str):
                monthly_cols.append(col)
        
        for _, row in df.iterrows():
            employee_name = str(row.get(name_col, "")) if name_col else ""
            if not employee_name or employee_name.lower() in ['nan', 'none', '']:
                continue
            
            position = str(row.get(position_col, "")) if position_col else ""
            
            # Get hourly rate if available
            hourly_cost = 0.0
            if rate_col and pd.notna(row.get(rate_col)):
                try:
                    rate_val = row[rate_col]
                    if isinstance(rate_val, str):
                        rate_val = rate_val.replace(",", "")
                    hourly_cost = float(rate_val)
                except (ValueError, TypeError):
                    pass
            
            # If no rate, estimate from monthly totals
            if hourly_cost == 0 and monthly_cols:
                monthly_values = []
                for col in monthly_cols:
                    val = row.get(col)
                    if pd.notna(val):
                        try:
                            if isinstance(val, str):
                                val = val.replace(",", "")
                            monthly_values.append(float(val))
                        except (ValueError, TypeError):
                            pass
                
                if monthly_values:
                    avg_monthly = sum(monthly_values) / len(monthly_values)
                    # Estimate hourly rate: monthly / hours * (1 + benefits)
                    hourly_cost = (avg_monthly / default_monthly_hours) * (1 + benefits_loading)
            
            # Use employee name as key (since monthly_salary may not have employee_id)
            lookup[employee_name] = {
                "hourly_cost": hourly_cost,
                "employment_type": "salaried",  # Assume salaried for monthly salary data
                "employee_name": employee_name,
                "position": position,
            }
        
        self.logger.info(f"Built compensation lookup from monthly_salary: {len(lookup)} employees")
        return lookup
    
    def _extract_shifts(self, df: pd.DataFrame, default_hours: float) -> List[Dict]:
        """Extract shift data from DataFrame."""
        shifts = []
        
        emp_id_col = self._find_column(df, ['employee_id', 'emp_id', 'מספר_עובד', '_meta_employee_id'])
        name_col = self._find_column(df, ['employee_name', '_meta_employee_name', 'שם_עובד', 'שם'])
        date_col = self._find_column(df, ['shift_date', 'תאריך', '_meta_date_range_start', 'date'])
        start_col = self._find_column(df, ['shift_start', 'כניסה', 'clock_in', 'entry_time', 'start_time'])
        end_col = self._find_column(df, ['shift_end', 'יציאה', 'clock_out', 'exit_time', 'end_time'])
        duration_col = self._find_column(df, ['duration_minutes', 'duration', 'משך_משמרת'])
        
        for _, row in df.iterrows():
            emp_id = str(row.get(emp_id_col, "")) if emp_id_col else ""
            if not emp_id:
                continue
            
            date_val = row.get(date_col) if date_col else None
            shift_date = normalize_date(date_val) if date_val else "unknown"
            
            start_time = str(row.get(start_col, "")) if start_col else ""
            end_time = str(row.get(end_col, "")) if end_col else ""
            
            start_parsed = parse_time(start_time)
            end_parsed = parse_time(end_time)
            
            # Calculate duration
            if duration_col and pd.notna(row.get(duration_col)):
                shift_hours = float(row[duration_col]) / 60.0
            elif start_parsed and end_parsed:
                diff = time_diff_minutes(start_parsed, end_parsed)
                if diff < 0:
                    diff += 24 * 60  # Handle overnight shifts
                shift_hours = diff / 60.0
            else:
                shift_hours = default_hours
            
            shifts.append({
                "employee_id": emp_id,
                "employee_name": str(row.get(name_col, "")) if name_col else "",
                "shift_date": shift_date,
                "shift_start": start_time,
                "shift_end": end_time,
                "shift_start_parsed": start_parsed,
                "shift_end_parsed": end_parsed,
                "shift_hours": shift_hours,
            })
        
        return shifts
    
    def _extract_procedures(self, df: pd.DataFrame) -> List[Dict]:
        """Extract procedure data from DataFrame."""
        procedures = []
        
        date_col = self._find_column(df, ['treatment_date', 'תאריך טיפול', 'תאריך', 'date'])
        time_col = self._find_column(df, ['treatment_start_time', 'שעות טיפול בפועל_start', 'start_time'])
        staff_col = self._find_column(df, ['treating_staff', 'צוות מטפל', 'staff_name', 'provider'])
        name_col = self._find_column(df, ['treatment_name', 'שם טיפול', 'procedure_name'])
        
        for _, row in df.iterrows():
            date_val = row.get(date_col) if date_col else None
            proc_date = normalize_date(date_val) if date_val else None
            
            if not proc_date:
                continue
            
            time_val = str(row.get(time_col, "")) if time_col else ""
            
            procedures.append({
                "procedure_date": proc_date,
                "procedure_time": time_val,
                "procedure_time_parsed": parse_time(time_val),
                "treating_staff": str(row.get(staff_col, "")) if staff_col else "",
                "treatment_name": str(row.get(name_col, "")) if name_col else "",
            })
        
        return procedures
    
    def _build_evidence(
        self,
        shift: Dict,
        hourly_cost: float,
        shift_hours: float,
        shift_cost: float,
        employment_type: str,
        procedure_count: int,
        avg_cost_per_procedure: Optional[float],
        detected_shift_type: str,
    ) -> str:
        """Build evidence string explaining the calculation."""
        parts = [f"[{detected_shift_type} SHIFT]"]
        
        start = shift.get("shift_start", "N/A")
        end = shift.get("shift_end", "N/A")
        date = shift.get("shift_date", "N/A")
        
        parts.append(f"Shift {start}-{end} on {date}.")
        parts.append(f"Duration: {shift_hours:.1f}h × ${hourly_cost:.2f}/h = ${shift_cost:.2f}")
        
        if employment_type == "salaried":
            parts.append("(rate includes 25% benefits loading)")
        elif employment_type == "contractor":
            parts.append("(contractor rate)")
        
        if procedure_count > 0:
            parts.append(f"Procedures: {procedure_count}.")
            if avg_cost_per_procedure:
                parts.append(f"Avg cost/procedure: ${avg_cost_per_procedure:.2f}")
        else:
            parts.append("No procedures matched to this shift.")
        
        return " ".join(parts)


# Register the block
@BlockRegistry.register(
    name="manpower_cost_per_shift",
    inputs=[
        {"name": "data", "ontology": DataType.CLASSIFIED_DATA, "required": True}
    ],
    outputs=[
        {"name": "result", "ontology": DataType.INSIGHT_RESULT}
    ],
    parameters=[
        {"name": "default_shift_hours", "type": "float", "default": 8.0,
         "description": "Default shift hours when duration cannot be calculated"},
        {"name": "benefits_loading", "type": "float", "default": 0.25,
         "description": "Benefits loading factor for salaried employees (0.25 = 25%)"},
    ],
    block_class=ManpowerCostBlock,
    description="Calculate manpower cost per shift and average cost per procedure",
)
def manpower_cost_per_shift(ctx: BlockContext) -> Dict[str, str]:
    """Calculate manpower cost per shift and average cost per procedure."""
    return ManpowerCostBlock(ctx).run()

