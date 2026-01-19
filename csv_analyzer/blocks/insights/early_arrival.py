"""
Early Arrival Block - Detect early employee arrivals.

Matches employee arrivals to medical procedures to determine
which employees arrived earlier than necessary.

Uses DuckDB SQL with ASOF JOIN for O(n log m) performance,
enabling efficient processing of large datasets.

Migrated from CodeInsightsRegistry to BlockRegistry.
"""

import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

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


class EarlyArrivalBlock(BaseBlock):
    """
    Early arrival detection block.
    
    Matches employee arrivals to medical procedures using DuckDB SQL ASOF JOIN
    for O(n log m) performance. Flags arrivals that are:
    - EARLY: Arrived too early (gap > max_early_minutes)
    - OK: Matched to procedure within threshold
    - UNCERTAIN: Multiple arrivals for same procedure
    - NO_PROCEDURES: No procedures for date/location
    """
    
    # Required document types for this insight
    REQUIRED_DOC_TYPES = ["employee_shifts", "medical_actions"]
    
    # Column candidate lists for flexible schema matching
    SHIFT_TIME_CANDIDATES = ['shift_start', 'כניסה', 'clock_in', 'entry_time', 'start_time']
    SHIFT_DATE_CANDIDATES = ['shift_date', 'תאריך', '_meta_date_range_start', 'date']
    SHIFT_NAME_CANDIDATES = ['employee_name', '_meta_employee_name', 'שם_עובד', 'שם', 'name']
    SHIFT_ID_CANDIDATES = ['employee_id', '_meta_employee_id', 'מספר_עובד', 'emp_id']
    SHIFT_LOC_CANDIDATES = ['department_code', 'location', 'הערה', 'מחלקה', 'clinical_notes']
    
    PROC_TIME_CANDIDATES = ['treatment_start_time', 'שעות טיפול בפועל_start', 'performed_datetime', 'start_time']
    PROC_DATE_CANDIDATES = ['treatment_date', 'תאריך טיפול', 'תאריך', 'date']
    PROC_CAT_CANDIDATES = ['treatment_category', 'קטגורית טיפול', 'category', 'department']
    PROC_NAME_CANDIDATES = ['treatment_name', 'שם טיפול', 'procedure_name', 'name', 'description']
    PROC_STAFF_CANDIDATES = ['treating_staff', 'צוות מטפל', 'staff_name', 'provider', 'doctor']
    
    def run(self) -> Dict[str, str]:
        """
        Match arrivals to procedures and detect early arrivals.
        
        Uses DuckDB SQL ASOF JOIN for efficient O(n log m) matching.
        
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
        
        return self._run_with_duckdb(
            shifts_df, procedures_df, classified_data, 
            max_early_minutes, multi_staff_procedures
        )
    
    def _run_with_duckdb(
        self,
        shifts_df: pd.DataFrame,
        procedures_df: pd.DataFrame,
        classified_data: Dict[str, pd.DataFrame],
        max_early_minutes: int,
        multi_staff_procedures: Optional[List[Dict]] = None,
    ) -> Dict[str, str]:
        """
        Run early arrival detection using DuckDB SQL for O(n log m) performance.
        
        Uses ASOF JOIN for efficient time-based matching instead of O(n*m) loops.
        """
        self.logger.info("Using DuckDB-optimized execution path")
        
        # Find column names
        shift_time_col = self._find_column(shifts_df, self.SHIFT_TIME_CANDIDATES)
        shift_date_col = self._find_column(shifts_df, self.SHIFT_DATE_CANDIDATES)
        shift_emp_name_col = self._find_column(shifts_df, self.SHIFT_NAME_CANDIDATES)
        shift_emp_id_col = self._find_column(shifts_df, self.SHIFT_ID_CANDIDATES)
        shift_loc_col = self._find_column(shifts_df, self.SHIFT_LOC_CANDIDATES)
        
        proc_time_col = self._find_column(procedures_df, self.PROC_TIME_CANDIDATES)
        proc_date_col = self._find_column(procedures_df, self.PROC_DATE_CANDIDATES)
        proc_cat_col = self._find_column(procedures_df, self.PROC_CAT_CANDIDATES)
        proc_name_col = self._find_column(procedures_df, self.PROC_NAME_CANDIDATES)
        proc_staff_col = self._find_column(procedures_df, self.PROC_STAFF_CANDIDATES)
        
        if not shift_time_col:
            raise ValueError(f"Could not find time column in shifts. Available: {list(shifts_df.columns)}")
        if not proc_time_col:
            raise ValueError(f"Could not find time column in procedures. Available: {list(procedures_df.columns)}")
        
        self.logger.info(f"Shift columns: time={shift_time_col}, date={shift_date_col}, name={shift_emp_name_col}")
        self.logger.info(f"Procedure columns: time={proc_time_col}, date={proc_date_col}")
        
        # Prepare data for DuckDB - normalize columns
        shifts_prepared = self._prepare_shifts_for_duckdb(
            shifts_df, shift_time_col, shift_date_col, 
            shift_emp_name_col, shift_emp_id_col, shift_loc_col
        )
        
        procedures_prepared = self._prepare_procedures_for_duckdb(
            procedures_df, proc_time_col, proc_date_col,
            proc_cat_col, proc_name_col, proc_staff_col
        )
        
        if shifts_prepared.empty:
            self.logger.warning("No valid shifts after preparation")
            empty_df = self._get_empty_result_df()
            result_uri = self.save_to_s3("result", empty_df)
            return {"result": result_uri}
        
        # Register tables with DuckDB
        self.duckdb.register("shifts", shifts_prepared)
        self.duckdb.register("procedures", procedures_prepared)
        
        # Execute ASOF JOIN query for time-based matching
        # This is O(n log m) instead of O(n*m)
        sql = f"""
        WITH 
        -- Prepare arrivals with parsed times
        arrivals AS (
            SELECT 
                employee_name,
                employee_id,
                shift_date,
                location,
                arrival_time,
                ROW_NUMBER() OVER (PARTITION BY shift_date, location ORDER BY arrival_time) as arrival_rank
            FROM shifts
            WHERE arrival_time IS NOT NULL
        ),
        
        -- Prepare procedures with parsed times  
        procs AS (
            SELECT
                procedure_date,
                procedure_time,
                treatment_name,
                category,
                treating_staff,
                ROW_NUMBER() OVER (PARTITION BY procedure_date ORDER BY procedure_time) as proc_rank
            FROM procedures
            WHERE procedure_time IS NOT NULL
        ),
        
        -- ASOF JOIN: For each arrival, find the nearest FUTURE procedure
        -- This efficiently matches arrivals to the closest procedure they could cover
        matched AS (
            SELECT 
                a.employee_name,
                a.employee_id,
                a.shift_date,
                a.location,
                a.arrival_time,
                p.procedure_time as matched_procedure_time,
                p.treatment_name as matched_treatment,
                p.category as matched_category,
                p.treating_staff,
                -- Calculate minutes between arrival and procedure
                -- Convert TIME to minutes since midnight for subtraction
                CASE 
                    WHEN p.procedure_time IS NOT NULL 
                    THEN (EXTRACT(HOUR FROM p.procedure_time) * 60 + EXTRACT(MINUTE FROM p.procedure_time))
                       - (EXTRACT(HOUR FROM a.arrival_time) * 60 + EXTRACT(MINUTE FROM a.arrival_time))
                    ELSE NULL 
                END as minutes_before_procedure
            FROM arrivals a
            ASOF LEFT JOIN procs p
                ON a.shift_date = p.procedure_date
                AND a.arrival_time <= p.procedure_time
        ),
        
        -- Calculate status based on time gap
        with_status AS (
            SELECT 
                *,
                CASE 
                    WHEN matched_procedure_time IS NULL THEN 'NO_PROCEDURES'
                    WHEN minutes_before_procedure > {max_early_minutes} THEN 'EARLY'
                    WHEN minutes_before_procedure >= 0 THEN 'OK'
                    ELSE 'NO_PROCEDURES'
                END as status,
                CASE 
                    WHEN minutes_before_procedure > {max_early_minutes}
                    THEN minutes_before_procedure - {max_early_minutes}
                    ELSE 0
                END as minutes_early
            FROM matched
        )
        
        SELECT 
            employee_name,
            employee_id,
            shift_date,
            location,
            CAST(arrival_time AS VARCHAR) as arrival_time,
            CAST(matched_procedure_time AS VARCHAR) as matched_procedure_time,
            matched_treatment,
            treating_staff,
            CAST(ROUND(minutes_before_procedure, 0) AS INTEGER) as minutes_before_procedure,
            CAST(ROUND(minutes_early, 0) AS INTEGER) as minutes_early,
            status,
            -- Build evidence string
            CASE 
                WHEN status = 'EARLY' THEN
                    'Arrived at ' || CAST(arrival_time AS VARCHAR) || 
                    COALESCE(' at ' || location, '') ||
                    ', nearest procedure at ' || CAST(matched_procedure_time AS VARCHAR) ||
                    ' (' || CAST(ROUND(minutes_before_procedure, 0) AS VARCHAR) || ' min gap > ' || 
                    '{max_early_minutes} allowed). Arrived ' || 
                    CAST(ROUND(minutes_early, 0) AS VARCHAR) || ' min early.' ||
                    COALESCE(' Treatment: ' || matched_treatment || '.', '')
                WHEN status = 'OK' THEN
                    'Arrived at ' || CAST(arrival_time AS VARCHAR) ||
                    COALESCE(' at ' || location, '') ||
                    ', covered procedure at ' || CAST(matched_procedure_time AS VARCHAR) ||
                    ' (' || CAST(ROUND(minutes_before_procedure, 0) AS VARCHAR) || ' min before).' ||
                    COALESCE(' Treatment: ' || matched_treatment || '.', '')
                WHEN status = 'NO_PROCEDURES' THEN
                    'Arrived at ' || CAST(arrival_time AS VARCHAR) ||
                    COALESCE(' at ' || location, '') ||
                    ', but no procedures found for this date/location.'
                ELSE 'Unknown status'
            END as evidence
        FROM with_status
        ORDER BY 
            CASE status 
                WHEN 'EARLY' THEN 0 
                WHEN 'UNCERTAIN' THEN 1 
                WHEN 'NO_PROCEDURES' THEN 2 
                ELSE 3 
            END,
            shift_date,
            arrival_time
        """
        
        self.logger.info("Executing DuckDB ASOF JOIN query...")
        result_df = self.duckdb.execute(sql).fetchdf()
        
        self.logger.info(f"DuckDB query returned {len(result_df)} rows")
        
        # Add cost calculation
        if not result_df.empty:
            result_df = self._add_cost_calculation(result_df, classified_data)
        
        # Log summary
        early_count = len(result_df[result_df['status'] == 'EARLY']) if not result_df.empty else 0
        self.logger.info(
            f"Early arrival analysis complete (DuckDB): "
            f"{len(result_df)} total, {early_count} early"
        )
        
        # Save result
        result_uri = self.save_to_s3("result", result_df)
        
        return {"result": result_uri}
    
    def _prepare_shifts_for_duckdb(
        self,
        df: pd.DataFrame,
        time_col: str,
        date_col: Optional[str],
        name_col: Optional[str],
        id_col: Optional[str],
        loc_col: Optional[str],
    ) -> pd.DataFrame:
        """Prepare shifts DataFrame for DuckDB with normalized columns."""
        result = pd.DataFrame()
        
        # Parse and normalize time
        result['arrival_time'] = df[time_col].apply(
            lambda x: parse_time(str(x)) if pd.notna(x) else None
        )
        
        # Convert datetime to time for DuckDB
        result['arrival_time'] = pd.to_datetime(
            result['arrival_time'].apply(
                lambda x: x.strftime('%H:%M:%S') if pd.notna(x) and x is not None else None
            ),
            format='%H:%M:%S',
            errors='coerce'
        ).dt.time
        
        # Normalize date
        if date_col and date_col in df.columns:
            result['shift_date'] = df[date_col].apply(
                lambda x: normalize_date(x) if pd.notna(x) else 'unknown'
            )
        else:
            result['shift_date'] = 'unknown'
        
        # Copy other columns
        result['employee_name'] = df[name_col].astype(str) if name_col and name_col in df.columns else ''
        result['employee_id'] = df[id_col].astype(str) if id_col and id_col in df.columns else ''
        result['location'] = df[loc_col].astype(str) if loc_col and loc_col in df.columns else 'UNKNOWN'
        
        # Clean up location
        result['location'] = result['location'].replace(['nan', 'None', ''], 'UNKNOWN')
        
        # Filter out rows with invalid times
        result = result[result['arrival_time'].notna()]
        
        self.logger.info(f"Prepared {len(result)} shifts for DuckDB")
        return result
    
    def _prepare_procedures_for_duckdb(
        self,
        df: pd.DataFrame,
        time_col: str,
        date_col: Optional[str],
        cat_col: Optional[str],
        name_col: Optional[str],
        staff_col: Optional[str],
    ) -> pd.DataFrame:
        """Prepare procedures DataFrame for DuckDB with normalized columns."""
        result = pd.DataFrame()
        
        # Parse and normalize time
        result['procedure_time'] = df[time_col].apply(
            lambda x: parse_time(str(x)) if pd.notna(x) else None
        )
        
        # Convert datetime to time for DuckDB
        result['procedure_time'] = pd.to_datetime(
            result['procedure_time'].apply(
                lambda x: x.strftime('%H:%M:%S') if pd.notna(x) and x is not None else None
            ),
            format='%H:%M:%S',
            errors='coerce'
        ).dt.time
        
        # Normalize date
        if date_col and date_col in df.columns:
            result['procedure_date'] = df[date_col].apply(
                lambda x: normalize_date(x) if pd.notna(x) else 'unknown'
            )
        else:
            result['procedure_date'] = 'unknown'
        
        # Copy other columns
        result['treatment_name'] = df[name_col].astype(str) if name_col and name_col in df.columns else ''
        result['category'] = df[cat_col].astype(str) if cat_col and cat_col in df.columns else ''
        result['treating_staff'] = df[staff_col].astype(str) if staff_col and staff_col in df.columns else ''
        
        # Filter out rows with invalid times
        result = result[result['procedure_time'].notna()]
        
        self.logger.info(f"Prepared {len(result)} procedures for DuckDB")
        return result
    
    def _get_empty_result_df(self) -> pd.DataFrame:
        """Get an empty result DataFrame with correct columns."""
        return pd.DataFrame(columns=[
            "employee_name", "employee_id", "shift_date", "location",
            "arrival_time", "matched_procedure_time", "matched_treatment",
            "treating_staff", "minutes_before_procedure", "minutes_early", 
            "status", "evidence"
        ])
    
    def _find_column(self, df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
        """Find first matching column from candidates."""
        cols = list(df.columns)
        for candidate in candidates:
            if candidate in cols:
                return candidate
        return None
    
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

