"""
Manpower Cost Block - Calculate shift costs and average cost per procedure.

Analyzes employee shifts and compensation data to calculate:
- Total manpower cost per shift
- Average cost per procedure performed during shift

Optimized with DuckDB SQL JOINs for large dataset performance.
"""

import logging
import re
from typing import Dict, List, Optional

import pandas as pd

from csv_analyzer.workflows.block import BlockRegistry
from csv_analyzer.workflows.ontology import DataType
from csv_analyzer.workflows.base_block import BaseBlock, BlockContext
import duckdb

logger = logging.getLogger(__name__)


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
        Calculate manpower costs per shift using DuckDB SQL.
        
        Returns:
            Dict with 'result' key containing S3 URI of results DataFrame
            Returns 'skipped': True if required doc types are missing
        """
        default_shift_hours = self.get_param("default_shift_hours", 8.0)
        benefits_loading = self.get_param("benefits_loading", 0.25)
        default_monthly_hours = self.get_param("default_monthly_hours", 186.0)
        
        self.logger.info(f"Running manpower cost analysis with DuckDB (default_hours={default_shift_hours})")
        
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
                "avg_cost_per_procedure", "detected_shift_type"
            ])
            result_uri = self.save_to_s3("result", empty_df)
            return {"result": result_uri, "skipped": True, "reason": f"Missing: {missing_types}"}
        
        # Check for compensation data
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
                "avg_cost_per_procedure", "detected_shift_type"
            ])
            result_uri = self.save_to_s3("result", empty_df)
            return {"result": result_uri, "skipped": True, "reason": f"Missing compensation data"}
        
        self.logger.info(f"Using compensation data from: {compensation_source}")
        
        shifts_df = classified_data.get("employee_shifts")
        procedures_df = classified_data.get("medical_actions")
        
        self.logger.info(f"Loaded {len(shifts_df)} shifts, {len(compensation_df)} compensation records")
        if procedures_df is not None:
            self.logger.info(f"Loaded {len(procedures_df)} procedures")
        
        # Identify column names
        shift_cols = self._identify_shift_columns(shifts_df)
        proc_cols = self._identify_procedure_columns(procedures_df) if procedures_df is not None else {}
        comp_cols = self._identify_compensation_columns(compensation_df, compensation_source)
        
        # Register DataFrames with DuckDB
        conn = duckdb.connect(":memory:")
        conn.register("shifts", shifts_df)
        conn.register("compensation", compensation_df)
        if procedures_df is not None:
            conn.register("procedures", procedures_df)
        
        # Execute with DuckDB
        result_df = self._run_with_duckdb(
            conn, shift_cols, proc_cols, comp_cols,
            compensation_source, default_shift_hours, benefits_loading, default_monthly_hours,
            has_procedures=(procedures_df is not None)
        )
        
        self.logger.info(
            f"Manpower cost analysis complete (DuckDB): {len(result_df)} shifts, "
            f"total cost: {result_df['shift_cost'].sum():,.0f}"
        )
        
        result_uri = self.save_to_s3("result", result_df)
        return {"result": result_uri}
    
    def _identify_shift_columns(self, df: pd.DataFrame) -> Dict[str, Optional[str]]:
        """Identify shift table column names."""
        return {
            "emp_id": self._find_column(df, ['employee_id', 'emp_id', 'מספר_עובד', '_meta_employee_id']),
            "emp_name": self._find_column(df, ['employee_name', '_meta_employee_name', 'שם_עובד', 'שם']),
            "date": self._find_column(df, ['shift_date', 'תאריך', '_meta_date_range_start', 'date']),
            "start": self._find_column(df, ['shift_start', 'כניסה', 'clock_in', 'entry_time', 'start_time']),
            "end": self._find_column(df, ['shift_end', 'יציאה', 'clock_out', 'exit_time', 'end_time']),
            "duration": self._find_column(df, ['duration_minutes', 'duration', 'משך_משמרת']),
        }
    
    def _identify_procedure_columns(self, df: pd.DataFrame) -> Dict[str, Optional[str]]:
        """Identify procedure table column names."""
        return {
            "date": self._find_column(df, ['treatment_date', 'תאריך טיפול', 'תאריך', 'date']),
            "time": self._find_column(df, ['treatment_start_time', 'שעות טיפול בפועל_start', 'start_time']),
            "staff": self._find_column(df, ['treating_staff', 'צוות מטפל', 'staff_name', 'provider']),
            "name": self._find_column(df, ['treatment_name', 'שם טיפול', 'procedure_name']),
        }
    
    def _identify_compensation_columns(self, df: pd.DataFrame, source: str) -> Dict[str, Optional[str]]:
        """Identify compensation table column names."""
        cols = {
            "emp_id": self._find_column(df, ['employee_id', 'emp_id', 'מספר_עובד', 'מזהה_עובד']),
            "emp_name": self._find_column(df, ['employee_name', 'name', 'שם_עובד', 'שם']),
            "position": self._find_column(df, ['position', 'job_title', 'תפקיד', 'role']),
            "emp_type": self._find_column(df, ['employment_type', 'type', 'סוג_העסקה']),
            "rate": self._find_column(df, ['hourly_rate', 'rate', 'תעריף_שעתי', 'תעריף', 'rate_primary']),
            "effective_rate": self._find_column(df, ['effective_hourly_cost', 'עלות_שעה_אפקטיבית']),
            "gross_salary": self._find_column(df, ['gross_salary', 'salary', 'שכר_ברוטו', 'משכורת']),
            "monthly_hours": self._find_column(df, ['monthly_hours', 'hours', 'שעות_חודשיות']),
        }
        
        if source == "employee_monthly_salary":
            monthly_cols = [str(c) for c in df.columns if re.match(r'^\d{1,2}\.\d{2}$', str(c))]
            cols["monthly_cols"] = monthly_cols
        else:
            cols["monthly_cols"] = []
        
        return cols
    
    def _run_with_duckdb(
        self,
        conn,
        shift_cols: Dict,
        proc_cols: Dict,
        comp_cols: Dict,
        compensation_source: str,
        default_shift_hours: float,
        benefits_loading: float,
        default_monthly_hours: float,
        has_procedures: bool,
    ) -> pd.DataFrame:
        """Execute manpower cost calculation using DuckDB SQL."""
        
        # Build compensation CTE based on source
        if compensation_source == "employee_monthly_salary" and comp_cols.get("monthly_cols"):
            monthly_cols = comp_cols["monthly_cols"]
            monthly_sum = " + ".join([f'COALESCE(CAST("{c}" AS DOUBLE), 0)' for c in monthly_cols])
            monthly_count = " + ".join([f'CASE WHEN COALESCE(CAST("{c}" AS DOUBLE), 0) > 0 THEN 1 ELSE 0 END' for c in monthly_cols])
            
            # Helper to safely cast with comma removal
            def safe_cast(col):
                return f"COALESCE(TRY_CAST(REPLACE(CAST(\"{col}\" AS VARCHAR), ',', '') AS DOUBLE), 0)"
            
            rate_cast = safe_cast(comp_cols['rate']) if comp_cols['rate'] else "0"
            
            comp_cte = f"""
            comp_data AS (
                SELECT 
                    COALESCE("{comp_cols['emp_name']}", '') as emp_key,
                    COALESCE("{comp_cols['emp_name']}", '') as employee_name,
                    COALESCE("{comp_cols['position']}", '') as position,
                    'salaried' as employment_type,
                    CASE 
                        WHEN {rate_cast} > 0 
                        THEN {rate_cast}
                        WHEN ({monthly_count}) > 0 
                        THEN (({monthly_sum}) / ({monthly_count})) / {default_monthly_hours} * {1 + benefits_loading}
                        ELSE 0 
                    END as hourly_cost
                FROM compensation
                WHERE "{comp_cols['emp_name']}" IS NOT NULL
            )
            """
            join_key = f"COALESCE(s.\"{shift_cols['emp_name']}\", '')"
        else:
            # Use TRY_CAST with comma removal for safety
            rate_expr = f"COALESCE(TRY_CAST(REPLACE(CAST(\"{comp_cols['rate']}\" AS VARCHAR), ',', '') AS DOUBLE), 0)" if comp_cols["rate"] else "0"
            effective_expr = f"TRY_CAST(REPLACE(CAST(\"{comp_cols['effective_rate']}\" AS VARCHAR), ',', '') AS DOUBLE)" if comp_cols["effective_rate"] else "NULL"
            
            comp_cte = f"""
            comp_data AS (
                SELECT 
                    COALESCE("{comp_cols['emp_id']}", '') as emp_key,
                    COALESCE("{comp_cols['emp_name']}", '') as employee_name,
                    COALESCE("{comp_cols['position']}", '') as position,
                    COALESCE("{comp_cols['emp_type']}", 'unknown') as employment_type,
                    COALESCE({effective_expr}, {rate_expr}, 0) as hourly_cost
                FROM compensation
                WHERE "{comp_cols['emp_id']}" IS NOT NULL
            )
            """
            join_key = f's."{shift_cols["emp_id"]}"'
        
        # Build shift hours calculation
        if shift_cols["duration"]:
            hours_expr = f'COALESCE(CAST(s."{shift_cols["duration"]}" AS DOUBLE) / 60.0, {default_shift_hours})'
        elif shift_cols["start"] and shift_cols["end"]:
            hours_expr = f"""
            CASE 
                WHEN s."{shift_cols['start']}" IS NOT NULL AND s."{shift_cols['end']}" IS NOT NULL THEN
                    CASE 
                        WHEN TRY_CAST(s."{shift_cols['end']}" AS TIME) >= TRY_CAST(s."{shift_cols['start']}" AS TIME) THEN
                            (EXTRACT(HOUR FROM TRY_CAST(s."{shift_cols['end']}" AS TIME)) - EXTRACT(HOUR FROM TRY_CAST(s."{shift_cols['start']}" AS TIME))) +
                            (EXTRACT(MINUTE FROM TRY_CAST(s."{shift_cols['end']}" AS TIME)) - EXTRACT(MINUTE FROM TRY_CAST(s."{shift_cols['start']}" AS TIME))) / 60.0
                        ELSE
                            24 + (EXTRACT(HOUR FROM TRY_CAST(s."{shift_cols['end']}" AS TIME)) - EXTRACT(HOUR FROM TRY_CAST(s."{shift_cols['start']}" AS TIME))) +
                            (EXTRACT(MINUTE FROM TRY_CAST(s."{shift_cols['end']}" AS TIME)) - EXTRACT(MINUTE FROM TRY_CAST(s."{shift_cols['start']}" AS TIME))) / 60.0
                    END
                ELSE {default_shift_hours}
            END
            """
        else:
            hours_expr = str(default_shift_hours)
        
        # Build shift type detection
        shift_type_expr = f"""
        CASE 
            WHEN TRY_CAST(s."{shift_cols['start']}" AS TIME) IS NULL THEN 'UNKNOWN'
            WHEN EXTRACT(HOUR FROM TRY_CAST(s."{shift_cols['start']}" AS TIME)) >= 5 AND EXTRACT(HOUR FROM TRY_CAST(s."{shift_cols['start']}" AS TIME)) < 12 THEN 'MORNING'
            WHEN EXTRACT(HOUR FROM TRY_CAST(s."{shift_cols['start']}" AS TIME)) >= 12 AND EXTRACT(HOUR FROM TRY_CAST(s."{shift_cols['start']}" AS TIME)) < 14 THEN 'MID_DAY'
            WHEN EXTRACT(HOUR FROM TRY_CAST(s."{shift_cols['start']}" AS TIME)) >= 14 AND EXTRACT(HOUR FROM TRY_CAST(s."{shift_cols['start']}" AS TIME)) < 17 THEN 'AFTERNOON'
            WHEN EXTRACT(HOUR FROM TRY_CAST(s."{shift_cols['start']}" AS TIME)) >= 17 AND EXTRACT(HOUR FROM TRY_CAST(s."{shift_cols['start']}" AS TIME)) < 21 THEN 'EVENING'
            ELSE 'NIGHT'
        END
        """ if shift_cols["start"] else "'UNKNOWN'"
        
        date_expr = f's."{shift_cols["date"]}"' if shift_cols["date"] else "'unknown'"
        
        # Build procedure matching
        if has_procedures and proc_cols:
            proc_date_expr = f'p."{proc_cols["date"]}"' if proc_cols["date"] else "'unknown'"
            # Skip time-based join for performance - just match by date
            time_join = ""
            
            proc_cte = f""",
            procedure_match AS (
                SELECT 
                    sb.shift_id,
                    COUNT(p."{proc_cols['date']}") as procedure_count
                FROM shift_base sb
                LEFT JOIN procedures p ON 
                    CAST({proc_date_expr} AS VARCHAR) = CAST(sb.shift_date AS VARCHAR)
                    {time_join}
                GROUP BY sb.shift_id
            )
            """
            proc_join = "LEFT JOIN procedure_match pm ON sb.shift_id = pm.shift_id"
            proc_count_expr = "COALESCE(pm.procedure_count, 0)"
        else:
            proc_cte = ""
            proc_join = ""
            proc_count_expr = "0"
        
        sql = f"""
        WITH {comp_cte},
        shift_base AS (
            SELECT 
                ROW_NUMBER() OVER () as shift_id,
                s."{shift_cols['emp_id']}" as employee_id,
                COALESCE(s."{shift_cols['emp_name']}", '') as shift_emp_name,
                {date_expr} as shift_date,
                s."{shift_cols['start']}" as shift_start,
                s."{shift_cols['end']}" as shift_end,
                {hours_expr} as shift_hours,
                {shift_type_expr} as detected_shift_type,
                c.employee_name,
                c.position,
                c.employment_type,
                COALESCE(c.hourly_cost, 0) as hourly_cost
            FROM shifts s
            LEFT JOIN comp_data c ON {join_key} = c.emp_key
            WHERE s."{shift_cols['emp_id']}" IS NOT NULL
        ){proc_cte}
        SELECT 
            sb.shift_date,
            sb.employee_id,
            COALESCE(sb.employee_name, sb.shift_emp_name) as employee_name,
            sb.position,
            sb.shift_start,
            sb.shift_end,
            ROUND(sb.shift_hours, 2) as shift_hours,
            ROUND(sb.hourly_cost, 2) as hourly_cost,
            sb.employment_type,
            ROUND(sb.shift_hours * sb.hourly_cost, 2) as shift_cost,
            {proc_count_expr} as procedure_count,
            CASE 
                WHEN {proc_count_expr} > 0 THEN ROUND((sb.shift_hours * sb.hourly_cost) / {proc_count_expr}, 2)
                ELSE NULL 
            END as avg_cost_per_procedure,
            sb.detected_shift_type
        FROM shift_base sb
        {proc_join}
        ORDER BY 
            CASE sb.detected_shift_type 
                WHEN 'MORNING' THEN 1 
                WHEN 'MID_DAY' THEN 2 
                WHEN 'AFTERNOON' THEN 3 
                WHEN 'EVENING' THEN 4 
                WHEN 'NIGHT' THEN 5 
                ELSE 6 
            END,
            sb.shift_date,
            sb.shift_start
        """
        
        return conn.execute(sql).fetchdf()
    
    def _find_column(self, df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
        """Find first matching column from candidates."""
        cols = list(df.columns)
        for candidate in candidates:
            if candidate in cols:
                return candidate
        return None


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

