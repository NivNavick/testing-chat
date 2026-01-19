"""
Gross Profit Block - Calculate gross profit per shift.

Calculates: Revenue - Physician Cost - Staff Cost = Gross Profit

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


class GrossProfitBlock(BaseBlock):
    """
    Calculate gross profit per shift.
    
    Formula: Gross Profit = Revenue - Physician Cost - Staff Cost
    
    Where:
    - Revenue: Sum of treatment_price for procedures during shift
    - Physician Cost: Sum of doctor_fee for procedures, OR allocated from compensation
    - Staff Cost: shift_hours × hourly_rate
    """
    
    REQUIRED_DOC_TYPES = ["employee_shifts", "medical_actions"]
    COMPENSATION_DOC_TYPES = ["employee_compensation", "employee_monthly_salary"]
    
    def run(self) -> Dict[str, str]:
        """
        Calculate gross profit per shift using DuckDB SQL JOINs.
        
        Returns:
            Dict with 'result' key containing S3 URI of results DataFrame
        """
        default_shift_hours = self.get_param("default_shift_hours", 8.0)
        benefits_loading = self.get_param("benefits_loading", 0.25)
        default_monthly_hours = self.get_param("default_monthly_hours", 186.0)
        
        self.logger.info("Running gross profit analysis with DuckDB")
        
        # Load classified data
        classified_data = self.load_classified_data("data")
        
        # Check for required document types
        missing_types = [dt for dt in self.REQUIRED_DOC_TYPES if dt not in classified_data]
        if missing_types:
            self.logger.warning(
                f"⚠️  Skipping gross_profit: missing required doc types: {missing_types}. "
                f"Available: {list(classified_data.keys())}"
            )
            empty_df = pd.DataFrame(columns=[
                "shift_date", "employee_id", "employee_name", "position",
                "employment_type", "detected_shift_type", "shift_hours",
                "shift_cost", "procedure_count", "total_revenue",
                "total_doctor_fee", "gross_profit", "profit_margin_pct"
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
                f"⚠️  Skipping gross_profit: no compensation data found. "
                f"Need one of: {self.COMPENSATION_DOC_TYPES}. Available: {list(classified_data.keys())}"
            )
            empty_df = pd.DataFrame(columns=[
                "shift_date", "employee_id", "employee_name", "position",
                "employment_type", "detected_shift_type", "shift_hours",
                "shift_cost", "procedure_count", "total_revenue",
                "total_doctor_fee", "gross_profit", "profit_margin_pct"
            ])
            result_uri = self.save_to_s3("result", empty_df)
            return {"result": result_uri, "skipped": True, "reason": f"Missing compensation data"}
        
        self.logger.info(f"Using compensation data from: {compensation_source}")
        
        shifts_df = classified_data.get("employee_shifts")
        procedures_df = classified_data.get("medical_actions")
        
        self.logger.info(
            f"Loaded {len(shifts_df)} shifts, {len(compensation_df)} compensation, "
            f"{len(procedures_df)} procedures"
        )
        
        # Identify column names
        shift_cols = self._identify_shift_columns(shifts_df)
        proc_cols = self._identify_procedure_columns(procedures_df)
        comp_cols = self._identify_compensation_columns(compensation_df, compensation_source)
        
        # Register DataFrames with DuckDB
        conn = duckdb.connect(":memory:")
        conn.register("shifts", shifts_df)
        conn.register("procedures", procedures_df)
        conn.register("compensation", compensation_df)
        
        # Build the SQL query
        result_df = self._run_with_duckdb(
            conn, shift_cols, proc_cols, comp_cols, 
            compensation_source, default_shift_hours, benefits_loading, default_monthly_hours
        )
        
        # Log summary
        if not result_df.empty:
            total_revenue = result_df["total_revenue"].sum()
            total_cost = result_df["shift_cost"].sum() + result_df["total_doctor_fee"].sum()
            total_profit = result_df["gross_profit"].sum()
            
            self.logger.info(
                f"Gross profit analysis complete (DuckDB): {len(result_df)} shifts. "
                f"Revenue: ${total_revenue:,.0f}, "
                f"Costs: ${total_cost:,.0f}, "
                f"Gross Profit: ${total_profit:,.0f}"
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
            "price": self._find_column(df, ['treatment_price', 'מחיר', 'price', 'amount', 'fee']),
            "doctor_fee": self._find_column(df, ['doctor_fee', 'שכר רופא', 'physician_fee']),
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
        
        # Find monthly columns for employee_monthly_salary
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
    ) -> pd.DataFrame:
        """Execute gross profit calculation using DuckDB SQL."""
        
        # Build compensation CTE based on source
        if compensation_source == "employee_monthly_salary" and comp_cols.get("monthly_cols"):
            # Calculate hourly rate from monthly salary
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
            # Standard compensation lookup - use TRY_CAST with comma removal
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
        
        # Build date normalization
        date_expr = f's."{shift_cols["date"]}"' if shift_cols["date"] else "'unknown'"
        proc_date_expr = f'p."{proc_cols["date"]}"' if proc_cols["date"] else "'unknown'"
        
        # Build price expressions
        price_expr = f'COALESCE(CAST(p."{proc_cols["price"]}" AS DOUBLE), 0)' if proc_cols["price"] else "0"
        doctor_fee_expr = f'COALESCE(CAST(p."{proc_cols["doctor_fee"]}" AS DOUBLE), 0)' if proc_cols["doctor_fee"] else "0"
        
        # Skip time-based join for performance - just match by date
        # Time window matching is slow with large procedure counts
        time_join = ""
        
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
        ),
        procedure_match AS (
            SELECT 
                sb.shift_id,
                COUNT(p."{proc_cols['date']}") as procedure_count,
                COALESCE(SUM({price_expr}), 0) as total_revenue,
                COALESCE(SUM({doctor_fee_expr}), 0) as total_doctor_fee
            FROM shift_base sb
            LEFT JOIN procedures p ON 
                CAST({proc_date_expr} AS VARCHAR) = CAST(sb.shift_date AS VARCHAR)
                {time_join}
            GROUP BY sb.shift_id
        )
        SELECT 
            sb.shift_date,
            sb.employee_id,
            COALESCE(sb.employee_name, sb.shift_emp_name) as employee_name,
            sb.position,
            sb.employment_type,
            sb.detected_shift_type,
            ROUND(sb.shift_hours, 2) as shift_hours,
            ROUND(sb.shift_hours * sb.hourly_cost, 2) as shift_cost,
            COALESCE(pm.procedure_count, 0) as procedure_count,
            ROUND(COALESCE(pm.total_revenue, 0), 2) as total_revenue,
            ROUND(COALESCE(pm.total_doctor_fee, 0), 2) as total_doctor_fee,
            ROUND(COALESCE(pm.total_revenue, 0) - COALESCE(pm.total_doctor_fee, 0) - (sb.shift_hours * sb.hourly_cost), 2) as gross_profit,
            CASE 
                WHEN COALESCE(pm.total_revenue, 0) > 0 THEN 
                    ROUND((COALESCE(pm.total_revenue, 0) - COALESCE(pm.total_doctor_fee, 0) - (sb.shift_hours * sb.hourly_cost)) / pm.total_revenue * 100, 1)
                ELSE 0 
            END as profit_margin_pct
        FROM shift_base sb
        LEFT JOIN procedure_match pm ON sb.shift_id = pm.shift_id
        ORDER BY gross_profit DESC
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
    name="gross_profit_per_shift",
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
    block_class=GrossProfitBlock,
    description="Calculate gross profit per shift (revenue - physician cost - staff cost)",
)
def gross_profit_per_shift(ctx: BlockContext) -> Dict[str, str]:
    """Calculate gross profit per shift."""
    return GrossProfitBlock(ctx).run()

