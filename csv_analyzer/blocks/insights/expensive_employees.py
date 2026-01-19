"""
Expensive Employees Block - Calculate and rank employees by total cost.

Analyzes employee monthly salary data to identify highest-cost employees
across various dimensions: total compensation, monthly averages, and role-based analysis.

Optimized with DuckDB SQL for large dataset performance.
"""

import logging
import re
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np

from csv_analyzer.workflows.block import BlockRegistry
from csv_analyzer.workflows.ontology import DataType
from csv_analyzer.workflows.base_block import BaseBlock, BlockContext
import duckdb

logger = logging.getLogger(__name__)


class ExpensiveEmployeesBlock(BaseBlock):
    """
    Calculate and rank employees by total compensation cost.
    
    Analyzes employee_monthly_salary data to:
    - Calculate total salary across all months per employee
    - Identify employees with dual rates
    - Rank by total cost, average monthly cost
    - Group by position/role for category analysis
    """
    
    # Required document types for this insight
    REQUIRED_DOC_TYPES = ["employee_monthly_salary"]
    
    def run(self) -> Dict[str, str]:
        """
        Analyze employee costs and generate rankings using DuckDB.
        
        Returns:
            Dict with 'result' key containing S3 URI of results DataFrame
            Returns 'skipped': True if required doc types are missing
        """
        # Get parameters
        top_n = self.get_param("top_n", 10)
        include_all = self.get_param("include_all", True)
        group_by_position = self.get_param("group_by_position", True)
        
        self.logger.info(f"Running expensive employees analysis with DuckDB (top_n={top_n})")
        
        # Load classified data
        classified_data = self.load_classified_data("data")
        
        # Check for required document types
        salary_df = classified_data.get("employee_monthly_salary")
        
        if salary_df is None:
            for doc_type, df in classified_data.items():
                if "salary" in doc_type.lower() or "compensation" in doc_type.lower():
                    salary_df = df
                    self.logger.info(f"Using fallback doc type: {doc_type}")
                    break
        
        if salary_df is None:
            self.logger.warning(
                f"⚠️  Skipping expensive_employees: missing required doc types: {self.REQUIRED_DOC_TYPES}. "
                f"Available: {list(classified_data.keys())}"
            )
            empty_df = pd.DataFrame(columns=[
                "cost_rank", "employee_name", "position", "city",
                "total_salary", "avg_monthly_salary", "months_active",
                "rate_primary", "rate_secondary", "has_dual_rate", "cost_percentile"
            ])
            result_uri = self.save_to_s3("result", empty_df)
            return {"result": result_uri, "skipped": True, "reason": f"Missing: {self.REQUIRED_DOC_TYPES}"}
        
        self.logger.info(f"Loaded salary data: {len(salary_df)} rows, columns: {list(salary_df.columns)}")
        
        # Identify columns
        name_col = self._find_column(salary_df, ['employee_name', 'שם עובד', 'שם_עובד', 'name'])
        position_col = self._find_column(salary_df, ['position', 'תפקיד', 'role', 'job_title'])
        city_col = self._find_column(salary_df, ['city', 'עיר', 'location', 'Unnamed: 3'])
        rate_col = self._find_column(salary_df, ['rate_primary', 'תעריף', 'hourly_rate', 'rate'])
        rate2_col = self._find_column(salary_df, ['rate_secondary', 'תעריף_משני'])
        payroll_col = self._find_column(salary_df, ['payroll_tag', 'תג בשכר', 'תג_בשכר'])
        
        # Identify monthly columns
        monthly_cols = self._find_monthly_columns(salary_df)
        
        self.logger.info(f"Found columns - name: {name_col}, position: {position_col}, "
                        f"rate: {rate_col}, monthly: {len(monthly_cols)} columns")
        
        if not monthly_cols:
            raise ValueError(f"No monthly salary columns found. Columns: {list(salary_df.columns)}")
        
        # Use DuckDB for aggregation
        conn = duckdb.connect(":memory:")
        conn.register("salary_data", salary_df)
        
        # Build the SQL query dynamically based on available columns
        # Helper to cast column with comma removal
        def cast_col(col):
            return f"COALESCE(TRY_CAST(REPLACE(CAST(\"{col}\" AS VARCHAR), ',', '') AS DOUBLE), 0)"
        
        # Build monthly sum expression with COALESCE for null handling
        monthly_sum_parts = [cast_col(col) for col in monthly_cols]
        monthly_sum_expr = " + ".join(monthly_sum_parts)
        
        # Build count of non-zero months
        monthly_count_parts = [f'CASE WHEN {cast_col(col)} > 0 THEN 1 ELSE 0 END' for col in monthly_cols]
        monthly_count_expr = " + ".join(monthly_count_parts)
        
        # Build max monthly expression
        monthly_max_parts = [cast_col(col) for col in monthly_cols]
        monthly_max_expr = f"GREATEST({', '.join(monthly_max_parts)})"
        
        # Build min non-zero monthly expression using CASE
        monthly_min_parts = [f'CASE WHEN {cast_col(col)} > 0 THEN {cast_col(col)} ELSE 999999999 END' for col in monthly_cols]
        monthly_min_expr = f"NULLIF(LEAST({', '.join(monthly_min_parts)}), 999999999)"
        
        sql = f"""
        WITH employee_totals AS (
            SELECT
                COALESCE("{name_col}", 'Unknown') as employee_name,
                COALESCE("{position_col}", '') as position,
                {f"COALESCE(\"{city_col}\", '')" if city_col else "''"} as city,
                {f"COALESCE(\"{payroll_col}\", '')" if payroll_col else "''"} as payroll_tag,
                {f'CAST("{rate_col}" AS DOUBLE)' if rate_col else 'NULL'} as rate_primary,
                {f'CAST("{rate2_col}" AS DOUBLE)' if rate2_col else 'NULL'} as rate_secondary,
                {f'(CAST("{rate2_col}" AS DOUBLE) IS NOT NULL AND CAST("{rate2_col}" AS DOUBLE) > 0)' if rate2_col else 'FALSE'} as has_dual_rate,
                ROUND({monthly_sum_expr}, 2) as total_salary,
                {monthly_count_expr} as months_active,
                {monthly_max_expr} as max_monthly_salary,
                COALESCE({monthly_min_expr}, 0) as min_monthly_salary
            FROM salary_data
            WHERE "{name_col}" IS NOT NULL
        ),
        ranked AS (
            SELECT
                *,
                ROUND(CASE WHEN months_active > 0 THEN total_salary / months_active ELSE 0 END, 2) as avg_monthly_salary,
                ROW_NUMBER() OVER (ORDER BY total_salary DESC) as cost_rank
            FROM employee_totals
        )
        SELECT
            cost_rank,
            employee_name,
            position,
            city,
            total_salary,
            avg_monthly_salary,
            months_active,
            rate_primary,
            rate_secondary,
            has_dual_rate,
            max_monthly_salary,
            min_monthly_salary,
            ROUND((COUNT(*) OVER () - cost_rank) * 100.0 / COUNT(*) OVER (), 1) as cost_percentile,
            payroll_tag
        FROM ranked
        ORDER BY cost_rank
        """
        
        if not include_all:
            sql += f" LIMIT {top_n}"
        
        result_df = conn.execute(sql).fetchdf()
        
        self.logger.info(
            f"Expensive employees analysis complete (DuckDB): "
            f"{len(result_df)} employees, "
            f"total cost: {result_df['total_salary'].sum():,.0f}"
        )
        
        # Position-based summary using DuckDB
        if group_by_position and position_col:
            position_sql = """
            SELECT
                position,
                ROUND(SUM(total_salary), 2) as position_total_cost,
                ROUND(AVG(total_salary), 2) as position_avg_cost,
                COUNT(*) as employee_count,
                ROUND(AVG(avg_monthly_salary), 2) as avg_monthly_per_employee,
                ROUND(AVG(rate_primary), 2) as avg_rate
            FROM ranked
            GROUP BY position
            ORDER BY position_total_cost DESC
            """
            # Need to run the CTE again for position summary
            full_position_sql = f"""
            WITH employee_totals AS (
                SELECT
                    COALESCE("{name_col}", 'Unknown') as employee_name,
                    COALESCE("{position_col}", '') as position,
                    {f'CAST("{rate_col}" AS DOUBLE)' if rate_col else 'NULL'} as rate_primary,
                    ROUND({monthly_sum_expr}, 2) as total_salary,
                    {monthly_count_expr} as months_active
                FROM salary_data
                WHERE "{name_col}" IS NOT NULL
            ),
            ranked AS (
                SELECT
                    *,
                    ROUND(CASE WHEN months_active > 0 THEN total_salary / months_active ELSE 0 END, 2) as avg_monthly_salary
                FROM employee_totals
            )
            {position_sql}
            """
            position_summary = conn.execute(full_position_sql).fetchdf()
            position_uri = self.save_to_s3("position_summary", position_summary)
            self.logger.info(f"Saved position summary: {len(position_summary)} positions")
        
        # Log top 5
        self.logger.info("Top 5 most expensive employees:")
        for _, row in result_df.head(5).iterrows():
            self.logger.info(
                f"  #{int(row['cost_rank'])}: {row['employee_name']} "
                f"({row['position']}) - {row['total_salary']:,.0f}"
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
    
    def _find_monthly_columns(self, df: pd.DataFrame) -> List[str]:
        """Find columns that represent monthly salary values."""
        monthly_cols = []
        
        # Pattern 1: M.YY format (e.g., 1.25, 2.25, 12.25)
        pattern1 = re.compile(r'^\d{1,2}\.\d{2}$')
        
        # Pattern 2: month_year format (e.g., jan_25, feb_25)
        pattern2 = re.compile(r'^(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[_-]?\d{2}$', re.IGNORECASE)
        
        for col in df.columns:
            col_str = str(col)
            if pattern1.match(col_str) or pattern2.match(col_str):
                monthly_cols.append(col)
        
        return monthly_cols
    
    def _safe_float(self, value) -> Optional[float]:
        """Safely convert value to float."""
        if pd.isna(value):
            return None
        try:
            # Handle string with commas
            if isinstance(value, str):
                value = value.replace(",", "")
            return float(value)
        except (ValueError, TypeError):
            return None


# Register the block
@BlockRegistry.register(
    name="expensive_employees",
    inputs=[
        {"name": "data", "ontology": DataType.CLASSIFIED_DATA, "required": True}
    ],
    outputs=[
        {"name": "result", "ontology": DataType.INSIGHT_RESULT},
        {"name": "position_summary", "ontology": DataType.INSIGHT_RESULT, "optional": True},
    ],
    parameters=[
        {"name": "top_n", "type": "integer", "default": 10, 
         "description": "Number of top employees to highlight"},
        {"name": "include_all", "type": "boolean", "default": True,
         "description": "Include all employees in output (not just top N)"},
        {"name": "group_by_position", "type": "boolean", "default": True,
         "description": "Generate position-based cost summary"},
    ],
    block_class=ExpensiveEmployeesBlock,
    description="Analyze employee salary data and rank by total compensation cost",
)
def expensive_employees(ctx: BlockContext) -> Dict[str, str]:
    """Analyze employee salary data and identify most expensive employees."""
    return ExpensiveEmployeesBlock(ctx).run()

