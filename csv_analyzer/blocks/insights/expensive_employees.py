"""
Expensive Employees Block - Calculate and rank employees by total cost.

Analyzes employee monthly salary data to identify highest-cost employees
across various dimensions: total compensation, monthly averages, and role-based analysis.

DEPRECATED: This pandas-based block has been replaced by a SQL version.
Use the sql_insight block with insight_name='expensive_employees' instead.

Migration path:
  OLD (this file):
    handler: expensive_employees
    
  NEW (SQL):
    handler: sql_insight
    parameters:
      insight_name: expensive_employees
      top_n: 10
      include_all: true

The SQL version is defined in:
  csv_analyzer/insights/definitions/expensive_employees.yaml
"""

import logging
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np

from csv_analyzer.workflows.block import BlockRegistry
from csv_analyzer.workflows.ontology import DataType
from csv_analyzer.workflows.base_block import BaseBlock, BlockContext

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
        Analyze employee costs and generate rankings.
        
        Returns:
            Dict with 'result' key containing S3 URI of results DataFrame
            Returns 'skipped': True if required doc types are missing
        """
        # Get parameters
        top_n = self.get_param("top_n", 10)  # Number of top employees to return
        include_all = self.get_param("include_all", True)  # Include all employees in output
        group_by_position = self.get_param("group_by_position", True)  # Add position-based analysis
        
        self.logger.info(f"Running expensive employees analysis (top_n={top_n})")
        
        # Load classified data
        classified_data = self.load_classified_data("data")
        
        # Check for required document types (with fallback check for compensation)
        salary_df = classified_data.get("employee_monthly_salary")
        
        if salary_df is None:
            # Try to find any salary-related data
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
            # Return empty result with skipped flag
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
        
        # Identify monthly columns (pattern: number.number like 1.25, 2.25, etc.)
        monthly_cols = self._find_monthly_columns(salary_df)
        
        self.logger.info(f"Found columns - name: {name_col}, position: {position_col}, "
                        f"rate: {rate_col}, monthly: {len(monthly_cols)} columns")
        
        if not monthly_cols:
            raise ValueError(f"No monthly salary columns found. Columns: {list(salary_df.columns)}")
        
        # Calculate totals for each employee
        results = []
        
        for idx, row in salary_df.iterrows():
            employee_name = str(row.get(name_col, "Unknown")) if name_col else f"Employee_{idx}"
            position = str(row.get(position_col, "")) if position_col else ""
            city = str(row.get(city_col, "")) if city_col else ""
            payroll_tag = str(row.get(payroll_col, "")) if payroll_col else ""
            
            # Get rates
            rate_primary = self._safe_float(row.get(rate_col)) if rate_col else None
            rate_secondary = self._safe_float(row.get(rate2_col)) if rate2_col else None
            
            # Calculate monthly totals
            monthly_values = []
            months_with_data = 0
            
            for month_col in monthly_cols:
                val = self._safe_float(row.get(month_col))
                if val is not None and val > 0:
                    monthly_values.append(val)
                    months_with_data += 1
                else:
                    monthly_values.append(0)
            
            total_salary = sum(monthly_values)
            avg_monthly = total_salary / months_with_data if months_with_data > 0 else 0
            max_monthly = max(monthly_values) if monthly_values else 0
            min_monthly_nonzero = min([v for v in monthly_values if v > 0]) if any(v > 0 for v in monthly_values) else 0
            
            # Determine if employee has dual rates
            has_dual_rate = rate_secondary is not None and rate_secondary > 0
            
            results.append({
                "employee_name": employee_name,
                "position": position,
                "city": city,
                "payroll_tag": payroll_tag,
                "rate_primary": rate_primary,
                "rate_secondary": rate_secondary,
                "has_dual_rate": has_dual_rate,
                "total_salary": round(total_salary, 2),
                "months_active": months_with_data,
                "avg_monthly_salary": round(avg_monthly, 2),
                "max_monthly_salary": round(max_monthly, 2),
                "min_monthly_salary": round(min_monthly_nonzero, 2),
            })
        
        # Create DataFrame and sort by total salary
        result_df = pd.DataFrame(results)
        result_df = result_df.sort_values("total_salary", ascending=False)
        
        # Add rank column
        result_df["cost_rank"] = range(1, len(result_df) + 1)
        
        # Calculate percentile
        if len(result_df) > 0:
            result_df["cost_percentile"] = (
                (len(result_df) - result_df["cost_rank"]) / len(result_df) * 100
            ).round(1)
        
        # Position-based summary
        if group_by_position and position_col:
            position_summary = result_df.groupby("position").agg({
                "total_salary": ["sum", "mean", "count"],
                "avg_monthly_salary": "mean",
                "rate_primary": "mean",
            }).round(2)
            position_summary.columns = [
                "position_total_cost", "position_avg_cost", "employee_count",
                "avg_monthly_per_employee", "avg_rate"
            ]
            position_summary = position_summary.reset_index()
            position_summary = position_summary.sort_values("position_total_cost", ascending=False)
            
            # Save position summary separately
            position_uri = self.save_to_s3("position_summary", position_summary)
            self.logger.info(f"Saved position summary: {len(position_summary)} positions")
        
        # Filter to top N if requested
        if not include_all:
            result_df = result_df.head(top_n)
        
        # Reorder columns for better readability
        column_order = [
            "cost_rank", "employee_name", "position", "city",
            "total_salary", "avg_monthly_salary", "months_active",
            "rate_primary", "rate_secondary", "has_dual_rate",
            "max_monthly_salary", "min_monthly_salary",
            "cost_percentile", "payroll_tag"
        ]
        result_df = result_df[[c for c in column_order if c in result_df.columns]]
        
        self.logger.info(
            f"Expensive employees analysis complete: "
            f"{len(result_df)} employees, "
            f"total cost: {result_df['total_salary'].sum():,.0f}"
        )
        
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

