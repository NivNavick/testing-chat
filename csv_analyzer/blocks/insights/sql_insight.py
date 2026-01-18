"""
SQL Insight Block - Execute YAML-defined DuckDB SQL insights.

This generic block can execute any insight defined in YAML files,
loading classified data into DuckDB and running the SQL query.

Usage in workflow YAML:
  - id: early_arrival
    handler: sql_insight
    inputs:
      - name: data
        source: classifier.classified_data
    parameters:
      insight_name: early_arrival_duckdb
      max_early_minutes: 30
"""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import duckdb
import pandas as pd

from csv_analyzer.workflows.block import BlockRegistry
from csv_analyzer.workflows.ontology import DataType
from csv_analyzer.workflows.base_block import BaseBlock, BlockContext
from csv_analyzer.insights.registry import InsightsRegistry
from csv_analyzer.insights.models import InsightDefinition, InsightType

logger = logging.getLogger(__name__)


class SQLInsightBlock(BaseBlock):
    """
    Execute a YAML-defined SQL insight using DuckDB.
    
    This block:
    1. Loads an insight definition from YAML by name
    2. Loads classified data from upstream blocks
    3. Registers each document type as a DuckDB table
    4. Applies parameter substitution to SQL
    5. Executes the SQL and returns results
    
    Features:
    - Supports all DuckDB SQL features (JOINs, window functions, CTEs)
    - Handles optional tables gracefully
    - Parameter substitution with {{param}} syntax
    - Optional clause handling with {? ... ?} syntax
    """
    
    def run(self) -> Dict[str, str]:
        """
        Execute the YAML-defined SQL insight.
        
        Required parameters:
            insight_name: Name of the insight to execute (from YAML)
            
        Optional parameters:
            Passed through to SQL template (e.g., max_early_minutes)
        
        Returns:
            Dict with 'result' key containing S3 URI of results DataFrame
        """
        # Get insight name
        insight_name = self.require_param("insight_name")
        
        self.logger.info(f"Executing SQL insight: {insight_name}")
        
        # Load insight definition from YAML
        registry = InsightsRegistry()
        registry.load_definitions()
        
        insight = registry.get(insight_name)
        if not insight:
            raise ValueError(f"Insight not found: {insight_name}")
        
        if insight.type != InsightType.SQL:
            raise ValueError(f"Insight '{insight_name}' is not a SQL insight (type={insight.type})")
        
        if not insight.sql:
            raise ValueError(f"Insight '{insight_name}' has no SQL query defined")
        
        self.logger.info(f"Loaded insight: {insight.name} v{insight.version}")
        self.logger.info(f"Required tables: {insight.requires}")
        
        # Load classified data from upstream
        classified_data = self.load_classified_data("data")
        
        if not classified_data:
            self.logger.warning("No classified data available")
            empty_df = pd.DataFrame()
            result_uri = self.save_to_s3("result", empty_df)
            return {"result": result_uri, "skipped": True, "reason": "No classified data"}
        
        self.logger.info(f"Available document types: {list(classified_data.keys())}")
        
        # Check for required tables
        missing_required = [
            t for t in insight.requires 
            if t not in classified_data
        ]
        
        if missing_required:
            self.logger.warning(
                f"⚠️  Skipping {insight_name}: missing required tables: {missing_required}. "
                f"Available: {list(classified_data.keys())}"
            )
            empty_df = pd.DataFrame()
            result_uri = self.save_to_s3("result", empty_df)
            return {
                "result": result_uri, 
                "skipped": True, 
                "reason": f"Missing required tables: {missing_required}"
            }
        
        # Create DuckDB connection and register tables
        conn = duckdb.connect(":memory:")
        
        for doc_type, df in classified_data.items():
            # Register DataFrame as table
            conn.register(doc_type, df)
            self.logger.info(f"Registered table '{doc_type}' with {len(df)} rows, {len(df.columns)} columns")
        
        # Build parameters dict from insight defaults + runtime overrides
        params = {}
        
        # Add insight parameter defaults
        for param_def in insight.parameters:
            params[param_def.name] = param_def.default
        
        # Override with runtime parameters from workflow
        for key in self.ctx.parameters:
            if key != "insight_name":  # Don't include the insight name itself
                params[key] = self.ctx.parameters[key]
        
        self.logger.info(f"Parameters: {params}")
        
        # Prepare SQL with parameter substitution
        prepared_sql = self._prepare_sql(insight.sql, params)
        
        self.logger.info(f"Executing SQL ({len(prepared_sql)} chars)")
        self.logger.debug(f"SQL:\n{prepared_sql}")
        
        # Execute SQL
        try:
            result_df = conn.execute(prepared_sql).df()
            self.logger.info(f"Query returned {len(result_df)} rows")
        except Exception as e:
            self.logger.error(f"SQL execution failed: {e}")
            self.logger.error(f"SQL:\n{prepared_sql}")
            raise
        finally:
            conn.close()
        
        # Save result
        result_uri = self.save_to_s3("result", result_df)
        
        return {"result": result_uri}
    
    def _prepare_sql(self, sql: str, parameters: Dict[str, Any]) -> str:
        """
        Prepare SQL query with parameter substitution.
        
        Replaces {{param_name}} placeholders with values.
        Handles optional WHERE clauses with {?condition?} syntax.
        
        Optional clauses (wrapped in {? ... ?}) are removed entirely if 
        the parameter inside them is None.
        
        Args:
            sql: Raw SQL template
            parameters: Dict of parameter values
            
        Returns:
            Prepared SQL with parameters substituted
        """
        result_sql = sql
        
        # First, handle optional clauses: {?...?}
        # If the clause contains a parameter that is None, remove the entire clause
        optional_pattern = r'\{\?\s*(.*?)\s*\?\}'
        
        def replace_optional(match):
            clause = match.group(1)
            # Find all {{param}} in this clause
            param_pattern = r'\{\{(\w+)\}\}'
            params_in_clause = re.findall(param_pattern, clause)
            
            # If any parameter is None, remove the entire clause
            for param_name in params_in_clause:
                if parameters.get(param_name) is None:
                    return ''  # Remove this optional clause
            
            # All parameters have values, so substitute them and keep the clause
            result_clause = clause
            for param_name in params_in_clause:
                value = parameters.get(param_name)
                if isinstance(value, str):
                    safe_value = f"'{value}'"
                elif isinstance(value, bool):
                    safe_value = 'true' if value else 'false'
                else:
                    safe_value = str(value)
                result_clause = result_clause.replace(f"{{{{{param_name}}}}}", safe_value)
            
            return result_clause
        
        result_sql = re.sub(optional_pattern, replace_optional, result_sql, flags=re.DOTALL)
        
        # Now replace any remaining {{param}} placeholders (non-optional ones)
        for name, value in parameters.items():
            placeholder = f"{{{{{name}}}}}"
            if placeholder in result_sql:
                if value is not None:
                    if isinstance(value, str):
                        safe_value = f"'{value}'"
                    elif isinstance(value, bool):
                        safe_value = 'true' if value else 'false'
                    else:
                        safe_value = str(value)
                    result_sql = result_sql.replace(placeholder, safe_value)
                else:
                    result_sql = result_sql.replace(placeholder, "NULL")
        
        # Clean up extra whitespace
        result_sql = re.sub(r'\n\s*\n', '\n', result_sql)
        result_sql = result_sql.strip()
        
        return result_sql


# Register the block
@BlockRegistry.register(
    name="sql_insight",
    inputs=[
        {"name": "data", "ontology": DataType.CLASSIFIED_DATA, "required": True}
    ],
    outputs=[
        {"name": "result", "ontology": DataType.INSIGHT_RESULT}
    ],
    parameters=[
        {
            "name": "insight_name", 
            "type": "string", 
            "required": True,
            "description": "Name of the YAML-defined insight to execute"
        },
        # Additional parameters are passed through to the SQL template
    ],
    block_class=SQLInsightBlock,
    description="Execute a YAML-defined DuckDB SQL insight",
)
def sql_insight(ctx: BlockContext) -> Dict[str, str]:
    """Execute a YAML-defined SQL insight."""
    return SQLInsightBlock(ctx).run()

