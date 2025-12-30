"""
Main Insights Engine - Orchestrates CSV loading, classification, and insight execution.
"""

import logging
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Optional, Union

import pandas as pd

from csv_analyzer.insights.data_store import DataStore
from csv_analyzer.insights.models import (
    DataStoreStatus,
    InsightDefinition,
    InsightResult,
    LoadedTable,
)
from csv_analyzer.insights.registry import InsightsRegistry

logger = logging.getLogger(__name__)


class InsightsEngine:
    """
    Main orchestrator for the Insights Engine.
    
    Combines:
    - CSV classification (using existing ClassificationEngine)
    - Data normalization and loading into DuckDB
    - Insight definition management
    - Query execution with parameterization
    
    Usage:
        engine = InsightsEngine(vertical="medical")
        
        # Load CSVs (auto-classifies and normalizes)
        engine.load_csv("shifts.csv")
        engine.load_csv("payroll.csv")
        
        # Run insights
        result = engine.run_insight("cost_per_shift")
        result.to_csv("output.csv")
        
        # Or run all applicable insights
        all_results = engine.run_all_insights()
    """
    
    def __init__(
        self,
        vertical: str = "medical",
        database_path: Optional[str] = None,
        definitions_path: Optional[Path] = None,
    ):
        """
        Initialize the Insights Engine.
        
        Args:
            vertical: The vertical context (e.g., "medical")
            database_path: Path to DuckDB file. None = in-memory.
            definitions_path: Path to insight YAML definitions. None = default.
        """
        self.vertical = vertical
        self.data_store = DataStore(database_path)
        self.registry = InsightsRegistry(definitions_path)
        
        # Lazy-load classification engine
        self._classification_engine = None
        self._embeddings_client = None
        
        # Load insight definitions
        self.registry.load_definitions()
        
        logger.info(
            f"InsightsEngine initialized: vertical={vertical}, "
            f"{len(self.registry.list_names())} insights available"
        )
    
    @property
    def classification_engine(self):
        """Lazy-load the classification engine."""
        if self._classification_engine is None:
            from csv_analyzer.engines.classification_engine import ClassificationEngine
            from csv_analyzer.multilingual_embeddings_client import (
                get_multilingual_embeddings_client,
            )
            
            self._embeddings_client = get_multilingual_embeddings_client()
            self._classification_engine = ClassificationEngine(self._embeddings_client)
        return self._classification_engine
    
    def load_csv(
        self,
        csv_file: Union[str, Path, BinaryIO, pd.DataFrame],
        document_type: Optional[str] = None,
        column_mappings: Optional[Dict[str, str]] = None,
        replace: bool = True,
    ) -> LoadedTable:
        """
        Load a CSV file into the data store.
        
        If document_type and column_mappings are not provided,
        the file will be classified automatically.
        
        Args:
            csv_file: Path to CSV, file object, or DataFrame
            document_type: Override classification (e.g., "employee_shifts")
            column_mappings: Override column mappings (original -> schema field)
            replace: If True, replace existing table. If False, append.
            
        Returns:
            LoadedTable with metadata about the loaded data
        """
        # Load the CSV
        if isinstance(csv_file, pd.DataFrame):
            df = csv_file
            source_file = "DataFrame"
        elif isinstance(csv_file, (str, Path)):
            df = pd.read_csv(csv_file)
            source_file = str(csv_file)
        else:
            df = pd.read_csv(csv_file)
            source_file = "file_object"
        
        logger.info(f"Loaded CSV: {source_file} ({len(df)} rows, {len(df.columns)} columns)")
        
        # Classify if not provided
        if document_type is None or column_mappings is None:
            classification = self.classification_engine.classify_hybrid(
                df,
                vertical=self.vertical,
            )
            
            if document_type is None:
                document_type = classification.document_type
                logger.info(
                    f"Classified as: {document_type} "
                    f"(confidence: {classification.final_score:.2f})"
                )
            
            if column_mappings is None:
                # Extract mappings from classification result
                column_mappings = {}
                for col_name, mapping_info in classification.suggested_mappings.items():
                    if isinstance(mapping_info, dict):
                        target = mapping_info.get("target") or mapping_info.get("field_name")
                        if target:
                            column_mappings[col_name] = target
                    elif mapping_info:
                        column_mappings[col_name] = str(mapping_info)
                
                logger.info(f"Column mappings: {column_mappings}")
            
            confidence = classification.final_score
        else:
            confidence = 1.0  # User-provided, assume confident
        
        if not document_type:
            raise ValueError(
                f"Could not classify CSV: {source_file}. "
                "Please provide document_type manually."
            )
        
        # Load into data store
        return self.data_store.load_dataframe(
            df=df,
            document_type=document_type,
            column_mappings=column_mappings or {},
            source_file=source_file,
            classification_confidence=confidence,
            replace=replace,
        )
    
    def load_csv_manual(
        self,
        csv_file: Union[str, Path, pd.DataFrame],
        document_type: str,
        column_mappings: Dict[str, str],
        replace: bool = True,
    ) -> LoadedTable:
        """
        Load a CSV with explicit document type and mappings (no classification).
        
        Faster than load_csv() when you know the document type.
        """
        if isinstance(csv_file, pd.DataFrame):
            df = csv_file
            source_file = "DataFrame"
        else:
            df = pd.read_csv(csv_file)
            source_file = str(csv_file)
        
        return self.data_store.load_dataframe(
            df=df,
            document_type=document_type,
            column_mappings=column_mappings,
            source_file=source_file,
            classification_confidence=1.0,
            replace=replace,
        )
    
    def run_insight(
        self,
        insight_name: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> InsightResult:
        """
        Execute an insight query.
        
        Args:
            insight_name: Name of the insight to run
            parameters: Optional parameters for the query
            
        Returns:
            InsightResult with data or error information
        """
        start_time = time.time()
        parameters = parameters or {}
        
        # Get insight definition
        insight = self.registry.get(insight_name)
        if not insight:
            return InsightResult(
                insight_name=insight_name,
                success=False,
                error=f"Insight '{insight_name}' not found. "
                      f"Available: {self.registry.list_names()}",
            )
        
        # Validate requirements
        loaded_tables = self.data_store.list_tables()
        validation = self.registry.validate_insight(insight_name, loaded_tables)
        
        if not validation["valid"]:
            return InsightResult(
                insight_name=insight_name,
                success=False,
                error=f"Missing required tables: {validation['missing_tables']}. "
                      f"Loaded tables: {loaded_tables}",
            )
        
        # Validate and coerce parameters
        try:
            validated_params = insight.validate_parameters(parameters)
        except ValueError as e:
            return InsightResult(
                insight_name=insight_name,
                success=False,
                error=str(e),
            )
        
        # Prepare SQL with parameter substitution
        sql = self._prepare_sql(insight.sql, validated_params)
        
        # Execute query
        try:
            result_df = self.data_store.execute(sql)
            execution_time = (time.time() - start_time) * 1000
            
            return InsightResult(
                insight_name=insight_name,
                success=True,
                data=result_df,
                row_count=len(result_df),
                executed_sql=sql,
                parameters_used=validated_params,
                execution_time_ms=execution_time,
            )
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            logger.error(f"Insight '{insight_name}' failed: {e}")
            return InsightResult(
                insight_name=insight_name,
                success=False,
                error=str(e),
                executed_sql=sql,
                parameters_used=validated_params,
                execution_time_ms=execution_time,
            )
    
    def _prepare_sql(self, sql: str, parameters: Dict[str, Any]) -> str:
        """
        Prepare SQL query with parameter substitution.
        
        Replaces {{param_name}} placeholders with values.
        Handles optional WHERE clauses with {?condition?} syntax.
        
        Optional clauses (wrapped in {? ... ?}) are removed entirely if 
        the parameter inside them is None.
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
                else:
                    safe_value = str(value)
                result_clause = result_clause.replace(f"{{{{{param_name}}}}}", safe_value)
            
            return result_clause
        
        result_sql = re.sub(optional_pattern, replace_optional, result_sql)
        
        # Now replace any remaining {{param}} placeholders (non-optional ones)
        for name, value in parameters.items():
            placeholder = f"{{{{{name}}}}}"
            if placeholder in result_sql:
                if value is not None:
                    if isinstance(value, str):
                        safe_value = f"'{value}'"
                    else:
                        safe_value = str(value)
                    result_sql = result_sql.replace(placeholder, safe_value)
                else:
                    result_sql = result_sql.replace(placeholder, "NULL")
        
        # Clean up extra whitespace
        result_sql = re.sub(r'\n\s*\n', '\n', result_sql)
        result_sql = result_sql.strip()
        
        return result_sql
    
    def run_all_insights(
        self,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, InsightResult]:
        """
        Run all insights that can be executed with currently loaded tables.
        
        Args:
            parameters: Optional parameters to apply to all insights
            
        Returns:
            Dict mapping insight names to their results
        """
        parameters = parameters or {}
        results = {}
        
        loaded_tables = self.data_store.list_tables()
        applicable_insights = self.registry.list_by_requirements(loaded_tables)
        
        logger.info(
            f"Running {len(applicable_insights)} applicable insights "
            f"(of {len(self.registry.list_names())} total)"
        )
        
        for insight in applicable_insights:
            result = self.run_insight(insight.name, parameters)
            results[insight.name] = result
            
            if result.success:
                logger.info(
                    f"  ✓ {insight.name}: {result.row_count} rows "
                    f"({result.execution_time_ms:.1f}ms)"
                )
            else:
                logger.warning(f"  ✗ {insight.name}: {result.error}")
        
        return results
    
    def list_available_insights(self) -> List[str]:
        """List insights that can run with currently loaded data."""
        loaded_tables = self.data_store.list_tables()
        applicable = self.registry.list_by_requirements(loaded_tables)
        return [i.name for i in applicable]
    
    def list_all_insights(self) -> List[InsightDefinition]:
        """List all registered insights."""
        return self.registry.list_all()
    
    def get_insight_info(self, name: str) -> Optional[InsightDefinition]:
        """Get details about a specific insight."""
        return self.registry.get(name)
    
    def get_status(self) -> DataStoreStatus:
        """Get status of loaded data."""
        return self.data_store.get_status()
    
    def get_connection(self):
        """
        Get raw DuckDB connection for custom queries.
        
        Use with caution - bypasses insight management.
        """
        return self.data_store.connection
    
    def execute_sql(self, sql: str) -> pd.DataFrame:
        """
        Execute arbitrary SQL against the data store.
        
        Useful for ad-hoc analysis beyond predefined insights.
        """
        return self.data_store.execute(sql)
    
    def export_all_results(
        self,
        output_dir: Union[str, Path],
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, str]:
        """
        Run all insights and export results to CSV files.
        
        Args:
            output_dir: Directory to save CSV files
            parameters: Optional parameters for insights
            
        Returns:
            Dict mapping insight names to output file paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        results = self.run_all_insights(parameters)
        output_files = {}
        
        for name, result in results.items():
            if result.success and result.data is not None:
                file_path = output_path / f"{name}.csv"
                result.to_csv(str(file_path))
                output_files[name] = str(file_path)
                logger.info(f"Exported {name} to {file_path}")
        
        return output_files
    
    def sample_loaded_data(self, document_type: str, limit: int = 5) -> pd.DataFrame:
        """Get sample data from a loaded table."""
        return self.data_store.sample_table(document_type, limit)
    
    def describe_table(self, document_type: str) -> pd.DataFrame:
        """Get schema of a loaded table."""
        return self.data_store.describe_table(document_type)
    
    def close(self) -> None:
        """Close the engine and release resources."""
        self.data_store.close()
    
    def __enter__(self) -> "InsightsEngine":
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

