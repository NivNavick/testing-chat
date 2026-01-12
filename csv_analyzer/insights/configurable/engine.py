"""
ConfigurableInsightEngine - High-level orchestrator for AI-driven insights.

This engine:
1. Loads insight configurations from PostgreSQL
2. Validates prerequisites (required tables)
3. Loads optional tables for context enrichment
4. Prepares table information for the InsightAgent
5. Executes insights and logs results
"""

import logging
from typing import Any, Dict, List, Optional
from uuid import UUID

import duckdb

from csv_analyzer.insights.configurable.models import (
    ConfigurableInsightDefinition,
    ConfigurableInsightResult,
    InsightExecution,
    InsightSummary,
    TableInfo,
)
from csv_analyzer.insights.configurable.repository import ConfigurableInsightRepository
from csv_analyzer.insights.configurable.enrichment import EmployeeContextEnricher
from csv_analyzer.intelligence.insight_agent import InsightAgent

logger = logging.getLogger(__name__)


class PrerequisiteError(Exception):
    """Raised when required tables are not available."""
    pass


class InsightNotFoundError(Exception):
    """Raised when insight is not found."""
    pass


class ConfigurableInsightEngine:
    """
    High-level orchestrator for configurable insights.
    
    Ties together:
    - InsightRepository: Configuration storage
    - InsightAgent: AI-powered query generation and execution
    - EmployeeContextEnricher: Employee metadata for context
    - DuckDB: Query execution
    
    Usage:
        engine = ConfigurableInsightEngine(duckdb_conn)
        
        # List available insights for loaded tables
        available = await engine.list_available_insights(["employee_shifts", "medical_actions"])
        
        # Run an insight
        result = await engine.run_insight("shift_fraud_detection", session_id)
    """
    
    def __init__(
        self,
        duckdb_connection: duckdb.DuckDBPyConnection,
        model: str = None,
        api_key: str = None,
    ):
        """
        Initialize the engine.
        
        Args:
            duckdb_connection: DuckDB connection with loaded session tables
            model: LLM model for AI operations (optional)
            api_key: OpenAI API key (optional)
        """
        self.conn = duckdb_connection
        self.repository = ConfigurableInsightRepository()
        self.agent = InsightAgent(
            duckdb_connection=duckdb_connection,
            model=model,
            api_key=api_key
        )
        self.enricher = EmployeeContextEnricher(duckdb_connection)
    
    # =========================================================================
    # Insight Listing
    # =========================================================================
    
    async def list_all_insights(
        self,
        category: Optional[str] = None,
        active_only: bool = True
    ) -> List[InsightSummary]:
        """
        List all insight definitions.
        
        Args:
            category: Optional filter by category
            active_only: If True, only return active insights
            
        Returns:
            List of InsightSummary objects
        """
        return await self.repository.list_insights(
            category=category,
            active_only=active_only
        )
    
    async def list_available_insights(
        self,
        available_tables: List[str]
    ) -> List[InsightSummary]:
        """
        List insights that can run with the available tables.
        
        Args:
            available_tables: Document types loaded in the session
            
        Returns:
            List of InsightSummary for runnable insights
        """
        return await self.repository.list_insights_for_tables(available_tables)
    
    async def get_insight(self, name: str) -> Optional[ConfigurableInsightDefinition]:
        """
        Get full insight definition with rules and severities.
        
        Args:
            name: Insight name
            
        Returns:
            ConfigurableInsightDefinition or None
        """
        return await self.repository.get_insight(name)
    
    # =========================================================================
    # Rule Management
    # =========================================================================
    
    async def update_rule(
        self,
        insight_name: str,
        rule_name: str,
        value: Any
    ) -> Dict[str, Any]:
        """
        Update a rule's value.
        
        Args:
            insight_name: Name of the insight
            rule_name: Name of the rule to update
            value: New value
            
        Returns:
            Updated rule info
        """
        rule = await self.repository.update_rule(insight_name, rule_name, value)
        return {
            "rule_name": rule.rule_name,
            "value": rule.current_value,
            "value_type": rule.value_type.value,
            "updated": True
        }
    
    async def reset_rule(
        self,
        insight_name: str,
        rule_name: str
    ) -> Dict[str, Any]:
        """
        Reset a rule to its default value.
        
        Args:
            insight_name: Name of the insight
            rule_name: Name of the rule to reset
            
        Returns:
            Reset rule info
        """
        rule = await self.repository.reset_rule(insight_name, rule_name)
        return {
            "rule_name": rule.rule_name,
            "value": rule.current_value,
            "default_value": rule.default_value,
            "reset": True
        }
    
    async def reset_all_rules(self, insight_name: str) -> List[Dict[str, Any]]:
        """Reset all rules for an insight to defaults."""
        rules = await self.repository.reset_all_rules(insight_name)
        return [
            {
                "rule_name": r.rule_name,
                "value": r.current_value,
                "default_value": r.default_value
            }
            for r in rules
        ]
    
    # =========================================================================
    # Insight Execution
    # =========================================================================
    
    async def run_insight(
        self,
        insight_name: str,
        session_id: Optional[UUID] = None,
        available_tables: Optional[List[str]] = None,
        table_mapping: Optional[Dict[str, str]] = None,
    ) -> ConfigurableInsightResult:
        """
        Execute an insight.
        
        Args:
            insight_name: Name of the insight to run
            session_id: Optional session ID for logging
            available_tables: Optional list of document types available.
                            If not provided, will be inferred from DuckDB.
            table_mapping: Optional dict mapping document_type -> actual DuckDB table name.
                            If provided, used to resolve actual table names.
                            
        Returns:
            ConfigurableInsightResult with flags and evidence
            
        Raises:
            InsightNotFoundError: If insight doesn't exist
            PrerequisiteError: If required tables are missing
        """
        # Load insight configuration
        insight = await self.repository.get_insight(insight_name)
        if not insight:
            raise InsightNotFoundError(f"Insight '{insight_name}' not found")
        
        if not insight.is_active:
            raise InsightNotFoundError(f"Insight '{insight_name}' is not active")
        
        # Get available tables if not provided
        if available_tables is None:
            available_tables = self._get_loaded_tables()
        
        # Check prerequisites
        missing = self._check_prerequisites(insight.required_tables, available_tables)
        if missing:
            raise PrerequisiteError(
                f"Missing required tables for '{insight_name}': {missing}. "
                f"Available tables: {available_tables}"
            )
        
        # Prepare table info for required tables
        tables = {}
        for doc_type in insight.required_tables:
            # Use mapping if provided, otherwise try to find table name
            if table_mapping and doc_type in table_mapping:
                actual_name = table_mapping[doc_type]
            else:
                actual_name = self._find_table_name(doc_type, available_tables)
            
            if actual_name:
                table_info = self.agent.get_table_info(actual_name)
                if table_info:
                    # Override document_type to use the required name for consistency
                    table_info.document_type = doc_type
                    tables[doc_type] = table_info
        
        if not tables:
            raise PrerequisiteError(
                f"Could not load table info for required tables: {insight.required_tables}"
            )
        
        # Load optional tables for context enrichment
        optional_tables_info = {}
        employee_context = None
        
        if insight.optional_tables:
            for opt_doc_type in insight.optional_tables:
                # Check if this optional table is available
                if table_mapping and opt_doc_type in table_mapping:
                    opt_actual_name = table_mapping[opt_doc_type]
                else:
                    opt_actual_name = self._find_table_name(opt_doc_type, available_tables)
                
                if opt_actual_name:
                    opt_table_info = self.agent.get_table_info(opt_actual_name)
                    if opt_table_info:
                        opt_table_info.document_type = opt_doc_type
                        optional_tables_info[opt_doc_type] = opt_table_info
                        tables[opt_doc_type] = opt_table_info  # Add to tables for AI
                        
                        # If it's employee metadata, enrich context
                        if "salary" in opt_doc_type.lower() or "employee" in opt_doc_type.lower():
                            self.enricher.load_employees(opt_actual_name)
                            employee_context = self.enricher.build_enrichment_context(
                                metadata_table=opt_actual_name
                            )
                            logger.info(f"Loaded employee context from {opt_actual_name}")
        
        logger.info(
            f"Running insight '{insight_name}' with tables: {list(tables.keys())}"
        )
        
        # Execute via agent (pass employee context if available)
        result = await self.agent.execute_insight(
            insight, 
            tables,
            employee_context=employee_context
        )
        
        # Log execution
        if result.success:
            await self._log_execution(insight, session_id, result)
        
        return result
    
    def _get_loaded_tables(self) -> List[str]:
        """Get list of tables currently loaded in DuckDB."""
        try:
            tables_df = self.conn.execute("SHOW TABLES").fetchdf()
            return tables_df['name'].tolist() if 'name' in tables_df.columns else []
        except Exception:
            return []
    
    def _check_prerequisites(
        self,
        required: List[str],
        available: List[str]
    ) -> List[str]:
        """
        Check which required tables are missing.
        
        Args:
            required: Required table/document types
            available: Available tables in DuckDB
            
        Returns:
            List of missing tables
        """
        available_lower = [t.lower() for t in available]
        missing = []
        
        for req in required:
            # Check exact match or if available contains the required type
            req_lower = req.lower()
            if req_lower not in available_lower:
                # Also check if any available table contains the document type
                found = any(req_lower in avail for avail in available_lower)
                if not found:
                    missing.append(req)
        
        return missing
    
    def _find_table_name(self, document_type: str, available: List[str]) -> Optional[str]:
        """
        Find the actual table name for a document type.
        
        Args:
            document_type: Document type from insight requirements
            available: Available table names
            
        Returns:
            Actual table name or None
        """
        doc_lower = document_type.lower()
        
        for table in available:
            # Exact match
            if table.lower() == doc_lower:
                return table
            # Contains match (for cases like "table_employee_shifts")
            if doc_lower in table.lower():
                return table
        
        return None
    
    async def _log_execution(
        self,
        insight: ConfigurableInsightDefinition,
        session_id: Optional[UUID],
        result: ConfigurableInsightResult
    ):
        """Log execution to audit table."""
        try:
            execution = InsightExecution(
                insight_id=insight.id,
                session_id=session_id,
                rules_snapshot=result.rules_used,
                correlations_found=[c.model_dump() for c in result.correlations_found],
                generated_sql=result.generated_sql,
                ai_explanation=result.ai_explanation,
                total_records=result.total_records,
                flagged_records=len(result.flagged_records),
                flags_by_severity=result.summary,
                execution_time_ms=result.execution_time_ms,
                success=result.success,
                error_message=result.error
            )
            
            await self.repository.log_execution(execution)
            
        except Exception as e:
            logger.error(f"Failed to log execution: {e}")
    
    # =========================================================================
    # Execution History
    # =========================================================================
    
    async def get_executions(
        self,
        insight_name: str,
        limit: int = 20
    ) -> List[InsightExecution]:
        """
        Get execution history for an insight.
        
        Args:
            insight_name: Name of the insight
            limit: Maximum number of executions to return
            
        Returns:
            List of InsightExecution, most recent first
        """
        return await self.repository.get_executions(insight_name, limit)
    
    async def get_latest_result(
        self,
        session_id: UUID,
        insight_name: str
    ) -> Optional[InsightExecution]:
        """
        Get the most recent execution for a session and insight.
        
        Args:
            session_id: Session UUID
            insight_name: Insight name
            
        Returns:
            Most recent InsightExecution or None
        """
        return await self.repository.get_execution_by_session(session_id, insight_name)


# ============================================================================
# Factory Function
# ============================================================================

def create_insight_engine(
    duckdb_connection: duckdb.DuckDBPyConnection,
    model: str = None,
    api_key: str = None
) -> ConfigurableInsightEngine:
    """
    Factory function to create a ConfigurableInsightEngine.
    
    Args:
        duckdb_connection: DuckDB connection
        model: Optional LLM model
        api_key: Optional OpenAI API key
        
    Returns:
        ConfigurableInsightEngine instance
    """
    return ConfigurableInsightEngine(
        duckdb_connection=duckdb_connection,
        model=model,
        api_key=api_key
    )

