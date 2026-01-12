"""
API Routes for Configurable Insights.

Provides REST endpoints for:
- Listing and viewing insight configurations
- Updating rule values
- Running insights on session data
- Viewing execution history
"""

import logging
from typing import Any, Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel

from csv_analyzer.insights.configurable import (
    ConfigurableInsightRepository,
    ConfigurableInsightDefinition,
    InsightSummary,
    InsightExecution,
    UpdateRuleRequest,
    PrerequisiteError,
    InsightNotFoundError,
)
from csv_analyzer.api.routes.pipeline import get_session, Session

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/insights", tags=["Configurable Insights"])


# ============================================================================
# Response Models
# ============================================================================

class RuleResponse(BaseModel):
    """Response for a single rule."""
    rule_name: str
    value_type: str
    current_value: Any
    default_value: Any
    min_value: Optional[Any] = None
    max_value: Optional[Any] = None
    description: str
    ai_hint: Optional[str] = None


class SeverityResponse(BaseModel):
    """Response for a severity mapping."""
    condition_name: str
    severity: str
    condition_description: str


class InsightDetailResponse(BaseModel):
    """Full insight detail response."""
    id: str
    name: str
    display_name: Optional[str]
    description: Optional[str]
    goal: str
    required_tables: List[str]
    category: Optional[str]
    is_active: bool
    rules: List[RuleResponse]
    severities: List[SeverityResponse]


class UpdateRuleResponse(BaseModel):
    """Response after updating a rule."""
    rule_name: str
    value: Any
    value_type: str
    updated: bool


class RunInsightResponse(BaseModel):
    """Response from running an insight."""
    insight_name: str
    success: bool
    error: Optional[str] = None
    total_records: int
    flagged_count: int
    summary: Dict[str, int]
    correlations_found: List[Dict[str, Any]]
    generated_sql: str
    ai_explanation: str
    rules_used: Dict[str, Any]
    execution_time_ms: float
    flags: List[Dict[str, Any]]


class ExecutionHistoryResponse(BaseModel):
    """Response for execution history."""
    id: str
    executed_at: str
    session_id: Optional[str]
    total_records: int
    flagged_records: int
    flags_by_severity: Dict[str, int]
    execution_time_ms: float
    success: bool
    error_message: Optional[str]


# ============================================================================
# Configuration Endpoints
# ============================================================================

@router.get("", response_model=List[InsightSummary])
async def list_insights(
    category: Optional[str] = None,
    active_only: bool = True
):
    """
    List all insight definitions.
    
    Args:
        category: Optional filter by category (e.g., "compliance", "payroll")
        active_only: If True, only return active insights
        
    Returns:
        List of insight summaries
    """
    try:
        return await ConfigurableInsightRepository.list_insights(
            category=category,
            active_only=active_only
        )
    except Exception as e:
        logger.error(f"Failed to list insights: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/available/{session_id}", response_model=List[InsightSummary])
async def list_available_insights(session_id: str):
    """
    List insights that can run with the tables loaded in a session.
    
    Args:
        session_id: Session ID to check
        
    Returns:
        List of insights whose required_tables are all available
    """
    try:
        session = get_session(session_id)
        
        # Get document types from loaded tables
        available_tables = [
            table.document_type 
            for table in session.tables.values()
        ]
        
        return await ConfigurableInsightRepository.list_insights_for_tables(available_tables)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to list available insights: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/{name}", response_model=InsightDetailResponse)
async def get_insight(name: str):
    """
    Get full insight definition with rules and severities.
    
    Args:
        name: Insight name
        
    Returns:
        Complete insight configuration
    """
    try:
        insight = await ConfigurableInsightRepository.get_insight(name)
        
        if not insight:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Insight '{name}' not found"
            )
        
        return InsightDetailResponse(
            id=str(insight.id),
            name=insight.name,
            display_name=insight.display_name,
            description=insight.description,
            goal=insight.goal,
            required_tables=insight.required_tables,
            category=insight.category,
            is_active=insight.is_active,
            rules=[
                RuleResponse(
                    rule_name=r.rule_name,
                    value_type=r.value_type.value,
                    current_value=r.current_value,
                    default_value=r.default_value,
                    min_value=r.min_value,
                    max_value=r.max_value,
                    description=r.description,
                    ai_hint=r.ai_hint
                )
                for r in insight.rules
            ],
            severities=[
                SeverityResponse(
                    condition_name=s.condition_name,
                    severity=s.severity.value,
                    condition_description=s.condition_description
                )
                for s in insight.severities
            ]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get insight '{name}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# ============================================================================
# Rule Management Endpoints
# ============================================================================

@router.put("/{name}/rules/{rule_name}", response_model=UpdateRuleResponse)
async def update_rule(name: str, rule_name: str, body: UpdateRuleRequest):
    """
    Update a rule's current value.
    
    Args:
        name: Insight name
        rule_name: Rule name to update
        body: New value
        
    Returns:
        Updated rule info
    """
    try:
        rule = await ConfigurableInsightRepository.update_rule(
            insight_name=name,
            rule_name=rule_name,
            value=body.value
        )
        
        return UpdateRuleResponse(
            rule_name=rule.rule_name,
            value=rule.current_value,
            value_type=rule.value_type.value,
            updated=True
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Failed to update rule '{rule_name}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/{name}/rules/{rule_name}/reset", response_model=UpdateRuleResponse)
async def reset_rule(name: str, rule_name: str):
    """
    Reset a rule to its default value.
    
    Args:
        name: Insight name
        rule_name: Rule name to reset
        
    Returns:
        Reset rule info
    """
    try:
        rule = await ConfigurableInsightRepository.reset_rule(
            insight_name=name,
            rule_name=rule_name
        )
        
        return UpdateRuleResponse(
            rule_name=rule.rule_name,
            value=rule.current_value,
            value_type=rule.value_type.value,
            updated=True
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Failed to reset rule '{rule_name}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/{name}/rules/reset-all")
async def reset_all_rules(name: str):
    """
    Reset all rules for an insight to their default values.
    
    Args:
        name: Insight name
        
    Returns:
        List of reset rules
    """
    try:
        rules = await ConfigurableInsightRepository.reset_all_rules(name)
        
        return {
            "insight": name,
            "rules_reset": len(rules),
            "rules": [
                {
                    "rule_name": r.rule_name,
                    "value": r.current_value,
                    "default_value": r.default_value
                }
                for r in rules
            ]
        }
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Failed to reset all rules for '{name}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# ============================================================================
# Execution Endpoints
# ============================================================================

@router.post("/sessions/{session_id}/run/{insight_name}", response_model=RunInsightResponse)
async def run_insight(session_id: str, insight_name: str):
    """
    Run an insight on session data.
    
    Args:
        session_id: Session with loaded tables
        insight_name: Name of the insight to run
        
    Returns:
        Insight execution results with flags
    """
    from csv_analyzer.insights.configurable.engine import create_insight_engine
    
    try:
        # Get session
        session = get_session(session_id)
        
        # Build mapping from document_type -> actual table_name
        # This maps the required_tables (document types) to actual DuckDB table names
        table_mapping = {
            table.document_type: table.table_name
            for table in session.tables.values()
        }
        
        # Get available document types
        available_tables = list(table_mapping.keys())
        
        # Create engine with session's DuckDB connection
        engine = create_insight_engine(session.duckdb_conn)
        
        # Run insight with table mapping
        result = await engine.run_insight(
            insight_name=insight_name,
            session_id=UUID(session_id) if session_id else None,
            available_tables=available_tables,
            table_mapping=table_mapping
        )
        
        return RunInsightResponse(
            insight_name=result.insight_name,
            success=result.success,
            error=result.error,
            total_records=result.total_records,
            flagged_count=len(result.flagged_records),
            summary=result.summary,
            correlations_found=[c.model_dump() for c in result.correlations_found],
            generated_sql=result.generated_sql,
            ai_explanation=result.ai_explanation,
            rules_used=result.rules_used,
            execution_time_ms=result.execution_time_ms,
            flags=[f.model_dump() for f in result.flagged_records]
        )
        
    except PrerequisiteError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except InsightNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to run insight '{insight_name}': {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# ============================================================================
# Execution History Endpoints
# ============================================================================

@router.get("/{name}/executions", response_model=List[ExecutionHistoryResponse])
async def get_executions(name: str, limit: int = 20):
    """
    Get execution history for an insight.
    
    Args:
        name: Insight name
        limit: Maximum number of executions to return
        
    Returns:
        List of executions, most recent first
    """
    try:
        executions = await ConfigurableInsightRepository.get_executions(name, limit)
        
        return [
            ExecutionHistoryResponse(
                id=str(e.id),
                executed_at=e.executed_at.isoformat() if e.executed_at else "",
                session_id=str(e.session_id) if e.session_id else None,
                total_records=e.total_records,
                flagged_records=e.flagged_records,
                flags_by_severity=e.flags_by_severity,
                execution_time_ms=e.execution_time_ms,
                success=e.success,
                error_message=e.error_message
            )
            for e in executions
        ]
        
    except Exception as e:
        logger.error(f"Failed to get executions for '{name}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/sessions/{session_id}/results/{insight_name}")
async def get_session_result(session_id: str, insight_name: str):
    """
    Get the most recent execution result for a session and insight.
    
    Args:
        session_id: Session ID
        insight_name: Insight name
        
    Returns:
        Most recent execution or 404 if not found
    """
    try:
        execution = await ConfigurableInsightRepository.get_execution_by_session(
            UUID(session_id),
            insight_name
        )
        
        if not execution:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No execution found for insight '{insight_name}' in session '{session_id}'"
            )
        
        return ExecutionHistoryResponse(
            id=str(execution.id),
            executed_at=execution.executed_at.isoformat() if execution.executed_at else "",
            session_id=str(execution.session_id) if execution.session_id else None,
            total_records=execution.total_records,
            flagged_records=execution.flagged_records,
            flags_by_severity=execution.flags_by_severity,
            execution_time_ms=execution.execution_time_ms,
            success=execution.success,
            error_message=execution.error_message
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get session result: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# ============================================================================
# Activation Endpoints
# ============================================================================

@router.post("/{name}/activate")
async def activate_insight(name: str):
    """Activate an insight."""
    try:
        success = await ConfigurableInsightRepository.set_active(name, True)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Insight '{name}' not found"
            )
        return {"insight": name, "is_active": True}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to activate insight '{name}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/{name}/deactivate")
async def deactivate_insight(name: str):
    """Deactivate an insight."""
    try:
        success = await ConfigurableInsightRepository.set_active(name, False)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Insight '{name}' not found"
            )
        return {"insight": name, "is_active": False}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to deactivate insight '{name}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

