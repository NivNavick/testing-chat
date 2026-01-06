"""
Pydantic models for API requests.
"""

from typing import Optional, List
from pydantic import BaseModel, Field


class CreateSessionRequest(BaseModel):
    """Request to create a new session."""
    vertical: str = Field(..., description="Business vertical (e.g., 'medical')")
    mode: str = Field(
        default="AUTO",
        description="Processing mode: AUTO, GUIDED, STRICT, DISCOVERY"
    )


class RunInsightRequest(BaseModel):
    """Request to run an insight."""
    insight_name: str = Field(..., description="Name of the insight to run")
    parameters: Optional[dict] = Field(
        default=None,
        description="Optional parameters for the insight"
    )


class GenerateInsightRequest(BaseModel):
    """Request to generate a custom insight using LLM."""
    description: str = Field(
        ...,
        description="Natural language description of the desired insight"
    )
    requires_tables: Optional[List[str]] = Field(
        default=None,
        description="List of table IDs to use for the insight"
    )


class AddRelationshipRequest(BaseModel):
    """Request to add a manual relationship between tables."""
    from_table_id: str = Field(..., description="Source table UUID")
    from_column: str = Field(..., description="Source column name")
    to_table_id: str = Field(..., description="Target table UUID")
    to_column: str = Field(..., description="Target column name")
    relationship_type: str = Field(
        default="ONE_TO_MANY",
        description="Relationship type: ONE_TO_ONE, ONE_TO_MANY, MANY_TO_MANY"
    )


class AddRuleRequest(BaseModel):
    """Request to add a manual business rule."""
    rule_type: str = Field(
        ...,
        description="Rule type: FORMULA, CONDITIONAL, LOOKUP"
    )
    description: str = Field(..., description="Rule description")
    applies_to_tables: List[str] = Field(
        default_factory=list,
        description="List of table IDs this rule applies to"
    )
    condition_sql: Optional[str] = Field(
        default=None,
        description="SQL condition for the rule"
    )
    formula_sql: Optional[str] = Field(
        default=None,
        description="SQL formula for the rule"
    )

