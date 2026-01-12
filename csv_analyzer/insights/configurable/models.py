"""
Pydantic models for the Configurable Insights Framework.

These models support the AI-driven insight system where:
- Insights are stored in PostgreSQL (not YAML)
- Rules have descriptions that AI interprets
- AI generates SQL queries dynamically
- Results include flagged records with severity levels
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


# ============================================================================
# Enums
# ============================================================================

class ValueType(str, Enum):
    """Supported value types for insight rules."""
    INTEGER = "integer"
    FLOAT = "float"
    STRING = "string"
    BOOLEAN = "boolean"


class Severity(str, Enum):
    """Severity levels for flagged records."""
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


# ============================================================================
# Rule and Severity Models
# ============================================================================

class InsightRule(BaseModel):
    """
    A configurable rule for an insight.
    
    The description field is KEY - AI reads this to understand
    how to apply the rule in the generated SQL.
    """
    id: Optional[UUID] = None
    rule_name: str = Field(..., description="Unique rule identifier within the insight")
    value_type: ValueType = Field(..., description="Data type of the rule value")
    current_value: Any = Field(..., description="Current configured value")
    default_value: Any = Field(..., description="Default value for reset")
    min_value: Optional[Any] = Field(None, description="Minimum allowed value (for validation)")
    max_value: Optional[Any] = Field(None, description="Maximum allowed value (for validation)")
    description: str = Field(..., description="AI reads this to understand how to apply the rule")
    ai_hint: Optional[str] = Field(None, description="Extra guidance for AI query generation")
    display_order: int = Field(0, description="Order for UI display")
    
    @field_validator('current_value', 'default_value', mode='before')
    @classmethod
    def coerce_value(cls, v, info):
        """Coerce string values from database to proper types."""
        if v is None:
            return v
        # Values come as strings from DB TEXT columns
        return v
    
    def get_typed_value(self) -> Any:
        """Get current_value converted to the proper type."""
        if self.current_value is None:
            return None
        
        if self.value_type == ValueType.INTEGER:
            return int(self.current_value)
        elif self.value_type == ValueType.FLOAT:
            return float(self.current_value)
        elif self.value_type == ValueType.BOOLEAN:
            if isinstance(self.current_value, bool):
                return self.current_value
            return str(self.current_value).lower() in ('true', '1', 'yes')
        else:
            return str(self.current_value)
    
    def validate_new_value(self, value: Any) -> Any:
        """Validate and coerce a new value for this rule."""
        if value is None:
            return self.default_value
        
        # Type coercion
        if self.value_type == ValueType.INTEGER:
            typed_value = int(value)
        elif self.value_type == ValueType.FLOAT:
            typed_value = float(value)
        elif self.value_type == ValueType.BOOLEAN:
            if isinstance(value, bool):
                typed_value = value
            else:
                typed_value = str(value).lower() in ('true', '1', 'yes')
        else:
            typed_value = str(value)
        
        # Range validation for numeric types
        if self.value_type in (ValueType.INTEGER, ValueType.FLOAT):
            if self.min_value is not None and typed_value < float(self.min_value):
                raise ValueError(f"Value {typed_value} is below minimum {self.min_value}")
            if self.max_value is not None and typed_value > float(self.max_value):
                raise ValueError(f"Value {typed_value} is above maximum {self.max_value}")
        
        return typed_value

    class Config:
        from_attributes = True


class SeverityMapping(BaseModel):
    """
    Maps a condition to a severity level.
    
    AI reads condition_description to understand when to apply this severity.
    """
    id: Optional[UUID] = None
    condition_name: str = Field(..., description="Unique condition identifier")
    severity: Severity = Field(..., description="Severity level to assign")
    condition_description: str = Field(..., description="AI reads this to know when to apply")
    display_order: int = Field(0, description="Order for UI display")

    class Config:
        from_attributes = True


# ============================================================================
# Insight Definition Model
# ============================================================================

class ConfigurableInsightDefinition(BaseModel):
    """
    Complete insight definition loaded from PostgreSQL.
    
    This is the main configuration object that contains:
    - The goal (natural language for AI)
    - Required tables (prerequisites)
    - Optional tables (context enrichment)
    - Configurable rules
    - Severity mappings
    """
    id: UUID
    name: str = Field(..., description="Unique insight identifier")
    display_name: Optional[str] = Field(None, description="Human-readable name")
    description: Optional[str] = Field(None, description="Detailed description")
    goal: str = Field(..., description="Natural language goal for AI to achieve")
    required_tables: List[str] = Field(..., description="Tables that must be uploaded")
    optional_tables: List[str] = Field(default_factory=list, description="Optional tables for context enrichment")
    category: Optional[str] = Field(None, description="Category for grouping")
    is_active: bool = Field(True, description="Whether insight is enabled")
    rules: List[InsightRule] = Field(default_factory=list, description="Configurable rules")
    severities: List[SeverityMapping] = Field(default_factory=list, description="Severity conditions")
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    def get_rule(self, rule_name: str) -> Optional[InsightRule]:
        """Get a rule by name."""
        for rule in self.rules:
            if rule.rule_name == rule_name:
                return rule
        return None
    
    def get_rule_value(self, rule_name: str, default: Any = None) -> Any:
        """Get typed value of a rule, or default if not found."""
        rule = self.get_rule(rule_name)
        if rule:
            return rule.get_typed_value()
        return default
    
    def get_severity(self, condition_name: str) -> Optional[Severity]:
        """Get severity for a condition."""
        for sev in self.severities:
            if sev.condition_name == condition_name:
                return sev.severity
        return None

    class Config:
        from_attributes = True


# ============================================================================
# Table Information Model
# ============================================================================

class TableInfo(BaseModel):
    """
    Information about a table available for insight execution.
    
    Includes schema and sample data for AI to understand the structure.
    """
    table_name: str = Field(..., description="Name of the table in DuckDB")
    document_type: str = Field(..., description="Classified document type")
    columns: List[Dict[str, Any]] = Field(..., description="Column definitions with types")
    sample_data: List[Dict[str, Any]] = Field(default_factory=list, description="Sample rows")
    row_count: int = Field(0, description="Total number of rows")
    
    def to_schema_string(self) -> str:
        """Format table schema for AI prompt."""
        lines = [f"Table: {self.table_name} ({self.row_count} rows)"]
        lines.append("Columns:")
        for col in self.columns:
            col_name = col.get('name', col.get('column_name', 'unknown'))
            col_type = col.get('type', col.get('dtype', 'unknown'))
            lines.append(f"  - {col_name} ({col_type})")
        
        if self.sample_data:
            lines.append("Sample values:")
            for i, row in enumerate(self.sample_data[:3]):
                lines.append(f"  Row {i+1}: {row}")
        
        return "\n".join(lines)


# ============================================================================
# Correlation Model (AI-discovered)
# ============================================================================

class Correlation(BaseModel):
    """
    A correlation between tables discovered by AI.
    
    Describes how two tables can be joined/related.
    """
    correlation_type: str = Field(..., description="Type: date_match, time_overlap, location_contains, exact_match")
    from_table: str = Field(..., description="Source table name")
    from_columns: List[str] = Field(..., description="Source column(s)")
    to_table: str = Field(..., description="Target table name")
    to_columns: List[str] = Field(..., description="Target column(s)")
    description: str = Field(..., description="How these relate")
    sql_hint: Optional[str] = Field(None, description="Suggested SQL pattern")
    confidence: float = Field(1.0, description="AI's confidence in this correlation")


# ============================================================================
# Result Models
# ============================================================================

class InsightFlag(BaseModel):
    """
    A flagged record from insight execution.
    
    Contains the severity, condition that triggered it, 
    the actual data, and an evidence explanation.
    """
    severity: Severity = Field(..., description="Severity level")
    condition: str = Field(..., description="Condition that triggered the flag")
    record_data: Dict[str, Any] = Field(..., description="The actual data row")
    evidence: str = Field(..., description="Human-readable explanation")
    
    class Config:
        from_attributes = True


class ConfigurableInsightResult(BaseModel):
    """
    Complete result from executing a configurable insight.
    
    Includes:
    - AI-generated artifacts (correlations, SQL, explanation)
    - Execution results (flagged records, summary)
    - Performance metrics
    """
    insight_name: str = Field(..., description="Name of the executed insight")
    
    # AI-generated artifacts
    correlations_found: List[Correlation] = Field(default_factory=list, description="How AI correlated tables")
    generated_sql: str = Field(..., description="The AI-generated SQL query")
    ai_explanation: str = Field(..., description="AI's explanation of the query")
    
    # Results
    total_records: int = Field(0, description="Total records analyzed")
    flagged_records: List[InsightFlag] = Field(default_factory=list, description="Records that were flagged")
    summary: Dict[str, int] = Field(default_factory=dict, description="Flags by severity: {CRITICAL: 1, WARNING: 2}")
    
    # Metadata
    rules_used: Dict[str, Any] = Field(default_factory=dict, description="Rule values at execution time")
    execution_time_ms: float = Field(0.0, description="Query execution time")
    executed_at: datetime = Field(default_factory=datetime.now)
    success: bool = Field(True, description="Whether execution succeeded")
    error: Optional[str] = Field(None, description="Error message if failed")
    
    def get_critical_flags(self) -> List[InsightFlag]:
        """Get only CRITICAL severity flags."""
        return [f for f in self.flagged_records if f.severity == Severity.CRITICAL]
    
    def get_warning_flags(self) -> List[InsightFlag]:
        """Get only WARNING severity flags."""
        return [f for f in self.flagged_records if f.severity == Severity.WARNING]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "insight_name": self.insight_name,
            "success": self.success,
            "error": self.error,
            "total_records": self.total_records,
            "flagged_count": len(self.flagged_records),
            "summary": self.summary,
            "correlations_found": [c.model_dump() for c in self.correlations_found],
            "generated_sql": self.generated_sql,
            "ai_explanation": self.ai_explanation,
            "rules_used": self.rules_used,
            "execution_time_ms": self.execution_time_ms,
            "executed_at": self.executed_at.isoformat(),
            "flags": [f.model_dump() for f in self.flagged_records],
        }

    class Config:
        from_attributes = True


# ============================================================================
# Execution Log Model
# ============================================================================

class InsightExecution(BaseModel):
    """
    Record of an insight execution (for audit log).
    """
    id: Optional[UUID] = None
    insight_id: UUID
    session_id: Optional[UUID] = None
    executed_at: datetime = Field(default_factory=datetime.now)
    rules_snapshot: Dict[str, Any] = Field(default_factory=dict)
    correlations_found: List[Dict[str, Any]] = Field(default_factory=list)
    generated_sql: Optional[str] = None
    ai_explanation: Optional[str] = None
    total_records: int = 0
    flagged_records: int = 0
    flags_by_severity: Dict[str, int] = Field(default_factory=dict)
    execution_time_ms: float = 0.0
    success: bool = True
    error_message: Optional[str] = None

    class Config:
        from_attributes = True


# ============================================================================
# Request/Response Models for API
# ============================================================================

class UpdateRuleRequest(BaseModel):
    """Request to update a rule's value."""
    value: Any = Field(..., description="New value for the rule")


class InsightSummary(BaseModel):
    """Summary view of an insight for listing."""
    id: UUID
    name: str
    display_name: Optional[str]
    category: Optional[str]
    is_active: bool
    required_tables: List[str]
    optional_tables: List[str] = Field(default_factory=list)
    rule_count: int
    severity_count: int
    created_at: Optional[datetime]
    updated_at: Optional[datetime]

    class Config:
        from_attributes = True


class RunInsightRequest(BaseModel):
    """Request to run an insight on session data."""
    parameters: Optional[Dict[str, Any]] = Field(None, description="Optional override parameters")

