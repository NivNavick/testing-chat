"""
Data models for the Insights Engine.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


class ParameterType(str, Enum):
    """Supported parameter types for insight queries."""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    DATE = "date"
    DATETIME = "datetime"
    BOOLEAN = "boolean"


@dataclass
class InsightParameter:
    """Definition of a parameter for an insight query."""
    name: str
    type: ParameterType
    required: bool = False
    default: Optional[Any] = None
    description: Optional[str] = None
    
    def validate(self, value: Any) -> Any:
        """Validate and coerce a parameter value."""
        if value is None:
            if self.required:
                raise ValueError(f"Required parameter '{self.name}' is missing")
            return self.default
        
        # Type coercion
        if self.type == ParameterType.STRING:
            return str(value)
        elif self.type == ParameterType.INTEGER:
            return int(value)
        elif self.type == ParameterType.FLOAT:
            return float(value)
        elif self.type == ParameterType.DATE:
            if isinstance(value, str):
                return value  # DuckDB handles date strings
            return str(value)
        elif self.type == ParameterType.DATETIME:
            if isinstance(value, str):
                return value
            return str(value)
        elif self.type == ParameterType.BOOLEAN:
            if isinstance(value, bool):
                return value
            return str(value).lower() in ('true', '1', 'yes')
        
        return value


@dataclass
class InsightDefinition:
    """
    Definition of an insight loaded from YAML.
    
    Attributes:
        name: Unique identifier for the insight
        description: Human-readable description
        version: Version of this insight definition
        requires: List of document types (tables) required
        sql: The SQL query to execute
        parameters: Optional parameters for filtering
        category: Optional category for grouping insights
        tags: Optional tags for discovery
    """
    name: str
    description: str
    sql: str
    requires: List[str]
    version: str = "1.0"
    parameters: List[InsightParameter] = field(default_factory=list)
    category: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    source_file: Optional[Path] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], source_file: Optional[Path] = None) -> "InsightDefinition":
        """Create an InsightDefinition from a dictionary (parsed YAML)."""
        parameters = []
        for param in data.get("parameters", []):
            parameters.append(InsightParameter(
                name=param["name"],
                type=ParameterType(param.get("type", "string")),
                required=param.get("required", False),
                default=param.get("default"),
                description=param.get("description"),
            ))
        
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            version=data.get("version", "1.0"),
            requires=data.get("requires", []),
            sql=data["sql"],
            parameters=parameters,
            category=data.get("category"),
            tags=data.get("tags", []),
            source_file=source_file,
        )
    
    def validate_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and coerce all parameters."""
        validated = {}
        for param_def in self.parameters:
            value = params.get(param_def.name)
            validated[param_def.name] = param_def.validate(value)
        return validated


@dataclass
class LoadedTable:
    """Information about a loaded table in the data store."""
    document_type: str
    source_file: str
    row_count: int
    columns: List[str]
    loaded_at: datetime = field(default_factory=datetime.now)
    classification_confidence: float = 0.0
    column_mappings: Dict[str, str] = field(default_factory=dict)


@dataclass
class InsightResult:
    """
    Result of running an insight.
    
    Attributes:
        insight_name: Name of the insight that was run
        success: Whether the insight ran successfully
        data: The result DataFrame (if successful)
        row_count: Number of rows returned
        executed_sql: The actual SQL that was executed
        parameters_used: Parameters that were applied
        error: Error message if failed
        execution_time_ms: How long the query took
    """
    insight_name: str
    success: bool
    data: Optional[pd.DataFrame] = None
    row_count: int = 0
    executed_sql: Optional[str] = None
    parameters_used: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    executed_at: datetime = field(default_factory=datetime.now)
    
    def to_csv(self, path: str) -> None:
        """Export results to CSV."""
        if self.data is not None:
            self.data.to_csv(path, index=False)
        else:
            raise ValueError(f"No data to export - insight failed: {self.error}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "insight_name": self.insight_name,
            "success": self.success,
            "row_count": self.row_count,
            "parameters_used": self.parameters_used,
            "error": self.error,
            "execution_time_ms": self.execution_time_ms,
            "executed_at": self.executed_at.isoformat(),
        }


@dataclass  
class DataStoreStatus:
    """Status of the data store."""
    tables: List[LoadedTable]
    total_rows: int
    database_path: Optional[str]
    
    def has_table(self, document_type: str) -> bool:
        """Check if a table is loaded."""
        return any(t.document_type == document_type for t in self.tables)
    
    def get_table(self, document_type: str) -> Optional[LoadedTable]:
        """Get info about a loaded table."""
        for t in self.tables:
            if t.document_type == document_type:
                return t
        return None
    
    def missing_tables(self, required: List[str]) -> List[str]:
        """Get list of required tables that are not loaded."""
        loaded = {t.document_type for t in self.tables}
        return [r for r in required if r not in loaded]

