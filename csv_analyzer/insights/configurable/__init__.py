"""
Configurable Insights Module.

This module provides the AI-driven configurable insights framework:
- InsightDefinition: Database-stored insight configurations
- InsightRule: Configurable rules with AI-interpretable descriptions
- InsightAgent: AI engine that generates queries dynamically
- ConfigurableInsightEngine: High-level orchestrator
"""

from csv_analyzer.insights.configurable.models import (
    InsightRule,
    SeverityMapping,
    ConfigurableInsightDefinition,
    InsightFlag,
    ConfigurableInsightResult,
    Correlation,
    TableInfo,
    InsightExecution,
    InsightSummary,
    ValueType,
    Severity,
    UpdateRuleRequest,
    RunInsightRequest,
)

from csv_analyzer.insights.configurable.repository import ConfigurableInsightRepository

from csv_analyzer.insights.configurable.engine import (
    ConfigurableInsightEngine,
    PrerequisiteError,
    InsightNotFoundError,
    create_insight_engine,
)

from csv_analyzer.insights.configurable.enrichment import (
    EmployeeContextEnricher,
    EmployeeContext,
    ROLE_TO_PROCEDURES,
    HEBREW_CITIES,
)

__all__ = [
    # Models
    "InsightRule",
    "SeverityMapping", 
    "ConfigurableInsightDefinition",
    "InsightFlag",
    "ConfigurableInsightResult",
    "Correlation",
    "TableInfo",
    "InsightExecution",
    "InsightSummary",
    "ValueType",
    "Severity",
    "UpdateRuleRequest",
    "RunInsightRequest",
    # Repository
    "ConfigurableInsightRepository",
    # Engine
    "ConfigurableInsightEngine",
    "PrerequisiteError",
    "InsightNotFoundError",
    "create_insight_engine",
    # Enrichment
    "EmployeeContextEnricher",
    "EmployeeContext",
    "ROLE_TO_PROCEDURES",
    "HEBREW_CITIES",
]

