"""
Insights Engine - Generic analytical insights over classified CSV data.

This module provides:
- DataStore: DuckDB-based storage for normalized CSV data
- InsightsRegistry: Discover and load insight definitions from YAML
- InsightsEngine: Main orchestrator for loading data and running insights
- CodeInsightsRegistry: Registry for Python code-based insights

Insight Types:
- SQL insights: Define logic in YAML with SQL queries
- Code insights: Define logic in Python handlers for complex algorithms

Usage:
    from csv_analyzer.insights import InsightsEngine
    
    engine = InsightsEngine(vertical="medical")
    engine.load_csv("shifts.csv")
    engine.load_csv("procedures.csv")
    
    # Run SQL or code-based insights with same interface
    results = engine.run_insight("early_arrival_matching")
    results.to_csv("output.csv")
"""

from csv_analyzer.insights.engine import InsightsEngine
from csv_analyzer.insights.data_store import DataStore
from csv_analyzer.insights.registry import InsightsRegistry
from csv_analyzer.insights.models import InsightDefinition, InsightResult, InsightType
from csv_analyzer.insights.code_insights import CodeInsightsRegistry

__all__ = [
    "InsightsEngine",
    "DataStore", 
    "InsightsRegistry",
    "InsightDefinition",
    "InsightResult",
    "InsightType",
    "CodeInsightsRegistry",
]

