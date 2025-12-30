"""
Insights Engine - Generic analytical insights over classified CSV data.

This module provides:
- DataStore: DuckDB-based storage for normalized CSV data
- InsightsRegistry: Discover and load insight definitions from YAML
- InsightsEngine: Main orchestrator for loading data and running insights

Usage:
    from csv_analyzer.insights import InsightsEngine
    
    engine = InsightsEngine(vertical="medical")
    engine.load_csv("shifts.csv")
    engine.load_csv("payroll.csv")
    
    results = engine.run_insight("cost_per_shift")
    results.to_csv("output.csv")
"""

from csv_analyzer.insights.engine import InsightsEngine
from csv_analyzer.insights.data_store import DataStore
from csv_analyzer.insights.registry import InsightsRegistry
from csv_analyzer.insights.models import InsightDefinition, InsightResult

__all__ = [
    "InsightsEngine",
    "DataStore", 
    "InsightsRegistry",
    "InsightDefinition",
    "InsightResult",
]

