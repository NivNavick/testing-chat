"""
AI Prompts for the Configurable Insights Framework.

This module contains prompt templates for:
- Correlation discovery: Finding how tables relate
- Query generation: Creating SQL with rules applied
"""

from csv_analyzer.intelligence.prompts.correlation import CORRELATION_PROMPT
from csv_analyzer.intelligence.prompts.query_generation import QUERY_GENERATION_PROMPT

__all__ = [
    "CORRELATION_PROMPT",
    "QUERY_GENERATION_PROMPT",
]

