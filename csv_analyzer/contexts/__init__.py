"""
Vertical Context System.

Provides domain-specific knowledge for enhanced column matching.
Each vertical (medical, finance, etc.) has its own context with:
- Semantic descriptions for schema fields
- Domain terminology for OpenAI prompts

Usage:
    from csv_analyzer.contexts import get_vertical_context
    
    context = get_vertical_context("medical")
    if context:
        # Expand embedding text with semantic context
        expanded = context.expand_embedding_text("department_code", base_text)
        
        # Get OpenAI prompt context
        openai_context = context.get_openai_context()
"""

from csv_analyzer.contexts.registry import (
    FieldContext,
    VerticalContext,
    VerticalContextRegistry,
    get_context_registry,
    get_vertical_context,
)

__all__ = [
    "FieldContext",
    "VerticalContext", 
    "VerticalContextRegistry",
    "get_context_registry",
    "get_vertical_context",
]

