"""
Column Mapping Conflict Resolvers.

This module provides the ConflictResolver class for handling cases where
multiple source columns want to map to the same target field.

Usage:
    from csv_analyzer.resolvers import ConflictResolver
    
    resolver = ConflictResolver(openai_fallback=fallback_service)
    result = resolver.resolve(
        column_matches=column_matches,
        winning_doc_type="employee_shifts",
    )
    
    # Access results
    print(result.mappings)    # Final mappings
    print(result.conflicts)   # Detected conflicts
"""

from .base import (
    MappingConflict,
    ResolvedMapping,
    ResolutionResult,
    ResolutionStrategy,
)
from .conflict_resolver import ConflictResolver

__all__ = [
    # Data classes
    "MappingConflict",
    "ResolvedMapping",
    "ResolutionResult",
    "ResolutionStrategy",
    # Resolver
    "ConflictResolver",
]

