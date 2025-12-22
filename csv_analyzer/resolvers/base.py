"""
Base data classes for column mapping conflict resolution.

These classes represent the core concepts:
- MappingConflict: When multiple source columns want the same target
- ResolvedMapping: The final mapping for a column (with conflict metadata)
- ResolutionResult: Complete result from the resolver
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class ResolutionStrategy(Enum):
    """How a mapping was resolved."""
    NO_CONFLICT = "no_conflict"          # Normal mapping, no conflict
    WINNER = "winner"                     # Won the conflict (highest confidence)
    REASSIGNED = "reassigned"             # Lost but got alternative target
    UNMAPPED = "unmapped"                 # Lost, no valid alternative
    OPENAI_ARBITRATED = "openai_arbitrated"  # OpenAI broke the tie


@dataclass
class MappingConflict:
    """
    Represents a conflict where multiple source columns want the same target field.
    
    Example:
        target_field: "employee_id"
        contenders: [
            {"source": "emp_num", "confidence": 0.92, "alternatives": [...]},
            {"source": "emp_code", "confidence": 0.88, "alternatives": [...]}
        ]
    """
    target_field: str
    target_required: bool
    contenders: List[Dict[str, Any]]  # Sorted by confidence (highest first)
    
    # Filled after resolution
    winner: Optional[str] = None
    resolution_strategy: ResolutionStrategy = ResolutionStrategy.NO_CONFLICT
    resolution_reason: str = ""
    
    @property
    def confidence_gap(self) -> float:
        """Gap between top two contenders. Large gap = clear winner."""
        if len(self.contenders) < 2:
            return 1.0
        return self.contenders[0]["confidence"] - self.contenders[1]["confidence"]
    
    @property
    def is_ambiguous(self) -> bool:
        """True if top contenders are too close to call (gap < 2%)."""
        return self.confidence_gap < 0.02
    
    @property
    def loser_sources(self) -> List[str]:
        """List of source columns that lost this conflict."""
        if not self.winner:
            return []
        return [c["source"] for c in self.contenders if c["source"] != self.winner]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_field": self.target_field,
            "target_required": self.target_required,
            "winner": self.winner,
            "losers": self.loser_sources,
            "confidence_gap": round(self.confidence_gap, 4),
            "resolution": self.resolution_strategy.value,
            "reason": self.resolution_reason,
        }


@dataclass
class ResolvedMapping:
    """
    Final resolved mapping for a single source column.
    
    Contains both the mapping result and metadata about any conflicts.
    """
    source_column: str
    target_field: Optional[str]
    confidence: float
    field_type: Optional[str] = None
    required: bool = False
    
    # Conflict metadata
    had_conflict: bool = False
    resolution_strategy: ResolutionStrategy = ResolutionStrategy.NO_CONFLICT
    original_target: Optional[str] = None  # If reassigned, what was first choice
    conflict_details: Optional[str] = None
    
    # Transformation info (preserved from original logic)
    transformation: Optional[Dict[str, Any]] = None
    
    # OpenAI metadata (preserved from original logic)
    source: str = "embeddings"  # "embeddings", "openai_fallback", "openai_verified"
    reason: Optional[str] = None
    attempts: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to the format expected by ScoringResult."""
        result = {
            "target_field": self.target_field,
            "confidence": round(self.confidence, 4) if self.confidence else 0.0,
            "field_type": self.field_type,
            "required": self.required,
        }
        
        # Add conflict info if present
        if self.had_conflict:
            result["conflict"] = {
                "resolution": self.resolution_strategy.value,
                "original_target": self.original_target,
                "details": self.conflict_details,
            }
        
        # Add transformation if present
        if self.transformation:
            result["transformation"] = self.transformation
        
        # Add source/reason if not default
        if self.source != "embeddings":
            result["source"] = self.source
        if self.reason:
            result["reason"] = self.reason
        if self.attempts:
            result["attempts"] = self.attempts
        
        return result


@dataclass
class ResolutionResult:
    """
    Complete result from the conflict resolver.
    
    Contains all mappings plus detected conflicts for audit/debugging.
    """
    mappings: Dict[str, ResolvedMapping]
    conflicts: List[MappingConflict]
    
    # Summary stats
    total_columns: int = 0
    mapped_count: int = 0
    conflict_count: int = 0
    unmapped_count: int = 0
    
    def to_suggested_mappings(self) -> Dict[str, Dict[str, Any]]:
        """Convert to the format expected by ScoringResult.suggested_mappings."""
        return {
            col: mapping.to_dict()
            for col, mapping in self.mappings.items()
        }
    
    def get_conflict_summary(self) -> List[Dict[str, Any]]:
        """Get a summary of all conflicts for logging/debugging."""
        return [c.to_dict() for c in self.conflicts]

