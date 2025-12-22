"""
End-of-Mapping Conflict Resolver.

Runs AFTER all column scoring is complete to detect and resolve cases
where multiple source columns want to map to the same target field.

Two-phase approach:
1. Detect all conflicts (full visibility)
2. Resolve with configurable strategy

Integrates with OpenAI fallback for ambiguous conflicts.
"""

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from .base import (
    MappingConflict,
    ResolvedMapping,
    ResolutionResult,
    ResolutionStrategy,
)

if TYPE_CHECKING:
    from csv_analyzer.services.openai_fallback import OpenAIFallbackService

logger = logging.getLogger(__name__)


class ConflictResolver:
    """
    End-of-mapping conflict resolver.
    
    Detects when multiple source columns want the same target field
    and resolves using confidence-based priority (highest wins).
    
    Features:
    - Full conflict visibility before resolution
    - Losers try alternative targets
    - Optional OpenAI arbitration for ambiguous cases
    - Detailed audit trail
    
    Usage:
        resolver = ConflictResolver(openai_fallback=fallback_service)
        result = resolver.resolve(
            column_matches={"emp_num": [...], "emp_code": [...]},
            winning_doc_type="employee_shifts",
        )
        
        # Access results
        print(result.mappings)   # Final mappings
        print(result.conflicts)  # Detected conflicts
    """
    
    # Minimum confidence to accept a mapping
    MIN_CONFIDENCE_THRESHOLD = 0.82
    
    # Gap below this = ambiguous (may need OpenAI)
    AMBIGUITY_THRESHOLD = 0.02
    
    def __init__(
        self,
        min_confidence: float = None,
        ambiguity_threshold: float = None,
        openai_fallback: Optional["OpenAIFallbackService"] = None,
    ):
        """
        Initialize the conflict resolver.
        
        Args:
            min_confidence: Minimum confidence to accept a mapping (default 0.82)
            ambiguity_threshold: Gap below which conflicts are ambiguous (default 0.02)
            openai_fallback: Optional OpenAI service for arbitrating ambiguous conflicts
        """
        self.min_confidence = min_confidence or self.MIN_CONFIDENCE_THRESHOLD
        self.ambiguity_threshold = ambiguity_threshold or self.AMBIGUITY_THRESHOLD
        self.openai_fallback = openai_fallback
    
    def resolve(
        self,
        column_matches: Dict[str, List[Dict]],
        winning_doc_type: str,
        column_profiles: Optional[List[Dict]] = None,
    ) -> ResolutionResult:
        """
        Main entry point: detect and resolve all conflicts.
        
        Args:
            column_matches: Raw matches from schema embeddings
                           {source_col: [{field_name, similarity, document_type, ...}, ...]}
            winning_doc_type: The document type that won classification
            column_profiles: Optional column profiles for additional context
            
        Returns:
            ResolutionResult with final mappings and conflict information
        """
        # Build profile lookup for additional context
        profile_lookup = self._build_profile_lookup(column_profiles)
        
        # Filter matches to winning document type only
        filtered_matches = self._filter_by_doc_type(column_matches, winning_doc_type)
        
        # ═══════════════════════════════════════════════════════════════
        # PHASE 1: Detect all conflicts
        # ═══════════════════════════════════════════════════════════════
        conflicts = self._detect_conflicts(filtered_matches)
        
        if conflicts:
            logger.info(f"🔍 Detected {len(conflicts)} mapping conflict(s)")
            for conflict in conflicts:
                sources = [c["source"] for c in conflict.contenders]
                logger.info(
                    f"   └─ '{conflict.target_field}' claimed by: {sources} "
                    f"(gap={conflict.confidence_gap:.1%})"
                )
        
        # ═══════════════════════════════════════════════════════════════
        # PHASE 2: Resolve conflicts and build final mappings
        # ═══════════════════════════════════════════════════════════════
        mappings = self._resolve_and_build_mappings(
            filtered_matches=filtered_matches,
            conflicts=conflicts,
            profile_lookup=profile_lookup,
        )
        
        # Calculate stats
        mapped_count = sum(1 for m in mappings.values() if m.target_field)
        unmapped_count = len(mappings) - mapped_count
        
        return ResolutionResult(
            mappings=mappings,
            conflicts=conflicts,
            total_columns=len(column_matches),
            mapped_count=mapped_count,
            conflict_count=len(conflicts),
            unmapped_count=unmapped_count,
        )
    
    def _build_profile_lookup(
        self,
        column_profiles: Optional[List[Dict]],
    ) -> Dict[str, Dict]:
        """Build a lookup dict from column name to profile."""
        if not column_profiles:
            return {}
        return {p.get("column_name", ""): p for p in column_profiles}
    
    def _filter_by_doc_type(
        self,
        column_matches: Dict[str, List[Dict]],
        doc_type: str,
    ) -> Dict[str, List[Dict]]:
        """Filter matches to only include the winning document type."""
        return {
            col: [m for m in matches if m.get("document_type") == doc_type]
            for col, matches in column_matches.items()
        }
    
    def _detect_conflicts(
        self,
        filtered_matches: Dict[str, List[Dict]],
    ) -> List[MappingConflict]:
        """
        PHASE 1: Find all target fields that have multiple source columns.
        
        A conflict occurs when:
        - Multiple columns have the SAME target as their best match
        - Both are above the confidence threshold
        """
        # Group by best target field
        target_to_sources: Dict[str, List[Dict]] = defaultdict(list)
        
        for col_name, matches in filtered_matches.items():
            if not matches:
                continue
            
            best = matches[0]
            best_confidence = best.get("similarity", 0)
            
            # Only consider if above threshold
            if best_confidence >= self.min_confidence:
                target_to_sources[best["field_name"]].append({
                    "source": col_name,
                    "confidence": best_confidence,
                    "required": best.get("required", False),
                    "field_type": best.get("field_type"),
                    "alternatives": matches[1:5],  # Keep top 5 alternatives
                })
        
        # Build conflict objects for targets with multiple sources
        conflicts = []
        for target_field, sources in target_to_sources.items():
            if len(sources) > 1:
                # Sort by confidence descending (highest first)
                sorted_sources = sorted(
                    sources, key=lambda x: x["confidence"], reverse=True
                )
                
                # Determine if target is required
                target_required = any(s.get("required", False) for s in sources)
                
                conflicts.append(MappingConflict(
                    target_field=target_field,
                    target_required=target_required,
                    contenders=sorted_sources,
                ))
        
        return conflicts
    
    def _resolve_and_build_mappings(
        self,
        filtered_matches: Dict[str, List[Dict]],
        conflicts: List[MappingConflict],
        profile_lookup: Dict[str, Dict],
    ) -> Dict[str, ResolvedMapping]:
        """
        PHASE 2: Resolve all conflicts and build the final mappings.
        
        Strategy:
        1. Resolve each conflict (pick winner)
        2. Claim target for winner
        3. Reassign losers to alternatives
        4. Build non-conflicting mappings
        """
        claimed_targets: Dict[str, str] = {}  # target -> source that owns it
        mappings: Dict[str, ResolvedMapping] = {}
        losers: Dict[str, MappingConflict] = {}  # source -> conflict they lost
        
        # Step 1: Resolve each conflict and track winners/losers
        for conflict in conflicts:
            winner = self._pick_winner(conflict)
            conflict.winner = winner
            
            # Winner claims the target
            claimed_targets[conflict.target_field] = winner
            
            # Track losers for reassignment
            for contender in conflict.contenders:
                if contender["source"] != winner:
                    losers[contender["source"]] = conflict
        
        # Step 2: Build mappings for conflict winners
        for conflict in conflicts:
            winner_source = conflict.winner
            winner_matches = filtered_matches.get(winner_source, [])
            
            if winner_matches:
                best = winner_matches[0]
                mappings[winner_source] = ResolvedMapping(
                    source_column=winner_source,
                    target_field=conflict.target_field,
                    confidence=best.get("similarity", 0),
                    field_type=best.get("field_type"),
                    required=conflict.target_required,
                    had_conflict=True,
                    resolution_strategy=conflict.resolution_strategy,
                    conflict_details=conflict.resolution_reason,
                )
        
        # Step 3: Build mappings for conflict losers (try alternatives)
        for loser_source, conflict in losers.items():
            loser_matches = filtered_matches.get(loser_source, [])
            mappings[loser_source] = self._reassign_loser(
                source=loser_source,
                matches=loser_matches,
                conflict=conflict,
                claimed=claimed_targets,
            )
        
        # Step 4: Build mappings for non-conflicting columns
        for col_name, matches in filtered_matches.items():
            if col_name in mappings:
                continue  # Already handled (winner or loser)
            
            if not matches:
                # No candidates at all
                mappings[col_name] = self._no_mapping(
                    source=col_name,
                    reason="No candidates for winning document type",
                )
                continue
            
            best = matches[0]
            best_confidence = best.get("similarity", 0)
            target = best.get("field_name")
            
            if best_confidence < self.min_confidence:
                # Below threshold
                mappings[col_name] = self._no_mapping(
                    source=col_name,
                    reason=f"Best match ({best_confidence:.0%}) below threshold ({self.min_confidence:.0%})",
                )
            elif target in claimed_targets:
                # Target already claimed by a conflict winner
                # This shouldn't happen normally, but handle it gracefully
                mappings[col_name] = self._reassign_loser(
                    source=col_name,
                    matches=matches,
                    conflict=None,
                    claimed=claimed_targets,
                )
            else:
                # Normal case: claim target
                claimed_targets[target] = col_name
                mappings[col_name] = ResolvedMapping(
                    source_column=col_name,
                    target_field=target,
                    confidence=best_confidence,
                    field_type=best.get("field_type"),
                    required=best.get("required", False),
                    resolution_strategy=ResolutionStrategy.NO_CONFLICT,
                )
        
        return mappings
    
    def _pick_winner(self, conflict: MappingConflict) -> str:
        """
        Decide the winner of a conflict.
        
        Strategy:
        1. If clear gap (>2%), highest confidence wins
        2. If ambiguous, try OpenAI arbitration
        3. Fall back to highest confidence
        """
        contenders = conflict.contenders
        top = contenders[0]
        gap = conflict.confidence_gap
        
        if conflict.is_ambiguous:
            # Ambiguous case - try OpenAI if available
            if self.openai_fallback and self.openai_fallback.is_available:
                winner = self._openai_arbitrate(conflict)
                if winner:
                    conflict.resolution_strategy = ResolutionStrategy.OPENAI_ARBITRATED
                    conflict.resolution_reason = (
                        f"Ambiguous (gap={gap:.1%}), OpenAI chose '{winner}'"
                    )
                    logger.info(
                        f"   └─ 🤖 OpenAI arbitrated: '{winner}' wins '{conflict.target_field}'"
                    )
                    return winner
            
            # No OpenAI or OpenAI didn't help - use highest confidence
            conflict.resolution_strategy = ResolutionStrategy.WINNER
            conflict.resolution_reason = (
                f"Ambiguous (gap={gap:.1%}), defaulted to highest confidence"
            )
            logger.info(
                f"   └─ ⚠️ Ambiguous conflict on '{conflict.target_field}', "
                f"chose '{top['source']}' (highest confidence)"
            )
        else:
            # Clear winner
            conflict.resolution_strategy = ResolutionStrategy.WINNER
            conflict.resolution_reason = f"Clear winner (gap={gap:.1%})"
            logger.info(
                f"   └─ ✅ '{top['source']}' wins '{conflict.target_field}' "
                f"({top['confidence']:.0%})"
            )
        
        return top["source"]
    
    def _openai_arbitrate(self, conflict: MappingConflict) -> Optional[str]:
        """
        Ask OpenAI to arbitrate an ambiguous conflict.
        
        Returns the winning source column name, or None if OpenAI can't decide.
        """
        try:
            # Get the OpenAI client from the fallback service
            client = getattr(self.openai_fallback, '_client', None)
            if not client:
                return None
            
            # Build context for OpenAI
            contender_info = []
            for c in conflict.contenders:
                contender_info.append({
                    "source_column": c["source"],
                    "confidence": c["confidence"],
                })
            
            # Call OpenAI to arbitrate
            # Note: This uses a simple prompt approach
            # You could extend OpenAIClient with a dedicated arbitrate_conflict method
            prompt = (
                f"I'm mapping CSV columns to a target schema field '{conflict.target_field}'.\n"
                f"Multiple source columns are claiming this target with similar confidence:\n\n"
            )
            for c in contender_info:
                prompt += f"- '{c['source_column']}' with {c['confidence']:.0%} confidence\n"
            
            prompt += (
                f"\nBased on the column names, which one most likely represents "
                f"'{conflict.target_field}'? Reply with ONLY the column name, nothing else."
            )
            
            # This is a simplified call - in production you'd use a structured approach
            response = client._client.chat.completions.create(
                model=client.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=50,
            )
            
            answer = response.choices[0].message.content.strip().strip("'\"")
            
            # Validate the answer is one of the contenders
            valid_sources = [c["source"] for c in conflict.contenders]
            if answer in valid_sources:
                return answer
            
            logger.warning(f"OpenAI returned invalid answer: {answer}")
            return None
            
        except Exception as e:
            logger.warning(f"OpenAI arbitration failed: {e}")
            return None
    
    def _reassign_loser(
        self,
        source: str,
        matches: List[Dict],
        conflict: Optional[MappingConflict],
        claimed: Dict[str, str],
    ) -> ResolvedMapping:
        """
        Try to assign a losing column to an alternative target.
        
        Iterates through the column's candidate list (skipping the first,
        which was the conflict) and finds the first unclaimed target.
        """
        original_target = matches[0]["field_name"] if matches else None
        
        # Try alternatives (skip first which was the conflict)
        for alt in matches[1:]:
            alt_target = alt.get("field_name")
            alt_confidence = alt.get("similarity", 0)
            
            # Must be above threshold and unclaimed
            if alt_confidence >= self.min_confidence and alt_target not in claimed:
                claimed[alt_target] = source
                
                winner_source = conflict.winner if conflict else claimed.get(original_target)
                logger.info(
                    f"   └─ 🔄 '{source}' reassigned: "
                    f"'{original_target}' → '{alt_target}' "
                    f"(lost to '{winner_source}')"
                )
                
                return ResolvedMapping(
                    source_column=source,
                    target_field=alt_target,
                    confidence=alt_confidence,
                    field_type=alt.get("field_type"),
                    required=alt.get("required", False),
                    had_conflict=True,
                    resolution_strategy=ResolutionStrategy.REASSIGNED,
                    original_target=original_target,
                    conflict_details=f"Lost '{original_target}' to '{winner_source}'",
                )
        
        # No valid alternative found
        winner_source = conflict.winner if conflict else claimed.get(original_target)
        logger.info(
            f"   └─ ❌ '{source}' unmapped: "
            f"lost '{original_target}' to '{winner_source}', no alternatives"
        )
        
        return ResolvedMapping(
            source_column=source,
            target_field=None,
            confidence=0.0,
            had_conflict=True,
            resolution_strategy=ResolutionStrategy.UNMAPPED,
            original_target=original_target,
            conflict_details=f"Lost '{original_target}' to '{winner_source}', no valid alternatives",
        )
    
    def _no_mapping(self, source: str, reason: str) -> ResolvedMapping:
        """Create a 'no mapping' result for a column."""
        return ResolvedMapping(
            source_column=source,
            target_field=None,
            confidence=0.0,
            resolution_strategy=ResolutionStrategy.UNMAPPED,
            conflict_details=reason,
        )

