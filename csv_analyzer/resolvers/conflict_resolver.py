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
import re
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
    from csv_analyzer.core.schema_registry import SchemaRegistry

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
    
    # Bonus added when column name exactly matches field name or alias
    NAME_MATCH_BONUS = 0.15
    
    # Penalty for type mismatches
    TYPE_MISMATCH_PENALTY = 0.15
    
    # Type compatibility matrix: source_type -> set of compatible target_types
    # Types not in compatible set get penalized
    TYPE_COMPATIBILITY = {
        # Datetime types
        "datetime": {"datetime", "string"},
        "date": {"date", "datetime", "string"},
        "time_of_day": {"datetime", "time", "string"},
        
        # Numeric types
        "integer": {"integer", "float", "number", "string"},
        "float": {"float", "integer", "number", "string"},
        "numeric": {"integer", "float", "number", "string"},
        
        # Text types
        "text": {"string", "text"},
        "id_like": {"string", "text", "integer"},
        "categorical": {"string", "text"},
        
        # Boolean
        "boolean": {"boolean", "string", "integer"},
        
        # Special
        "empty": set(),  # Empty columns match nothing well
        "unknown": {"string", "text"},  # Unknown defaults to string
    }
    
    def __init__(
        self,
        min_confidence: float = None,
        ambiguity_threshold: float = None,
        openai_fallback: Optional["OpenAIFallbackService"] = None,
        schema_registry: Optional["SchemaRegistry"] = None,
    ):
        """
        Initialize the conflict resolver.
        
        Args:
            min_confidence: Minimum confidence to accept a mapping (default 0.82)
            ambiguity_threshold: Gap below which conflicts are ambiguous (default 0.02)
            openai_fallback: Optional OpenAI service for arbitrating ambiguous conflicts
            schema_registry: Optional schema registry for name matching
        """
        self.min_confidence = min_confidence or self.MIN_CONFIDENCE_THRESHOLD
        self.ambiguity_threshold = ambiguity_threshold or self.AMBIGUITY_THRESHOLD
        self.openai_fallback = openai_fallback
        self.schema_registry = schema_registry
        self._field_names_cache: Dict[str, set] = {}  # target_field -> set of normalized names/aliases
    
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
        
        # Apply type compatibility adjustments to confidences
        filtered_matches = self._apply_type_adjustments(filtered_matches, profile_lookup)
        
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
    
    def _apply_type_adjustments(
        self,
        filtered_matches: Dict[str, List[Dict]],
        profile_lookup: Dict[str, Dict],
    ) -> Dict[str, List[Dict]]:
        """
        Apply type compatibility adjustments to match confidences.
        
        Penalizes matches where column type doesn't match field type.
        Re-sorts matches by adjusted confidence.
        """
        adjusted_matches = {}
        
        for col_name, matches in filtered_matches.items():
            # Get source column type
            profile = profile_lookup.get(col_name, {})
            source_type = profile.get("detected_type", "unknown")
            
            adjusted_list = []
            for match in matches:
                # Copy match to avoid modifying original
                adjusted_match = dict(match)
                
                target_type = match.get("field_type", "string")
                original_conf = match.get("similarity", 0)
                
                # Apply type adjustment
                adjusted_conf = self._get_type_adjusted_confidence(
                    original_conf, source_type, target_type
                )
                
                if adjusted_conf != original_conf:
                    adjusted_match["similarity"] = adjusted_conf
                    adjusted_match["original_similarity"] = original_conf
                    adjusted_match["type_penalty_applied"] = True
                    logger.debug(
                        f"'{col_name}' ({source_type}) → '{match.get('field_name')}' ({target_type}): "
                        f"{original_conf:.0%} → {adjusted_conf:.0%} (type mismatch)"
                    )
                
                adjusted_list.append(adjusted_match)
            
            # Re-sort by adjusted confidence
            adjusted_list.sort(key=lambda x: x.get("similarity", 0), reverse=True)
            adjusted_matches[col_name] = adjusted_list
        
        return adjusted_matches
    
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
        3. Build non-conflicting mappings FIRST (so they claim their targets)
        4. THEN reassign losers to alternatives (won't steal from non-conflicting)
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
        
        # Step 3: Build mappings for NON-CONFLICTING columns FIRST
        # This ensures they claim their rightful targets before losers try alternatives
        for col_name, matches in filtered_matches.items():
            if col_name in mappings:
                continue  # Already handled (winner)
            if col_name in losers:
                continue  # Handle later
            
            if not matches:
                mappings[col_name] = self._no_mapping(
                    source=col_name,
                    reason="No candidates for winning document type",
                )
                continue
            
            best = matches[0]
            best_confidence = best.get("similarity", 0)
            target = best.get("field_name")
            
            if best_confidence < self.min_confidence:
                mappings[col_name] = self._no_mapping(
                    source=col_name,
                    reason=f"Best match ({best_confidence:.0%}) below threshold ({self.min_confidence:.0%})",
                )
            elif target in claimed_targets:
                # Shouldn't happen, but handle gracefully
                mappings[col_name] = self._reassign_loser(
                    source=col_name,
                    matches=matches,
                    conflict=None,
                    claimed=claimed_targets,
                    all_column_matches=filtered_matches,
                    profile=profile_lookup.get(col_name),
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
        
        # Step 4: NOW reassign losers (after non-conflicting columns claimed their targets)
        for loser_source, conflict in losers.items():
            loser_matches = filtered_matches.get(loser_source, [])
            mappings[loser_source] = self._reassign_loser(
                source=loser_source,
                matches=loser_matches,
                conflict=conflict,
                claimed=claimed_targets,
                all_column_matches=filtered_matches,
                profile=profile_lookup.get(loser_source),
            )
        
        return mappings
    
    def _normalize_name(self, name: str) -> str:
        """Normalize a column/field name for comparison."""
        # Remove common separators, lowercase
        normalized = re.sub(r'[_\-\s]+', '', name.lower())
        return normalized
    
    def _get_field_name_and_aliases(self, target_field: str, vertical: str = "medical") -> tuple:
        """
        Get field name and aliases separately for priority matching.
        
        Returns:
            (normalized_field_name, set_of_normalized_aliases)
        """
        cache_key = f"{vertical}:{target_field}"
        
        if cache_key in self._field_names_cache:
            return self._field_names_cache[cache_key]
        
        field_name = self._normalize_name(target_field)
        aliases = set()
        
        if self.schema_registry:
            for schema in self.schema_registry.get_schemas_by_vertical(vertical):
                field = schema.get_field(target_field)
                if field:
                    for alias in field.aliases:
                        aliases.add(self._normalize_name(alias))
                    break
        
        self._field_names_cache[cache_key] = (field_name, aliases)
        return field_name, aliases
    
    def _get_name_match_priority(self, source_column: str, target_field: str) -> int:
        """
        Get name match priority for a source column.
        
        Returns:
            2 = Exact field name match (highest priority)
            1 = Alias match (medium priority)
            0 = No name match (use embedding confidence)
        """
        normalized_source = self._normalize_name(source_column)
        field_name, aliases = self._get_field_name_and_aliases(target_field)
        
        if normalized_source == field_name:
            return 2  # Exact field name match
        elif normalized_source in aliases:
            return 1  # Alias match
        else:
            return 0  # No name match
    
    def _is_exact_name_match(self, source_column: str, target_field: str) -> bool:
        """Check if source column name matches target field name or alias."""
        return self._get_name_match_priority(source_column, target_field) > 0
    
    def _is_type_compatible(self, source_type: str, target_type: str) -> bool:
        """
        Check if source column type is compatible with target field type.
        
        Args:
            source_type: Detected type from column profiler (e.g., "datetime", "integer")
            target_type: Field type from schema (e.g., "datetime", "string")
            
        Returns:
            True if types are compatible, False otherwise
        """
        if not source_type or not target_type:
            return True  # Can't check, assume compatible
        
        source_type = source_type.lower()
        target_type = target_type.lower()
        
        # Exact match is always compatible
        if source_type == target_type:
            return True
        
        # Check compatibility matrix
        compatible_types = self.TYPE_COMPATIBILITY.get(source_type, {"string"})
        return target_type in compatible_types
    
    def _get_type_adjusted_confidence(
        self,
        base_confidence: float,
        source_type: str,
        target_type: str,
    ) -> float:
        """
        Adjust confidence based on type compatibility.
        
        - Compatible types: no change
        - Incompatible types: apply penalty
        
        Args:
            base_confidence: Original embedding confidence
            source_type: Detected column type
            target_type: Schema field type
            
        Returns:
            Adjusted confidence score
        """
        if self._is_type_compatible(source_type, target_type):
            return base_confidence
        
        # Apply penalty for type mismatch
        adjusted = base_confidence - self.TYPE_MISMATCH_PENALTY
        logger.debug(
            f"Type mismatch penalty: {source_type} → {target_type}, "
            f"{base_confidence:.0%} → {adjusted:.0%}"
        )
        return max(0.0, adjusted)  # Don't go below 0
    
    def _pick_winner(self, conflict: MappingConflict) -> str:
        """
        Decide the winner of a conflict.
        
        Strategy (in priority order):
        1. Exact FIELD NAME match beats everything
        2. ALIAS match beats non-matching columns
        3. If multiple at same priority level, highest confidence wins
        4. If ambiguous, try OpenAI arbitration
        5. Fall back to highest confidence
        """
        contenders = conflict.contenders
        target_field = conflict.target_field
        
        # Calculate name match priority for each contender
        # Priority: 2 = field name match, 1 = alias match, 0 = no match
        prioritized = []
        for contender in contenders:
            source = contender["source"]
            priority = self._get_name_match_priority(source, target_field)
            prioritized.append({
                **contender,
                "name_priority": priority,
            })
        
        # Sort by: name_priority DESC, then confidence DESC
        prioritized.sort(key=lambda x: (x["name_priority"], x["confidence"]), reverse=True)
        
        top = prioritized[0]
        top_priority = top["name_priority"]
        
        # Check if top has clear name priority advantage
        if top_priority > 0:
            # Count how many have same priority
            same_priority = [p for p in prioritized if p["name_priority"] == top_priority]
            
            if top_priority == 2:
                # Exact field name match
                if len(same_priority) == 1:
                    winner = top["source"]
                    conflict.resolution_strategy = ResolutionStrategy.WINNER
                    conflict.resolution_reason = f"Exact field name match: '{winner}' = '{target_field}'"
                    logger.info(
                        f"   └─ 🎯 '{winner}' wins '{target_field}' (exact field name match)"
                    )
                    return winner
                else:
                    # Multiple field name matches (shouldn't happen, but handle it)
                    winner = same_priority[0]["source"]
                    conflict.resolution_strategy = ResolutionStrategy.WINNER
                    conflict.resolution_reason = f"Best among field name matches"
                    logger.info(
                        f"   └─ 🎯 '{winner}' wins '{target_field}' (best field name match)"
                    )
                    return winner
            
            elif top_priority == 1:
                # Alias match - check if any have higher priority (field name)
                field_name_matches = [p for p in prioritized if p["name_priority"] == 2]
                if field_name_matches:
                    # Field name match beats alias
                    winner = field_name_matches[0]["source"]
                    conflict.resolution_strategy = ResolutionStrategy.WINNER
                    conflict.resolution_reason = f"Field name match beats alias match"
                    logger.info(
                        f"   └─ 🎯 '{winner}' wins '{target_field}' (field name > alias)"
                    )
                    return winner
                
                if len(same_priority) == 1:
                    # Only one alias match
                    winner = top["source"]
                    conflict.resolution_strategy = ResolutionStrategy.WINNER
                    conflict.resolution_reason = f"Alias match: '{winner}' matches alias of '{target_field}'"
                    logger.info(
                        f"   └─ 🏷️ '{winner}' wins '{target_field}' (alias match)"
                    )
                    return winner
                else:
                    # Multiple alias matches - pick highest confidence
                    winner = same_priority[0]["source"]
                    conflict.resolution_strategy = ResolutionStrategy.WINNER
                    conflict.resolution_reason = (
                        f"Multiple alias matches, '{winner}' has highest confidence"
                    )
                    logger.info(
                        f"   └─ 🏷️ '{winner}' wins '{target_field}' "
                        f"(best among {len(same_priority)} alias matches)"
                    )
                    return winner
        
        # No name matches - use confidence-based resolution
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
                        f"   └─ 🤖 OpenAI arbitrated: '{winner}' wins '{target_field}'"
                    )
                    return winner
            
            # No OpenAI or OpenAI didn't help - use highest confidence
            conflict.resolution_strategy = ResolutionStrategy.WINNER
            conflict.resolution_reason = (
                f"Ambiguous (gap={gap:.1%}), defaulted to highest confidence"
            )
            logger.info(
                f"   └─ ⚠️ Ambiguous conflict on '{target_field}', "
                f"chose '{top['source']}' (highest confidence)"
            )
        else:
            # Clear winner by confidence
            conflict.resolution_strategy = ResolutionStrategy.WINNER
            conflict.resolution_reason = f"Clear winner (gap={gap:.1%})"
            logger.info(
                f"   └─ ✅ '{top['source']}' wins '{target_field}' "
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
            response = client.client.chat.completions.create(
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
        all_column_matches: Optional[Dict[str, List[Dict]]] = None,
        profile: Optional[Dict] = None,
    ) -> ResolvedMapping:
        """
        Try to assign a losing column to an alternative target.
        
        IMPORTANT: Only reassign to a target if:
        1. No other column has a BETTER claim
        2. OpenAI validates the alternative (if available)
        
        Args:
            source: Source column name
            matches: This column's match candidates
            conflict: The conflict this column lost (if any)
            claimed: Currently claimed targets
            all_column_matches: ALL columns' matches (to check for better claims)
            profile: Column profile with sample values (for OpenAI validation)
        """
        original_target = matches[0]["field_name"] if matches else None
        original_confidence = matches[0].get("similarity", 0) if matches else 0
        
        # Try alternatives (skip first which was the conflict)
        for alt in matches[1:]:
            alt_target = alt.get("field_name")
            alt_confidence = alt.get("similarity", 0)
            
            # Must be above threshold
            if alt_confidence < self.min_confidence:
                continue
            
            # Must be unclaimed
            if alt_target in claimed:
                continue
            
            # Check if another column has this as their BEST match
            if all_column_matches:
                better_claimant = self._find_better_claimant(
                    target=alt_target,
                    my_source=source,
                    my_confidence=alt_confidence,
                    all_matches=all_column_matches,
                    claimed=claimed,
                )
                if better_claimant:
                    logger.debug(
                        f"   └─ Skipping '{alt_target}' for '{source}': "
                        f"'{better_claimant}' has better claim"
                    )
                    continue
            
            # NEW: Validate alternative with OpenAI if available
            if self.openai_fallback and self.openai_fallback.is_available:
                is_valid = self._validate_alternative_with_openai(
                    source_column=source,
                    original_target=original_target,
                    alternative_target=alt_target,
                    profile=profile,
                )
                if not is_valid:
                    logger.info(
                        f"   └─ 🤖 OpenAI rejected '{source}' → '{alt_target}' "
                        f"(not semantically appropriate)"
                    )
                    continue
            
            # Safe to claim this alternative
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
    
    def _validate_alternative_with_openai(
        self,
        source_column: str,
        original_target: str,
        alternative_target: str,
        profile: Optional[Dict] = None,
    ) -> bool:
        """
        Ask OpenAI if an alternative target is semantically appropriate.
        
        This is called when a column loses a conflict and tries to reassign
        to an alternative target. OpenAI validates if this makes sense.
        
        Args:
            source_column: Source column name (e.g., "Worker_Code")
            original_target: The target it originally wanted (e.g., "employee_id")
            alternative_target: The alternative being considered (e.g., "shift_id")
            profile: Column profile with sample values
            
        Returns:
            True if alternative is semantically appropriate, False otherwise
        """
        try:
            client = getattr(self.openai_fallback, '_client', None)
            if not client:
                return True  # Can't validate, assume OK
            
            # Get sample values for context
            samples = []
            if profile:
                samples = profile.get("sample_values", [])[:5]
            samples_str = ", ".join(str(s) for s in samples) if samples else "N/A"
            
            # Get field descriptions if available
            original_desc = ""
            alt_desc = ""
            if self.schema_registry:
                for schema in self.schema_registry.get_all_schemas():
                    orig_field = schema.get_field(original_target)
                    alt_field = schema.get_field(alternative_target)
                    if orig_field:
                        original_desc = orig_field.description
                    if alt_field:
                        alt_desc = alt_field.description
                    if original_desc and alt_desc:
                        break
            
            prompt = f"""I'm mapping CSV columns to a target schema. A column lost a conflict and is trying to reassign to an alternative target.

Column: "{source_column}"
Sample values: {samples_str}

Original target (lost): "{original_target}"
  Description: {original_desc or 'N/A'}

Proposed alternative: "{alternative_target}"
  Description: {alt_desc or 'N/A'}

Question: Is "{alternative_target}" a semantically appropriate mapping for a column named "{source_column}" with the sample values shown?

Answer ONLY "yes" or "no" followed by a brief reason.
Example: "no - Worker_Code is an employee identifier, not a shift identifier"
"""
            
            response = client.client.chat.completions.create(
                model=client.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=100,
            )
            
            answer = response.choices[0].message.content.strip().lower()
            
            # Parse yes/no from response
            if answer.startswith("yes"):
                logger.debug(f"OpenAI approved '{source_column}' → '{alternative_target}'")
                return True
            else:
                logger.debug(f"OpenAI rejected '{source_column}' → '{alternative_target}': {answer}")
                return False
                
        except Exception as e:
            logger.warning(f"OpenAI validation failed: {e}")
            return True  # On error, assume OK
    
    def _find_better_claimant(
        self,
        target: str,
        my_source: str,
        my_confidence: float,
        all_matches: Dict[str, List[Dict]],
        claimed: Dict[str, str],
    ) -> Optional[str]:
        """
        Check if another column has a better claim on a target.
        
        A column has a "better claim" if:
        1. The target is their BEST (first) match
        2. Their confidence is >= my confidence
        3. They haven't already been assigned elsewhere
        
        Returns the source column name that has better claim, or None.
        """
        for other_source, other_matches in all_matches.items():
            # Skip myself
            if other_source == my_source:
                continue
            
            # Skip if this column already claimed something
            if other_source in [v for v in claimed.values()]:
                continue
            
            # Skip if no matches
            if not other_matches:
                continue
            
            # Check if target is their BEST match
            other_best = other_matches[0]
            if other_best.get("field_name") != target:
                continue
            
            other_confidence = other_best.get("similarity", 0)
            
            # They have better or equal claim if confidence is higher
            # (equal goes to them since it's their primary target)
            if other_confidence >= my_confidence:
                return other_source
        
        return None
    
    def _no_mapping(self, source: str, reason: str) -> ResolvedMapping:
        """Create a 'no mapping' result for a column."""
        return ResolvedMapping(
            source_column=source,
            target_field=None,
            confidence=0.0,
            resolution_strategy=ResolutionStrategy.UNMAPPED,
            conflict_details=reason,
        )

