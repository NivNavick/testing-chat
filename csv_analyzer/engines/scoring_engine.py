"""
Hybrid Scoring Engine.

Combines document-level similarity (PostgreSQL) with column-level matching (ChromaDB)
to produce a final classification score.

Supports OpenAI fallback for columns that can't be confidently mapped by embeddings.
Supports transformation detection for unit/format conversions.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from csv_analyzer.core.schema_embeddings import SchemaEmbeddingsService
from csv_analyzer.core.schema_registry import SchemaRegistry, get_schema_registry

if TYPE_CHECKING:
    from csv_analyzer.services.dspy_service import DSPyClassificationService
    from csv_analyzer.contexts.registry import VerticalContext

logger = logging.getLogger(__name__)

# Import transformation detection (optional - graceful degradation)
try:
    from csv_analyzer.transformations.detector import detect_transformation, TransformationResult
    TRANSFORMATIONS_AVAILABLE = True
except ImportError:
    TRANSFORMATIONS_AVAILABLE = False
    logger.debug("Transformations module not available")


@dataclass
class ColumnMatch:
    """Match result for a single column."""
    source_column: str
    source_type: str
    target_field: Optional[str]
    target_type: Optional[str]
    document_type: str
    similarity: float  # Type-adjusted similarity
    required: bool = False
    type_compatibility: float = 1.0  # Type compatibility score (0.4 to 1.0)
    raw_similarity: float = 0.0  # Original embedding similarity before type adjustment


@dataclass
class ScoringResult:
    """Complete scoring result."""
    # Final classification
    document_type: str
    vertical: str
    final_score: float
    
    # Component scores
    document_score: float  # From PostgreSQL ground truth
    column_score: float    # From ChromaDB schema matching
    coverage_score: float  # Required field coverage
    
    # Detailed column mappings
    column_matches: Dict[str, List[ColumnMatch]]
    
    # Suggested mappings (best match per column)
    suggested_mappings: Dict[str, Dict[str, Any]]
    
    # Score breakdown by document type
    all_scores: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Mapping conflicts: target_field -> list of source columns that map to it
    mapping_conflicts: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_type": self.document_type,
            "vertical": self.vertical,
            "final_score": self.final_score,
            "component_scores": {
                "document_similarity": self.document_score,
                "column_matching": self.column_score,
                "field_coverage": self.coverage_score,
            },
            "suggested_mappings": self.suggested_mappings,
            "mapping_conflicts": self.mapping_conflicts,
            "all_document_type_scores": self.all_scores,
        }


class ScoringEngine:
    """
    Hybrid scoring engine that combines document and column level scores.
    
    Formula:
        final_score = α * document_score + β * column_score + γ * coverage_score
    
    Where:
        - document_score: Similarity from PostgreSQL ground truth
        - column_score: Average column-to-schema-field similarity from ChromaDB
        - coverage_score: Percentage of required fields matched
    """
    
    # Default weights (can be tuned)
    WEIGHT_DOCUMENT = 0.3  # α
    WEIGHT_COLUMN = 0.5    # β
    WEIGHT_COVERAGE = 0.2  # γ
    
    def __init__(
        self,
        schema_embeddings_service: SchemaEmbeddingsService,
        schema_registry: Optional[SchemaRegistry] = None,
        weight_document: float = None,
        weight_column: float = None,
        weight_coverage: float = None,
        dspy_service: Optional["DSPyClassificationService"] = None,
        dspy_verify_all: bool = False,
        vertical_context: Optional["VerticalContext"] = None,
    ):
        """
        Initialize the scoring engine.
        
        Args:
            schema_embeddings_service: ChromaDB service for column matching
            schema_registry: Schema registry for field information
            weight_document: Weight for document similarity (default 0.3)
            weight_column: Weight for column matching (default 0.5)
            weight_coverage: Weight for field coverage (default 0.2)
            dspy_service: Optional DSPy service for column classification
            dspy_verify_all: If True, verify ALL matches with DSPy (not just low-confidence)
            vertical_context: Optional vertical context for domain terminology
        """
        self.schema_service = schema_embeddings_service
        self.schema_registry = schema_registry or get_schema_registry()
        
        self.weight_document = weight_document or self.WEIGHT_DOCUMENT
        self.weight_column = weight_column or self.WEIGHT_COLUMN
        self.weight_coverage = weight_coverage or self.WEIGHT_COVERAGE
        
        self.dspy_service = dspy_service
        self.dspy_verify_all = dspy_verify_all
        self.vertical_context = vertical_context
    
    def score(
        self,
        column_profiles: List[Dict[str, Any]],
        document_similarity_results: List[Dict[str, Any]],
        vertical: Optional[str] = None,
    ) -> ScoringResult:
        """
        Calculate hybrid score combining document and column level matching.
        
        Args:
            column_profiles: Column profiles from column_profiler
            document_similarity_results: Results from PostgreSQL similarity search
            vertical: Optional vertical filter
            
        Returns:
            ScoringResult with final classification and detailed breakdown
        """
        # 1. Calculate document-level scores
        doc_type_doc_scores = self._calculate_document_scores(
            document_similarity_results
        )
        
        # 2. Calculate column-level scores (also caches embeddings)
        column_results = self.schema_service.score_columns_against_schemas(
            columns=column_profiles,
            vertical=vertical,
        )
        
        doc_type_col_scores = column_results["document_type_scores"]
        column_matches = column_results["column_matches"]
        column_embeddings = column_results.get("column_embeddings", {})
        
        # 3. Calculate coverage scores
        doc_type_coverage = self._calculate_coverage_scores(
            column_matches=column_matches,
            vertical=vertical or self._get_vertical_from_results(document_similarity_results),
        )
        
        # 4. Combine scores
        all_doc_types = set(doc_type_doc_scores.keys()) | set(doc_type_col_scores.keys())
        
        combined_scores = {}
        for doc_type in all_doc_types:
            doc_score = doc_type_doc_scores.get(doc_type, 0)
            col_score = doc_type_col_scores.get(doc_type, {}).get("score", 0)
            cov_score = doc_type_coverage.get(doc_type, 0)
            
            final = (
                self.weight_document * doc_score +
                self.weight_column * col_score +
                self.weight_coverage * cov_score
            )
            
            combined_scores[doc_type] = {
                "final_score": round(final, 4),
                "document_score": round(doc_score, 4),
                "column_score": round(col_score, 4),
                "coverage_score": round(cov_score, 4),
            }
        
        # 5. Find winner
        if combined_scores:
            winner = max(combined_scores, key=lambda x: combined_scores[x]["final_score"])
            winner_scores = combined_scores[winner]
        else:
            winner = None
            winner_scores = {
                "final_score": 0,
                "document_score": 0,
                "column_score": 0,
                "coverage_score": 0,
            }
        
        # 5b. Second query: Get column matches filtered to winning document type
        # Uses cached embeddings to avoid regenerating them
        winning_vertical = self._get_vertical_for_doc_type(winner, vertical, document_similarity_results)
        if winner and column_embeddings:
            final_column_matches = self.schema_service.get_column_matches_for_document_type(
                columns=column_profiles,
                column_embeddings=column_embeddings,
                document_type=winner,
                vertical=winning_vertical,
                n_results=5,
            )
            logger.info(f"Second query: Retrieved column matches filtered to '{winner}'")
        else:
            final_column_matches = column_matches
        
        # 6. Build suggested mappings (with optional OpenAI fallback)
        # Use final_column_matches which is filtered to winning doc type
        suggested_mappings = self._build_suggested_mappings(
            column_matches=final_column_matches,
            winning_doc_type=winner,
            column_profiles=column_profiles,
        )
        
        # 7. Detect mapping conflicts (multiple source columns → same target field)
        mapping_conflicts = self._detect_mapping_conflicts(suggested_mappings)
        
        return ScoringResult(
            document_type=winner,
            vertical=winning_vertical,
            final_score=winner_scores["final_score"],
            document_score=winner_scores["document_score"],
            column_score=winner_scores["column_score"],
            coverage_score=winner_scores["coverage_score"],
            column_matches=self._convert_column_matches(final_column_matches),
            suggested_mappings=suggested_mappings,
            all_scores=combined_scores,
            mapping_conflicts=mapping_conflicts,
        )
    
    def _calculate_document_scores(
        self,
        document_results: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """Calculate document-type scores from PostgreSQL results."""
        scores = {}
        
        for result in document_results:
            doc_type = result.get("document_type")
            similarity = result.get("similarity", 0)
            
            if doc_type not in scores:
                scores[doc_type] = []
            scores[doc_type].append(similarity)
        
        # Average similarities per document type
        return {
            doc_type: sum(sims) / len(sims)
            for doc_type, sims in scores.items()
        }
    
    def _calculate_coverage_scores(
        self,
        column_matches: Dict[str, List[Dict]],
        vertical: str,
    ) -> Dict[str, float]:
        """Calculate required field coverage for each document type."""
        coverage = {}
        
        # Get all document types and their required fields
        schemas = self.schema_registry.get_schemas_by_vertical(vertical)
        
        for schema in schemas:
            required_fields = {f.name for f in schema.get_required_fields()}
            
            if not required_fields:
                coverage[schema.name] = 1.0
                continue
            
            # Find which required fields have a good match
            matched_required = set()
            
            for col_name, matches in column_matches.items():
                for match in matches:
                    if (match["document_type"] == schema.name and 
                        match["field_name"] in required_fields and
                        match["similarity"] > 0.5):  # Threshold for "good" match
                        matched_required.add(match["field_name"])
            
            coverage[schema.name] = len(matched_required) / len(required_fields)
        
        return coverage
    
    # Minimum confidence to suggest a mapping (below this = "no mapping")
    MIN_CONFIDENCE_THRESHOLD = 0.82
    
    # Minimum gap between best and 2nd best match to be confident
    # If best=83% and 2nd=82%, the match is ambiguous
    MIN_CONFIDENCE_GAP = 0.02
    
    # Number of candidates to send to OpenAI fallback
    OPENAI_TOP_K_CANDIDATES = 5
    
    def _build_suggested_mappings(
        self,
        column_matches: Dict[str, List[Dict]],
        winning_doc_type: Optional[str],
        column_profiles: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Build suggested column mappings for the winning document type.
        
        Two modes:
        1. Normal mode (openai_verify_all=False):
           - Only send low-confidence/ambiguous matches to OpenAI
        
        2. Verify-all mode (openai_verify_all=True):
           - Verify EVERY match with OpenAI
           - If rejected, try next candidate
           - Ensures semantic correctness, not just similarity
        """
        suggestions = {}
        unmapped_columns = []  # Columns to send to OpenAI fallback
        columns_to_verify = []  # Columns to verify with OpenAI (verify-all mode)
        
        # Build a lookup for column profiles
        profile_lookup = {}
        if column_profiles:
            for p in column_profiles:
                profile_lookup[p.get("column_name", "")] = p
        
        for col_name, matches in column_matches.items():
            # Filter matches for winning doc type
            if winning_doc_type:
                relevant_matches = [
                    m for m in matches
                    if m["document_type"] == winning_doc_type
                ]
            else:
                relevant_matches = matches
            
            if not relevant_matches:
                suggestions[col_name] = self._no_mapping()
                continue
            
            best = relevant_matches[0]
            best_confidence = best["similarity"]
            needs_fallback = False
            
            # Check 1: Is confidence high enough?
            if best_confidence < self.MIN_CONFIDENCE_THRESHOLD:
                logger.debug(f"Column '{col_name}': best match '{best['field_name']}' "
                           f"below threshold ({best_confidence:.1%} < {self.MIN_CONFIDENCE_THRESHOLD:.0%})")
                needs_fallback = True
            
            # Check 2: Is there a clear winner? (sufficient gap to 2nd best)
            if not needs_fallback and len(relevant_matches) > 1:
                second_best = relevant_matches[1]
                gap = best_confidence - second_best["similarity"]
                
                # If both match to DIFFERENT fields and gap is too small, it's ambiguous
                if (second_best["field_name"] != best["field_name"] and 
                    gap < self.MIN_CONFIDENCE_GAP):
                    logger.debug(f"Column '{col_name}': ambiguous match between "
                               f"'{best['field_name']}' ({best_confidence:.1%}) and "
                               f"'{second_best['field_name']}' ({second_best['similarity']:.1%})")
                    needs_fallback = True
            
            profile = profile_lookup.get(col_name, {})
            
            if self.dspy_verify_all and self.dspy_service and self.dspy_service.is_available:
                # Verify-all mode: verify EVERY match, even high-confidence ones
                columns_to_verify.append({
                    "column_name": col_name,
                    "column_type": profile.get("detected_type", "unknown"),
                    "sample_values": profile.get("sample_values", [])[:5],
                    "candidates": relevant_matches[:self.OPENAI_TOP_K_CANDIDATES],
                })
                # Placeholder - will be filled after verification
                suggestions[col_name] = self._no_mapping()
            elif needs_fallback:
                # Normal mode: only fallback for low-confidence/ambiguous
                unmapped_columns.append({
                    "column_name": col_name,
                    "column_type": profile.get("detected_type", "unknown"),
                    "sample_values": profile.get("sample_values", [])[:5],
                    "candidates": relevant_matches[:self.OPENAI_TOP_K_CANDIDATES],
                })
                # Placeholder - will be filled by OpenAI or left as no_mapping
                suggestions[col_name] = self._no_mapping()
            else:
                # Good match (no verification needed in normal mode)
                suggestion = {
                    "target_field": best["field_name"],
                    "confidence": best_confidence,
                    "field_type": best["field_type"],
                    "required": best["required"],
                    "sources": ["embeddings"],  # High confidence, no DSPy needed
                    "source_type": best.get("source_type", "unknown"),
                    "type_compatibility": best.get("type_compatibility", 1.0),
                    "raw_similarity": best.get("raw_similarity", best_confidence),
                }
                
                # Check for transformation needs
                transformation = self._detect_transformation_for_match(
                    col_name=col_name,
                    profile=profile,
                    target_field=best["field_name"],
                    vertical=self._get_vertical_from_doc_type(winning_doc_type),
                )
                if transformation:
                    suggestion["transformation"] = transformation
                
                suggestions[col_name] = suggestion
        
        # Handle verify-all mode (PARALLEL)
        if columns_to_verify:
            logger.info(f"🚀 Verifying {len(columns_to_verify)} column mappings in parallel...")
            
            # Use parallel verification for speed
            parallel_results = self.dspy_service.verify_columns_parallel(
                columns_to_verify=columns_to_verify,
                document_type=winning_doc_type or "unknown",
                schema_registry=self.schema_registry,
                vertical=self._get_vertical_from_doc_type(winning_doc_type),
                vertical_context=self.vertical_context,
                max_workers=4,  # Parallel workers
            )
            
            # Process results
            for col_name, result in parallel_results.items():
                if result.get("target_field"):
                    suggestion = {
                        "target_field": result["target_field"],
                        "confidence": result.get("confidence", 0.8),
                        "field_type": result.get("field_type"),
                        "required": result.get("required", False),
                        "sources": ["embeddings", "dspy"],  # Both contributed
                        "attempts": result.get("attempts", 1),
                        "reason": result.get("reason", ""),
                    }
                    
                    # Check for transformation needs
                    transformation = self._detect_transformation_for_match(
                        col_name=col_name,
                        profile=profile_lookup.get(col_name, {}),
                        target_field=result["target_field"],
                        vertical=self._get_vertical_from_doc_type(winning_doc_type),
                    )
                    if transformation:
                        suggestion["transformation"] = transformation
                    
                    suggestions[col_name] = suggestion
                else:
                    suggestions[col_name] = {
                        "target_field": None,
                        "confidence": 0.0,
                        "field_type": None,
                        "required": False,
                        "sources": ["embeddings", "dspy"],  # Both tried, no match
                        "reason": result.get("reason", "No valid match found"),
                    }
        
        # Handle normal fallback mode for unmapped columns (PARALLEL)
        elif unmapped_columns and self.dspy_service and self.dspy_service.is_available:
            logger.info(f"🚀 Classifying {len(unmapped_columns)} unmapped columns with DSPy in parallel")
            
            fallback_results = self.dspy_service.classify_columns_parallel(
                columns=unmapped_columns,
                document_type=winning_doc_type or "unknown",
                max_workers=4,
            )
            
            # Merge fallback results
            for col_name, result in fallback_results.items():
                if result.get("target_field"):
                    suggestion = {
                        "target_field": result["target_field"],
                        "confidence": result.get("confidence", 0.75),
                        "field_type": result.get("field_type"),
                        "required": result.get("required", False),
                        "sources": ["dspy"],  # Unmapped column, classified by DSPy alone
                        "reason": result.get("reason", ""),
                    }
                    
                    # Check for transformation needs
                    transformation = self._detect_transformation_for_match(
                        col_name=col_name,
                        profile=profile_lookup.get(col_name, {}),
                        target_field=result["target_field"],
                        vertical=self._get_vertical_from_doc_type(winning_doc_type),
                    )
                    if transformation:
                        suggestion["transformation"] = transformation
                    
                    suggestions[col_name] = suggestion
                    logger.info(f"OpenAI fallback mapped '{col_name}' → '{result['target_field']}'")
        elif unmapped_columns:
            logger.debug(f"{len(unmapped_columns)} columns remain unmapped (OpenAI fallback not available)")
        
        return suggestions
    
    def _no_mapping(self) -> Dict[str, Any]:
        """Return a 'no mapping' result."""
        return {
            "target_field": None,
            "confidence": 0.0,
            "field_type": None,
            "required": False,
        }
    
    def _detect_mapping_conflicts(
        self,
        suggested_mappings: Dict[str, Dict[str, Any]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Detect cases where multiple source columns map to the same target field.
        
        Returns a dict where:
        - Key: target field name (the schema field)
        - Value: list of source columns that map to it, sorted by confidence desc
        
        Only includes targets with 2+ source columns (actual conflicts).
        """
        from collections import defaultdict
        
        # Invert the mapping: target_field -> list of source columns
        target_to_sources = defaultdict(list)
        
        for source_col, mapping in suggested_mappings.items():
            target = mapping.get("target_field")
            if target:
                target_to_sources[target].append({
                    "source_column": source_col,
                    "confidence": mapping.get("confidence", 0),
                    "field_type": mapping.get("field_type"),
                    "required": mapping.get("required", False),
                    "sources": mapping.get("sources", ["embeddings"]),
                    "reason": mapping.get("reason", ""),
                    "attempts": mapping.get("attempts"),
                    "transformation": mapping.get("transformation"),
                })
        
        # Filter to only conflicts (2+ sources for same target)
        conflicts = {}
        for target, sources in target_to_sources.items():
            if len(sources) > 1:
                # Sort by confidence descending
                sources.sort(key=lambda x: x.get("confidence", 0), reverse=True)
                conflicts[target] = sources
        
        if conflicts:
            logger.info(f"⚠️  Detected {len(conflicts)} mapping conflicts (multiple columns → same target)")
            for target, sources in conflicts.items():
                source_names = [s["source_column"] for s in sources]
                logger.debug(f"   {target} ← {source_names}")
        
        return conflicts
    
    def _convert_column_matches(
        self,
        column_matches: Dict[str, List[Dict]],
    ) -> Dict[str, List[ColumnMatch]]:
        """Convert raw matches to ColumnMatch objects."""
        result = {}
        for col_name, matches in column_matches.items():
            result[col_name] = [
                ColumnMatch(
                    source_column=col_name,
                    source_type=m.get("source_type", "unknown"),
                    target_field=m["field_name"],
                    target_type=m["field_type"],
                    document_type=m["document_type"],
                    similarity=m["similarity"],
                    required=m["required"],
                    type_compatibility=m.get("type_compatibility", 1.0),
                    raw_similarity=m.get("raw_similarity", m["similarity"]),
                )
                for m in matches
            ]
        return result
    
    def _get_vertical_from_results(
        self,
        document_results: List[Dict[str, Any]],
    ) -> str:
        """Extract vertical from document results."""
        for result in document_results:
            if "vertical" in result:
                return result["vertical"]
        return "medical"  # Default
    
    def _get_vertical_for_doc_type(
        self,
        doc_type: Optional[str],
        vertical_hint: Optional[str],
        document_results: List[Dict[str, Any]],
    ) -> str:
        """Get vertical for a document type."""
        if vertical_hint:
            return vertical_hint
        
        # Try to find from document results
        for result in document_results:
            if result.get("document_type") == doc_type and "vertical" in result:
                return result["vertical"]
        
        # Try schema registry
        for vertical, schemas in self.schema_registry.schemas.items():
            if doc_type in schemas:
                return vertical
        
        return "medical"  # Default
    
    def _get_vertical_from_doc_type(self, doc_type: Optional[str]) -> str:
        """Get vertical for a document type from schema registry."""
        if not doc_type:
            return "medical"
        
        for vertical, schemas in self.schema_registry.schemas.items():
            if doc_type in schemas:
                return vertical
        
        return "medical"
    
    def _detect_transformation_for_match(
        self,
        col_name: str,
        profile: Dict[str, Any],
        target_field: str,
        vertical: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Detect if a transformation is needed for a column-to-field match.
        
        Args:
            col_name: Source column name
            profile: Column profile with sample values
            target_field: Target field name
            vertical: Vertical (e.g., "medical")
            
        Returns:
            Transformation info dict or None if no transformation needed
        """
        if not TRANSFORMATIONS_AVAILABLE:
            return None
        
        # Find all schemas in the vertical and look for the target field
        schemas = self.schema_registry.get_schemas_by_vertical(vertical)
        
        for schema in schemas:
            field = schema.get_field(target_field)
            if field is None:
                continue
            
            # Check if field has unit requirements
            if field.target_unit and field.accepts_units:
                # Get OpenAI client for unit detection if available
                openai_client = None
                if self.dspy_service and self.dspy_service.is_available:
                    openai_client = getattr(self.dspy_service, '_classifier', None)
                
                # Detect transformation
                result = detect_transformation(
                    column_name=col_name,
                    sample_values=profile.get("sample_values", [])[:5],
                    target_field=target_field,
                    target_unit=field.target_unit,
                    accepts_units=field.accepts_units,
                    openai_client=openai_client,
                )
                
                if result.needs_transformation and result.transformation:
                    logger.info(
                        f"🔄 Transformation detected: '{col_name}' needs "
                        f"{result.source_unit} → {result.target_unit} "
                        f"({result.transformation.formula_description})"
                    )
                    return result.to_dict()
            
            # Check if field has format requirements
            if field.target_format and field.accepts_formats:
                # TODO: Implement format detection
                pass
            
            # Found the field, no need to check other schemas
            break
        
        return None
