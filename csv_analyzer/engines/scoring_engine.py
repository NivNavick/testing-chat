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
from csv_analyzer.resolvers import ConflictResolver, ResolutionResult

if TYPE_CHECKING:
    from csv_analyzer.services.openai_fallback import OpenAIFallbackService
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
    similarity: float
    required: bool = False


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
        openai_fallback: Optional["OpenAIFallbackService"] = None,
        openai_verify_all: bool = False,
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
            openai_fallback: Optional OpenAI fallback service for unmapped columns
            openai_verify_all: If True, verify ALL matches with OpenAI (not just low-confidence)
            vertical_context: Optional vertical context for domain terminology
        """
        self.schema_service = schema_embeddings_service
        self.schema_registry = schema_registry or get_schema_registry()
        
        self.weight_document = weight_document or self.WEIGHT_DOCUMENT
        self.weight_column = weight_column or self.WEIGHT_COLUMN
        self.weight_coverage = weight_coverage or self.WEIGHT_COVERAGE
        
        self.openai_fallback = openai_fallback
        self.openai_verify_all = openai_verify_all
        self.vertical_context = vertical_context
        
        # Initialize conflict resolver for end-of-mapping conflict detection
        self.conflict_resolver = ConflictResolver(
            openai_fallback=openai_fallback,
        )
    
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
        
        # 2. Calculate column-level scores
        column_results = self.schema_service.score_columns_against_schemas(
            columns=column_profiles,
            vertical=vertical,
        )
        
        doc_type_col_scores = column_results["document_type_scores"]
        column_matches = column_results["column_matches"]
        
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
        
        # 6. Build suggested mappings (with optional OpenAI fallback and conflict resolution)
        # First apply OpenAI fallback/verification if enabled
        enhanced_matches = self._apply_openai_enhancements(
            column_matches=column_matches,
            winning_doc_type=winner,
            column_profiles=column_profiles,
        )
        
        # Then resolve conflicts using end-of-mapping resolver
        resolution_result = self.conflict_resolver.resolve(
            column_matches=enhanced_matches,
            winning_doc_type=winner,
            column_profiles=column_profiles,
        )
        
        # Log conflict summary
        if resolution_result.conflicts:
            logger.info(
                f"Resolved {resolution_result.conflict_count} mapping conflicts, "
                f"{resolution_result.mapped_count}/{resolution_result.total_columns} columns mapped"
            )
        
        # Convert to suggested mappings format
        suggested_mappings = resolution_result.to_suggested_mappings()
        
        # 7. Get vertical
        winning_vertical = self._get_vertical_for_doc_type(winner, vertical, document_similarity_results)
        
        return ScoringResult(
            document_type=winner,
            vertical=winning_vertical,
            final_score=winner_scores["final_score"],
            document_score=winner_scores["document_score"],
            column_score=winner_scores["column_score"],
            coverage_score=winner_scores["coverage_score"],
            column_matches=self._convert_column_matches(column_matches),
            suggested_mappings=suggested_mappings,
            all_scores=combined_scores,
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
    
    def _apply_openai_enhancements(
        self,
        column_matches: Dict[str, List[Dict]],
        winning_doc_type: Optional[str],
        column_profiles: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, List[Dict]]:
        """
        Apply OpenAI fallback/verification to enhance column matches.
        
        This method runs BEFORE conflict resolution to improve match quality.
        It modifies the matches in-place or returns enhanced matches.
        
        Two modes:
        1. Normal mode (openai_verify_all=False):
           - Only send low-confidence/ambiguous matches to OpenAI
        
        2. Verify-all mode (openai_verify_all=True):
           - Verify EVERY match with OpenAI
           - If rejected, try next candidate
        
        Returns:
            Enhanced column_matches dict with OpenAI improvements applied
        """
        if not self.openai_fallback or not self.openai_fallback.is_available:
            return column_matches
        
        # Build a lookup for column profiles
        profile_lookup = {}
        if column_profiles:
            for p in column_profiles:
                profile_lookup[p.get("column_name", "")] = p
        
        # Identify columns needing OpenAI help
        unmapped_columns = []  # Low-confidence columns for fallback
        columns_to_verify = []  # Columns for verify-all mode
        
        for col_name, matches in column_matches.items():
            # Filter matches for winning doc type
            if winning_doc_type:
                relevant_matches = [
                    m for m in matches
                    if m.get("document_type") == winning_doc_type
                ]
            else:
                relevant_matches = matches
            
            if not relevant_matches:
                continue
            
            best = relevant_matches[0]
            best_confidence = best.get("similarity", 0)
            needs_fallback = False
            
            # Check 1: Is confidence high enough?
            if best_confidence < self.MIN_CONFIDENCE_THRESHOLD:
                logger.debug(f"Column '{col_name}': best match '{best['field_name']}' "
                           f"below threshold ({best_confidence:.1%} < {self.MIN_CONFIDENCE_THRESHOLD:.0%})")
                needs_fallback = True
            
            # Check 2: Is there a clear winner? (sufficient gap to 2nd best)
            if not needs_fallback and len(relevant_matches) > 1:
                second_best = relevant_matches[1]
                gap = best_confidence - second_best.get("similarity", 0)
                
                if (second_best.get("field_name") != best.get("field_name") and 
                    gap < self.MIN_CONFIDENCE_GAP):
                    logger.debug(f"Column '{col_name}': ambiguous match between "
                               f"'{best['field_name']}' ({best_confidence:.1%}) and "
                               f"'{second_best['field_name']}' ({second_best.get('similarity', 0):.1%})")
                    needs_fallback = True
            
            profile = profile_lookup.get(col_name, {})
            
            if self.openai_verify_all:
                columns_to_verify.append({
                    "column_name": col_name,
                    "column_type": profile.get("detected_type", "unknown"),
                    "sample_values": profile.get("sample_values", [])[:5],
                    "candidates": relevant_matches[:self.OPENAI_TOP_K_CANDIDATES],
                })
            elif needs_fallback:
                unmapped_columns.append({
                    "column_name": col_name,
                    "column_type": profile.get("detected_type", "unknown"),
                    "sample_values": profile.get("sample_values", [])[:5],
                    "candidates": relevant_matches[:self.OPENAI_TOP_K_CANDIDATES],
                })
        
        # Make a copy of column_matches to modify
        enhanced_matches = {k: list(v) for k, v in column_matches.items()}
        
        # Handle verify-all mode
        if columns_to_verify:
            logger.info(f"🔍 Verifying {len(columns_to_verify)} column mappings with OpenAI...")
            
            for col_info in columns_to_verify:
                col_name = col_info["column_name"]
                
                result = self.openai_fallback.verify_and_find_match(
                    column_name=col_name,
                    column_type=col_info["column_type"],
                    sample_values=col_info["sample_values"],
                    candidates=col_info["candidates"],
                    document_type=winning_doc_type or "unknown",
                    schema_registry=self.schema_registry,
                    vertical=self._get_vertical_from_doc_type(winning_doc_type),
                    vertical_context=self.vertical_context,
                )
                
                if result.get("target_field"):
                    # Update the match with OpenAI's verified result
                    # Insert at front with boosted confidence
                    verified_match = {
                        "field_name": result["target_field"],
                        "similarity": result.get("confidence", 0.9),
                        "document_type": winning_doc_type,
                        "field_type": result.get("field_type"),
                        "required": result.get("required", False),
                        "source": "openai_verified",
                        "reason": result.get("reason", ""),
                    }
                    # Put verified match first
                    enhanced_matches[col_name] = [verified_match] + [
                        m for m in enhanced_matches.get(col_name, [])
                        if m.get("field_name") != result["target_field"]
                    ]
        
        # Handle normal fallback mode
        elif unmapped_columns:
            logger.info(f"Sending {len(unmapped_columns)} unmapped columns to OpenAI fallback")
            
            fallback_results = self.openai_fallback.process_unmapped_columns(
                unmapped_columns=unmapped_columns,
                document_type=winning_doc_type or "unknown",
            )
            
            for col_name, result in fallback_results.items():
                if result.get("target_field"):
                    # Insert OpenAI's suggestion at front
                    fallback_match = {
                        "field_name": result["target_field"],
                        "similarity": result.get("confidence", 0.75),
                        "document_type": winning_doc_type,
                        "field_type": result.get("field_type"),
                        "required": result.get("required", False),
                        "source": "openai_fallback",
                        "reason": result.get("reason", ""),
                    }
                    enhanced_matches[col_name] = [fallback_match] + [
                        m for m in enhanced_matches.get(col_name, [])
                        if m.get("field_name") != result["target_field"]
                    ]
                    logger.info(f"OpenAI fallback mapped '{col_name}' → '{result['target_field']}'")
        
        return enhanced_matches
    
    
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
                    source_type="",  # Not available here
                    target_field=m["field_name"],
                    target_type=m["field_type"],
                    document_type=m["document_type"],
                    similarity=m["similarity"],
                    required=m["required"],
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
                if self.openai_fallback and self.openai_fallback.is_available:
                    openai_client = getattr(self.openai_fallback, '_client', None)
                
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
