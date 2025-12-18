"""
Hybrid Scoring Engine.

Combines document-level similarity (PostgreSQL) with column-level matching (ChromaDB)
to produce a final classification score.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from csv_analyzer.core.schema_embeddings import SchemaEmbeddingsService
from csv_analyzer.core.schema_registry import SchemaRegistry, get_schema_registry

logger = logging.getLogger(__name__)


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
    ):
        """
        Initialize the scoring engine.
        
        Args:
            schema_embeddings_service: ChromaDB service for column matching
            schema_registry: Schema registry for field information
            weight_document: Weight for document similarity (default 0.3)
            weight_column: Weight for column matching (default 0.5)
            weight_coverage: Weight for field coverage (default 0.2)
        """
        self.schema_service = schema_embeddings_service
        self.schema_registry = schema_registry or get_schema_registry()
        
        self.weight_document = weight_document or self.WEIGHT_DOCUMENT
        self.weight_column = weight_column or self.WEIGHT_COLUMN
        self.weight_coverage = weight_coverage or self.WEIGHT_COVERAGE
    
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
        
        # 6. Build suggested mappings
        suggested_mappings = self._build_suggested_mappings(
            column_matches=column_matches,
            winning_doc_type=winner,
        )
        
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
    
    def _build_suggested_mappings(
        self,
        column_matches: Dict[str, List[Dict]],
        winning_doc_type: Optional[str],
    ) -> Dict[str, Dict[str, Any]]:
        """Build suggested column mappings for the winning document type."""
        suggestions = {}
        
        for col_name, matches in column_matches.items():
            # Filter matches for winning doc type
            if winning_doc_type:
                relevant_matches = [
                    m for m in matches
                    if m["document_type"] == winning_doc_type
                ]
            else:
                relevant_matches = matches
            
            if relevant_matches:
                best = relevant_matches[0]
                suggestions[col_name] = {
                    "target_field": best["field_name"],
                    "confidence": best["similarity"],
                    "field_type": best["field_type"],
                    "required": best["required"],
                }
            else:
                suggestions[col_name] = {
                    "target_field": None,
                    "confidence": 0.0,
                    "field_type": None,
                    "required": False,
                }
        
        return suggestions
    
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
