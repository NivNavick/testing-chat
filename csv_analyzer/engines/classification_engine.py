"""
CSV Classification Engine.

Classifies unknown CSVs by finding similar ground truth examples
using embedding similarity search.
"""

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from csv_analyzer.columns_analyzer import profile_dataframe
from csv_analyzer.core.text_representation import (
    csv_to_text_representation,
    column_to_embedding_text,
)
from csv_analyzer.db.repositories.ground_truth_repo import (
    ColumnMappingKBRepository,
    GroundTruthRepository,
    VerticalRepository,
)
from csv_analyzer.multilingual_embeddings_client import MultilingualEmbeddingsClient

logger = logging.getLogger(__name__)


class ClassificationResult:
    """Result of CSV classification."""
    
    def __init__(
        self,
        document_type: Optional[str],
        vertical: Optional[str],
        confidence: float,
        suggested_mappings: Dict[str, Dict[str, Any]],
        similar_examples: List[Dict],
        column_profiles: List[Dict],
        text_representation: str,
    ):
        self.document_type = document_type
        self.vertical = vertical
        self.confidence = confidence
        self.suggested_mappings = suggested_mappings
        self.similar_examples = similar_examples
        self.column_profiles = column_profiles
        self.text_representation = text_representation
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_type": self.document_type,
            "vertical": self.vertical,
            "confidence": self.confidence,
            "suggested_mappings": self.suggested_mappings,
            "similar_examples": self.similar_examples,
            "column_profiles": [
                {
                    "column_name": p["column_name"],
                    "detected_type": p["detected_type"],
                    "sample_values": p.get("sample_values", [])[:5],
                }
                for p in self.column_profiles
            ],
        }
    
    def __repr__(self):
        return (
            f"ClassificationResult(document_type={self.document_type!r}, "
            f"vertical={self.vertical!r}, confidence={self.confidence:.2f})"
        )


class ClassificationEngine:
    """
    Engine for classifying unknown CSVs.
    
    Uses embedding similarity to find the most similar ground truth examples
    and aggregates their document types via weighted voting.
    
    Usage:
        engine = ClassificationEngine(embeddings_client)
        
        result = engine.classify("unknown_file.csv")
        print(result.document_type)  # "employee_shifts"
        print(result.confidence)     # 0.89
        print(result.suggested_mappings)  # {"emp_id": {"target": "employee_id", ...}}
    """
    
    def __init__(self, embeddings_client: MultilingualEmbeddingsClient):
        """
        Initialize the classification engine.
        
        Args:
            embeddings_client: Multilingual embeddings client
        """
        self.embeddings_client = embeddings_client
    
    def classify(
        self,
        csv_file: Union[str, Path, BinaryIO, pd.DataFrame],
        vertical: Optional[str] = None,
        k: int = 5,
    ) -> ClassificationResult:
        """
        Classify an unknown CSV by finding similar ground truth examples.
        
        Args:
            csv_file: Path to CSV, file object, or DataFrame
            vertical: Optional vertical to filter by (e.g., "medical")
            k: Number of similar examples to retrieve
            
        Returns:
            ClassificationResult with document type, confidence, and mappings
        """
        logger.info(f"Classifying CSV: {csv_file if not isinstance(csv_file, pd.DataFrame) else 'DataFrame'}")
        
        # 1. Load CSV if needed
        if isinstance(csv_file, pd.DataFrame):
            df = csv_file
        elif isinstance(csv_file, (str, Path)):
            df = pd.read_csv(csv_file)
        else:
            df = pd.read_csv(csv_file)
        
        logger.info(f"Loaded CSV with {len(df)} rows and {len(df.columns)} columns")
        
        # 2. Profile columns
        column_profiles = profile_dataframe(df)
        logger.info(f"Profiled {len(column_profiles)} columns")
        
        # 3. Generate text representation
        text_repr = csv_to_text_representation(column_profiles)
        logger.info(f"Generated text representation ({len(text_repr)} chars)")
        
        # 4. Generate query embedding
        query_embedding = self._create_query_embedding(text_repr)
        if query_embedding is None:
            logger.error("Failed to generate embedding")
            return ClassificationResult(
                document_type=None,
                vertical=None,
                confidence=0.0,
                suggested_mappings={},
                similar_examples=[],
                column_profiles=column_profiles,
                text_representation=text_repr,
            )
        
        # 5. Get vertical ID if filtering
        vertical_id = None
        if vertical:
            v = VerticalRepository.get_by_name(vertical)
            if v:
                vertical_id = v["id"]
            else:
                logger.warning(f"Vertical '{vertical}' not found, searching all verticals")
        
        # 6. Find similar ground truth records
        similar = GroundTruthRepository.find_similar(
            query_embedding=np.array(query_embedding),
            vertical_id=vertical_id,
            limit=k,
        )
        
        if not similar:
            logger.warning("No similar ground truth records found")
            return ClassificationResult(
                document_type=None,
                vertical=None,
                confidence=0.0,
                suggested_mappings={},
                similar_examples=[],
                column_profiles=column_profiles,
                text_representation=text_repr,
            )
        
        logger.info(f"Found {len(similar)} similar records")
        
        # 7. Aggregate results via weighted voting
        result = self._aggregate_results(similar, column_profiles)
        result.column_profiles = column_profiles
        result.text_representation = text_repr
        
        logger.info(f"Classification: {result.document_type} (confidence: {result.confidence:.2f})")
        
        return result
    
    def classify_with_column_suggestions(
        self,
        csv_file: Union[str, Path, BinaryIO, pd.DataFrame],
        vertical: Optional[str] = None,
        k: int = 5,
    ) -> ClassificationResult:
        """
        Classify CSV and also suggest column mappings using column-level embeddings.
        
        This provides more accurate column mappings by also querying the
        column_mappings_kb table for each column.
        """
        # First do standard classification
        result = self.classify(csv_file, vertical, k)
        
        if not result.document_type:
            return result
        
        # Enhance column mappings with column-level similarity
        vertical_obj = VerticalRepository.get_by_name(result.vertical)
        vertical_id = vertical_obj["id"] if vertical_obj else None
        
        enhanced_mappings = {}
        for profile in result.column_profiles:
            col_name = profile["column_name"]
            
            # Get suggestion from document-level similarity
            doc_level_suggestion = result.suggested_mappings.get(col_name, {})
            
            # Get suggestion from column-level similarity
            col_suggestion = self._suggest_column_mapping(
                profile, vertical_id
            )
            
            # Merge suggestions (prefer higher confidence)
            if col_suggestion and col_suggestion.get("confidence", 0) > doc_level_suggestion.get("confidence", 0):
                enhanced_mappings[col_name] = col_suggestion
            elif doc_level_suggestion:
                enhanced_mappings[col_name] = doc_level_suggestion
            else:
                enhanced_mappings[col_name] = {
                    "target": None,
                    "confidence": 0.0,
                    "source": "no_match",
                }
        
        result.suggested_mappings = enhanced_mappings
        return result
    
    def _create_query_embedding(self, text: str) -> Optional[List[float]]:
        """
        Create embedding for a query (document to be classified).
        
        Uses "query: " prefix for e5 models.
        """
        prefixed_text = f"query: {text}"
        
        if hasattr(self.embeddings_client, '_model') and self.embeddings_client._model:
            embedding = self.embeddings_client._model.encode(
                prefixed_text,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            return embedding.tolist()
        
        return self.embeddings_client.create_embedding(text)
    
    def _aggregate_results(
        self,
        similar: List[Dict],
        column_profiles: List[Dict],
    ) -> ClassificationResult:
        """
        Aggregate K nearest neighbors into final classification.
        
        Uses weighted voting based on similarity scores.
        """
        # Weighted voting for document type
        votes = defaultdict(float)
        vertical_votes = defaultdict(float)
        
        for record in similar:
            similarity = record.get("similarity", 0)
            doc_type = record["document_type"]
            vertical = record["vertical"]
            
            votes[doc_type] += similarity
            vertical_votes[vertical] += similarity
        
        # Find winner
        winner_doc_type = max(votes, key=votes.get) if votes else None
        winner_vertical = max(vertical_votes, key=vertical_votes.get) if vertical_votes else None
        
        # Calculate confidence
        total_weight = sum(votes.values())
        confidence = votes[winner_doc_type] / total_weight if total_weight > 0 else 0
        
        # Merge column mappings from matching examples
        source_columns = {p["column_name"] for p in column_profiles}
        suggested_mappings = self._merge_mappings(
            similar_records=[r for r in similar if r["document_type"] == winner_doc_type],
            source_columns=source_columns,
        )
        
        # Format similar examples for output
        similar_examples = [
            {
                "external_id": r["external_id"],
                "document_type": r["document_type"],
                "vertical": r["vertical"],
                "similarity": round(r.get("similarity", 0), 3),
            }
            for r in similar
        ]
        
        return ClassificationResult(
            document_type=winner_doc_type,
            vertical=winner_vertical,
            confidence=round(confidence, 3),
            suggested_mappings=suggested_mappings,
            similar_examples=similar_examples,
            column_profiles=[],  # Will be set by caller
            text_representation="",  # Will be set by caller
        )
    
    def _merge_mappings(
        self,
        similar_records: List[Dict],
        source_columns: set,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Merge column mappings from similar ground truth records.
        
        For each source column, finds the most common target mapping
        from the similar records.
        """
        # Count occurrences of each mapping
        mapping_counts = defaultdict(lambda: defaultdict(float))
        
        for record in similar_records:
            mappings = record.get("column_mappings", {})
            similarity = record.get("similarity", 1.0)
            
            for source, target in mappings.items():
                mapping_counts[source][target] += similarity
        
        # Build result for columns in the input CSV
        result = {}
        for source_col in source_columns:
            if source_col in mapping_counts:
                # Find best target
                targets = mapping_counts[source_col]
                best_target = max(targets, key=targets.get)
                total_weight = sum(targets.values())
                conf = targets[best_target] / total_weight if total_weight > 0 else 0
                
                result[source_col] = {
                    "target": best_target,
                    "confidence": round(conf, 3),
                    "source": "document_similarity",
                }
            else:
                result[source_col] = {
                    "target": None,
                    "confidence": 0.0,
                    "source": "no_match",
                }
        
        return result
    
    def _suggest_column_mapping(
        self,
        column_profile: Dict,
        vertical_id: Optional[int],
    ) -> Optional[Dict[str, Any]]:
        """
        Suggest mapping for a single column using column-level embeddings.
        """
        # Generate text for this column
        col_text = column_to_embedding_text(column_profile)
        
        # Generate query embedding
        embedding = self._create_query_embedding(col_text)
        if embedding is None:
            return None
        
        # Find similar columns
        similar = ColumnMappingKBRepository.find_similar_columns(
            query_embedding=np.array(embedding),
            vertical_id=vertical_id,
            limit=3,
        )
        
        if not similar:
            return None
        
        # Return top match
        top = similar[0]
        return {
            "target": top["target_field"],
            "confidence": round(top.get("similarity", 0), 3),
            "source": "column_similarity",
            "matched_column": top["source_column_name"],
        }
