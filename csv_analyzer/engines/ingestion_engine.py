"""
Ground Truth Ingestion Engine.

Handles the process of:
1. Reading a CSV file
2. Profiling its columns
3. Generating text representation
4. Creating embeddings
5. Storing in PostgreSQL with pgvector
"""

import logging
import uuid
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
    DocumentTypeRepository,
    GroundTruthRecord,
    GroundTruthRepository,
    VerticalRepository,
)
from csv_analyzer.multilingual_embeddings_client import MultilingualEmbeddingsClient

logger = logging.getLogger(__name__)


class IngestionEngine:
    """
    Engine for ingesting ground truth CSV files.
    
    Usage:
        engine = IngestionEngine(embeddings_client)
        
        gt_id = engine.ingest_csv(
            csv_file="hospital_shifts.csv",
            vertical="medical",
            document_type="employee_shifts",
            column_mappings={"emp_id": "employee_id", ...}
        )
    """
    
    def __init__(self, embeddings_client: MultilingualEmbeddingsClient):
        """
        Initialize the ingestion engine.
        
        Args:
            embeddings_client: Multilingual embeddings client for generating vectors
        """
        self.embeddings_client = embeddings_client
    
    def ingest_csv(
        self,
        csv_file: Union[str, Path, BinaryIO, pd.DataFrame],
        vertical: str,
        document_type: str,
        column_mappings: Dict[str, str],
        external_id: Optional[str] = None,
        source_description: Optional[str] = None,
        labeler: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> int:
        """
        Ingest a CSV file as ground truth.
        
        Args:
            csv_file: Path to CSV file, file-like object, or DataFrame
            vertical: Vertical name (e.g., "medical")
            document_type: Document type name (e.g., "employee_shifts")
            column_mappings: Dict mapping source columns to target fields
                            e.g., {"emp_id": "employee_id", "work_date": "shift_date"}
            external_id: Optional custom ID (auto-generated if not provided)
            source_description: Optional description of the data source
            labeler: Optional name/email of person who labeled this
            notes: Optional notes about this ground truth
            
        Returns:
            Database ID of the created ground truth record
        """
        logger.info(f"Ingesting CSV: {csv_file if not isinstance(csv_file, pd.DataFrame) else 'DataFrame'}")
        
        # 1. Read CSV or use DataFrame directly
        if isinstance(csv_file, pd.DataFrame):
            df = csv_file
            file_name = "dataframe_input.csv"
        elif isinstance(csv_file, (str, Path)):
            df = pd.read_csv(csv_file)
            file_name = Path(csv_file).name
        else:
            df = pd.read_csv(csv_file)
            file_name = "uploaded_file.csv"
        
        logger.info(f"Read CSV with {len(df)} rows and {len(df.columns)} columns")
        
        # 2. Profile columns
        column_profiles = profile_dataframe(df)
        logger.info(f"Profiled {len(column_profiles)} columns")
        
        # 3. Generate text representation
        text_repr = csv_to_text_representation(column_profiles)
        logger.info(f"Generated text representation ({len(text_repr)} chars)")
        
        # 4. Generate embedding (use passage prefix for documents)
        embedding = self._create_passage_embedding(text_repr)
        if embedding is None:
            raise RuntimeError("Failed to generate embedding")
        logger.info(f"Generated embedding ({len(embedding)} dimensions)")
        
        # 5. Get/create vertical and document type
        vertical_id = VerticalRepository.get_or_create(vertical)
        doc_type_id = DocumentTypeRepository.get_or_create(vertical_id, document_type)
        
        # 6. Generate external ID if not provided
        if external_id is None:
            external_id = f"gt_{vertical}_{document_type}_{uuid.uuid4().hex[:8]}"
        
        # 7. Check if already exists
        if GroundTruthRepository.exists(external_id):
            raise ValueError(f"Ground truth with external_id '{external_id}' already exists")
        
        # 8. Create record
        record = GroundTruthRecord(
            external_id=external_id,
            document_type_id=doc_type_id,
            source_description=source_description or f"Ingested from {file_name}",
            row_count=len(df),
            column_count=len(df.columns),
            column_profiles=column_profiles,
            text_representation=text_repr,
            embedding=np.array(embedding),
            column_mappings=column_mappings,
            labeler=labeler,
            notes=notes,
        )
        
        # 9. Insert into database
        gt_id = GroundTruthRepository.insert(record)
        logger.info(f"✅ Inserted ground truth record: {external_id} (ID: {gt_id})")
        
        # 10. Also store individual column mappings in knowledge base
        self._store_column_mappings(
            column_profiles=column_profiles,
            column_mappings=column_mappings,
            vertical_id=vertical_id,
            document_type_id=doc_type_id,
        )
        
        return gt_id
    
    def _create_passage_embedding(self, text: str) -> Optional[List[float]]:
        """
        Create embedding for a passage (document to be stored).
        
        Uses "passage: " prefix for e5 models.
        """
        # The embeddings client already handles prefix based on length,
        # but for documents we always want passage prefix
        prefixed_text = f"passage: {text}"
        
        # Use the raw encode method to ensure passage prefix
        if hasattr(self.embeddings_client, '_model') and self.embeddings_client._model:
            embedding = self.embeddings_client._model.encode(
                prefixed_text,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            return embedding.tolist()
        
        # Fallback to standard method
        return self.embeddings_client.create_embedding(text)
    
    def _store_column_mappings(
        self,
        column_profiles: List[Dict[str, Any]],
        column_mappings: Dict[str, str],
        vertical_id: int,
        document_type_id: int,
    ):
        """
        Store individual column mappings in the knowledge base.
        
        This enables column-level similarity search for mapping suggestions.
        """
        # Build a lookup of column profiles by name
        profile_by_name = {p["column_name"]: p for p in column_profiles}
        
        for source_col, target_field in column_mappings.items():
            profile = profile_by_name.get(source_col)
            if not profile:
                logger.warning(f"Column '{source_col}' not found in profiles, skipping")
                continue
            
            # Generate text for this column
            col_text = column_to_embedding_text(profile)
            
            # Generate embedding
            embedding = self._create_passage_embedding(col_text)
            if embedding is None:
                logger.warning(f"Failed to generate embedding for column '{source_col}'")
                continue
            
            # Store in knowledge base
            ColumnMappingKBRepository.insert_or_update(
                vertical_id=vertical_id,
                document_type_id=document_type_id,
                source_column_name=source_col,
                source_column_type=profile.get("detected_type", "unknown"),
                sample_values=profile.get("sample_values", [])[:10],
                target_field=target_field,
                column_text_representation=col_text,
                embedding=np.array(embedding),
            )
        
        logger.info(f"Stored {len(column_mappings)} column mappings in knowledge base")
    
    def ingest_batch(
        self,
        records: List[Dict[str, Any]],
    ) -> List[int]:
        """
        Ingest multiple ground truth records.
        
        Args:
            records: List of dicts with keys:
                - csv_file: Path to CSV
                - vertical: Vertical name
                - document_type: Document type name
                - column_mappings: Column mappings dict
                - (optional) external_id, source_description, labeler, notes
                
        Returns:
            List of created record IDs
        """
        ids = []
        for i, record in enumerate(records):
            try:
                gt_id = self.ingest_csv(**record)
                ids.append(gt_id)
            except Exception as e:
                logger.error(f"Failed to ingest record {i}: {e}")
                ids.append(None)
        
        return ids
