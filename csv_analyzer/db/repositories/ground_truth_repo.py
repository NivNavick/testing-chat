"""
Repository for ground truth records and related entities.

Handles CRUD operations and vector similarity search for:
- Verticals
- Document types
- Ground truth records
- Column mappings knowledge base
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from psycopg2.extras import RealDictCursor

from csv_analyzer.db.connection import Database

logger = logging.getLogger(__name__)


@dataclass
class GroundTruthRecord:
    """Data class for ground truth records."""
    external_id: str
    document_type_id: int
    source_description: str
    row_count: int
    column_count: int
    column_profiles: List[Dict[str, Any]]
    text_representation: str
    embedding: np.ndarray
    column_mappings: Dict[str, str]
    labeler: Optional[str] = None
    notes: Optional[str] = None
    id: Optional[int] = None


class VerticalRepository:
    """Repository for verticals (medical, banking, etc.)."""
    
    @staticmethod
    def get_or_create(name: str, description: str = None) -> int:
        """
        Get vertical ID by name, or create if not exists.
        
        Args:
            name: Vertical name (e.g., "medical")
            description: Optional description
            
        Returns:
            Vertical ID
        """
        with Database.get_connection() as conn:
            with conn.cursor() as cur:
                # Try to get existing
                cur.execute(
                    "SELECT id FROM verticals WHERE name = %s",
                    (name,)
                )
                row = cur.fetchone()
                if row:
                    return row[0]
                
                # Create new
                cur.execute(
                    "INSERT INTO verticals (name, description) VALUES (%s, %s) RETURNING id",
                    (name, description)
                )
                return cur.fetchone()[0]
    
    @staticmethod
    def get_by_name(name: str) -> Optional[Dict]:
        """Get vertical by name."""
        with Database.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    "SELECT * FROM verticals WHERE name = %s",
                    (name,)
                )
                row = cur.fetchone()
                return dict(row) if row else None
    
    @staticmethod
    def list_all() -> List[Dict]:
        """List all verticals."""
        with Database.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT * FROM verticals ORDER BY name")
                return [dict(row) for row in cur.fetchall()]


class DocumentTypeRepository:
    """Repository for document types (employee_shifts, medical_actions, etc.)."""
    
    @staticmethod
    def get_or_create(vertical_id: int, name: str, description: str = None) -> int:
        """
        Get document type ID, or create if not exists.
        
        Args:
            vertical_id: Vertical ID
            name: Document type name
            description: Optional description
            
        Returns:
            Document type ID
        """
        with Database.get_connection() as conn:
            with conn.cursor() as cur:
                # Try to get existing
                cur.execute(
                    "SELECT id FROM document_types WHERE vertical_id = %s AND name = %s",
                    (vertical_id, name)
                )
                row = cur.fetchone()
                if row:
                    return row[0]
                
                # Create new
                cur.execute(
                    """INSERT INTO document_types (vertical_id, name, description) 
                       VALUES (%s, %s, %s) RETURNING id""",
                    (vertical_id, name, description)
                )
                return cur.fetchone()[0]
    
    @staticmethod
    def get_by_name(vertical_name: str, doc_type_name: str) -> Optional[Dict]:
        """Get document type by vertical and document type names."""
        with Database.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """SELECT dt.* FROM document_types dt
                       JOIN verticals v ON dt.vertical_id = v.id
                       WHERE v.name = %s AND dt.name = %s""",
                    (vertical_name, doc_type_name)
                )
                row = cur.fetchone()
                return dict(row) if row else None
    
    @staticmethod
    def list_by_vertical(vertical_id: int) -> List[Dict]:
        """List all document types for a vertical."""
        with Database.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    "SELECT * FROM document_types WHERE vertical_id = %s ORDER BY name",
                    (vertical_id,)
                )
                return [dict(row) for row in cur.fetchall()]


class GroundTruthRepository:
    """Repository for ground truth records with vector similarity search."""
    
    @staticmethod
    def insert(record: GroundTruthRecord) -> int:
        """
        Insert a ground truth record.
        
        Args:
            record: GroundTruthRecord instance
            
        Returns:
            Inserted record ID
        """
        with Database.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO ground_truth (
                        external_id, document_type_id, source_description,
                        row_count, column_count, column_profiles,
                        text_representation, embedding, column_mappings,
                        labeler, notes
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    )
                    RETURNING id""",
                    (
                        record.external_id,
                        record.document_type_id,
                        record.source_description,
                        record.row_count,
                        record.column_count,
                        json.dumps(record.column_profiles),
                        record.text_representation,
                        record.embedding.tolist() if isinstance(record.embedding, np.ndarray) else record.embedding,
                        json.dumps(record.column_mappings),
                        record.labeler,
                        record.notes,
                    )
                )
                record_id = cur.fetchone()[0]
                logger.info(f"Inserted ground truth record: {record.external_id} (ID: {record_id})")
                return record_id
    
    @staticmethod
    def find_similar(
        query_embedding: np.ndarray,
        vertical_id: Optional[int] = None,
        limit: int = 5
    ) -> List[Dict]:
        """
        Find K most similar ground truth records using cosine similarity.
        
        Args:
            query_embedding: Query embedding vector (1024 dims)
            vertical_id: Optional vertical ID to filter by
            limit: Number of results to return
            
        Returns:
            List of similar records with similarity scores
        """
        embedding_list = query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else query_embedding
        
        with Database.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                if vertical_id:
                    cur.execute(
                        """SELECT 
                            gt.id,
                            gt.external_id,
                            dt.name as document_type,
                            v.name as vertical,
                            gt.column_mappings,
                            gt.text_representation,
                            gt.column_profiles,
                            1 - (gt.embedding <=> %s::vector) as similarity
                        FROM ground_truth gt
                        JOIN document_types dt ON gt.document_type_id = dt.id
                        JOIN verticals v ON dt.vertical_id = v.id
                        WHERE v.id = %s
                        ORDER BY gt.embedding <=> %s::vector
                        LIMIT %s""",
                        (embedding_list, vertical_id, embedding_list, limit)
                    )
                else:
                    cur.execute(
                        """SELECT 
                            gt.id,
                            gt.external_id,
                            dt.name as document_type,
                            v.name as vertical,
                            gt.column_mappings,
                            gt.text_representation,
                            gt.column_profiles,
                            1 - (gt.embedding <=> %s::vector) as similarity
                        FROM ground_truth gt
                        JOIN document_types dt ON gt.document_type_id = dt.id
                        JOIN verticals v ON dt.vertical_id = v.id
                        ORDER BY gt.embedding <=> %s::vector
                        LIMIT %s""",
                        (embedding_list, embedding_list, limit)
                    )
                
                results = []
                for row in cur.fetchall():
                    row_dict = dict(row)
                    # Parse JSON fields
                    if isinstance(row_dict.get("column_mappings"), str):
                        row_dict["column_mappings"] = json.loads(row_dict["column_mappings"])
                    if isinstance(row_dict.get("column_profiles"), str):
                        row_dict["column_profiles"] = json.loads(row_dict["column_profiles"])
                    results.append(row_dict)
                
                return results
    
    @staticmethod
    def get_by_external_id(external_id: str) -> Optional[Dict]:
        """Get ground truth record by external ID."""
        with Database.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    "SELECT * FROM ground_truth WHERE external_id = %s",
                    (external_id,)
                )
                row = cur.fetchone()
                if row:
                    row_dict = dict(row)
                    if isinstance(row_dict.get("column_mappings"), str):
                        row_dict["column_mappings"] = json.loads(row_dict["column_mappings"])
                    if isinstance(row_dict.get("column_profiles"), str):
                        row_dict["column_profiles"] = json.loads(row_dict["column_profiles"])
                    return row_dict
                return None
    
    @staticmethod
    def exists(external_id: str) -> bool:
        """Check if a ground truth record exists."""
        with Database.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT 1 FROM ground_truth WHERE external_id = %s",
                    (external_id,)
                )
                return cur.fetchone() is not None
    
    @staticmethod
    def count() -> int:
        """Count total ground truth records."""
        with Database.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM ground_truth")
                return cur.fetchone()[0]
    
    @staticmethod
    def list_all() -> List[Dict]:
        """List all ground truth records (without embeddings)."""
        with Database.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """SELECT 
                        gt.id, gt.external_id, gt.source_description,
                        gt.row_count, gt.column_count,
                        dt.name as document_type, v.name as vertical,
                        gt.labeler, gt.created_at
                    FROM ground_truth gt
                    JOIN document_types dt ON gt.document_type_id = dt.id
                    JOIN verticals v ON dt.vertical_id = v.id
                    ORDER BY gt.created_at DESC"""
                )
                return [dict(row) for row in cur.fetchall()]


class ColumnMappingKBRepository:
    """Repository for column mapping knowledge base."""
    
    @staticmethod
    def insert_or_update(
        vertical_id: int,
        document_type_id: int,
        source_column_name: str,
        source_column_type: str,
        sample_values: List[str],
        target_field: str,
        column_text_representation: str,
        embedding: np.ndarray,
    ) -> int:
        """
        Insert or update a column mapping in the knowledge base.
        If the mapping exists, increment occurrence count.
        
        Returns:
            Record ID
        """
        embedding_list = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
        
        with Database.get_connection() as conn:
            with conn.cursor() as cur:
                # Try to update existing
                cur.execute(
                    """UPDATE column_mappings_kb 
                       SET occurrence_count = occurrence_count + 1,
                           updated_at = NOW()
                       WHERE vertical_id = %s 
                         AND document_type_id = %s 
                         AND source_column_name = %s 
                         AND target_field = %s
                       RETURNING id""",
                    (vertical_id, document_type_id, source_column_name, target_field)
                )
                row = cur.fetchone()
                if row:
                    return row[0]
                
                # Insert new
                cur.execute(
                    """INSERT INTO column_mappings_kb (
                        vertical_id, document_type_id,
                        source_column_name, source_column_type, sample_values,
                        target_field, column_text_representation, embedding
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id""",
                    (
                        vertical_id,
                        document_type_id,
                        source_column_name,
                        source_column_type,
                        json.dumps(sample_values),
                        target_field,
                        column_text_representation,
                        embedding_list,
                    )
                )
                return cur.fetchone()[0]
    
    @staticmethod
    def find_similar_columns(
        query_embedding: np.ndarray,
        vertical_id: Optional[int] = None,
        document_type_id: Optional[int] = None,
        limit: int = 5
    ) -> List[Dict]:
        """
        Find similar columns in the knowledge base.
        
        Args:
            query_embedding: Embedding of the column to match
            vertical_id: Optional vertical filter
            document_type_id: Optional document type filter
            limit: Number of results
            
        Returns:
            List of similar column mappings with similarity scores
        """
        embedding_list = query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else query_embedding
        
        with Database.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                where_clauses = []
                params = [embedding_list]
                
                if vertical_id:
                    where_clauses.append("vertical_id = %s")
                    params.append(vertical_id)
                
                if document_type_id:
                    where_clauses.append("document_type_id = %s")
                    params.append(document_type_id)
                
                where_sql = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""
                
                params.extend([embedding_list, limit])
                
                cur.execute(
                    f"""SELECT 
                        id, source_column_name, source_column_type,
                        target_field, occurrence_count,
                        1 - (embedding <=> %s::vector) as similarity
                    FROM column_mappings_kb
                    {where_sql}
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s""",
                    params
                )
                
                return [dict(row) for row in cur.fetchall()]
