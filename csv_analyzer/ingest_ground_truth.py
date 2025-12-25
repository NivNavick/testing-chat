#!/usr/bin/env python3
"""
Ground Truth Ingestion Script.

This script ingests labeled CSV files as ground truth for training
the CSV classification system.

Usage:
    # Ingest a single CSV with column mappings
    python ingest_ground_truth.py --csv data/shifts.csv \
        --vertical medical \
        --document-type employee_shifts \
        --mappings '{"emp_id": "employee_id", "work_date": "shift_date"}'
    
    # List all ground truth records
    python ingest_ground_truth.py --list
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from csv_analyzer.db.connection import init_database
from csv_analyzer.engines.ingestion_engine import IngestionEngine
from csv_analyzer.multilingual_embeddings_client import get_multilingual_embeddings_client
from csv_analyzer.db.repositories.ground_truth_repo import GroundTruthRepository

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def ingest_single_csv(args):
    """Ingest a single CSV file with explicit column mappings."""
    if not args.csv:
        logger.error("--csv is required")
        return 1
    
    if not args.vertical:
        logger.error("--vertical is required")
        return 1
    
    if not args.document_type:
        logger.error("--document-type is required")
        return 1
    
    if not args.mappings:
        logger.error("--mappings is required (JSON string)")
        return 1
    
    try:
        column_mappings = json.loads(args.mappings)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in --mappings: {e}")
        return 1
    
    # Initialize database
    logger.info("Initializing database...")
    init_database(
        host="localhost",
        port=5432,
        database="csv_mapping",
        user="postgres",
        password="postgres",
        run_migrations=True,
    )
    
    # Initialize embeddings client
    logger.info("Loading embedding model...")
    embeddings_client = get_multilingual_embeddings_client()
    
    if not embeddings_client.is_available:
        logger.error("Embedding model not available")
        return 1
    
    # Create ingestion engine
    engine = IngestionEngine(embeddings_client)
    
    # Ingest CSV
    try:
        gt_id = engine.ingest_csv(
            csv_file=args.csv,
            vertical=args.vertical,
            document_type=args.document_type,
            column_mappings=column_mappings,
            external_id=args.external_id,
            source_description=args.description,
            labeler=args.labeler,
            notes=args.notes,
        )
        logger.info(f"✅ Successfully ingested ground truth (ID: {gt_id})")
        return 0
    except Exception as e:
        logger.error(f"Failed to ingest: {e}")
        return 1


def list_ground_truth():
    """List all ground truth records."""
    # Initialize database
    init_database(
        host="localhost",
        port=5432,
        database="csv_mapping",
        user="postgres",
        password="postgres",
        run_migrations=False,
    )
    
    records = GroundTruthRepository.list_all()
    
    if not records:
        print("No ground truth records found.")
        return 0
    
    print(f"\n{'='*80}")
    print(f"{'ID':<5} {'External ID':<35} {'Type':<25} {'Vertical':<10}")
    print(f"{'='*80}")
    
    for r in records:
        print(f"{r['id']:<5} {r['external_id']:<35} {r['document_type']:<25} {r['vertical']:<10}")
    
    print(f"{'='*80}")
    print(f"Total: {len(records)} records")
    
    return 0


def delete_ground_truth(external_id: str):
    """Delete a ground truth record by external ID."""
    from csv_analyzer.db.connection import Database
    
    # Initialize database
    init_database(
        host="localhost",
        port=5432,
        database="csv_mapping",
        user="postgres",
        password="postgres",
        run_migrations=False,
    )
    
    # Check if exists
    if not GroundTruthRepository.exists(external_id):
        logger.error(f"Ground truth record not found: {external_id}")
        return 1
    
    # Delete
    with Database.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM ground_truth WHERE external_id = %s",
                (external_id,)
            )
            logger.info(f"✅ Deleted ground truth record: {external_id}")
    
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Ingest ground truth CSV files for classification training"
    )
    
    parser.add_argument(
        "--csv",
        type=str,
        help="Path to CSV file to ingest"
    )
    parser.add_argument(
        "--vertical",
        type=str,
        help="Vertical name (e.g., 'medical', 'banking')"
    )
    parser.add_argument(
        "--document-type",
        type=str,
        dest="document_type",
        help="Document type (e.g., 'employee_shifts', 'staff_clinical_procedures')"
    )
    parser.add_argument(
        "--mappings",
        type=str,
        help="Column mappings as JSON string, e.g., '{\"emp_id\": \"employee_id\"}'"
    )
    parser.add_argument(
        "--external-id",
        type=str,
        dest="external_id",
        help="Custom external ID (auto-generated if not provided)"
    )
    parser.add_argument(
        "--description",
        type=str,
        help="Description of the data source"
    )
    parser.add_argument(
        "--labeler",
        type=str,
        help="Name/email of person who labeled this"
    )
    parser.add_argument(
        "--notes",
        type=str,
        help="Additional notes"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all ground truth records"
    )
    parser.add_argument(
        "--delete",
        type=str,
        help="Delete a ground truth record by external ID"
    )
    
    args = parser.parse_args()
    
    if args.list:
        return list_ground_truth()
    elif args.delete:
        return delete_ground_truth(args.delete)
    elif args.csv:
        return ingest_single_csv(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
