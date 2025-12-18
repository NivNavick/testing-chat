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
    
    # Ingest sample ground truth (for testing)
    python ingest_ground_truth.py --sample
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
    """Ingest a single CSV file."""
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


def ingest_sample_ground_truth():
    """Ingest sample ground truth data for testing."""
    import pandas as pd
    import io
    
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
    logger.info("Loading embedding model (this may take a minute)...")
    embeddings_client = get_multilingual_embeddings_client()
    
    if not embeddings_client.is_available:
        logger.error("Embedding model not available")
        return 1
    
    # Create ingestion engine
    engine = IngestionEngine(embeddings_client)
    
    # Sample 1: English employee shifts
    sample1_csv = """emp_id,shift_date,start_time,end_time,department
E001,2024-01-15,08:00,16:00,Emergency
E002,2024-01-15,09:00,17:00,ICU
E003,2024-01-15,07:30,15:30,Surgery
E001,2024-01-16,08:00,16:00,Emergency
E002,2024-01-16,09:00,17:00,ICU
"""
    
    sample1_mappings = {
        "emp_id": "employee_id",
        "shift_date": "shift_date",
        "start_time": "shift_start",
        "end_time": "shift_end",
        "department": "department_code",
    }
    
    # Sample 2: Hebrew employee shifts (different column names)
    sample2_csv = """מספר_עובד,תאריך,שעת_התחלה,שעת_סיום,מחלקה
W001,2024-02-01,08:00,16:00,חדר_מיון
W002,2024-02-01,09:00,17:00,טיפול_נמרץ
W003,2024-02-01,07:00,15:00,ניתוחים
W001,2024-02-02,08:00,16:00,חדר_מיון
"""
    
    sample2_mappings = {
        "מספר_עובד": "employee_id",
        "תאריך": "shift_date",
        "שעת_התחלה": "shift_start",
        "שעת_סיום": "shift_end",
        "מחלקה": "department_code",
    }
    
    # Sample 3: Custom column names (employee shifts)
    sample3_csv = """staff_code,work_dt,clock_in,clock_out,unit_code,hrs
STF001,2024-03-01,08:00,16:00,ER,8
STF002,2024-03-01,09:00,17:00,ICU,8
STF003,2024-03-01,06:00,14:00,OR,8
STF001,2024-03-02,08:00,16:00,ER,8
"""
    
    sample3_mappings = {
        "staff_code": "employee_id",
        "work_dt": "shift_date",
        "clock_in": "shift_start",
        "clock_out": "shift_end",
        "unit_code": "department_code",
        "hrs": "duration_minutes",
    }
    
    # Sample 4: Medical actions
    sample4_csv = """action_id,patient_num,doctor_id,procedure_code,procedure_name,action_date,dept
ACT001,P12345,D001,CPT99213,Office Visit,2024-01-15 09:30:00,Cardiology
ACT002,P12346,D002,CPT99214,Office Visit Extended,2024-01-15 10:00:00,Neurology
ACT003,P12345,D001,CPT93000,ECG,2024-01-15 09:45:00,Cardiology
"""
    
    sample4_mappings = {
        "action_id": "action_id",
        "patient_num": "patient_id",
        "doctor_id": "performer_id",
        "procedure_code": "action_code",
        "procedure_name": "action_name",
        "action_date": "performed_at",
        "dept": "department_code",
    }
    
    # Sample 5: Hebrew medical actions
    sample5_csv = """מזהה_פעולה,מספר_מטופל,מזהה_רופא,קוד_פעולה,שם_פעולה,תאריך_ביצוע,מחלקה
P001,M100,R01,12345,בדיקת דם,2024-02-10 08:00:00,מעבדה
P002,M101,R02,12346,צילום חזה,2024-02-10 09:00:00,רדיולוגיה
P003,M100,R01,12347,בדיקת שתן,2024-02-10 08:30:00,מעבדה
"""
    
    sample5_mappings = {
        "מזהה_פעולה": "action_id",
        "מספר_מטופל": "patient_id",
        "מזהה_רופא": "performer_id",
        "קוד_פעולה": "action_code",
        "שם_פעולה": "action_name",
        "תאריך_ביצוע": "performed_at",
        "מחלקה": "department_code",
    }
    
    samples = [
        {
            "csv_data": sample1_csv,
            "vertical": "medical",
            "document_type": "employee_shifts",
            "column_mappings": sample1_mappings,
            "external_id": "gt_sample_shifts_english",
            "source_description": "Sample English employee shifts",
        },
        {
            "csv_data": sample2_csv,
            "vertical": "medical",
            "document_type": "employee_shifts",
            "column_mappings": sample2_mappings,
            "external_id": "gt_sample_shifts_hebrew",
            "source_description": "Sample Hebrew employee shifts (משמרות עובדים)",
        },
        {
            "csv_data": sample3_csv,
            "vertical": "medical",
            "document_type": "employee_shifts",
            "column_mappings": sample3_mappings,
            "external_id": "gt_sample_shifts_custom",
            "source_description": "Sample employee shifts with custom column names",
        },
        {
            "csv_data": sample4_csv,
            "vertical": "medical",
            "document_type": "medical_actions",
            "column_mappings": sample4_mappings,
            "external_id": "gt_sample_actions_english",
            "source_description": "Sample English medical actions",
        },
        {
            "csv_data": sample5_csv,
            "vertical": "medical",
            "document_type": "medical_actions",
            "column_mappings": sample5_mappings,
            "external_id": "gt_sample_actions_hebrew",
            "source_description": "Sample Hebrew medical actions (פעולות רפואיות)",
        },
    ]
    
    success_count = 0
    for sample in samples:
        # Check if already exists
        if GroundTruthRepository.exists(sample["external_id"]):
            logger.info(f"Skipping {sample['external_id']} (already exists)")
            continue
        
        try:
            # Create DataFrame from CSV string
            df = pd.read_csv(io.StringIO(sample["csv_data"]))
            
            # Save to temp file (ingestion expects file path or DataFrame)
            gt_id = engine.ingest_csv(
                csv_file=df,  # Pass DataFrame directly
                vertical=sample["vertical"],
                document_type=sample["document_type"],
                column_mappings=sample["column_mappings"],
                external_id=sample["external_id"],
                source_description=sample["source_description"],
                labeler="sample_data",
            )
            logger.info(f"✅ Ingested: {sample['external_id']} (ID: {gt_id})")
            success_count += 1
        except Exception as e:
            logger.error(f"Failed to ingest {sample['external_id']}: {e}")
    
    logger.info(f"\n{'='*50}")
    logger.info(f"Ingested {success_count}/{len(samples)} sample ground truth records")
    logger.info(f"Total ground truth records: {GroundTruthRepository.count()}")
    
    return 0


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
    print(f"{'ID':<5} {'External ID':<35} {'Type':<20} {'Vertical':<10}")
    print(f"{'='*80}")
    
    for r in records:
        print(f"{r['id']:<5} {r['external_id']:<35} {r['document_type']:<20} {r['vertical']:<10}")
    
    print(f"{'='*80}")
    print(f"Total: {len(records)} records")
    
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
        help="Document type (e.g., 'employee_shifts', 'medical_actions')"
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
        "--sample",
        action="store_true",
        help="Ingest sample ground truth data for testing"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all ground truth records"
    )
    
    args = parser.parse_args()
    
    if args.list:
        return list_ground_truth()
    elif args.sample:
        return ingest_sample_ground_truth()
    else:
        return ingest_single_csv(args)


if __name__ == "__main__":
    sys.exit(main())
