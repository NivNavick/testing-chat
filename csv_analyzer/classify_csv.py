#!/usr/bin/env python3
"""
CSV Classification Script.

This script classifies an unknown CSV file by finding similar
ground truth examples and suggesting document type and column mappings.

Usage:
    # Classify a CSV file
    python classify_csv.py --csv data/unknown_file.csv
    
    # Classify with vertical filter
    python classify_csv.py --csv data/unknown_file.csv --vertical medical
    
    # Output as JSON
    python classify_csv.py --csv data/unknown_file.csv --json
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from csv_analyzer.db.connection import init_database
from csv_analyzer.engines.classification_engine import ClassificationEngine
from csv_analyzer.multilingual_embeddings_client import get_multilingual_embeddings_client
from csv_analyzer.db.repositories.ground_truth_repo import GroundTruthRepository

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def classify_csv(args):
    """Classify a CSV file."""
    if not args.csv:
        logger.error("--csv is required")
        return 1
    
    csv_path = Path(args.csv)
    if not csv_path.exists():
        logger.error(f"File not found: {args.csv}")
        return 1
    
    # Initialize database
    logger.info("Connecting to database...")
    init_database(
        host="localhost",
        port=5432,
        database="csv_mapping",
        user="postgres",
        password="postgres",
        run_migrations=False,
    )
    
    # Check if we have ground truth
    gt_count = GroundTruthRepository.count()
    if gt_count == 0:
        logger.warning("No ground truth records found!")
        logger.warning("Run 'python ingest_ground_truth.py --sample' first to add sample data.")
        return 1
    
    logger.info(f"Found {gt_count} ground truth records")
    
    # Initialize embeddings client
    logger.info("Loading embedding model...")
    embeddings_client = get_multilingual_embeddings_client()
    
    if not embeddings_client.is_available:
        logger.error("Embedding model not available")
        return 1
    
    # Create classification engine
    engine = ClassificationEngine(embeddings_client)
    
    # Classify
    logger.info(f"Classifying: {args.csv}")
    
    if args.hybrid:
        # Use hybrid scoring (document + column level)
        result = engine.classify_hybrid(
            csv_file=csv_path,
            vertical=args.vertical,
            k=args.k,
        )
        # Output results
        if args.json:
            print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
        else:
            print_hybrid_result(result)
    elif args.enhanced:
        result = engine.classify_with_column_suggestions(
            csv_file=csv_path,
            vertical=args.vertical,
            k=args.k,
        )
        if args.json:
            print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
        else:
            print_result(result)
    else:
        result = engine.classify(
            csv_file=csv_path,
            vertical=args.vertical,
            k=args.k,
        )
        if args.json:
            print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
        else:
            print_result(result)
    
    return 0


def print_result(result):
    """Print classification result in a human-readable format."""
    print("\n" + "="*70)
    print("CLASSIFICATION RESULT")
    print("="*70)
    
    # Main classification
    confidence_bar = "█" * int(result.confidence * 20) + "░" * (20 - int(result.confidence * 20))
    
    print(f"\n📄 Document Type: {result.document_type or 'Unknown'}")
    print(f"🏢 Vertical:      {result.vertical or 'Unknown'}")
    print(f"📊 Confidence:    [{confidence_bar}] {result.confidence:.1%}")
    
    # Similar examples
    if result.similar_examples:
        print("\n" + "-"*70)
        print("Similar Ground Truth Examples:")
        print("-"*70)
        for i, ex in enumerate(result.similar_examples, 1):
            print(f"  {i}. {ex['external_id']}")
            print(f"     Type: {ex['document_type']} | Similarity: {ex['similarity']:.1%}")
    
    # Column mappings
    if result.suggested_mappings:
        print("\n" + "-"*70)
        print("Suggested Column Mappings:")
        print("-"*70)
        print(f"{'Source Column':<25} {'→':<3} {'Target Field':<25} {'Confidence':<12}")
        print("-"*70)
        
        for source, mapping in result.suggested_mappings.items():
            target = mapping.get("target") or "(no mapping)"
            conf = mapping.get("confidence", 0)
            conf_str = f"{conf:.0%}" if conf > 0 else "-"
            
            # Color code by confidence
            if conf >= 0.8:
                status = "✅"
            elif conf >= 0.5:
                status = "⚠️"
            else:
                status = "❓"
            
            print(f"{source:<25} {'→':<3} {target:<25} {conf_str:<10} {status}")
    
    # Input column profiles
    if result.column_profiles:
        print("\n" + "-"*70)
        print("Input CSV Column Profiles:")
        print("-"*70)
        for p in result.column_profiles:
            samples = p.get("sample_values", [])[:3]
            samples_str = ", ".join(str(s) for s in samples)
            if len(samples_str) > 40:
                samples_str = samples_str[:37] + "..."
            print(f"  • {p['column_name']} ({p['detected_type']}): {samples_str}")
    
    print("\n" + "="*70)


def print_hybrid_result(result):
    """Print hybrid classification result with detailed scoring breakdown."""
    print("\n" + "="*70)
    print("HYBRID CLASSIFICATION RESULT")
    print("="*70)
    
    # Main classification
    final_bar = "█" * int(result.final_score * 20) + "░" * (20 - int(result.final_score * 20))
    
    print(f"\n📄 Document Type: {result.document_type or 'Unknown'}")
    print(f"🏢 Vertical:      {result.vertical or 'Unknown'}")
    print(f"📊 Final Score:   [{final_bar}] {result.final_score:.1%}")
    
    # Score breakdown
    print("\n" + "-"*70)
    print("Score Breakdown:")
    print("-"*70)
    doc_bar = "█" * int(result.document_score * 15) + "░" * (15 - int(result.document_score * 15))
    col_bar = "█" * int(result.column_score * 15) + "░" * (15 - int(result.column_score * 15))
    cov_bar = "█" * int(result.coverage_score * 15) + "░" * (15 - int(result.coverage_score * 15))
    
    print(f"  📁 Document Similarity (30%):  [{doc_bar}] {result.document_score:.1%}")
    print(f"  📋 Column Matching (50%):      [{col_bar}] {result.column_score:.1%}")
    print(f"  ✓  Field Coverage (20%):       [{cov_bar}] {result.coverage_score:.1%}")
    
    # All document type scores
    if result.all_scores:
        print("\n" + "-"*70)
        print("All Document Type Scores:")
        print("-"*70)
        print(f"{'Document Type':<25} {'Final':<10} {'Doc':<10} {'Col':<10} {'Cov':<10}")
        print("-"*70)
        
        sorted_scores = sorted(
            result.all_scores.items(),
            key=lambda x: x[1].get("final_score", 0),
            reverse=True
        )
        
        for doc_type, scores in sorted_scores:
            marker = "→" if doc_type == result.document_type else " "
            print(
                f"{marker} {doc_type:<23} "
                f"{scores.get('final_score', 0):.1%}     "
                f"{scores.get('document_score', 0):.1%}     "
                f"{scores.get('column_score', 0):.1%}     "
                f"{scores.get('coverage_score', 0):.1%}"
            )
    
    # Similar ground truth examples
    if result.similar_examples:
        print("\n" + "-"*70)
        print("Similar Ground Truth Examples:")
        print("-"*70)
        for i, ex in enumerate(result.similar_examples, 1):
            print(f"  {i}. {ex['external_id']}")
            print(f"     Type: {ex['document_type']} | Similarity: {ex['similarity']:.1%}")
    
    # Column mappings with improved display
    if result.suggested_mappings:
        print("\n" + "-"*70)
        print("Suggested Column Mappings (from Schema Matching):")
        print("-"*70)
        print(f"{'Source Column':<25} {'→':<3} {'Target Field':<25} {'Confidence':<12}")
        print("-"*70)
        
        for source, mapping in result.suggested_mappings.items():
            target = mapping.get("target_field") or mapping.get("target") or "(no mapping)"
            conf = mapping.get("confidence", 0)
            conf_str = f"{conf:.0%}" if conf > 0 else "-"
            required = "⚡" if mapping.get("required") else ""
            
            # Color code by confidence
            if conf >= 0.8:
                status = "✅"
            elif conf >= 0.5:
                status = "⚠️"
            else:
                status = "❓"
            
            print(f"{source:<25} {'→':<3} {target:<23}{required} {conf_str:<10} {status}")
    
    # Input column profiles
    if result.column_profiles:
        print("\n" + "-"*70)
        print("Input CSV Column Profiles:")
        print("-"*70)
        for p in result.column_profiles:
            samples = p.get("sample_values", [])[:3]
            samples_str = ", ".join(str(s) for s in samples)
            if len(samples_str) > 40:
                samples_str = samples_str[:37] + "..."
            print(f"  • {p['column_name']} ({p['detected_type']}): {samples_str}")
    
    print("\n" + "="*70)


def demo_classification():
    """Run a demo classification with synthetic data."""
    import pandas as pd
    import io
    
    # Initialize database
    logger.info("Connecting to database...")
    init_database(
        host="localhost",
        port=5432,
        database="csv_mapping",
        user="postgres",
        password="postgres",
        run_migrations=False,
    )
    
    # Check if we have ground truth
    gt_count = GroundTruthRepository.count()
    if gt_count == 0:
        logger.warning("No ground truth records found!")
        logger.warning("Run 'python ingest_ground_truth.py --sample' first.")
        return 1
    
    # Initialize embeddings client
    logger.info("Loading embedding model...")
    embeddings_client = get_multilingual_embeddings_client()
    
    if not embeddings_client.is_available:
        logger.error("Embedding model not available")
        return 1
    
    # Create test CSVs
    test_cases = [
        {
            "name": "English shifts (slightly different columns)",
            "csv": """employee_number,date,in_time,out_time,dept_code
EMP100,2024-05-01,08:30,17:00,Radiology
EMP101,2024-05-01,07:00,15:30,Pharmacy
EMP102,2024-05-01,09:00,17:30,Lab
"""
        },
        {
            "name": "Hebrew shifts (unseen)",
            "csv": """קוד_עובד,תאריך_עבודה,כניסה,יציאה,יחידה
ע001,2024-06-01,08:00,16:00,מיון
ע002,2024-06-01,09:00,17:00,פנימית
ע003,2024-06-01,07:00,15:00,כירורגיה
"""
        },
        {
            "name": "Medical actions (English)",
            "csv": """id,patient,provider,code,description,datetime,location
1001,PT100,DR01,99213,Office Visit,2024-04-01 10:00:00,Clinic A
1002,PT101,DR02,99214,Extended Visit,2024-04-01 11:00:00,Clinic B
1003,PT100,DR01,93000,ECG Test,2024-04-01 10:30:00,Clinic A
"""
        },
    ]
    
    engine = ClassificationEngine(embeddings_client)
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n{'#'*70}")
        print(f"# TEST CASE {i}: {test['name']}")
        print(f"{'#'*70}")
        
        df = pd.read_csv(io.StringIO(test["csv"]))
        result = engine.classify(df)
        print_result(result)
    
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Classify an unknown CSV file"
    )
    
    parser.add_argument(
        "--csv",
        type=str,
        help="Path to CSV file to classify"
    )
    parser.add_argument(
        "--vertical",
        type=str,
        help="Filter by vertical (e.g., 'medical')"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of similar examples to retrieve (default: 5)"
    )
    parser.add_argument(
        "--enhanced",
        action="store_true",
        help="Use enhanced column mapping with column-level embeddings (PostgreSQL only)"
    )
    parser.add_argument(
        "--hybrid",
        action="store_true",
        help="Use hybrid scoring: document similarity (PostgreSQL) + column matching (ChromaDB)"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run demo classification with test data"
    )
    
    args = parser.parse_args()
    
    if args.demo:
        return demo_classification()
    elif args.csv:
        return classify_csv(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
