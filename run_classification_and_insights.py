#!/usr/bin/env python3
"""
Classify CSVs and Run Insights.

This script:
1. Classifies unknown CSV files using the classification engine
2. Logs document types and column mappings for observability
3. Loads classified data into the insights engine
4. Runs all insights and exports results

Usage:
    # Basic usage
    python run_classification_and_insights.py \
        --csv tests/insights_test_data/employee_shifts.csv \
        --csv tests/insights_test_data/employee_compensation.csv \
        --csv tests/insights_test_data/staff_clinical_procedures.csv \
        --vertical medical \
        --hybrid
    
    # With DSPy verification
    python run_classification_and_insights.py \
        --csv path/to/unknown.csv \
        --vertical medical \
        --hybrid \
        --dspy-verify
    
    # Custom output directory
    python run_classification_and_insights.py \
        --csv path/to/data.csv \
        --vertical medical \
        --output-dir results/2024-01-01
"""

import argparse
import json
import logging
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from csv_analyzer.db.connection import init_database
from csv_analyzer.db.repositories.ground_truth_repo import GroundTruthRepository
from csv_analyzer.engines.classification_engine import ClassificationEngine
from csv_analyzer.insights import InsightsEngine
from csv_analyzer.multilingual_embeddings_client import get_multilingual_embeddings_client

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ClassificationResult:
    """Store classification results for logging."""
    def __init__(self, csv_path: Path, result):
        self.csv_path = csv_path
        self.result = result
        self.document_type = result.document_type
        # Handle both HybridClassificationResult (has final_score) and ClassificationResult (has confidence)
        self.confidence = getattr(result, 'final_score', getattr(result, 'confidence', 0.0))
        self.vertical = result.vertical
        self.column_mappings = self._extract_mappings(result)
    
    def _extract_mappings(self, result) -> Dict[str, str]:
        """Extract column mappings from result."""
        mappings = {}
        if hasattr(result, 'suggested_mappings') and result.suggested_mappings:
            for source, mapping in result.suggested_mappings.items():
                target = mapping.get('target_field') or mapping.get('target')
                if target:
                    mappings[source] = target
        return mappings
    
    def to_dict(self) -> dict:
        """Convert to dictionary for logging."""
        return {
            'file': str(self.csv_path),
            'filename': self.csv_path.name,
            'document_type': self.document_type,
            'confidence': f"{self.confidence:.1%}",
            'vertical': self.vertical,
            'column_mappings': self.column_mappings,
            'mapped_columns': len(self.column_mappings),
        }


def print_classification_summary(results: List[ClassificationResult]):
    """Print a summary of all classifications."""
    print("\n" + "=" * 70)
    print("CLASSIFICATION SUMMARY")
    print("=" * 70)
    
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result.csv_path.name}")
        print(f"   Document Type: {result.document_type}")
        print(f"   Confidence:    {result.confidence:.1%}")
        print(f"   Vertical:      {result.vertical}")
        print(f"   Columns:       {len(result.column_mappings)} mapped")
        
        if result.column_mappings:
            print(f"   Mappings:")
            for source, target in list(result.column_mappings.items())[:5]:
                print(f"      {source:<25} → {target}")
            if len(result.column_mappings) > 5:
                print(f"      ... and {len(result.column_mappings) - 5} more")


def save_classification_log(results: List[ClassificationResult], output_dir: Path):
    """Save classification results to JSON log file."""
    log_file = output_dir / "classification_log.json"
    
    log_data = {
        'timestamp': datetime.now().isoformat(),
        'total_files': len(results),
        'classifications': [r.to_dict() for r in results]
    }
    
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Classification log saved to {log_file}")
    return log_file


def main():
    parser = argparse.ArgumentParser(
        description="Classify CSVs and run insights analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--csv",
        action="append",
        required=True,
        help="Path to CSV file to classify (can specify multiple times)"
    )
    parser.add_argument(
        "--vertical",
        type=str,
        required=True,
        help="Vertical domain (e.g., 'medical'). REQUIRED."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for results (default: csv_analyzer/insights_output/<timestamp>)"
    )
    parser.add_argument(
        "--hybrid",
        action="store_true",
        help="Use hybrid scoring: document similarity + column matching (recommended)"
    )
    parser.add_argument(
        "--dspy-verify",
        action="store_true",
        help="Verify column mappings with DSPy (requires OPENAI_API_KEY)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="LLM model for DSPy (default: gpt-4o-mini)"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of similar examples to retrieve (default: 5)"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("CLASSIFICATION + INSIGHTS PIPELINE")
    print("=" * 70)
    print(f"Vertical: {args.vertical}")
    print(f"Mode: {'Hybrid' if args.hybrid else 'Standard'}")
    print(f"CSV files: {len(args.csv)}")
    for csv_file in args.csv:
        print(f"  - {csv_file}")
    print()
    
    # ========================================================================
    # STEP 1: Initialize Database and Classification Engine
    # ========================================================================
    print("=" * 70)
    print("STEP 1: INITIALIZING CLASSIFICATION ENGINE")
    print("=" * 70)
    
    # Initialize database
    logger.info("Connecting to PostgreSQL database...")
    try:
        init_database(
            host="localhost",
            port=5432,
            database="csv_mapping",
            user="postgres",
            password="postgres",
            run_migrations=False,
        )
        print("✓ Database connected")
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        print("\n❌ DATABASE CONNECTION REQUIRED")
        print("Please ensure PostgreSQL is running with the csv_mapping database.")
        print("Run: python ingest_ground_truth.py --sample")
        return 1
    
    # Check ground truth data
    gt_count = GroundTruthRepository.count()
    if gt_count == 0:
        print("\n❌ NO GROUND TRUTH DATA")
        print("Run: python ingest_ground_truth.py --sample")
        return 1
    print(f"✓ Found {gt_count} ground truth records")
    
    # Initialize embeddings client
    logger.info("Loading embedding model...")
    embeddings_client = get_multilingual_embeddings_client()
    
    if not embeddings_client.is_available:
        logger.error("Embedding model not available")
        return 1
    print("✓ Embeddings client ready")
    
    # Initialize DSPy service (optional)
    dspy_service = None
    if args.dspy_verify:
        logger.info("Initializing DSPy service...")
        try:
            from csv_analyzer.services.dspy_service import create_dspy_service
            
            compiled_path = Path(__file__).parent / "csv_analyzer" / "models" / "dspy_compiled"
            dspy_compiled_path = compiled_path if compiled_path.exists() else None
            
            dspy_service = create_dspy_service(
                model=f"openai/{args.model}",
                compiled_path=dspy_compiled_path,
                enabled=True,
            )
            
            if dspy_service.is_available:
                print(f"✓ DSPy service enabled (model: {args.model})")
            else:
                logger.warning("DSPy not available (set OPENAI_API_KEY)")
                dspy_service = None
        except Exception as e:
            logger.warning(f"Failed to initialize DSPy: {e}")
            dspy_service = None
    
    # Create classification engine
    classification_engine = ClassificationEngine(
        embeddings_client,
        dspy_service=dspy_service,
        dspy_verify_all=args.dspy_verify,
    )
    print("✓ Classification engine ready")
    
    # ========================================================================
    # STEP 2: Classify All CSVs
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 2: CLASSIFYING CSV FILES")
    print("=" * 70)
    
    classification_results = []
    
    for csv_file in args.csv:
        csv_path = Path(csv_file)
        
        if not csv_path.exists():
            print(f"\n❌ File not found: {csv_file}")
            continue
        
        print(f"\n📄 Processing: {csv_path.name}")
        print("-" * 70)
        
        try:
            # Classify the CSV
            if args.hybrid:
                logger.info("Running hybrid classification...")
                result = classification_engine.classify_hybrid(
                    csv_file=str(csv_path),
                    vertical=args.vertical,
                    k=args.k,
                )
            else:
                logger.info("Running standard classification...")
                result = classification_engine.classify(
                    csv_file=str(csv_path),
                    vertical=args.vertical,
                    k=args.k,
                )
            
            # Store result
            classification_result = ClassificationResult(csv_path, result)
            classification_results.append(classification_result)
            
            # Print immediate feedback
            print(f"✓ Document Type: {result.document_type}")
            confidence = getattr(result, 'final_score', result.confidence)
            print(f"✓ Confidence:    {confidence:.1%}")
            print(f"✓ Mappings:      {len(classification_result.column_mappings)} columns")
            
            # Show top mappings
            for i, (source, target) in enumerate(list(classification_result.column_mappings.items())[:3]):
                print(f"     {source} → {target}")
            if len(classification_result.column_mappings) > 3:
                print(f"     ... and {len(classification_result.column_mappings) - 3} more")
            
        except Exception as e:
            logger.error(f"Classification failed for {csv_path.name}: {e}")
            import traceback
            traceback.print_exc()
            print(f"❌ Classification failed: {e}")
            continue
    
    if not classification_results:
        print("\n❌ No files were successfully classified")
        return 1
    
    # Print summary
    print_classification_summary(classification_results)
    
    # ========================================================================
    # STEP 3: Load Data into Insights Engine
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 3: LOADING DATA INTO INSIGHTS ENGINE")
    print("=" * 70)
    
    insights_engine = InsightsEngine(vertical=args.vertical)
    
    loaded_count = 0
    for classification_result in classification_results:
        csv_path = classification_result.csv_path
        doc_type = classification_result.document_type
        mappings = classification_result.column_mappings
        
        print(f"\n📥 Loading: {csv_path.name}")
        print(f"   Type: {doc_type}")
        print(f"   Mappings: {len(mappings)} columns")
        
        try:
            insights_engine.load_csv_manual(
                csv_file=str(csv_path),
                document_type=doc_type,
                column_mappings=mappings,
            )
            print("   ✓ Loaded successfully")
            loaded_count += 1
        except Exception as e:
            logger.error(f"Failed to load {csv_path.name}: {e}")
            print(f"   ❌ Load failed: {e}")
            import traceback
            traceback.print_exc()
    
    if loaded_count == 0:
        print("\n❌ No data was loaded into insights engine")
        return 1
    
    # Show data store status
    status = insights_engine.get_status()
    print(f"\n📊 Data Store Status:")
    print(f"   Tables:     {len(status.tables)}")
    print(f"   Total rows: {status.total_rows}")
    
    for table in status.tables:
        print(f"   - {table.document_type}: {table.row_count} rows, {len(table.columns)} columns")
    
    # ========================================================================
    # STEP 4: Run All Insights
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 4: RUNNING INSIGHTS")
    print("=" * 70)
    
    results = insights_engine.run_all_insights()
    
    success_count = 0
    fail_count = 0
    
    for name, result in results.items():
        print(f"\n{'─' * 70}")
        print(f"📋 {name.upper()}")
        print(f"{'─' * 70}")
        
        if result.success:
            print(f"✓ {result.row_count} rows ({result.execution_time_ms:.1f}ms)")
            success_count += 1
            
            if result.row_count > 0 and result.row_count <= 10:
                # Show preview for small results
                print("\nPreview:")
                print(result.data.to_string(index=False, max_colwidth=40))
            elif result.row_count > 0:
                # Show just first few rows
                print(f"\nFirst 5 rows (of {result.row_count}):")
                print(result.data.head(5).to_string(index=False, max_colwidth=40))
        else:
            print(f"✗ Error: {result.error}")
            fail_count += 1
    
    print(f"\n{'=' * 70}")
    print(f"Insights Results: {success_count} succeeded, {fail_count} failed")
    print(f"{'=' * 70}")
    
    # ========================================================================
    # STEP 5: Export Results
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 5: EXPORTING RESULTS")
    print("=" * 70)
    
    # Create output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        run_id = datetime.now().strftime("%Y-%m-%d_%H%M%S") + "_" + str(uuid.uuid4())[:8]
        output_dir = Path("csv_analyzer/insights_output") / run_id
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Output directory: {output_dir}")
    
    # Save classification log
    log_file = save_classification_log(classification_results, output_dir)
    try:
        print(f"✓ Classification log: {log_file.relative_to(Path.cwd())}")
    except ValueError:
        print(f"✓ Classification log: {log_file}")
    
    # Export insights
    print("\n💾 Exporting insights to CSV...")
    exported = insights_engine.export_all_results(output_dir)
    
    for name, path in exported.items():
        path_obj = Path(path)
        try:
            print(f"  → {path_obj.relative_to(Path.cwd())}")
        except ValueError:
            print(f"  → {path_obj}")
    
    # Create latest symlink
    if not args.output_dir:  # Only create symlink for auto-generated paths
        latest_link = output_dir.parent / "latest"
        if latest_link.is_symlink() or latest_link.exists():
            latest_link.unlink()
        latest_link.symlink_to(output_dir.name)
        try:
            print(f"  → {latest_link.relative_to(Path.cwd())} -> {output_dir.name}")
        except ValueError:
            print(f"  → {latest_link} -> {output_dir.name}")
    
    # Close insights engine
    insights_engine.close()
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "=" * 70)
    print("✅ PIPELINE COMPLETE")
    print("=" * 70)
    print(f"Classified:  {len(classification_results)} files")
    print(f"Loaded:      {loaded_count} files")
    print(f"Insights:    {success_count} succeeded, {fail_count} failed")
    print(f"Output:      {output_dir}")
    print(f"Log:         {log_file.name}")
    print("=" * 70)
    
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

