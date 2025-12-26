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
    
    # Classify with DSPy for column verification
    python classify_csv.py --csv data/unknown_file.csv --hybrid --dspy
    
    # Output as JSON
    python classify_csv.py --csv data/unknown_file.csv --json
"""

import argparse
import json
import logging
import os
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
    
    # Vertical is required for hybrid mode
    if args.hybrid and not args.vertical:
        logger.error("--vertical is required for hybrid classification")
        logger.error("Available verticals: medical")
        logger.error("Usage: python classify_csv.py --csv <file> --hybrid --vertical medical")
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
    dspy_service = None
    dspy_verify_all = getattr(args, 'openai_verify', False)  # --dspy-verify flag
    
    # Enable DSPy for LLM-based column classification
    if getattr(args, 'dspy', False) or dspy_verify_all:
        try:
            from csv_analyzer.services.dspy_service import create_dspy_service
            
            # Check for compiled model
            compiled_path = Path(__file__).parent / "models" / "dspy_compiled"
            dspy_compiled_path = None
            if compiled_path.exists():
                dspy_compiled_path = compiled_path
                logger.info(f"📦 Loading compiled DSPy model from {compiled_path}")
            
            dspy_service = create_dspy_service(
                model=f"openai/{getattr(args, 'model', 'gpt-4o-mini')}",
                compiled_path=dspy_compiled_path,
                enabled=True,
            )
            if dspy_service.is_available:
                if dspy_verify_all:
                    logger.info("🔍 DSPy verify-all mode enabled (will verify every mapping)")
                else:
                    logger.info("🚀 DSPy classification enabled")
            else:
                logger.warning("DSPy not available (set OPENAI_API_KEY env var)")
                dspy_service = None
                dspy_verify_all = False
        except Exception as e:
            logger.warning(f"Failed to initialize DSPy: {e}")
            dspy_verify_all = False
    
    engine = ClassificationEngine(
        embeddings_client,
        dspy_service=dspy_service,
        dspy_verify_all=dspy_verify_all,
    )
    
    # Classify
    logger.info(f"Classifying: {args.csv}")
    
    if args.hybrid:
        # Use hybrid scoring (document + column level)
        result = engine.classify_hybrid(
            csv_file=csv_path,
            vertical=args.vertical,
            k=args.k,
            force_reindex=getattr(args, 'reindex', False),
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
    
    # Column mappings with candidates displayed under each mapping
    if result.suggested_mappings:
        print("\n" + "-"*70)
        print("Suggested Column Mappings (from Schema Matching):")
        print("-"*70)
        
        # Get candidates lookup and winning document type
        column_candidates = getattr(result, 'column_candidates', {}) or {}
        winning_doc_type = result.document_type
        
        for source, mapping in result.suggested_mappings.items():
            target = mapping.get("target_field") or mapping.get("target") or "(no mapping)"
            conf = mapping.get("confidence", 0)
            conf_str = f"{conf:.0%}" if conf > 0 else "-"
            required = "⚡" if mapping.get("required") else ""
            
            # Get type compatibility info
            source_type = mapping.get("source_type", "")
            target_type = mapping.get("field_type", "")
            type_compat = mapping.get("type_compatibility", 1.0)
            raw_sim = mapping.get("raw_similarity", conf)
            
            # Get DSPy-related info
            mapping_source = mapping.get("source", "embeddings")
            dspy_reason = mapping.get("reason", "")
            dspy_attempts = mapping.get("attempts")
            
            # Color code by confidence
            if conf >= 0.8:
                status = "✅"
            elif conf >= 0.5:
                status = "⚠️"
            else:
                status = "❓"
            
            # Add source indicator
            if mapping_source == "dspy_verified":
                source_tag = " [DSPy verified]"
            elif mapping_source == "dspy":
                source_tag = " [DSPy]"
            else:
                source_tag = ""
            
            # Type compatibility indicator
            if type_compat < 0.95 and type_compat > 0:
                if type_compat >= 0.80:
                    type_tag = f" 🟡 type:{source_type}→{target_type}"
                else:
                    type_tag = f" 🔴 type:{source_type}→{target_type}"
            else:
                type_tag = ""
            
            print(f"\n  {source:<23} {'→':<3} {target:<21}{required} {conf_str:<8} {status}{source_tag}{type_tag}")
            
            # Show DSPy reasoning if available
            if dspy_reason:
                print(f"      💬 DSPy: {dspy_reason}")
            if dspy_attempts:
                print(f"      🔄 Attempts: {dspy_attempts} candidate(s) tried")
            
            # Show transformation info if needed
            transformation = mapping.get("transformation")
            if transformation and transformation.get("needs_transformation"):
                source_unit = transformation.get("source_unit", "?")
                target_unit = transformation.get("target_unit", "?")
                formula = transformation.get("formula", "?")
                func_name = transformation.get("transform_function", "?")
                trans_conf = transformation.get("confidence", "medium")
                trans_reason = transformation.get("reason", "")
                
                conf_icon = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(trans_conf, "⚪")
                print(f"      🔄 Transform: {source_unit} → {target_unit} ({formula})")
                print(f"         Function: {func_name}")
                if trans_reason:
                    print(f"         {conf_icon} {trans_reason}")
            
            # Show candidates for this column
            candidates = column_candidates.get(source, [])
            if candidates:
                # Filter candidates for winning document type
                relevant_candidates = []
                for c in candidates:
                    # Handle both ColumnMatch objects and dicts
                    if hasattr(c, 'document_type'):
                        doc_type = c.document_type
                        field_name = c.target_field
                        similarity = c.similarity
                        req = c.required
                        type_compat = getattr(c, 'type_compatibility', 1.0)
                        raw_sim = getattr(c, 'raw_similarity', similarity)
                        source_type = getattr(c, 'source_type', '')
                        target_type = getattr(c, 'target_type', '')
                    else:
                        doc_type = c.get("document_type", "")
                        field_name = c.get("field_name", c.get("target_field", ""))
                        similarity = c.get("similarity", 0)
                        req = c.get("required", False)
                        type_compat = c.get("type_compatibility", 1.0)
                        raw_sim = c.get("raw_similarity", similarity)
                        source_type = c.get("source_type", "")
                        target_type = c.get("field_type", "")
                    
                    if winning_doc_type and doc_type == winning_doc_type:
                        relevant_candidates.append({
                            "field_name": field_name,
                            "similarity": similarity,
                            "required": req,
                            "type_compatibility": type_compat,
                            "raw_similarity": raw_sim,
                            "source_type": source_type,
                            "field_type": target_type,
                        })
                
                if relevant_candidates:
                    print("      Candidates:")
                    for i, cand in enumerate(relevant_candidates[:5]):
                        field_name = cand["field_name"]
                        similarity = cand["similarity"]
                        req = cand["required"]
                        type_compat = cand.get("type_compatibility", 1.0)
                        raw_sim = cand.get("raw_similarity", similarity)
                        source_type = cand.get("source_type", "?")
                        target_type = cand.get("field_type", "?")
                        
                        # Determine if this is the selected one
                        is_selected = (field_name == target)
                        marker = "→" if is_selected else " "
                        req_icon = "⚡" if req else ""
                        
                        # Type compatibility indicator
                        if type_compat >= 0.95:
                            type_icon = "🟢"
                        elif type_compat >= 0.80:
                            type_icon = "🟡"
                        else:
                            type_icon = "🔴"
                        
                        # Add reasoning based on embedding threshold
                        if i == 0:  # Best candidate
                            if similarity >= 0.82:
                                emb_reason = "✓ above threshold"
                            else:
                                emb_reason = "✗ below 82% threshold"
                        elif i == 1 and len(relevant_candidates) > 1:
                            gap = relevant_candidates[0]["similarity"] - similarity
                            if gap < 0.02:
                                emb_reason = f"⚠ ambiguous (gap {gap:.1%})"
                            else:
                                emb_reason = ""
                        else:
                            emb_reason = ""
                        
                        # Show if this was rejected/accepted by DSPy
                        if mapping_source in ("dspy_verified", "dspy"):
                            if is_selected:
                                dspy_status = " ← accepted"
                            elif i < (dspy_attempts or 1):
                                dspy_status = " ← rejected"
                            else:
                                dspy_status = ""
                        else:
                            dspy_status = ""
                        
                        # Type info string
                        type_str = f"{type_icon} {source_type}→{target_type}" if type_compat < 1.0 else ""
                        reason_str = f"  {emb_reason}" if emb_reason else ""
                        print(f"      {marker} {i+1}. {field_name:<18} {similarity:>5.1%} {req_icon:<2}{type_str}{reason_str}{dspy_status}")
                else:
                    # Show all candidates if none match winning doc type
                    print("      Candidates (other doc types):")
                    for i, c in enumerate(candidates[:3]):
                        if hasattr(c, 'target_field'):
                            field_name = c.target_field
                            similarity = c.similarity
                            doc_type = c.document_type
                        else:
                            field_name = c.get("field_name", c.get("target_field", ""))
                            similarity = c.get("similarity", 0)
                            doc_type = c.get("document_type", "")
                        print(f"        {i+1}. {field_name:<18} {similarity:>5.1%}  ({doc_type})")
            else:
                print("      Candidates: (none found)")
    
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
        help="Vertical domain (e.g., 'medical'). REQUIRED for --hybrid mode."
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
        "--dspy",
        action="store_true",
        dest="dspy",
        help="Use DSPy for LLM-based column classification (for low-confidence matches)"
    )
    parser.add_argument(
        "--dspy-verify",
        action="store_true",
        dest="openai_verify",
        help="Verify ALL column mappings with DSPy (rejects wrong matches, tries next candidate)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        dest="model",
        help="LLM model to use (default: gpt-4o-mini)"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON"
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Force reindex of schema embeddings in ChromaDB (use after schema changes)"
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
