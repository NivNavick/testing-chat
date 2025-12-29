#!/usr/bin/env python3
"""
Threshold Analysis Script.

Analyzes embedding scores across multiple CSVs to determine optimal
threshold for sending columns to DSPy.

Usage:
    python analyze_thresholds.py
"""

import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from csv_analyzer.db.connection import init_database
from csv_analyzer.multilingual_embeddings_client import get_multilingual_embeddings_client
from csv_analyzer.core.schema_embeddings import SchemaEmbeddingsService
from csv_analyzer.core.schema_registry import get_schema_registry
from csv_analyzer.columns_analyzer import profile_dataframe

# Configure logging
logging.basicConfig(
    level=logging.WARNING,  # Reduce noise
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class ColumnScoreData:
    """Score data for a single column."""
    csv_file: str
    column_name: str
    column_type: str
    sample_values: List[str]
    
    # Top candidates with scores
    candidates: List[Dict[str, Any]]
    
    # Derived metrics
    best_score: float = 0.0
    second_best_score: float = 0.0
    score_gap: float = 0.0
    best_field: str = ""
    second_best_field: str = ""
    
    def __post_init__(self):
        if self.candidates:
            self.best_score = self.candidates[0].get("similarity", 0)
            self.best_field = self.candidates[0].get("field_name", "")
            if len(self.candidates) > 1:
                self.second_best_score = self.candidates[1].get("similarity", 0)
                self.second_best_field = self.candidates[1].get("field_name", "")
                self.score_gap = self.best_score - self.second_best_score


@dataclass
class ThresholdAnalysis:
    """Analysis results for a specific threshold."""
    threshold: float
    gap_threshold: float
    
    total_columns: int = 0
    columns_above_threshold: int = 0
    columns_below_threshold: int = 0
    columns_ambiguous: int = 0  # Above threshold but gap too small
    columns_to_dspy: int = 0  # Total that would be sent to DSPy
    
    # Score distributions
    scores_above: List[float] = field(default_factory=list)
    scores_below: List[float] = field(default_factory=list)
    
    @property
    def dspy_percentage(self) -> float:
        return (self.columns_to_dspy / self.total_columns * 100) if self.total_columns else 0


def collect_scores(
    csv_files: List[Path],
    embeddings_service: SchemaEmbeddingsService,
    vertical: str = "medical",
) -> List[ColumnScoreData]:
    """
    Collect embedding scores for all columns in all CSV files.
    """
    all_scores = []
    
    for csv_path in csv_files:
        print(f"\n📄 Processing: {csv_path.name}")
        
        try:
            # Load CSV and analyze columns
            df = pd.read_csv(csv_path, nrows=500)  # Sample for speed
            column_profiles = profile_dataframe(df)
            
            if not column_profiles:
                print(f"   ⚠️  No columns found")
                continue
            
            # Get embeddings and matches
            results = embeddings_service.score_columns_against_schemas(
                columns=column_profiles,
                vertical=vertical,
            )
            
            column_matches = results.get("column_matches", {})
            
            # Collect scores for each column
            for col in column_profiles:
                col_name = col["column_name"]
                matches = column_matches.get(col_name, [])
                
                score_data = ColumnScoreData(
                    csv_file=csv_path.name,
                    column_name=col_name,
                    column_type=col.get("detected_type", "unknown"),
                    sample_values=col.get("sample_values", [])[:3],
                    candidates=matches[:5],
                )
                all_scores.append(score_data)
                
            print(f"   ✓ {len(column_profiles)} columns processed")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            continue
    
    return all_scores


def analyze_threshold(
    scores: List[ColumnScoreData],
    threshold: float,
    gap_threshold: float = 0.02,
) -> ThresholdAnalysis:
    """
    Analyze how many columns would be sent to DSPy at a given threshold.
    """
    analysis = ThresholdAnalysis(
        threshold=threshold,
        gap_threshold=gap_threshold,
        total_columns=len(scores),
    )
    
    for score in scores:
        if score.best_score < threshold:
            analysis.columns_below_threshold += 1
            analysis.columns_to_dspy += 1
            analysis.scores_below.append(score.best_score)
        elif score.score_gap < gap_threshold and len(score.candidates) > 1:
            # Above threshold but ambiguous
            analysis.columns_ambiguous += 1
            analysis.columns_to_dspy += 1
            analysis.scores_above.append(score.best_score)
        else:
            analysis.columns_above_threshold += 1
            analysis.scores_above.append(score.best_score)
    
    return analysis


def print_score_distribution(scores: List[ColumnScoreData]):
    """Print score distribution histogram."""
    print("\n" + "="*70)
    print("SCORE DISTRIBUTION")
    print("="*70)
    
    # Create buckets
    buckets = {
        "90-100%": [],
        "85-90%": [],
        "82-85%": [],
        "80-82%": [],
        "75-80%": [],
        "70-75%": [],
        "60-70%": [],
        "50-60%": [],
        "<50%": [],
    }
    
    for score in scores:
        s = score.best_score
        if s >= 0.90:
            buckets["90-100%"].append(score)
        elif s >= 0.85:
            buckets["85-90%"].append(score)
        elif s >= 0.82:
            buckets["82-85%"].append(score)
        elif s >= 0.80:
            buckets["80-82%"].append(score)
        elif s >= 0.75:
            buckets["75-80%"].append(score)
        elif s >= 0.70:
            buckets["70-75%"].append(score)
        elif s >= 0.60:
            buckets["60-70%"].append(score)
        elif s >= 0.50:
            buckets["50-60%"].append(score)
        else:
            buckets["<50%"].append(score)
    
    max_count = max(len(b) for b in buckets.values()) if buckets else 1
    
    for bucket_name, bucket_scores in buckets.items():
        count = len(bucket_scores)
        bar_len = int(count / max_count * 40) if max_count else 0
        bar = "█" * bar_len
        pct = count / len(scores) * 100 if scores else 0
        print(f"{bucket_name:>10} | {bar:<40} {count:>3} ({pct:>5.1f}%)")
    
    print(f"\nTotal columns: {len(scores)}")
    
    # Show some examples in each critical bucket
    print("\n" + "-"*70)
    print("EXAMPLES BY SCORE RANGE")
    print("-"*70)
    
    critical_buckets = ["82-85%", "80-82%", "75-80%", "70-75%"]
    for bucket_name in critical_buckets:
        examples = buckets.get(bucket_name, [])[:3]
        if examples:
            print(f"\n{bucket_name}:")
            for ex in examples:
                samples = ", ".join(str(s)[:15] for s in ex.sample_values[:2])
                print(f"  • {ex.column_name} ({ex.column_type})")
                print(f"    → {ex.best_field} ({ex.best_score:.1%})")
                if ex.second_best_field:
                    print(f"    → {ex.second_best_field} ({ex.second_best_score:.1%}) [gap: {ex.score_gap:.1%}]")
                print(f"    Samples: {samples}")


def print_threshold_comparison(scores: List[ColumnScoreData]):
    """Compare different threshold values."""
    print("\n" + "="*70)
    print("THRESHOLD COMPARISON")
    print("="*70)
    
    thresholds = [0.70, 0.75, 0.78, 0.80, 0.82, 0.85, 0.88, 0.90]
    gap_thresholds = [0.01, 0.02, 0.03, 0.05]
    
    print(f"\n{'Threshold':>10} {'Gap':>6} {'→DSPy':>8} {'%DSPy':>8} {'Below':>8} {'Ambig':>8} {'Above':>8}")
    print("-"*70)
    
    best_threshold = None
    best_score = float('inf')
    
    for threshold in thresholds:
        for gap in gap_thresholds:
            analysis = analyze_threshold(scores, threshold, gap)
            
            # Heuristic: We want ~20-30% going to DSPy for good accuracy
            # Too low = missing corrections, too high = expensive
            target_pct = 25
            score = abs(analysis.dspy_percentage - target_pct)
            
            if gap == 0.02:  # Only consider standard gap
                print(
                    f"{threshold:>10.0%} {gap:>6.0%} "
                    f"{analysis.columns_to_dspy:>8} {analysis.dspy_percentage:>7.1f}% "
                    f"{analysis.columns_below_threshold:>8} {analysis.columns_ambiguous:>8} "
                    f"{analysis.columns_above_threshold:>8}"
                )
                
                if score < best_score:
                    best_score = score
                    best_threshold = threshold
    
    return best_threshold


def print_columns_near_threshold(
    scores: List[ColumnScoreData],
    threshold: float = 0.82,
    margin: float = 0.05,
):
    """Print columns that are close to the threshold (critical zone)."""
    print("\n" + "="*70)
    print(f"COLUMNS IN CRITICAL ZONE ({threshold-margin:.0%} - {threshold+margin:.0%})")
    print("="*70)
    
    critical = [
        s for s in scores 
        if (threshold - margin) <= s.best_score <= (threshold + margin)
    ]
    
    # Sort by score
    critical.sort(key=lambda x: x.best_score, reverse=True)
    
    print(f"\n{len(critical)} columns in critical zone:\n")
    
    for s in critical:
        status = "✅ KEEP" if s.best_score >= threshold else "🔄 DSPy"
        ambig = " ⚠️AMBIG" if s.score_gap < 0.02 and len(s.candidates) > 1 else ""
        
        samples = ", ".join(str(v)[:12] for v in s.sample_values[:2])
        print(f"{status}{ambig}")
        print(f"  File: {s.csv_file}")
        print(f"  Column: {s.column_name} ({s.column_type})")
        print(f"  Score: {s.best_score:.1%} → {s.best_field}")
        if s.second_best_field:
            print(f"  2nd: {s.second_best_score:.1%} → {s.second_best_field} (gap: {s.score_gap:.1%})")
        print(f"  Samples: {samples}")
        print()


def main():
    print("="*70)
    print("THRESHOLD ANALYSIS FOR DSPY CANDIDATE FILTERING")
    print("="*70)
    
    # Initialize database
    print("\n🔧 Initializing...")
    init_database(
        host="localhost",
        port=5432,
        database="csv_mapping",
        user="postgres",
        password="postgres",
        run_migrations=False,
    )
    
    # Initialize embeddings
    print("🔧 Loading embedding model...")
    embeddings_client = get_multilingual_embeddings_client()
    
    if not embeddings_client.is_available:
        print("❌ Embedding model not available")
        return 1
    
    # Initialize schema embeddings service
    schema_service = SchemaEmbeddingsService(
        embeddings_client=embeddings_client,
        schema_registry=get_schema_registry(),
    )
    
    # Ensure schemas are indexed
    schema_service.index_all_schemas(force_reindex=False)
    
    # Test files - diverse selection
    test_dir = Path(__file__).parent / "data" / "unknown_samples"
    
    test_files = [
        # English standard
        "hospital_a_shifts.csv",
        "staff_medical_actions.csv",
        "procedures_log.csv",
        
        # Hebrew
        "hebrew_shifts_test.csv",
        "nurse_schedule_hebrew.csv",
        "clinic_schedule_hebrew.csv",
        "hebrew_ambiguous.csv",
        
        # Creative/weird names
        "creative_employee_shifts.csv",
        "custom_weird_names.csv",
        "creative_shift_names.csv",
        
        # Edge cases
        "hard_conflicts.csv",
        "conflict_test_shifts.csv",
        
        # Other types
        "lab_test_results.csv",
        "patient_vitals_fahrenheit.csv",
        "hospital_shifts_hours.csv",
    ]
    
    csv_files = []
    for f in test_files:
        path = test_dir / f
        if path.exists():
            csv_files.append(path)
        else:
            print(f"⚠️  File not found: {f}")
    
    print(f"\n📁 Found {len(csv_files)} test files")
    
    # Collect scores
    print("\n" + "="*70)
    print("COLLECTING EMBEDDING SCORES")
    print("="*70)
    
    all_scores = collect_scores(csv_files, schema_service, vertical="medical")
    
    print(f"\n✅ Collected scores for {len(all_scores)} columns")
    
    # Print distribution
    print_score_distribution(all_scores)
    
    # Compare thresholds
    recommended = print_threshold_comparison(all_scores)
    
    # Show critical zone columns
    print_columns_near_threshold(all_scores, threshold=0.82)
    
    # Final recommendation
    print("\n" + "="*70)
    print("RECOMMENDATION")
    print("="*70)
    
    # Calculate stats at current vs recommended
    current = analyze_threshold(all_scores, 0.82, 0.02)
    
    print(f"""
Current threshold: 0.82 (82%)
  → {current.columns_to_dspy} columns sent to DSPy ({current.dspy_percentage:.1f}%)
  → {current.columns_above_threshold} columns use embeddings only
  → {current.columns_ambiguous} columns ambiguous (gap < 2%)
""")
    
    # Test a few alternatives
    alternatives = [0.78, 0.80, 0.85]
    print("Alternative thresholds:")
    for alt in alternatives:
        alt_analysis = analyze_threshold(all_scores, alt, 0.02)
        print(f"  {alt:.0%}: {alt_analysis.columns_to_dspy} to DSPy ({alt_analysis.dspy_percentage:.1f}%)")
    
    # Export detailed data for manual review
    output_file = Path(__file__).parent / "threshold_analysis_results.json"
    
    export_data = {
        "total_columns": len(all_scores),
        "test_files": [str(f.name) for f in csv_files],
        "current_threshold": 0.82,
        "columns": [
            {
                "file": s.csv_file,
                "column": s.column_name,
                "type": s.column_type,
                "best_score": round(s.best_score, 4),
                "best_field": s.best_field,
                "second_score": round(s.second_best_score, 4),
                "second_field": s.second_best_field,
                "gap": round(s.score_gap, 4),
                "would_dspy_at_82": s.best_score < 0.82 or (s.score_gap < 0.02 and s.second_best_field),
                "samples": s.sample_values[:2],
            }
            for s in all_scores
        ]
    }
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n📊 Detailed results exported to: {output_file}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

