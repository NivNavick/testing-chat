#!/usr/bin/env python3
"""
DSPy Optimizer Script.

Compiles the DSPy column classifier using ground truth examples.
This optimizes the prompts for better accuracy.

Usage:
    # Optimize with all ground truth (default)
    python optimize_dspy.py

    # Optimize with specific vertical
    python optimize_dspy.py --vertical medical
    
    # Optimize with specific document type
    python optimize_dspy.py --document-type employee_shifts
    
    # Specify output directory
    python optimize_dspy.py --output ./compiled_model
    
    # Use a different teacher model
    python optimize_dspy.py --teacher-model openai/gpt-4o
"""

import argparse
import logging
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

try:
    import dspy
    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False
    logger.error("dspy-ai not installed. Run: pip install dspy-ai")


def accuracy_metric(example, prediction, trace=None) -> float:
    """
    Metric for column classification accuracy.
    
    Returns 1.0 if the prediction matches the expected target field.
    """
    expected = example.target_field
    predicted = getattr(prediction, 'target_field', None)
    
    if predicted == "none":
        predicted = None
    
    if expected == predicted:
        return 1.0
    
    # Partial credit for close matches
    if expected and predicted:
        expected_lower = expected.lower().replace("_", "")
        predicted_lower = predicted.lower().replace("_", "")
        if expected_lower == predicted_lower:
            return 0.9  # Case/underscore mismatch
    
    return 0.0


def verify_metric(example, prediction, trace=None) -> float:
    """
    Metric for verification accuracy.
    
    Returns 1.0 if is_correct matches the expected value.
    """
    expected = example.is_correct
    predicted = getattr(prediction, 'is_correct', None)
    
    return 1.0 if expected == predicted else 0.0


def optimize(
    vertical: str = None,
    document_type: str = None,
    output_dir: Path = None,
    teacher_model: str = None,
    student_model: str = None,
    max_examples: int = 50,
    num_threads: int = 4,
):
    """
    Optimize DSPy column classifier using ground truth.
    
    Args:
        vertical: Filter to specific vertical (e.g., "medical")
        document_type: Filter to specific document type
        output_dir: Directory to save compiled model
        teacher_model: Model to use for generating examples (default: gpt-4o)
        student_model: Model to optimize (default: gpt-4o-mini)
        max_examples: Maximum training examples to use
        num_threads: Number of parallel threads for optimization
    """
    if not DSPY_AVAILABLE:
        logger.error("DSPy not available")
        return 1
    
    from csv_analyzer.services.dspy_training import load_training_data
    from csv_analyzer.services.dspy_classifier import (
        DSPyColumnClassifier,
        ClassifyColumnSignature,
        VerifyMappingSignature,
    )
    
    # Set defaults
    output_dir = output_dir or Path(__file__).parent / "models" / "dspy_compiled"
    teacher_model = teacher_model or "openai/gpt-4o"
    student_model = student_model or "openai/gpt-4o-mini"
    
    logger.info("=" * 60)
    logger.info("DSPy Column Classifier Optimization")
    logger.info("=" * 60)
    logger.info(f"Teacher model: {teacher_model}")
    logger.info(f"Student model: {student_model}")
    logger.info(f"Output directory: {output_dir}")
    
    # 1. Load training data
    logger.info("\n📂 Loading training data from ground_truth/...")
    
    verticals = [vertical] if vertical else None
    doc_types = [document_type] if document_type else None
    
    training_data = load_training_data()
    
    logger.info(f"Loaded {len(training_data.column_examples)} column examples")
    logger.info(f"Loaded {len(training_data.verify_examples)} verification examples")
    
    if len(training_data.column_examples) == 0:
        logger.error("No training examples found!")
        logger.error("Make sure you have CSV files in ground_truth/medical/employee_shifts/")
        return 1
    
    # 2. Convert to DSPy examples
    logger.info("\n🔄 Converting to DSPy format...")
    dspy_examples = training_data.to_dspy_examples()
    
    classify_train = dspy_examples["classify"][:max_examples]
    verify_train = dspy_examples["verify"][:max_examples]
    
    logger.info(f"Classification examples: {len(classify_train)}")
    logger.info(f"Verification examples: {len(verify_train)}")
    
    # 3. Configure DSPy with teacher model
    logger.info(f"\n🤖 Configuring DSPy with {teacher_model}...")
    
    teacher = dspy.LM(teacher_model, temperature=0.7)
    student = dspy.LM(student_model, temperature=0.1)
    
    # Use teacher for optimization
    dspy.configure(lm=teacher)
    
    # 4. Create student predictors
    logger.info("\n📚 Creating predictors...")
    
    classify_predictor = dspy.ChainOfThought(ClassifyColumnSignature)
    verify_predictor = dspy.ChainOfThought(VerifyMappingSignature)
    
    # 5. Optimize classification predictor
    logger.info("\n🎯 Optimizing ClassifyColumn predictor...")
    logger.info(f"Using {len(classify_train)} examples")
    
    classify_optimizer = dspy.BootstrapFewShot(
        metric=accuracy_metric,
        max_bootstrapped_demos=4,
        max_labeled_demos=4,
        max_rounds=1,
        num_threads=num_threads,
    )
    
    try:
        optimized_classify = classify_optimizer.compile(
            classify_predictor,
            trainset=classify_train,
        )
        logger.info("✅ ClassifyColumn optimization complete")
    except Exception as e:
        logger.error(f"ClassifyColumn optimization failed: {e}")
        optimized_classify = classify_predictor
    
    # 6. Optimize verification predictor
    logger.info("\n🎯 Optimizing VerifyMapping predictor...")
    logger.info(f"Using {len(verify_train)} examples")
    
    verify_optimizer = dspy.BootstrapFewShot(
        metric=verify_metric,
        max_bootstrapped_demos=4,
        max_labeled_demos=4,
        max_rounds=1,
        num_threads=num_threads,
    )
    
    try:
        optimized_verify = verify_optimizer.compile(
            verify_predictor,
            trainset=verify_train,
        )
        logger.info("✅ VerifyMapping optimization complete")
    except Exception as e:
        logger.error(f"VerifyMapping optimization failed: {e}")
        optimized_verify = verify_predictor
    
    # 7. Save compiled models
    logger.info(f"\n💾 Saving compiled models to {output_dir}...")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    optimized_classify.save(output_dir / "classify_predictor.json")
    optimized_verify.save(output_dir / "verify_predictor.json")
    
    logger.info("=" * 60)
    logger.info("✅ Optimization complete!")
    logger.info("=" * 60)
    logger.info(f"\nTo use the optimized model:")
    logger.info(f"  from csv_analyzer.services.dspy_classifier import DSPyColumnClassifier")
    logger.info(f"  classifier = DSPyColumnClassifier.load('{output_dir}')")
    
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Optimize DSPy column classifier using ground truth"
    )
    
    parser.add_argument(
        "--vertical",
        type=str,
        help="Filter to specific vertical (e.g., 'medical')"
    )
    parser.add_argument(
        "--document-type",
        type=str,
        dest="document_type",
        help="Filter to specific document type"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output directory for compiled model"
    )
    parser.add_argument(
        "--teacher-model",
        type=str,
        default="openai/gpt-4o",
        help="Teacher model for generating examples (default: openai/gpt-4o)"
    )
    parser.add_argument(
        "--student-model",
        type=str,
        default="openai/gpt-4o-mini",
        help="Student model to optimize (default: openai/gpt-4o-mini)"
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=50,
        help="Maximum training examples (default: 50)"
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Number of parallel threads (default: 4)"
    )
    
    args = parser.parse_args()
    
    return optimize(
        vertical=args.vertical,
        document_type=args.document_type,
        output_dir=args.output,
        teacher_model=args.teacher_model,
        student_model=args.student_model,
        max_examples=args.max_examples,
        num_threads=args.threads,
    )


if __name__ == "__main__":
    sys.exit(main())

