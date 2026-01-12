#!/usr/bin/env python3
"""
Test script for employee_monthly_salary preprocessing and classification.

Tests:
1. Loading the schema YAML
2. Applying split_delimiter transform to תעריף column
3. Verifying rate_primary and rate_secondary extraction
"""

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import logging

from csv_analyzer.preprocessing.transformers import ValueTransformer, TransformConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_split_delimiter_transform():
    """Test the split_delimiter transform on תעריף column."""
    print("\n" + "="*70)
    print("TEST: split_delimiter transform for תעריף column")
    print("="*70)
    
    # Create test data with various rate formats
    test_data = {
        "שם עובד": ["איאד שאהין", "אדיסה ביווטה", "הודא אבו נבות", "מחאגנה מנאל", "מיטל מילוחין"],
        "תעריף": ["73/82", "47", "73/77", "", "45"],
        "תפקיד": ["אח/אחות", "צוות רפואי", "אח/אחות גסטרו", "אח/אחות גסטרו", "טכנאית עיניים"],
    }
    df = pd.DataFrame(test_data)
    
    print("\nInput DataFrame:")
    print(df.to_string(index=False))
    
    # Create transformer and config
    transformer = ValueTransformer()
    config = TransformConfig(
        source_column="תעריף",
        transform_type="split_delimiter",
        pattern=r"^(\d+)(?:/(\d+))?$",
        output_columns=[
            {"name": "rate_primary", "group": 1},
            {"name": "rate_secondary", "group": 2},
        ],
    )
    
    # Apply transform
    result = transformer.apply_all(df, [config])
    
    print(f"\nTransformations applied: {result.transformations_applied}")
    print(f"Rows affected: {result.rows_affected}")
    print(f"Columns added: {result.columns_added}")
    print(f"Errors: {result.errors}")
    
    print("\nOutput DataFrame:")
    print(result.df.to_string(index=False))
    
    # Verify results
    assert "rate_primary" in result.df.columns, "rate_primary column not created"
    assert "rate_secondary" in result.df.columns, "rate_secondary column not created"
    
    # Check specific values
    row_dual = result.df[result.df["שם עובד"] == "איאד שאהין"].iloc[0]
    assert row_dual["rate_primary"] == 73, f"Expected 73, got {row_dual['rate_primary']}"
    assert row_dual["rate_secondary"] == 82, f"Expected 82, got {row_dual['rate_secondary']}"
    
    row_single = result.df[result.df["שם עובד"] == "אדיסה ביווטה"].iloc[0]
    assert row_single["rate_primary"] == 47, f"Expected 47, got {row_single['rate_primary']}"
    assert pd.isna(row_single["rate_secondary"]) or row_single["rate_secondary"] is None, \
        f"Expected None/NaN, got {row_single['rate_secondary']}"
    
    print("\n✅ split_delimiter transform working correctly!")
    return True


def test_clean_number_transform():
    """Test the clean_number transform on salary columns."""
    print("\n" + "="*70)
    print("TEST: clean_number transform for comma-formatted numbers")
    print("="*70)
    
    # Create test data with comma-formatted numbers
    test_data = {
        "שם עובד": ["ליבנת שלום", "מירב וולובצ'יק", "מיטל מילוחין"],
        "8.25": ["9,135", "7,808", "1,026"],
        "9.25": ["9,781", "6,796", "567"],
    }
    df = pd.DataFrame(test_data)
    
    print("\nInput DataFrame:")
    print(df.to_string(index=False))
    
    # Create transformer and config
    transformer = ValueTransformer()
    configs = [
        TransformConfig(
            source_column="8.25",
            transform_type="clean_number",
            pattern=r",",
            replacement="",
        ),
        TransformConfig(
            source_column="9.25",
            transform_type="clean_number",
            pattern=r",",
            replacement="",
        ),
    ]
    
    # Apply transforms
    result = transformer.apply_all(df, configs)
    
    print(f"\nTransformations applied: {result.transformations_applied}")
    print(f"Rows affected: {result.rows_affected}")
    print(f"Columns modified: {result.columns_modified}")
    print(f"Errors: {result.errors}")
    
    print("\nOutput DataFrame:")
    print(result.df.to_string(index=False))
    
    # Verify results
    assert result.df["8.25"].iloc[0] == 9135, f"Expected 9135, got {result.df['8.25'].iloc[0]}"
    assert result.df["9.25"].iloc[1] == 6796, f"Expected 6796, got {result.df['9.25'].iloc[1]}"
    
    print("\n✅ clean_number transform working correctly!")
    return True


def test_real_file():
    """Test preprocessing on the actual salary file."""
    print("\n" + "="*70)
    print("TEST: Real salary file preprocessing")
    print("="*70)
    
    salary_file = Path("/Users/nivnavick/Downloads/monthly_salary - Sheet1 (1).csv")
    if not salary_file.exists():
        print(f"⚠️  Salary file not found: {salary_file}")
        return False
    
    # Load the file
    df = pd.read_csv(salary_file)
    print(f"\nLoaded file: {len(df)} rows, {len(df.columns)} columns")
    print(f"Columns: {list(df.columns)}")
    
    # Apply split_delimiter transform
    transformer = ValueTransformer()
    config = TransformConfig(
        source_column="תעריף",
        transform_type="split_delimiter",
        pattern=r"^(\d+)(?:/(\d+))?$",
        output_columns=[
            {"name": "rate_primary", "group": 1},
            {"name": "rate_secondary", "group": 2},
        ],
    )
    
    result = transformer.apply_all(df, [config])
    
    print(f"\nAfter split_delimiter transform:")
    print(f"  Rows affected: {result.rows_affected}")
    print(f"  Columns added: {result.columns_added}")
    
    # Show rows with dual rates
    dual_rate_rows = result.df[result.df["rate_secondary"].notna()]
    print(f"\nEmployees with dual rates ({len(dual_rate_rows)}):")
    print(dual_rate_rows[["שם עובד", "תפקיד", "תעריף", "rate_primary", "rate_secondary"]].to_string(index=False))
    
    print("\n✅ Real file preprocessing working!")
    return True


def main():
    """Run all tests."""
    print("="*70)
    print("EMPLOYEE MONTHLY SALARY PREPROCESSING TESTS")
    print("="*70)
    
    tests = [
        test_split_delimiter_transform,
        test_clean_number_transform,
        test_real_file,
    ]
    
    passed = 0
    failed = 0
    
    for test_fn in tests:
        try:
            if test_fn():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"\n❌ Test failed with exception: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "="*70)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("="*70)
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

