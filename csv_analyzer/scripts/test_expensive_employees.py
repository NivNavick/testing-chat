#!/usr/bin/env python3
"""
Test script for expensive_employees block.

Tests the block directly without running the full workflow.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import logging
import tempfile
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_expensive_employees_block():
    """Test the expensive employees block with sample data."""
    print("\n" + "="*70)
    print("TEST: ExpensiveEmployeesBlock")
    print("="*70)
    
    from csv_analyzer.workflows.base_block import BlockContext
    from csv_analyzer.blocks.insights.expensive_employees import ExpensiveEmployeesBlock
    
    # Create test salary data
    test_data = {
        "שם עובד": ["מרים שלף", "אנה פדרנקו", "נאשף ראבב", "איאד שאהין", "סופיה משה"],
        "תפקיד": ["אח/אחות גסטרו", "אח/אחות גסטרו", "אח/אחות גסטרו", "אח/אחות", "מזכירה רפואית"],
        "עיר": ["בת ים", "בת ים", "בת ים", "בת ים", "בת ים"],
        "תעריף": ["100", "85", "110", "73/82", "50"],
        "rate_primary": [100, 85, 110, 73, 50],
        "rate_secondary": [None, None, None, 82, None],
        "1.25": [17782, 11683, 12583, 7565, 7412],
        "2.25": [10460, 9399, 10946, 6715, 7435],
        "3.25": [15126, 13060, 13228, 6937, 7593],
        "4.25": [13040, 9973, 11391, 6367, 8719],
        "5.25": [12481, 14009, 2274, 9631, 6993],
    }
    df = pd.DataFrame(test_data)
    
    print("\nTest Data:")
    print(df.to_string(index=False))
    
    # Create temp directory for storage
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Save classified data
        classified_path = Path(tmp_dir) / "classified_data"
        classified_path.mkdir(parents=True)
        
        # Save salary data as JSON
        salary_json = df.to_dict(orient="records")
        salary_path = classified_path / "employee_monthly_salary.json"
        salary_path.write_text(json.dumps(salary_json, ensure_ascii=False))
        
        # Create manifest
        manifest = {"employee_monthly_salary": str(salary_path)}
        manifest_path = Path(tmp_dir) / "manifest.json"
        manifest_path.write_text(json.dumps(manifest))
        
        # Create block context
        ctx = BlockContext(
            inputs={"data": str(manifest_path)},
            params={
                "top_n": 10,
                "include_all": True,
                "group_by_position": True,
            },
            workflow_run_id="test_run",
            block_id="analyze_costs",
            local_storage_path=tmp_dir,
        )
        
        # Run block
        block = ExpensiveEmployeesBlock(ctx)
        result = block.run()
        
        print(f"\nBlock returned: {result}")
        
        # Load result
        result_path = result.get("result")
        if result_path:
            result_data = json.loads(Path(result_path).read_text())
            result_df = pd.DataFrame(result_data)
            
            print("\n" + "="*70)
            print("RESULTS: Employee Cost Rankings")
            print("="*70)
            print(result_df.to_string(index=False))
            
            # Verify results
            assert len(result_df) == 5, f"Expected 5 rows, got {len(result_df)}"
            assert "cost_rank" in result_df.columns
            assert "total_salary" in result_df.columns
            
            # Top employee should be מרים שלף (highest total)
            top_emp = result_df.iloc[0]["employee_name"]
            print(f"\nTop employee: {top_emp}")
            print(f"Total cost: {result_df.iloc[0]['total_salary']:,.0f}")
            
            print("\n✅ ExpensiveEmployeesBlock test passed!")
            return True
    
    return False


def test_with_real_file():
    """Test with the actual salary file."""
    print("\n" + "="*70)
    print("TEST: Real Salary File")
    print("="*70)
    
    salary_file = Path("/Users/nivnavick/Downloads/monthly_salary - Sheet1 (1).csv")
    if not salary_file.exists():
        print(f"⚠️  Salary file not found: {salary_file}")
        return False
    
    from csv_analyzer.workflows.base_block import BlockContext
    from csv_analyzer.blocks.insights.expensive_employees import ExpensiveEmployeesBlock
    from csv_analyzer.preprocessing.transformers import ValueTransformer, TransformConfig
    
    # Load and preprocess
    df = pd.read_csv(salary_file)
    print(f"\nLoaded file: {len(df)} rows, {len(df.columns)} columns")
    
    # Apply split_delimiter transform for תעריף
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
    df = result.df
    
    # Create temp directory for storage
    with tempfile.TemporaryDirectory() as tmp_dir:
        import json
        
        # Save classified data
        classified_path = Path(tmp_dir) / "classified_data"
        classified_path.mkdir(parents=True)
        
        # Save salary data as JSON
        salary_json = df.to_dict(orient="records")
        salary_path = classified_path / "employee_monthly_salary.json"
        salary_path.write_text(json.dumps(salary_json, ensure_ascii=False, default=str))
        
        # Create manifest
        manifest = {"employee_monthly_salary": str(salary_path)}
        manifest_path = Path(tmp_dir) / "manifest.json"
        manifest_path.write_text(json.dumps(manifest))
        
        # Create block context
        ctx = BlockContext(
            inputs={"data": str(manifest_path)},
            params={
                "top_n": 10,
                "include_all": True,
                "group_by_position": True,
            },
            workflow_run_id="test_run",
            block_id="analyze_costs",
            local_storage_path=tmp_dir,
        )
        
        # Run block
        block = ExpensiveEmployeesBlock(ctx)
        result = block.run()
        
        # Load result
        result_path = result.get("result")
        if result_path:
            result_data = json.loads(Path(result_path).read_text())
            result_df = pd.DataFrame(result_data)
            
            print("\n" + "="*70)
            print("TOP 10 MOST EXPENSIVE EMPLOYEES")
            print("="*70)
            
            top_10 = result_df.head(10)
            for _, row in top_10.iterrows():
                dual_rate = " (dual rate)" if row.get("has_dual_rate") else ""
                print(f"#{int(row['cost_rank']):2d}: {row['employee_name']:20s} | "
                      f"{row['position']:20s} | "
                      f"Total: {row['total_salary']:>10,.0f} | "
                      f"Avg: {row['avg_monthly_salary']:>8,.0f}{dual_rate}")
            
            print("\n" + "-"*70)
            print(f"Total employees analyzed: {len(result_df)}")
            print(f"Grand total salary cost: {result_df['total_salary'].sum():,.0f}")
            print(f"Average employee cost: {result_df['total_salary'].mean():,.0f}")
            
            print("\n✅ Real file test passed!")
            return True
    
    return False


def main():
    """Run all tests."""
    print("="*70)
    print("EXPENSIVE EMPLOYEES BLOCK TESTS")
    print("="*70)
    
    tests = [
        test_expensive_employees_block,
        test_with_real_file,
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

