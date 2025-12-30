"""
Test script for the Insights Engine.

Run with: PYTHONPATH=. python tests/test_insights_engine.py
"""

from pathlib import Path

from csv_analyzer.insights import InsightsEngine


def main():
    print("=" * 70)
    print("INSIGHTS ENGINE TEST")
    print("=" * 70)
    
    # Path to test data
    test_data_dir = Path(__file__).parent / "insights_test_data"
    
    # Initialize engine
    engine = InsightsEngine(vertical="medical")
    
    # Load test CSVs with manual mappings (no classification needed)
    print("\n📂 Loading test data...")
    
    # Load compensation
    engine.load_csv_manual(
        csv_file=str(test_data_dir / "employee_compensation.csv"),
        document_type="employee_compensation",
        column_mappings={}  # Column names already match schema
    )
    print("  ✓ Loaded employee_compensation.csv")
    
    # Load shifts
    engine.load_csv_manual(
        csv_file=str(test_data_dir / "employee_shifts.csv"),
        document_type="employee_shifts",
        column_mappings={}
    )
    print("  ✓ Loaded employee_shifts.csv")
    
    # Load procedures
    engine.load_csv_manual(
        csv_file=str(test_data_dir / "staff_clinical_procedures.csv"),
        document_type="staff_clinical_procedures",
        column_mappings={}
    )
    print("  ✓ Loaded staff_clinical_procedures.csv")
    
    # Status
    status = engine.get_status()
    print(f"\n📊 Loaded {len(status.tables)} tables, {status.total_rows} total rows")
    
    # Run all insights
    print("\n" + "=" * 70)
    print("RUNNING ALL INSIGHTS")
    print("=" * 70)
    
    results = engine.run_all_insights()
    
    for name, result in results.items():
        print(f"\n{'─' * 70}")
        print(f"📋 {name.upper()}")
        print(f"{'─' * 70}")
        
        if result.success:
            print(f"✓ {result.row_count} rows ({result.execution_time_ms:.1f}ms)")
            if result.row_count > 0:
                print()
                # Print first 10 rows
                print(result.data.head(10).to_string(index=False))
                if result.row_count > 10:
                    print(f"... and {result.row_count - 10} more rows")
        else:
            print(f"✗ Error: {result.error}")
    
    # Export all results
    output_dir = Path(__file__).parent / "insights_test_output"
    print(f"\n💾 Exporting to {output_dir}...")
    exported = engine.export_all_results(output_dir)
    for name, path in exported.items():
        print(f"  → {path}")
    
    engine.close()
    
    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()

