"""
Example usage of the Insights Engine.

This script demonstrates how to:
1. Load CSVs with manual or automatic classification
2. List available insights
3. Run specific insights
4. Export results to CSV

Note: Auto-classification requires the PostgreSQL database to be running.
      For testing without DB, use load_csv_manual() with explicit mappings.
"""

import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

from csv_analyzer.insights import InsightsEngine


def run_manual_mode():
    """
    Demo using manual column mappings (no database required).
    Good for testing and when you know your data structure.
    """
    print("=" * 60)
    print("INSIGHTS ENGINE - MANUAL MODE (No DB Required)")
    print("=" * 60)
    
    engine = InsightsEngine(vertical="medical")
    
    # Define sample data paths
    data_dir = Path(__file__).parent / "data" / "unknown_samples"
    
    # Load shifts with explicit mappings
    print("\n📂 Loading CSVs with manual mappings...")
    
    shifts_file = data_dir / "employee_shifts" / "hospital_a_shifts.csv"
    if shifts_file.exists():
        # hospital_a_shifts.csv columns: employee_number,work_date,time_in,time_out,unit,hours_worked
        result = engine.load_csv_manual(
            csv_file=str(shifts_file),
            document_type="employee_shifts",
            column_mappings={
                "employee_number": "employee_id",
                "work_date": "shift_date",
                "time_in": "shift_start",
                "time_out": "shift_end",
                "unit": "department_code",
                "hours_worked": "duration_minutes",  # Will be treated as hours
            }
        )
        print(f"  ✓ Loaded shifts: {result.row_count} rows")
    
    # Load compensation
    compensation_file = data_dir / "employee_compensation" / "team_beta_payroll.csv"
    if compensation_file.exists():
        # team_beta_payroll.csv columns: staff_id,full_name,emp_status,base_salary,hourly_rate,monthly_hours,loaded_cost,department,position
        result = engine.load_csv_manual(
            csv_file=str(compensation_file),
            document_type="employee_compensation",
            column_mappings={
                "staff_id": "employee_id",
                "full_name": "employee_name",
                "emp_status": "employment_type",
                "base_salary": "gross_salary",
                "hourly_rate": "hourly_rate",
                "monthly_hours": "monthly_hours",
                "loaded_cost": "effective_hourly_cost",
                "department": "department",
                "position": "position",
            }
        )
        print(f"  ✓ Loaded compensation: {result.row_count} rows")
    
    # Load procedures
    procedures_file = data_dir / "staff_clinical_procedures" / "procedures_log.csv"
    if procedures_file.exists():
        # procedures_log.csv columns: procedure_id,patient_mrn,physician_id,cpt_code,procedure_desc,service_datetime,service_dept,icd10_code
        result = engine.load_csv_manual(
            csv_file=str(procedures_file),
            document_type="staff_clinical_procedures",
            column_mappings={
                "procedure_id": "procedure_id",
                "patient_mrn": "patient_ref",
                "physician_id": "staff_id",
                "cpt_code": "billing_code",
                "procedure_desc": "procedure_description",
                "service_datetime": "performed_datetime",
                "service_dept": "location",
                "icd10_code": "diagnosis_code",
            }
        )
        print(f"  ✓ Loaded procedures: {result.row_count} rows")
    
    run_insights(engine)
    engine.close()


def run_auto_mode():
    """
    Demo using automatic classification (requires PostgreSQL database).
    """
    print("=" * 60)
    print("INSIGHTS ENGINE - AUTO CLASSIFICATION MODE")
    print("=" * 60)
    
    # Check if database is available
    try:
        from csv_analyzer.db.connection import Database
        Database.init()
        print("✓ Database connected")
    except Exception as e:
        print(f"✗ Database not available: {e}")
        print("  Run in manual mode instead.")
        return
    
    engine = InsightsEngine(vertical="medical")
    
    data_dir = Path(__file__).parent / "data" / "unknown_samples"
    
    print("\n📂 Loading CSVs with auto-classification...")
    
    for subdir, doc_type in [
        ("employee_shifts", None),
        ("employee_compensation", None),
        ("staff_clinical_procedures", None),
    ]:
        folder = data_dir / subdir
        if folder.exists():
            for csv_file in folder.glob("*.csv"):
                try:
                    result = engine.load_csv(str(csv_file))
                    print(f"  ✓ {csv_file.name} → {result.document_type} ({result.row_count} rows)")
                except Exception as e:
                    print(f"  ✗ {csv_file.name}: {e}")
                break  # Just load one per type for demo
    
    run_insights(engine)
    engine.close()


def run_insights(engine: InsightsEngine):
    """Run insights using the loaded engine."""
    
    # Show status
    print("\n📊 Data Store Status:")
    status = engine.get_status()
    for table in status.tables:
        cols = ", ".join(table.columns[:4]) + ("..." if len(table.columns) > 4 else "")
        print(f"  - {table.document_type}: {table.row_count} rows [{cols}]")
    
    # List all insights
    print("\n📋 All Insight Definitions:")
    for insight in engine.list_all_insights():
        print(f"  - {insight.name}")
        print(f"    {insight.description}")
        print(f"    Requires: {', '.join(insight.requires)}")
    
    # List insights that can run
    print("\n✅ Insights Ready to Run:")
    available = engine.list_available_insights()
    for name in available:
        print(f"  - {name}")
    
    if not available:
        print("  (None - required tables not loaded)")
        return
    
    # Run cost_per_shift
    if "cost_per_shift" in available:
        print("\n" + "─" * 40)
        print("🔍 Running 'cost_per_shift' insight...")
        result = engine.run_insight("cost_per_shift")
        
        if result.success:
            print(f"  ✓ {result.row_count} rows ({result.execution_time_ms:.1f}ms)")
            print("\n" + result.data.to_string(index=False))
        else:
            print(f"  ✗ {result.error}")
    
    # Run shift_summary
    if "shift_summary" in available:
        print("\n" + "─" * 40)
        print("🔍 Running 'shift_summary' insight...")
        result = engine.run_insight("shift_summary")
        
        if result.success:
            print(f"  ✓ {result.row_count} rows ({result.execution_time_ms:.1f}ms)")
            print("\n" + result.data.to_string(index=False))
        else:
            print(f"  ✗ {result.error}")
    
    # Run avg_cost_per_procedure
    if "avg_cost_per_procedure" in available:
        print("\n" + "─" * 40)
        print("🔍 Running 'avg_cost_per_procedure' insight...")
        result = engine.run_insight("avg_cost_per_procedure")
        
        if result.success:
            print(f"  ✓ {result.row_count} rows ({result.execution_time_ms:.1f}ms)")
            print("\n" + result.data.to_string(index=False))
        else:
            print(f"  ✗ {result.error}")
    
    # Export all
    print("\n" + "─" * 40)
    print("🚀 Running ALL applicable insights...")
    all_results = engine.run_all_insights()
    
    for name, result in all_results.items():
        icon = "✓" if result.success else "✗"
        if result.success:
            print(f"  {icon} {name}: {result.row_count} rows ({result.execution_time_ms:.1f}ms)")
        else:
            print(f"  {icon} {name}: {result.error}")
    
    # Export to CSV
    output_dir = Path(__file__).parent / "insights_output"
    print(f"\n💾 Exporting to {output_dir}...")
    exported = engine.export_all_results(output_dir)
    for name, path in exported.items():
        print(f"  → {path}")
    
    # Custom SQL
    print("\n🔧 Custom SQL Query:")
    try:
        df = engine.execute_sql("""
            SELECT 
                department_code,
                COUNT(*) as shifts,
                COUNT(DISTINCT employee_id) as employees
            FROM employee_shifts
            GROUP BY department_code
            ORDER BY shifts DESC
        """)
        print(df.to_string(index=False))
    except Exception as e:
        print(f"  Error: {e}")
    
    print("\n" + "=" * 60)
    print("DONE!")


if __name__ == "__main__":
    import sys
    
    if "--auto" in sys.argv:
        run_auto_mode()
    else:
        # Default to manual mode (no DB required)
        run_manual_mode()
