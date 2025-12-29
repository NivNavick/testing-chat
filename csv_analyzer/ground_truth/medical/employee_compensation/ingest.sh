#!/bin/bash
# Ingest ground truth files for employee_compensation
# Run from csv_analyzer directory: bash ground_truth/medical/employee_compensation/ingest.sh

set -e
cd "$(dirname "$0")/../../.."

echo "Ingesting employee_compensation ground truth..."

# standard_english.csv - Standard English format
poetry run python ingest_ground_truth.py \
  --csv ground_truth/medical/employee_compensation/standard_english.csv \
  --vertical medical \
  --document-type employee_compensation \
  --mappings '{"emp_id": "employee_id", "name": "employee_name", "employment_type": "employment_type", "gross_salary": "gross_salary", "hourly_rate": "hourly_rate", "monthly_hours": "monthly_hours", "effective_hourly_cost": "effective_hourly_cost", "department": "department", "position": "position", "start_date": "start_date", "currency": "currency"}' \
  --external-id "gt_employee_compensation_english" \
  --description "Standard English employee compensation data with salaried and contractors"

# hebrew_compensation.csv - Hebrew format
poetry run python ingest_ground_truth.py \
  --csv ground_truth/medical/employee_compensation/hebrew_compensation.csv \
  --vertical medical \
  --document-type employee_compensation \
  --mappings '{"מספר_עובד": "employee_id", "שם_עובד": "employee_name", "סוג_העסקה": "employment_type", "שכר_ברוטו": "gross_salary", "תעריף_שעתי": "hourly_rate", "שעות_חודשיות": "monthly_hours", "עלות_שעה_אפקטיבית": "effective_hourly_cost", "מחלקה": "department", "תפקיד": "position", "תאריך_התחלה": "start_date", "מטבע": "currency"}' \
  --external-id "gt_employee_compensation_hebrew" \
  --description "Hebrew employee compensation data (שכיר/קבלן)"

# detailed_rates.csv - Alternative English column names
poetry run python ingest_ground_truth.py \
  --csv ground_truth/medical/employee_compensation/detailed_rates.csv \
  --vertical medical \
  --document-type employee_compensation \
  --mappings '{"employee_number": "employee_id", "full_name": "employee_name", "worker_type": "employment_type", "base_salary": "gross_salary", "cost_per_hour": "hourly_rate", "work_hours": "monthly_hours", "total_hourly_cost": "effective_hourly_cost", "dept": "department", "job_title": "position", "hire_date": "start_date", "curr": "currency"}' \
  --external-id "gt_employee_compensation_detailed" \
  --description "English compensation with alternative column names"

# hebrew_detailed.csv - Alternative Hebrew column names  
poetry run python ingest_ground_truth.py \
  --csv ground_truth/medical/employee_compensation/hebrew_detailed.csv \
  --vertical medical \
  --document-type employee_compensation \
  --mappings '{"קוד_עובד": "employee_id", "שם": "employee_name", "סוג": "employment_type", "משכורת": "gross_salary", "עלות_לשעה": "hourly_rate", "שעות": "monthly_hours", "עלות_שעה_כוללת": "effective_hourly_cost", "יחידה": "department", "משרה": "position", "תחילת_עבודה": "start_date", "מטבע": "currency"}' \
  --external-id "gt_employee_compensation_hebrew_detailed" \
  --description "Hebrew compensation with alternative terminology"

echo "✅ All employee_compensation ground truth files ingested!"

