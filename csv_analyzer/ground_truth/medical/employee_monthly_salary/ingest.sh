#!/bin/bash
# Ingest ground truth files for employee_monthly_salary
# Run from csv_analyzer directory: bash ground_truth/medical/employee_monthly_salary/ingest.sh

set -e
cd "$(dirname "$0")/../../.."

echo "Ingesting employee_monthly_salary ground truth..."

# hebrew_monthly_salary.csv - Main Hebrew salary file with role types and monthly values
poetry run python ingest_ground_truth.py \
  --csv ground_truth/medical/employee_monthly_salary/hebrew_monthly_salary.csv \
  --vertical medical \
  --document-type employee_monthly_salary \
  --mappings '{"תפקיד": "position", "שם עובד": "employee_name", "תג בשכר": "payroll_tag", "עיר": "city", "תעריף": "rate_primary"}' \
  --external-id "gt_employee_monthly_salary_hebrew" \
  --description "Hebrew monthly salary with role types, dual rates, and monthly compensation values"

# dual_rate_salary.csv - Focus on employees with dual rate values (e.g., 73/82)
poetry run python ingest_ground_truth.py \
  --csv ground_truth/medical/employee_monthly_salary/dual_rate_salary.csv \
  --vertical medical \
  --document-type employee_monthly_salary \
  --mappings '{"תפקיד": "position", "שם עובד": "employee_name", "תג בשכר": "payroll_tag", "עיר": "city", "תעריף": "rate_primary"}' \
  --external-id "gt_employee_monthly_salary_dual_rate" \
  --description "Salary file specifically showcasing dual rate format (base/overtime)"

# english_monthly_salary.csv - English version for multi-language support
poetry run python ingest_ground_truth.py \
  --csv ground_truth/medical/employee_monthly_salary/english_monthly_salary.csv \
  --vertical medical \
  --document-type employee_monthly_salary \
  --mappings '{"position": "position", "employee_name": "employee_name", "payroll_tag": "payroll_tag", "city": "city", "rate": "rate_primary"}' \
  --external-id "gt_employee_monthly_salary_english" \
  --description "English version of monthly salary data"

echo "✅ All employee_monthly_salary ground truth files ingested!"

