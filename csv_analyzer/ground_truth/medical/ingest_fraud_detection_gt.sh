#!/bin/bash
# Ingest ground truth for the 3 fraud detection files
# Run from the csv_analyzer directory

cd "$(dirname "$0")/../.."

echo "=========================================="
echo "Ingesting Employee Shifts (353_shifts)"
echo "=========================================="
python ingest_ground_truth.py \
    --csv ground_truth/medical/employee_shifts/353_shifts_full.csv \
    --vertical medical \
    --document-type employee_shifts \
    --external-id "353_shifts_gastro_dec2025" \
    --description "Gastro department shift schedule Dec 2025 - Iad Shahin" \
    --labeler "system" \
    --notes "Hebrew format with time ranges, day abbreviations" \
    --mappings '{
        "תאריך": "shift_date",
        "סוג": "shift_type",
        "כניסה": "shift_start",
        "יציאה": "shift_end",
        "סה\"כ שעות": "duration_minutes",
        "הפסקה": "break_minutes",
        "הערה": "department_code"
    }'

echo ""
echo "=========================================="
echo "Ingesting Medical Actions"
echo "=========================================="
python ingest_ground_truth.py \
    --csv ground_truth/medical/staff_clinical_procedures/medical_actions_hebrew.csv \
    --vertical medical \
    --document-type staff_clinical_procedures \
    --external-id "medical_actions_gastro_dec2025" \
    --description "Gastro procedures and billing - Dec 2025" \
    --labeler "system" \
    --notes "Hebrew format with time ranges, includes התחשבנויות to filter" \
    --mappings '{
        "שם טיפול": "procedure_description",
        "קוד טיפול": "billing_code",
        "תאריך טיפול": "performed_datetime",
        "שעות טיפול בפועל": "procedure_time_range",
        "קטגורית טיפול": "procedure_category",
        "מחלקה": "location",
        "מחיר": "price",
        "צוות מטפל": "staff_id"
    }'

echo ""
echo "=========================================="
echo "Ingesting Monthly Salary"
echo "=========================================="
python ingest_ground_truth.py \
    --csv ground_truth/medical/employee_compensation/monthly_salary_hebrew.csv \
    --vertical medical \
    --document-type employee_compensation \
    --external-id "monthly_salary_gastro_2025" \
    --description "Monthly salary data with hourly rates - 2025" \
    --labeler "system" \
    --notes "Hebrew format with salary tag containing location" \
    --mappings '{
        "תפקיד": "position",
        "שם עובד": "employee_name",
        "תג בשכר": "salary_tag",
        "תעריף": "hourly_rate"
    }'

echo ""
echo "Done! Run --list to see all ground truth records."

