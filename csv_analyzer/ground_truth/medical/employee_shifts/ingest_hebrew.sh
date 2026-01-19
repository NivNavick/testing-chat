#!/bin/bash
# Ingest Hebrew ground truth files for employee_shifts
# Run from csv_analyzer directory: bash ground_truth/medical/employee_shifts/ingest_hebrew.sh

set -e
cd "$(dirname "$0")/../../.."

echo "Ingesting Hebrew employee_shifts ground truth..."

# hebrew_schedule_v1.csv - Uses קוד_עובד, תאריך_עבודה, שעת_כניסה/יציאה, סוג_משמרת
poetry run python ingest_ground_truth.py \
  --csv ground_truth/medical/employee_shifts/hebrew_schedule_v1.csv \
  --vertical medical \
  --document-type employee_shifts \
  --mappings '{"קוד_עובד": "employee_id", "תאריך_עבודה": "shift_date", "שעת_כניסה": "shift_start", "שעת_יציאה": "shift_end", "שם_מחלקה": "department_code", "סוג_משמרת": "shift_type"}' \
  --external-id "gt_employee_shifts_hebrew_v1" \
  --description "Hebrew schedule with entry/exit times and shift types"

# hebrew_schedule_v2.csv - Uses זיהוי_עובד, יום_עבודה, התחלה/סיום, תורנות
poetry run python ingest_ground_truth.py \
  --csv ground_truth/medical/employee_shifts/hebrew_schedule_v2.csv \
  --vertical medical \
  --document-type employee_shifts \
  --mappings '{"זיהוי_עובד": "employee_id", "יום_עבודה": "shift_date", "התחלה": "shift_start", "סיום": "shift_end", "יחידה": "department_code", "תורנות": "shift_type"}' \
  --external-id "gt_employee_shifts_hebrew_v2" \
  --description "Hebrew schedule with alternative terminology"

# hebrew_schedule_v3.csv - Uses ת_עובד, זמן_התחלה/סיום, אגף, סוג_תורנות
poetry run python ingest_ground_truth.py \
  --csv ground_truth/medical/employee_shifts/hebrew_schedule_v3.csv \
  --vertical medical \
  --document-type employee_shifts \
  --mappings '{"ת_עובד": "employee_id", "תאריך": "shift_date", "זמן_התחלה": "shift_start", "זמן_סיום": "shift_end", "אגף": "department_code", "סוג_תורנות": "shift_type", "שם_עובד": "employee_name"}' \
  --external-id "gt_employee_shifts_hebrew_v3" \
  --description "Hebrew schedule with employee names and wing terminology"

# hebrew_roster.csv - Uses מס_עובד, תאריך_משמרת, כניסה/יציאה, סיווג_משמרת
poetry run python ingest_ground_truth.py \
  --csv ground_truth/medical/employee_shifts/hebrew_roster.csv \
  --vertical medical \
  --document-type employee_shifts \
  --mappings '{"מס_עובד": "employee_id", "תאריך_משמרת": "shift_date", "כניסה": "shift_start", "יציאה": "shift_end", "מחלקת_עבודה": "department_code", "סיווג_משמרת": "shift_type"}' \
  --external-id "gt_employee_shifts_hebrew_roster" \
  --description "Hebrew roster with abbreviated column names"

# hebrew_detailed.csv - Full details with actual hours and supervisor flag
poetry run python ingest_ground_truth.py \
  --csv ground_truth/medical/employee_shifts/hebrew_detailed.csv \
  --vertical medical \
  --document-type employee_shifts \
  --mappings '{"קוד": "employee_id", "תאריך": "shift_date", "שעת_הגעה": "shift_start", "שעת_עזיבה": "shift_end", "שעות_מתוכננות": "duration_minutes", "שעות_בפועל": "actual_hours", "מחלקה": "department_code", "סוג": "shift_type", "האם_אחמש": "is_supervisor"}' \
  --external-id "gt_employee_shifts_hebrew_detailed" \
  --description "Hebrew detailed schedule with actual hours and supervisor flag"

# bat_yam_roster.csv - Staffing roster format with roles (no specific times)
# Uses shift codes (בוקר/ערב) instead of specific start/end times
poetry run python ingest_ground_truth.py \
  --csv ground_truth/medical/employee_shifts/bat_yam_roster.csv \
  --vertical medical \
  --document-type employee_shifts \
  --mappings '{"חטיבה": "department_code", "עובד": "employee_name", "ק.משמרת": "shift_type", "שם משמרת": "role", "יום בחודש": "shift_date"}' \
  --external-id "gt_employee_shifts_bat_yam_roster" \
  --description "Bat Yam staffing roster December 2025 - roster format with roles and shift codes"

echo "✅ All Hebrew ground truth files ingested!"

