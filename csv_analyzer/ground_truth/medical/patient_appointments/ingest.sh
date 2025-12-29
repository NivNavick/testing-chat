#!/bin/bash
# Ingest ground truth files for patient_appointments
# Run from csv_analyzer directory: bash ground_truth/medical/patient_appointments/ingest.sh

set -e
cd "$(dirname "$0")/../../.."

echo "Ingesting patient_appointments ground truth..."

# standard_english.csv - Standard English format
poetry run python ingest_ground_truth.py \
  --csv ground_truth/medical/patient_appointments/standard_english.csv \
  --vertical medical \
  --document-type patient_appointments \
  --mappings '{"appt_id": "appointment_id", "patient_id": "patient_id", "patient_name": "patient_name", "appt_date": "appointment_date", "appt_time": "appointment_time", "duration": "duration_minutes", "doctor_name": "doctor_name", "department": "department", "appt_type": "appointment_type", "status": "status", "room": "room"}' \
  --external-id "gt_patient_appointments_english" \
  --description "Standard English patient appointments"

# hebrew_appointments.csv - Hebrew format
poetry run python ingest_ground_truth.py \
  --csv ground_truth/medical/patient_appointments/hebrew_appointments.csv \
  --vertical medical \
  --document-type patient_appointments \
  --mappings '{"מזהה_תור": "appointment_id", "מספר_מטופל": "patient_id", "שם_מטופל": "patient_name", "תאריך_תור": "appointment_date", "שעת_תור": "appointment_time", "משך": "duration_minutes", "שם_רופא": "doctor_name", "מחלקה": "department", "סוג_תור": "appointment_type", "סטטוס": "status", "חדר": "room"}' \
  --external-id "gt_patient_appointments_hebrew" \
  --description "Hebrew patient appointments (תורים)"

# clinic_schedule.csv - Alternative English with more fields
poetry run python ingest_ground_truth.py \
  --csv ground_truth/medical/patient_appointments/clinic_schedule.csv \
  --vertical medical \
  --document-type patient_appointments \
  --mappings '{"booking_id": "appointment_id", "patient_num": "patient_id", "full_name": "patient_name", "visit_date": "appointment_date", "scheduled_time": "appointment_time", "end_time": "end_time", "physician": "doctor_name", "clinic": "department", "visit_type": "appointment_type", "booking_status": "status", "room_number": "room", "phone": "phone"}' \
  --external-id "gt_patient_appointments_clinic" \
  --description "Clinic schedule with booking terminology"

# hebrew_clinic_visits.csv - Alternative Hebrew terminology
poetry run python ingest_ground_truth.py \
  --csv ground_truth/medical/patient_appointments/hebrew_clinic_visits.csv \
  --vertical medical \
  --document-type patient_appointments \
  --mappings '{"קוד_תור": "appointment_id", "מזהה_מטופל": "patient_id", "שם": "patient_name", "תאריך": "appointment_date", "שעה": "appointment_time", "משך_תור": "duration_minutes", "רופא": "doctor_name", "מרפאה": "department", "סוג": "appointment_type", "מצב": "status", "מיקום": "room", "טלפון": "phone"}' \
  --external-id "gt_patient_appointments_hebrew_clinic" \
  --description "Hebrew clinic visits with alternative terminology"

echo "✅ All patient_appointments ground truth files ingested!"

