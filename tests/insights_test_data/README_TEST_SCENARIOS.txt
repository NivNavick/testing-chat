TEST DATA SCENARIOS FOR INSIGHTS ENGINE
========================================

Files:
- employee_compensation.csv (10 employees)
- employee_shifts.csv (25 shifts over 3 days)
- staff_clinical_procedures.csv (50 procedures)

All files use matching employee_id/staff_id format: EMP001-EMP010

=== LATENESS SCENARIOS (employee_shifts.csv) ===

ON TIME (within acceptable range):
- EMP001 on 2024-06-01: arrived 07:55 for 08:00 shift (5 min early) ✓
- EMP002 on 2024-06-01: arrived 08:00 for 08:00 shift (exactly on time) ✓
- EMP003 on 2024-06-01: arrived 06:55 for 07:00 shift (5 min early) ✓
- EMP004 on 2024-06-01: arrived 15:00 for 15:00 shift (on time) ✓
- EMP005 on 2024-06-01: arrived 08:55 for 09:00 shift (5 min early) ✓

LATE ARRIVALS (should be flagged):
- EMP001 on 2024-06-02: arrived 08:25 for 08:00 shift (25 min LATE) ⚠️
- EMP002 on 2024-06-02: arrived 08:45 for 08:00 shift (45 min LATE) ⚠️
- EMP003 on 2024-06-02: arrived 07:35 for 07:00 shift (35 min LATE) ⚠️
- EMP004 on 2024-06-02: arrived 15:50 for 15:00 shift (50 min LATE) ⚠️
- EMP005 on 2024-06-03: arrived 10:15 for 09:00 shift (75 min LATE) ⚠️
- EMP007 on 2024-06-02: arrived 23:40 for 23:00 shift (40 min LATE) ⚠️
- EMP009 on 2024-06-02: arrived 12:30 for 12:00 shift (30 min LATE) ⚠️

TOO EARLY (more than 30 min before shift):
- EMP001 on 2024-06-03: arrived 07:30 for 08:00 shift (30 min early - borderline)
- EMP003 on 2024-06-03: arrived 14:30 for 15:00 shift (30 min early - borderline)
- EMP004 on 2024-06-03: arrived 06:00 for 07:00 shift (60 min early) ⚠️

EARLY DEPARTURES:
- EMP002 on 2024-06-02: left 15:30 for 16:00 shift end (30 min early) ⚠️
- EMP006 on 2024-06-02: left 15:30 for 16:00 shift end (30 min early) ⚠️
- EMP010 on 2024-06-02: left 14:30 for 15:00 shift end (30 min early) ⚠️


=== ATTENDANCE DISCREPANCY SCENARIOS ===

NO ACTIVITY RECORDED (shift exists but no procedures):
- EMP007 (Nurse Yael Mor, ICU): Has night shifts but NO procedures in the data
  Should be flagged as "NO_ACTIVITY_RECORDED"

ACTIVITY TIMING GAPS:
- EMP001 on 2024-06-01: Shift 08:00-16:00, first procedure at 09:30 (90 min gap)
- EMP002 on 2024-06-01: Shift 08:00-16:00, first procedure at 09:00 (60 min gap)
- EMP008 on 2024-06-01: Shift 07:00-19:00, last procedure at 15:00 (4 hour gap)


=== COST ANALYSIS SCENARIOS ===

SALARIED EMPLOYEES (effective_hourly_cost = gross_salary * 1.25 / 186):
- EMP001: 45000 * 1.25 / 186 = 302.42/hour
- EMP002: 52000 * 1.25 / 186 = 349.46/hour
- EMP008: 55000 * 1.25 / 186 = 369.62/hour (highest)

CONTRACTORS (hourly_rate = effective_hourly_cost):
- EMP004: 95.00/hour (lowest cost)
- EMP006: 75.00/hour
- EMP009: 110.00/hour

Expected daily costs (8-hour shifts):
- Cardiology: EMP001 + EMP002 = (302.42 + 349.46) * 8 = 5,215.04
- Surgery (extended): EMP008 = 369.62 * 12 = 4,435.44


=== REVENUE/PROFIT SCENARIOS ===

HIGH REVENUE PROCEDURES:
- EMP008 (Surgery): 4500 + 4500 + 3800 + 3200 + 1200 = 17,200 over 2 days
- EMP005 (Orthopedics): 2500 + 350 + 350 + 2500 = 5,700 over 2 days

LOW REVENUE PROCEDURES:
- EMP003 (Emergency): mostly 85-175 office visits
- EMP010 (Surgery RN): consults at 85-125 each

PROFIT MARGIN EXAMPLES:
- EMP008 day 1: Revenue 8,900 - Cost 4,435 = Profit 4,465 (50% margin)
- EMP004 day 1: Revenue 235 - Cost 760 = Loss -525 (contractor, low-value procedures)


=== DEPARTMENTS ===

- Cardiology: EMP001, EMP002 (physicians, high cost, high revenue)
- Emergency: EMP003, EMP004, EMP009 (mixed staff, volume-based)
- Orthopedics: EMP005 (surgeon, high-value procedures)
- Radiology: EMP006 (tech, moderate volume)
- ICU: EMP007 (no procedures - should flag)
- Surgery: EMP008, EMP010 (surgeon + RN support)

