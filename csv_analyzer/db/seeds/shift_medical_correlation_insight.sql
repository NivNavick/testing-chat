-- Seed Data: Shift-Medical Correlation Insight (Fraud Detection)
-- 
-- Uses CANONICAL columns from schema transformations:
-- - employee_shifts: shift_start, shift_end, day_of_month, city_code
-- - staff_clinical_procedures: procedure_start_time, procedure_end_time, city_code
-- - employee_compensation: city_code, role_category
--
-- This insight is designed to work with CANONICALIZED data where:
-- - Time ranges have been split into start/end columns
-- - City codes have been extracted and normalized (bat_yam, hadera, etc.)
-- - Billing entries have been filtered out

-- ============================================================================
-- 1. Insert the Insight Definition
-- ============================================================================
INSERT INTO insight_definitions (name, display_name, description, goal, required_tables, optional_tables, category)
VALUES (
    'shift_medical_correlation',
    'התאמת משמרות לפעולות רפואיות',
    'Correlates employee working hours with medical actions to detect fraud patterns. Uses canonicalized data for reliable SQL generation.',
    'Correlate employee working hours with medical actions to detect potential fraud patterns:

DETECTION GOALS:
1. Medical actions recorded >30 minutes BEFORE shift start (early_arrival)
2. Medical actions recorded AFTER shift end + tolerance (late_departure)  
3. Shifts with ZERO medical activity at the employee''s assigned location (no_activity_at_branch)
4. Shifts with fewer procedures than expected (low_activity)
5. Medical actions at DIFFERENT location than employee''s shift location (location_mismatch)

CANONICAL COLUMNS AVAILABLE:
The data has been pre-processed with the following transformations:
- employee_shifts has: shift_start (TIME), shift_end (TIME), day_of_month (INTEGER), city_code (STRING like "bat_yam")
- staff_clinical_procedures has: procedure_start_time (TIME), procedure_end_time (TIME), city_code (STRING)
- employee_compensation has: city_code (STRING), role_category (STRING like "gastro_nurse")

LOCATION CORRELATION:
- Use city_code column directly for exact location matching
- Normalized city codes: bat_yam, hadera, jerusalem, tel_aviv, haifa, beer_sheva

TIME CORRELATION:
- Compare procedure_start_time with shift_start/shift_end
- Use TIME arithmetic for DuckDB: shift_start - INTERVAL ''30 minutes''

DATE CORRELATION:
- Use day_of_month from shifts to match with DAY(performed_datetime) from procedures
- This handles the Hebrew date format "ה - 04" which has been parsed to just the day number',
    ARRAY['employee_shifts', 'staff_clinical_procedures'],
    ARRAY['employee_compensation'],
    'fraud_detection'
)
ON CONFLICT (name) DO UPDATE SET
    display_name = EXCLUDED.display_name,
    description = EXCLUDED.description,
    goal = EXCLUDED.goal,
    required_tables = EXCLUDED.required_tables,
    optional_tables = EXCLUDED.optional_tables,
    category = EXCLUDED.category,
    updated_at = NOW();

-- ============================================================================
-- 2. Insert Configurable Rules
-- ============================================================================

-- Rule: Early Arrival Tolerance
INSERT INTO insight_rules (insight_id, rule_name, value_type, current_value, default_value, min_value, max_value, description, ai_hint, display_order)
SELECT 
    id,
    'early_arrival_tolerance_minutes',
    'integer',
    '30',
    '30',
    '0',
    '120',
    'Maximum minutes BEFORE shift start that a medical action can occur without being flagged. If a procedure is recorded more than this many minutes before the shift starts, flag as CRITICAL early arrival.',
    'Use TIME comparison: procedure_start_time < shift_start - INTERVAL ''X minutes'' means early arrival.',
    1
FROM insight_definitions WHERE name = 'shift_medical_correlation'
ON CONFLICT (insight_id, rule_name) DO UPDATE SET
    current_value = EXCLUDED.current_value,
    description = EXCLUDED.description,
    ai_hint = EXCLUDED.ai_hint;

-- Rule: Late Departure Tolerance
INSERT INTO insight_rules (insight_id, rule_name, value_type, current_value, default_value, min_value, max_value, description, ai_hint, display_order)
SELECT 
    id,
    'late_departure_tolerance_minutes',
    'integer',
    '15',
    '15',
    '0',
    '60',
    'Maximum minutes AFTER shift end that a medical action can occur without being flagged. Procedures shortly after clock-out are normal (finishing up).',
    'Use TIME comparison: procedure_end_time > shift_end + INTERVAL ''X minutes'' for late flags.',
    2
FROM insight_definitions WHERE name = 'shift_medical_correlation'
ON CONFLICT (insight_id, rule_name) DO UPDATE SET
    current_value = EXCLUDED.current_value,
    description = EXCLUDED.description,
    ai_hint = EXCLUDED.ai_hint;

-- Rule: Minimum Expected Procedures
INSERT INTO insight_rules (insight_id, rule_name, value_type, current_value, default_value, min_value, max_value, description, ai_hint, display_order)
SELECT 
    id,
    'min_expected_procedures',
    'integer',
    '1',
    '1',
    '0',
    '50',
    'Minimum number of procedures expected per shift. Shifts with 0 procedures at the branch are flagged as WARNING (no_activity_at_branch). Shifts with fewer than this threshold are flagged as INFO (low_activity).',
    'COUNT procedures per shift: 0 = WARNING no_activity, < threshold = INFO low_activity.',
    3
FROM insight_definitions WHERE name = 'shift_medical_correlation'
ON CONFLICT (insight_id, rule_name) DO UPDATE SET
    current_value = EXCLUDED.current_value,
    description = EXCLUDED.description,
    ai_hint = EXCLUDED.ai_hint;

-- Rule: Minimum Shift Duration
INSERT INTO insight_rules (insight_id, rule_name, value_type, current_value, default_value, min_value, max_value, description, ai_hint, display_order)
SELECT 
    id,
    'min_shift_duration_hours',
    'float',
    '2.0',
    '2.0',
    '0.5',
    '12.0',
    'Only analyze shifts longer than this duration (hours). Short shifts may not generate procedure records and would create false positives.',
    'Filter: DATEDIFF(''minute'', shift_start, shift_end) / 60.0 >= this value.',
    4
FROM insight_definitions WHERE name = 'shift_medical_correlation'
ON CONFLICT (insight_id, rule_name) DO UPDATE SET
    current_value = EXCLUDED.current_value,
    description = EXCLUDED.description,
    ai_hint = EXCLUDED.ai_hint;

-- Rule: Location Match Required
INSERT INTO insight_rules (insight_id, rule_name, value_type, current_value, default_value, description, ai_hint, display_order)
SELECT 
    id,
    'location_match_required',
    'boolean',
    'true',
    'true',
    'If true, only match procedures where city_code matches between shifts and procedures tables. If false, consider all procedures regardless of location.',
    'JOIN condition: AND shifts.city_code = procedures.city_code when this is true.',
    5
FROM insight_definitions WHERE name = 'shift_medical_correlation'
ON CONFLICT (insight_id, rule_name) DO UPDATE SET
    current_value = EXCLUDED.current_value,
    description = EXCLUDED.description,
    ai_hint = EXCLUDED.ai_hint;

-- ============================================================================
-- 3. Insert Severity Mappings
-- ============================================================================

-- Severity: Early Arrival (CRITICAL)
INSERT INTO insight_severities (insight_id, condition_name, severity, condition_description, display_order)
SELECT 
    id,
    'early_arrival',
    'CRITICAL',
    'Medical action recorded more than early_arrival_tolerance_minutes BEFORE the shift start time. This indicates the employee may have clocked in procedure time before actually starting their shift - potential time fraud.',
    1
FROM insight_definitions WHERE name = 'shift_medical_correlation'
ON CONFLICT (insight_id, condition_name) DO UPDATE SET
    severity = EXCLUDED.severity,
    condition_description = EXCLUDED.condition_description;

-- Severity: No Activity at Branch (WARNING)
INSERT INTO insight_severities (insight_id, condition_name, severity, condition_description, display_order)
SELECT 
    id,
    'no_activity_at_branch',
    'WARNING',
    'Employee worked a shift at branch X but has ZERO procedures recorded at that branch on that day. May indicate: (1) documentation gap, (2) non-clinical duties, (3) working at wrong location, or (4) fraudulent shift reporting.',
    2
FROM insight_definitions WHERE name = 'shift_medical_correlation'
ON CONFLICT (insight_id, condition_name) DO UPDATE SET
    severity = EXCLUDED.severity,
    condition_description = EXCLUDED.condition_description;

-- Severity: Location Mismatch (WARNING)
INSERT INTO insight_severities (insight_id, condition_name, severity, condition_description, display_order)
SELECT 
    id,
    'location_mismatch',
    'WARNING',
    'Medical procedures were recorded at a DIFFERENT location than where the employee was scheduled to work. Employee shift says "bat_yam" but procedures recorded at "hadera". May indicate travel, location confusion, or suspicious activity.',
    3
FROM insight_definitions WHERE name = 'shift_medical_correlation'
ON CONFLICT (insight_id, condition_name) DO UPDATE SET
    severity = EXCLUDED.severity,
    condition_description = EXCLUDED.condition_description;

-- Severity: Low Activity (INFO)
INSERT INTO insight_severities (insight_id, condition_name, severity, condition_description, display_order)
SELECT 
    id,
    'low_activity',
    'INFO',
    'Employee has fewer procedures than expected (min_expected_procedures) for their shift duration. Worth reviewing but may have legitimate explanations like training, administrative duties, or slow day.',
    4
FROM insight_definitions WHERE name = 'shift_medical_correlation'
ON CONFLICT (insight_id, condition_name) DO UPDATE SET
    severity = EXCLUDED.severity,
    condition_description = EXCLUDED.condition_description;

-- Severity: Late Departure (INFO)
INSERT INTO insight_severities (insight_id, condition_name, severity, condition_description, display_order)
SELECT 
    id,
    'late_departure',
    'INFO',
    'Medical action recorded more than late_departure_tolerance_minutes AFTER shift end. May indicate overtime, delayed documentation, or schedule adjustments. Lower priority than early arrival.',
    5
FROM insight_definitions WHERE name = 'shift_medical_correlation'
ON CONFLICT (insight_id, condition_name) DO UPDATE SET
    severity = EXCLUDED.severity,
    condition_description = EXCLUDED.condition_description;

