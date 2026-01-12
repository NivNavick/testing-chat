-- Seed Data: Shift Fraud Detection Insight
-- This is an example insight demonstrating the framework capabilities
-- 
-- Goal: Detect potential fraud by comparing employee shift hours 
--       against medical activity at the same location/time

-- ============================================================================
-- 1. Insert the Insight Definition
-- ============================================================================
INSERT INTO insight_definitions (name, display_name, description, goal, required_tables, optional_tables, category)
VALUES (
    'shift_fraud_detection',
    'זיהוי הונאת שעות עבודה',
    'Detects potential fraud by comparing employee shift hours against medical procedure records at the same location and time. When employee metadata is available, validates employee exists and uses their role for smarter matching.',
    'Analyze employee shift records against medical procedure records to detect potential fraud.
Find shifts where the employee claimed work hours but there is no corresponding medical 
activity at the same location during those hours.

IMPORTANT CONTEXT:
- The employee name will NOT appear in medical_actions records
- Correlation must be done via LOCATION (partial string match, e.g., "בת ים" appears in both)
- Correlation must also use TIME OVERLAP (shift entry/exit hours vs procedure hours on same date)
- The medical_actions.hours field contains a time RANGE in format "HH:MM - HH:MM"
- Location fields may have different formats - use substring matching for Hebrew city names

EMPLOYEE METADATA (when employee_salary table is available):
- Validate that the employee exists in the salary/metadata table
- Get employee role (תפקיד) like "אח/אחות גסטרו" (gastro nurse) or "טכנאית עיניים" (eye technician)
- Match employee role to appropriate medical department - e.g., gastro nurse should match gastro procedures
- A "גסטרו" nurse should only be flagged if there are no גסטרו procedures, not general procedures

The goal is to identify shifts where:
1. An employee claims to have worked specific hours at a location
2. But there is little or no ROLE-APPROPRIATE medical activity at that location during those hours
3. This could indicate fraudulent time reporting',
    ARRAY['employee_shifts', 'medical_actions'],
    ARRAY['employee_salary'],
    'compliance'
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

-- Rule: Early Tolerance
INSERT INTO insight_rules (insight_id, rule_name, value_type, current_value, default_value, min_value, max_value, description, ai_hint, display_order)
SELECT 
    id,
    'early_tolerance_minutes',
    'integer',
    '30',
    '30',
    '0',
    '120',
    'Medical activity occurring up to this many minutes BEFORE the shift start time should still be considered valid. Employees may arrive early and start working before their official clock-in time.',
    'When checking time overlap, subtract this value from the shift start time to create an expanded valid window.',
    1
FROM insight_definitions WHERE name = 'shift_fraud_detection'
ON CONFLICT (insight_id, rule_name) DO UPDATE SET
    current_value = EXCLUDED.current_value,
    description = EXCLUDED.description,
    ai_hint = EXCLUDED.ai_hint;

-- Rule: Late Tolerance
INSERT INTO insight_rules (insight_id, rule_name, value_type, current_value, default_value, min_value, max_value, description, ai_hint, display_order)
SELECT 
    id,
    'late_tolerance_minutes',
    'integer',
    '15',
    '15',
    '0',
    '60',
    'Medical activity occurring up to this many minutes AFTER the shift end time should still be considered valid. Employees may finish procedures after their official clock-out time.',
    'When checking time overlap, add this value to the shift end time to create an expanded valid window.',
    2
FROM insight_definitions WHERE name = 'shift_fraud_detection'
ON CONFLICT (insight_id, rule_name) DO UPDATE SET
    current_value = EXCLUDED.current_value,
    description = EXCLUDED.description,
    ai_hint = EXCLUDED.ai_hint;

-- Rule: Minimum Procedures Per Shift
INSERT INTO insight_rules (insight_id, rule_name, value_type, current_value, default_value, min_value, max_value, description, ai_hint, display_order)
SELECT 
    id,
    'min_procedures_per_shift',
    'integer',
    '1',
    '1',
    '0',
    '50',
    'The minimum number of medical procedures expected during a valid shift. If fewer procedures are found at the location during the shift hours, the shift is considered suspicious.',
    'Use this as the threshold: 0 procedures = CRITICAL (no_activity), below this threshold = WARNING (low_activity).',
    3
FROM insight_definitions WHERE name = 'shift_fraud_detection'
ON CONFLICT (insight_id, rule_name) DO UPDATE SET
    current_value = EXCLUDED.current_value,
    description = EXCLUDED.description,
    ai_hint = EXCLUDED.ai_hint;

-- Rule: Minimum Shift Hours
INSERT INTO insight_rules (insight_id, rule_name, value_type, current_value, default_value, min_value, max_value, description, ai_hint, display_order)
SELECT 
    id,
    'min_shift_hours',
    'float',
    '2.0',
    '2.0',
    '0.5',
    '12.0',
    'Only analyze shifts longer than this duration (in hours). Very short shifts may not generate procedure records and should be excluded from fraud analysis to reduce false positives.',
    'Add a WHERE clause to filter: total_hours >= this value. Calculate total_hours from entry and exit times.',
    4
FROM insight_definitions WHERE name = 'shift_fraud_detection'
ON CONFLICT (insight_id, rule_name) DO UPDATE SET
    current_value = EXCLUDED.current_value,
    description = EXCLUDED.description,
    ai_hint = EXCLUDED.ai_hint;

-- Rule: Location Match Strategy
INSERT INTO insight_rules (insight_id, rule_name, value_type, current_value, default_value, description, ai_hint, display_order)
SELECT 
    id,
    'location_match_strategy',
    'string',
    'substring',
    'substring',
    'How to match locations between shift records and medical records. Options:
- "substring": Look for partial matches (e.g., "בת ים" appears in both strings) - RECOMMENDED for Hebrew
- "exact": Require exact string match
- "city_extract": Extract city name from both and match

Since location formats differ between files (shifts may have "בת ים - עצמאיים" while medical has "בסט מדיקל -בת ים"), substring matching is usually best.',
    'Extract a common location key from both tables. For substring mode, look for common Hebrew city names like "בת ים", "חדרה", "ירושלים" that appear in both location fields.',
    5
FROM insight_definitions WHERE name = 'shift_fraud_detection'
ON CONFLICT (insight_id, rule_name) DO UPDATE SET
    current_value = EXCLUDED.current_value,
    description = EXCLUDED.description,
    ai_hint = EXCLUDED.ai_hint;

-- ============================================================================
-- 3. Insert Severity Mappings
-- ============================================================================

-- Severity: No Activity (CRITICAL)
INSERT INTO insight_severities (insight_id, condition_name, severity, condition_description, display_order)
SELECT 
    id,
    'no_activity',
    'CRITICAL',
    'ZERO medical procedures found at the matching location during the shift time window (including early/late tolerances). This is the strongest indicator of potential fraud - the employee claimed work hours but there is no evidence of any activity at that location during those hours.',
    1
FROM insight_definitions WHERE name = 'shift_fraud_detection'
ON CONFLICT (insight_id, condition_name) DO UPDATE SET
    severity = EXCLUDED.severity,
    condition_description = EXCLUDED.condition_description;

-- Severity: Low Activity (WARNING)
INSERT INTO insight_severities (insight_id, condition_name, severity, condition_description, display_order)
SELECT 
    id,
    'low_activity',
    'WARNING',
    'Some medical procedures found at the location during the shift, but fewer than the min_procedures_per_shift threshold. This may indicate partial fraud, documentation gaps, or that the employee was performing non-clinical duties.',
    2
FROM insight_definitions WHERE name = 'shift_fraud_detection'
ON CONFLICT (insight_id, condition_name) DO UPDATE SET
    severity = EXCLUDED.severity,
    condition_description = EXCLUDED.condition_description;

-- Severity: Location Not Matched (WARNING)
INSERT INTO insight_severities (insight_id, condition_name, severity, condition_description, display_order)
SELECT 
    id,
    'location_not_matched',
    'WARNING',
    'Could not find any medical records at a location matching the shift location using the configured matching strategy. This may indicate data quality issues, a new location not yet in the medical system, or suspicious location reporting.',
    3
FROM insight_definitions WHERE name = 'shift_fraud_detection'
ON CONFLICT (insight_id, condition_name) DO UPDATE SET
    severity = EXCLUDED.severity,
    condition_description = EXCLUDED.condition_description;

-- Severity: Activity Outside Tolerance (INFO)
INSERT INTO insight_severities (insight_id, condition_name, severity, condition_description, display_order)
SELECT 
    id,
    'activity_outside_tolerance',
    'INFO',
    'Medical activity was found at the location, but it occurred outside the shift window plus the configured tolerances. Worth reviewing but not necessarily suspicious - may indicate schedule changes or data entry timing issues.',
    4
FROM insight_definitions WHERE name = 'shift_fraud_detection'
ON CONFLICT (insight_id, condition_name) DO UPDATE SET
    severity = EXCLUDED.severity,
    condition_description = EXCLUDED.condition_description;

