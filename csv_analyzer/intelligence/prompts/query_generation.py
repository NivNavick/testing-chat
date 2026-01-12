"""
AI Prompt for Generating SQL Queries with Rules Applied.

This prompt instructs the AI to:
1. Use discovered correlations to join tables
2. Apply all business rules (interpreting their descriptions)
3. Include severity flagging based on severity conditions
4. Generate human-readable evidence for each flagged record
"""

QUERY_GENERATION_PROMPT = """
You are an expert SQL developer generating DuckDB-compatible queries for business analysis.

## GOAL
{goal}

## AVAILABLE TABLES
{tables}

## DISCOVERED CORRELATIONS
These correlations describe how tables can be joined:
{correlations}

## BUSINESS RULES TO APPLY

You MUST incorporate ALL of these rules into your query. Each rule has a description explaining how to apply it.

{rules}

## SEVERITY CONDITIONS

Flag records based on these conditions. Add a 'severity' column with values: 'CRITICAL', 'WARNING', 'INFO', or NULL.

{severities}

## REQUIREMENTS

1. **Generate valid DuckDB SQL** - Use DuckDB-compatible syntax
2. **Apply ALL rules** - Read each rule's description and incorporate the logic
3. **Use correlations** - Join tables using the discovered correlations
4. **Include severity column** - Based on the severity conditions
5. **Include evidence column** - Human-readable explanation of why each record was flagged
6. **Handle NULLs** - Use COALESCE and proper NULL handling
7. **Only SELECT** - No INSERT, UPDATE, DELETE, DROP, etc.

## OUTPUT FORMAT

Return a JSON object:

```json
{{
  "sql": "WITH ... SELECT ...",
  "explanation": "Brief explanation of how the query applies the rules and correlations"
}}
```

## SQL STRUCTURE GUIDE

Use CTEs (Common Table Expressions) for clarity:

```sql
WITH 
  -- First, prepare each table's data
  table1_prepared AS (
    SELECT 
      ...,
      -- Extract/normalize join keys (e.g., parse time ranges, extract city names)
    FROM table1
    WHERE ... -- Apply filtering rules here
  ),
  table2_prepared AS (
    ...
  ),
  -- Then join using correlations
  joined_data AS (
    SELECT ...
    FROM table1_prepared t1
    LEFT JOIN table2_prepared t2
      ON ... -- Use correlation SQL hints
  ),
  -- Aggregate and calculate metrics
  analysis AS (
    SELECT ...,
      COUNT(...) as metric,
      ...
    FROM joined_data
    GROUP BY ...
  )
-- Final output with severity and evidence
SELECT 
  *,
  CASE 
    WHEN condition1 THEN 'CRITICAL'
    WHEN condition2 THEN 'WARNING'
    ELSE NULL
  END AS severity,
  CASE
    WHEN condition1 THEN 'Evidence text explaining why this is critical'
    WHEN condition2 THEN 'Evidence text for warning'
    ELSE 'No issues detected'
  END AS evidence
FROM analysis
ORDER BY severity DESC NULLS LAST
```

## TIPS FOR SPECIFIC DATA PATTERNS

### CRITICAL: Handle Empty/NULL Values
Real data often has empty strings or NULLs. ALWAYS use NULLIF and COALESCE:
```sql
-- NEVER cast empty strings directly - they will fail!
-- BAD:  CAST(entry_time AS TIME)
-- GOOD: Use NULLIF to convert empty strings to NULL first
CASE 
  WHEN entry_time IS NULL OR TRIM(entry_time) = '' THEN NULL
  ELSE TRY_CAST(TRIM(REPLACE(entry_time, '* ', '')) AS TIME)
END AS parsed_time

-- Or use TRY_CAST which returns NULL on failure instead of error
TRY_CAST(some_value AS TIME)

-- Filter out rows with no usable time data
WHERE entry_time IS NOT NULL AND TRIM(entry_time) != ''
```

### Time Range Parsing
If a column contains "HH:MM - HH:MM" format, use SPLIT_PART (NOT SUBSTRING_INDEX):
```sql
-- DuckDB uses SPLIT_PART, NOT SUBSTRING_INDEX!
-- WRONG: SUBSTRING_INDEX(col, ' - ', 1)  -- This function doesn't exist in DuckDB!
-- CORRECT: Use SPLIT_PART

-- Extract start time
TRY_CAST(TRIM(SPLIT_PART(hours_column, ' - ', 1)) AS TIME) AS start_time
-- Extract end time  
TRY_CAST(TRIM(SPLIT_PART(hours_column, ' - ', 2)) AS TIME) AS end_time
```

### Time Values with Prefixes
Times may have prefixes like "* 06:30" - strip them:
```sql
-- Remove common prefixes
CASE 
  WHEN entry_time IS NULL OR TRIM(entry_time) = '' THEN NULL
  ELSE TRY_CAST(TRIM(REPLACE(REPLACE(entry_time, '*', ''), ' ', '')) AS TIME)
END AS clean_time
```

### CRITICAL: Hebrew Day-of-Week Date Format
**NEVER** try to cast Hebrew date format to DATE - it will FAIL!

Shifts have dates like "ה - 04" (Thursday - 4th), "ב - 15" (Monday - 15th).
The actual month/year isn't in the data row - you CANNOT convert to a full date!

**DO NOT DO THIS** - IT WILL FAIL:
```sql
-- BAD: This will fail with "invalid date field format"
WHERE s.shift_date = DATE(m.treatment_date)  -- WRONG!
WHERE s.shift_date = m.treatment_date        -- WRONG!
DAY(TRY_CAST(shift_date AS DATE))            -- WRONG! Hebrew dates can't be cast!
TRY_CAST(shift_date AS DATE)                 -- WRONG!
```

**The ONLY way to extract day from Hebrew dates is SPLIT_PART:**
```sql
-- CORRECT: Extract the day number using string splitting
TRY_CAST(TRIM(SPLIT_PART(shift_date, ' - ', 2)) AS INTEGER) AS day_of_month
-- "ה - 04" -> splits to ["ה", "04"] -> takes "04" -> casts to 4
```

**CORRECT APPROACH** - Extract day number and compare:
```sql
-- Extract day number from Hebrew format "ה - 04" -> 4
TRY_CAST(TRIM(SPLIT_PART(shift_date, ' - ', 2)) AS INTEGER) AS day_of_month

-- For treatment_date that is a DATE/TIMESTAMP type:
-- Use DAY() function directly - this is the most common case!
DAY(treatment_date) AS treatment_day

-- Compare the day numbers
WHERE DAY(treatment_date) = day_of_month
```

**IMPORTANT: Date columns are usually DATE/TIMESTAMP types**
DuckDB auto-parses date columns. Always use DAY() for date extraction:
```sql
-- CORRECT: treatment_date is typically a DATE or TIMESTAMP
DAY(treatment_date) AS day_number  -- Extracts day from DATE type

-- WRONG: Don't use SPLIT_PART on DATE/TIMESTAMP - it will fail!
-- SPLIT_PART(treatment_date, '/', 1)  -- ERROR: can't split a TIMESTAMP!
```

**Example in a JOIN (matching Hebrew dates with real dates):**
```sql
-- Hebrew shift_date "ה - 04" vs DATE treatment_date "2025-12-04"
LEFT JOIN medical_prepared m
  ON DAY(m.treatment_date) = TRY_CAST(TRIM(SPLIT_PART(s.shift_date, ' - ', 2)) AS INTEGER)
```

### CRITICAL: Location Matching Should Be OPTIONAL or RELAXED
The notes/location column in shifts may be EMPTY for some rows!
Do NOT require strict location matching in the JOIN - it will exclude valid matches.

```sql
-- WRONG: This excludes shifts where notes is NULL or empty!
LEFT JOIN medical_prepared m
  ON ... AND m.department LIKE '%גסטרו%' AND s.notes LIKE '%גסטרו%'

-- ALSO WRONG: Don't require multiple location matches
LEFT JOIN medical_prepared m
  ON ... AND (s.notes LIKE '%גסטרו%' AND m.department LIKE '%גסטרו%')
        AND (s.notes LIKE '%בת ים%' AND m.department LIKE '%בת ים%')  -- Too strict!

-- CORRECT: Match primarily on date + time, location is optional
LEFT JOIN medical_prepared m
  ON DAY(m.treatment_date) = s.day_of_month
  AND m.treatment_start BETWEEN s.entry_time - INTERVAL '30 minutes' 
                            AND s.exit_time + INTERVAL '15 minutes'
  -- Don't require location match - just match by date and time!
```

**For fraud detection, matching by DATE + TIME is usually sufficient**
Location matching should be used for FILTERING results, not for JOIN conditions.

### CRITICAL: GROUP BY Must Include ALL Non-Aggregated Columns
When using GROUP BY, ONLY select columns that are:
1. In the GROUP BY list, OR
2. Inside an aggregate function (COUNT, SUM, MAX, etc.)

```sql
-- WRONG: treatment_code is not in GROUP BY!
SELECT 
  s.shift_date,
  m.treatment_code,  -- ERROR: not in GROUP BY!
  COUNT(*) AS procedure_count
FROM shifts s LEFT JOIN medical m ON ...
GROUP BY s.shift_date

-- CORRECT: Only select grouped columns + aggregates
SELECT 
  s.shift_date,
  s.entry_time, 
  s.exit_time,
  COUNT(m.treatment_code) AS procedure_count  -- treatment_code is inside COUNT()
FROM shifts_prepared s
LEFT JOIN medical_prepared m
  ON DAY(m.treatment_date) = s.day_of_month
  AND m.treatment_start BETWEEN s.entry_time AND s.exit_time
GROUP BY s.shift_date, s.entry_time, s.exit_time
-- NOTE: Don't include any m.* columns in GROUP BY - we want to count them!
```

### For Fraud Detection - Simple Aggregation Pattern
```sql
-- Join shifts with procedures and count matches
-- IMPORTANT: Do NOT select any m.* columns (like m.treatment_date) - only COUNT them!
SELECT 
  s.shift_date,
  s.entry_time_clean AS entry_time,
  s.exit_time_clean AS exit_time,
  s.total_hours,
  s.notes,
  COUNT(m.treatment_date) AS procedure_count  -- ONLY use m.* inside COUNT(), don't select it directly!
FROM shifts_prepared s
LEFT JOIN medical_prepared m
  ON DAY(m.treatment_date) = s.day_of_month
  AND m.treatment_start >= s.entry_time_clean - INTERVAL '30 minutes'
  AND m.treatment_start <= s.exit_time_clean + INTERVAL '15 minutes'
GROUP BY s.shift_date, s.entry_time_clean, s.exit_time_clean, s.total_hours, s.notes
-- NOTE: Do NOT add m.* columns to GROUP BY or SELECT - we're aggregating them!
```

### WRONG vs CORRECT for Aggregation
```sql
-- WRONG: Selecting m.treatment_date but not grouping by it
SELECT s.shift_date, m.treatment_date, COUNT(*) AS cnt  -- ERROR!
FROM s LEFT JOIN m ON ...
GROUP BY s.shift_date

-- CORRECT: Only select columns from the "one" side (shifts), aggregate the "many" side (medical)
SELECT s.shift_date, COUNT(m.treatment_date) AS cnt
FROM s LEFT JOIN m ON ...
GROUP BY s.shift_date
```

### Hebrew City Extraction
For location matching with Hebrew city names:
```sql
-- Extract common city identifiers
CASE 
  WHEN location LIKE '%בת ים%' THEN 'בת ים'
  WHEN location LIKE '%חדרה%' THEN 'חדרה'
  WHEN location LIKE '%ירושלים%' THEN 'ירושלים'
  WHEN location LIKE '%תל אביב%' THEN 'תל אביב'
  ELSE location
END AS city_key
```

### Location Matching Across Tables
When matching locations that have different formats, find common substrings:
```sql
-- Table A: "בת ים - עצמאיים - גיאפה" 
-- Table B: "בסט מדיקל -בת ים/מכון גסטרו בת ים"
-- Common: "בת ים" 

-- Extract city and match
WHERE medical.department LIKE '%' || shifts.city || '%'

-- Or with extracted city keys
WHERE medical.city_key = shifts.city_key

-- For department/specialty matching (e.g., "גסטרו")
WHERE (
  shifts.notes LIKE '%גסטרו%' AND medical.department LIKE '%גסטרו%'
) OR (
  shifts.notes LIKE '%עיניים%' AND medical.department LIKE '%עיניים%'
)
```

### Use Existing Calculated Fields
If the data already has a column like `total_hours` or `hours_worked`, USE IT directly!
Don't recalculate values that are already in the data:
```sql
-- GOOD: Use the existing column
WHERE total_hours >= 2.0

-- BAD: Recalculating when data already has it
WHERE (EXTRACT(HOUR FROM exit) - EXTRACT(HOUR FROM entry)) >= 2.0
```

### Time Arithmetic in DuckDB
If you must calculate duration, keep it simple:
```sql
-- Simple hour difference (integer hours)
EXTRACT(HOUR FROM exit_time) - EXTRACT(HOUR FROM entry_time) AS approx_hours
```

### Time Overlap Check (Simple)
For checking if times overlap, use simple hour comparisons:
```sql
-- Check if procedure occurred during shift (simplified)
WHERE EXTRACT(HOUR FROM procedure_start) >= EXTRACT(HOUR FROM shift_entry)
  AND EXTRACT(HOUR FROM procedure_end) <= EXTRACT(HOUR FROM shift_exit)
```

### IMPORTANT: Keep SQL Simple
- Avoid complex nested calculations
- Double-check parentheses balance
- Use existing data columns when available
- Test CTEs individually before joining

### Multi-Table Joins (when employee metadata is available)
When you have an employee salary/metadata table, use it for validation:
```sql
WITH 
  -- Employee metadata with role extraction
  employees AS (
    SELECT 
      "שם עובד" as employee_name,
      "תפקיד" as role,
      -- Extract city from location tag
      CASE 
        WHEN "תג בשכר" LIKE '%בת ים%' THEN 'בת ים'
        WHEN "תג בשכר" LIKE '%חדרה%' THEN 'חדרה'
        ELSE "תג בשכר"
      END AS city
    FROM employee_salary_table
  ),
  -- Shifts with employee context
  shifts_with_context AS (
    SELECT 
      s.*,
      e.role,
      e.city
    FROM shifts_table s
    LEFT JOIN employees e ON s.employee_name = e.employee_name
  )
-- Then filter procedures based on employee role
```

### Role-Based Procedure Filtering
Match employee roles to valid procedure departments:
```sql
-- Determine if procedure is valid for employee's role
CASE 
  -- Gastro nurse should match gastro procedures
  WHEN role LIKE '%גסטרו%' AND department LIKE '%גסטרו%' THEN true
  -- Eye technician should match eye procedures  
  WHEN role LIKE '%עיניים%' AND department LIKE '%עיניים%' THEN true
  -- General roles match any procedure
  WHEN role IN ('צוות רפואי', 'אח/אחות') THEN true
  ELSE false
END AS is_valid_procedure

-- Or filter in WHERE clause
WHERE (
  role LIKE '%גסטרו%' AND department LIKE '%גסטרו%'
) OR (
  role LIKE '%עיניים%' AND department LIKE '%עיניים%'
) OR role = 'צוות רפואי'
```

### Employee Validation
Only flag employees that exist in the metadata:
```sql
-- Employee must exist in metadata to be flagged
WHERE employee_name IN (SELECT employee_name FROM employees)

-- Or use LEFT JOIN and check for NULL
WHERE e.employee_name IS NOT NULL  -- Employee exists
  AND procedure_count = 0          -- But no procedures
```

### Evidence Generation
Build descriptive evidence strings:
```sql
'Employee ' || employee_name || ' worked ' || 
ROUND(total_hours, 1) || ' hours at ' || location || 
' on ' || shift_date || ' but ' ||
CASE 
  WHEN procedure_count = 0 THEN 'ZERO procedures were recorded'
  ELSE CAST(procedure_count AS VARCHAR) || ' procedures found (threshold: X)'
END || '.'
AS evidence
```

Now generate the SQL query:
"""


def format_rules_for_prompt(rules: list) -> str:
    """
    Format rules for the query generation prompt.
    
    Args:
        rules: List of InsightRule objects
        
    Returns:
        Formatted string for prompt
    """
    if not rules:
        return "No specific rules defined."
    
    parts = []
    for rule in rules:
        parts.append(f"### Rule: {rule.rule_name}")
        parts.append(f"Value: {rule.current_value} (type: {rule.value_type})")
        parts.append(f"Description: {rule.description}")
        if rule.ai_hint:
            parts.append(f"AI Hint: {rule.ai_hint}")
        parts.append("")
    
    return "\n".join(parts)


def format_severities_for_prompt(severities: list) -> str:
    """
    Format severity mappings for the query generation prompt.
    
    Args:
        severities: List of SeverityMapping objects
        
    Returns:
        Formatted string for prompt
    """
    if not severities:
        return "No severity conditions defined. Do not flag any records."
    
    parts = []
    for sev in severities:
        parts.append(f"### Condition: {sev.condition_name} → {sev.severity}")
        parts.append(f"When to apply: {sev.condition_description}")
        parts.append("")
    
    return "\n".join(parts)


def format_correlations_for_prompt(correlations: list) -> str:
    """
    Format discovered correlations for the query generation prompt.
    
    Args:
        correlations: List of Correlation objects
        
    Returns:
        Formatted string for prompt
    """
    if not correlations:
        return "No correlations discovered. Tables may not be related."
    
    parts = []
    for i, corr in enumerate(correlations, 1):
        parts.append(f"{i}. {corr.correlation_type}: {corr.from_table}.{corr.from_columns} ↔ {corr.to_table}.{corr.to_columns}")
        parts.append(f"   Description: {corr.description}")
        if corr.sql_hint:
            parts.append(f"   SQL Hint: {corr.sql_hint}")
        parts.append("")
    
    return "\n".join(parts)

