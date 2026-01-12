"""
AI Prompt for Discovering Table Correlations.

This prompt helps the AI identify how multiple tables can be joined/related,
even when there are no explicit foreign key columns.

The AI looks for:
- Date/time field matches
- Time range overlaps
- Location/string containment (partial matches)
- ID field matches
- Semantic relationships
"""

CORRELATION_PROMPT = """
You are an expert data analyst tasked with discovering how multiple database tables relate to each other.

## GOAL
{goal}

## AVAILABLE TABLES

{tables}

## YOUR TASK

Analyze these tables and identify ALL possible correlations between them. 
A correlation is a way to join or relate records from different tables.

### Correlation Types to Look For

1. **date_match**: Date/datetime fields that can be matched directly
   - Example: shifts.shift_date = medical.treatment_date

2. **time_overlap**: Time ranges that may overlap
   - Example: shifts.entry_time-exit_time overlaps with procedures.start_time-end_time
   - Note: One table might have time ranges in a SINGLE column like "HH:MM - HH:MM"

3. **location_contains**: Location strings where one contains part of another
   - Example: "בת ים" appears in both shifts.location and medical.department
   - This is common for Hebrew text where formats differ between systems
   - **Look for common city names**: בת ים, חדרה, תל אביב, ירושלים

4. **exact_match**: Fields that should match exactly
   - Example: shifts.employee_id = payroll.employee_id

5. **employee_name_match**: Employee names that link across files
   - Example: shifts.employee_name = salary.employee_name
   - Use this to get employee role, department, and validate employment

6. **role_to_department**: Employee role determines valid departments
   - Example: employee role "אח/אחות גסטרו" → procedures.department LIKE '%גסטרו%'
   - Requires employee metadata table with role information

7. **semantic_relationship**: Fields that relate conceptually
   - Example: shifts.employee_name might relate to medical.staff_team

### CRITICAL: Verify Actual Data Values Match!

Before suggesting a correlation, CHECK THE SAMPLE DATA to verify values can actually match:

1. **Look at actual values** in both columns you want to correlate
2. **Check if values from one column appear in the other** (exact or partial)
3. **Don't assume column names mean the data will match** - verify with sample values!

Example verification:
- Table A has `notes` with values: "גסטרו", "עיניים"
- Table B has `department` with values: "בסט מדיקל -בת ים/מכון גסטרו בת ים"
- ✅ "גסטרו" DOES appear inside "מכון גסטרו בת ים" → correlation is valid
- ❌ If values don't overlap at all → don't suggest this correlation!

### Semantic Location Matching (CRITICAL for Hebrew data)

When comparing location/department/address strings, look for COMMON SUBSTRINGS:

1. **Extract Hebrew city names**: בת ים, חדרה, תל אביב, ירושלים, חיפה, באר שבע
2. **Check if the SAME city/keyword appears in BOTH columns' sample values**
3. **Use substring matching** in SQL: `col_a LIKE '%keyword%' AND col_b LIKE '%keyword%'`

Example - Finding the common link:
- Table A `location_tag`: "בת ים - עצמאיים - גיאפה" 
- Table B `department`: "בסט מדיקל -בת ים/מכון גסטרו בת ים"
- Common substring: "בת ים" appears in BOTH!
- SQL: `department LIKE '%בת ים%'` to filter medical actions at same location

Example - Matching department/specialty:
- Table A `notes`: "גסטרו" (gastro department indicator)
- Table B `department`: "בסט מדיקל -בת ים/מכון גסטרו בת ים"
- Common substring: "גסטרו" appears in BOTH!
- SQL: `department LIKE '%גסטרו%'` to filter procedures by specialty

### Employee Metadata Correlation (when employee_salary/metadata table exists)

When an employee metadata table is available:

1. **employee_name exact match**: Link shifts.employee_name = metadata.employee_name
2. **Role-based filtering**: Use employee role (תפקיד) to filter valid procedures
   - "אח/אחות גסטרו" (Gastro nurse) → procedures in גסטרו department
   - "טכנאית עיניים" (Eye technician) → procedures in עיניים department
3. **Location from metadata**: Get employee's work location from metadata table

Example role-to-department mapping:
- Role "אח/אחות גסטרו" → look for procedures where department contains "גסטרו"
- Role "טכנאית עיניים" → look for procedures where department contains "עיניים"
- Role "צוות רפואי" → any medical procedure is valid

### Important Considerations

- Column names may be in Hebrew, Arabic, or English
- Time ranges might be stored as single strings like "15:30 - 21:00"
- Location formats often differ between files (one may have full address, another just city)
- **Look for COMMON SUBSTRINGS** (city names, department names) across location columns
- Employee names may link to a metadata table that provides their role/department
- **ALWAYS verify sample data values can actually match before suggesting a correlation**

## OUTPUT FORMAT

Return a JSON object with discovered correlations:

```json
{{
  "correlations": [
    {{
      "correlation_type": "date_match|time_overlap|location_contains|exact_match|semantic_relationship",
      "from_table": "table_name",
      "from_columns": ["column1", "column2"],
      "to_table": "other_table_name",
      "to_columns": ["column1"],
      "description": "Human-readable explanation of how these relate",
      "sql_hint": "Suggested SQL JOIN condition or pattern",
      "confidence": 0.0-1.0
    }}
  ],
  "notes": "Any important observations about the data that affect correlation"
}}
```

### Example Correlations

For shifts and medical_actions where employee name is NOT in medical:

```json
{{
  "correlations": [
    {{
      "correlation_type": "date_match",
      "from_table": "employee_shifts",
      "from_columns": ["shift_date"],
      "to_table": "medical_actions",
      "to_columns": ["treatment_date"],
      "description": "Shift date matches treatment date",
      "sql_hint": "shifts.shift_date = medical.treatment_date",
      "confidence": 0.95
    }},
    {{
      "correlation_type": "location_contains",
      "from_table": "employee_shifts", 
      "from_columns": ["location"],
      "to_table": "medical_actions",
      "to_columns": ["department"],
      "description": "Both contain Hebrew city name like 'בת ים'",
      "sql_hint": "medical.department LIKE '%' || extract_city(shifts.location) || '%'",
      "confidence": 0.8
    }},
    {{
      "correlation_type": "time_overlap",
      "from_table": "employee_shifts",
      "from_columns": ["entry_time", "exit_time"],
      "to_table": "medical_actions",
      "to_columns": ["hours"],
      "description": "Shift hours overlap with procedure hours (format: HH:MM - HH:MM)",
      "sql_hint": "Parse hours column and check if between entry_time and exit_time",
      "confidence": 0.85
    }}
  ],
  "notes": "Employee name does not appear in medical_actions. Correlation must use date + location + time overlap."
}}
```

Now analyze the provided tables and discover correlations:
"""


def format_tables_for_correlation_prompt(tables: dict) -> str:
    """
    Format table information for the correlation prompt.
    
    Args:
        tables: Dict mapping document_type -> TableInfo
        
    Returns:
        Formatted string for prompt
    """
    # Common Hebrew keywords to look for in location/department matching
    HEBREW_KEYWORDS = ['בת ים', 'חדרה', 'תל אביב', 'ירושלים', 'חיפה', 'באר שבע', 
                       'גסטרו', 'עיניים', 'רפואה', 'מדיקל', 'קפסולה']
    
    parts = []
    all_keywords_found = {}  # Track keywords found across all tables
    
    for doc_type, table_info in tables.items():
        # IMPORTANT: Use the actual DuckDB table name for SQL queries
        actual_table_name = table_info.table_name
        parts.append(f"### Table: {actual_table_name}")
        parts.append(f"Document Type: {doc_type}")
        parts.append(f"Row Count: {table_info.row_count}")
        parts.append("")
        parts.append("**IMPORTANT: Use table name '{0}' in all SQL queries**".format(actual_table_name))
        parts.append("")
        parts.append("Columns:")
        for col in table_info.columns:
            col_name = col.get('name', col.get('column_name', 'unknown'))
            col_type = col.get('type', col.get('dtype', 'unknown'))
            parts.append(f"  - {col_name} ({col_type})")
        
        if table_info.sample_data:
            parts.append("")
            parts.append("Sample Data (first 5 rows):")
            for i, row in enumerate(table_info.sample_data[:5]):
                # Truncate long values
                truncated = {k: str(v)[:60] + ('...' if len(str(v)) > 60 else '') 
                            for k, v in row.items()}
                parts.append(f"  {i+1}. {truncated}")
            
            # Extract distinct values from string columns for correlation verification
            parts.append("")
            parts.append("Distinct Values (for correlation verification):")
            string_cols = [col.get('name', col.get('column_name')) 
                          for col in table_info.columns 
                          if 'VARCHAR' in str(col.get('type', '')).upper() or 
                             'TEXT' in str(col.get('type', '')).upper() or
                             'STRING' in str(col.get('type', '')).upper()]
            
            for col_name in string_cols[:8]:  # Increased to 8 columns
                values = set()
                for row in table_info.sample_data[:15]:  # Check more rows
                    val = row.get(col_name)
                    if val and str(val).strip() and str(val) != 'None':
                        values.add(str(val)[:50])
                if values:
                    parts.append(f"  {col_name}: {list(values)[:6]}")
            
            # Find Hebrew keywords in this table's data
            parts.append("")
            parts.append("Hebrew Keywords Found (for location/department matching):")
            keywords_in_table = set()
            for row in table_info.sample_data:
                for col_name, val in row.items():
                    if val:
                        val_str = str(val)
                        for keyword in HEBREW_KEYWORDS:
                            if keyword in val_str:
                                keywords_in_table.add(keyword)
                                # Track which tables have which keywords
                                if keyword not in all_keywords_found:
                                    all_keywords_found[keyword] = []
                                if actual_table_name not in all_keywords_found[keyword]:
                                    all_keywords_found[keyword].append(actual_table_name)
            
            if keywords_in_table:
                parts.append(f"  Found: {list(keywords_in_table)}")
            else:
                parts.append("  None detected")
        
        parts.append("")
        parts.append("---")
        parts.append("")
    
    # Add summary of keywords found in multiple tables (potential correlations!)
    if all_keywords_found:
        parts.append("### CORRELATION HINTS - Keywords Found in Multiple Tables:")
        for keyword, tables_list in all_keywords_found.items():
            if len(tables_list) > 1:
                parts.append(f"  '{keyword}' appears in: {tables_list}")
                parts.append(f"    → Use: table_a.column LIKE '%{keyword}%' AND table_b.column LIKE '%{keyword}%'")
        parts.append("")
    
    return "\n".join(parts)

