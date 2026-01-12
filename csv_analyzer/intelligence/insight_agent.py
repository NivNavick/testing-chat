"""
InsightAgent - AI-driven agent for cross-file insights.

This agent:
1. Discovers correlations between tables using AI
2. Generates SQL queries with business rules applied
3. Validates generated SQL for safety
4. Executes queries and flags results

No hardcoded rules - everything is AI-interpreted based on:
- The insight's goal (natural language)
- Rule descriptions
- Severity condition descriptions
"""

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional

import duckdb
import pandas as pd

from csv_analyzer.core.config import get_settings
from csv_analyzer.insights.configurable.models import (
    ConfigurableInsightDefinition,
    ConfigurableInsightResult,
    Correlation,
    InsightFlag,
    TableInfo,
    Severity,
)
from csv_analyzer.intelligence.prompts.correlation import (
    CORRELATION_PROMPT,
    format_tables_for_correlation_prompt,
)
from csv_analyzer.intelligence.prompts.query_generation import (
    QUERY_GENERATION_PROMPT,
    format_rules_for_prompt,
    format_severities_for_prompt,
    format_correlations_for_prompt,
)

logger = logging.getLogger(__name__)


class SQLValidationError(Exception):
    """Raised when generated SQL fails validation."""
    pass


class InsightExecutionError(Exception):
    """Raised when insight execution fails."""
    pass


class InsightAgent:
    """
    AI-driven agent for executing configurable insights.
    
    Independent from SmartAnalyzer - handles cross-file insights
    with configurable rules and severity flagging.
    """
    
    # Dangerous SQL keywords that are not allowed
    # Note: REPLACE is NOT included because it's a legitimate string function
    # CREATE is blocked so "CREATE OR REPLACE" won't work anyway
    DANGEROUS_KEYWORDS = [
        'DROP', 'DELETE', 'UPDATE', 'INSERT', 'TRUNCATE', 'ALTER', 
        'CREATE', 'GRANT', 'REVOKE', 'EXEC', 'EXECUTE'
    ]
    
    def __init__(
        self,
        duckdb_connection: duckdb.DuckDBPyConnection,
        model: str = None,
        api_key: str = None,
    ):
        """
        Initialize the InsightAgent.
        
        Args:
            duckdb_connection: DuckDB connection with loaded tables
            model: LLM model to use (default from settings)
            api_key: OpenAI API key (default from settings/env)
        """
        self.conn = duckdb_connection
        self.settings = get_settings()
        self.model = model or self.settings.llm_model
        self.api_key = api_key or self.settings.openai_api_key
        self._client = None
    
    def _get_client(self):
        """Lazy-load OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError("openai package not installed. Run: pip install openai")
        return self._client
    
    async def execute_insight(
        self,
        insight: ConfigurableInsightDefinition,
        tables: Dict[str, TableInfo],
        employee_context: Optional[Dict[str, Any]] = None,
    ) -> ConfigurableInsightResult:
        """
        Execute an insight using AI-generated SQL.
        
        Full flow:
        1. Discover correlations between tables (AI)
        2. Generate SQL query with rules applied (AI)
        3. Validate SQL (safety check)
        4. Execute against DuckDB
        5. Extract flagged results
        
        Args:
            insight: The insight definition with goal, rules, severities
            tables: Dict of table_name -> TableInfo with schemas and samples
            employee_context: Optional employee metadata context for enrichment
            
        Returns:
            ConfigurableInsightResult with flags and evidence
        """
        start_time = time.time()
        
        # Build rules snapshot for logging
        rules_snapshot = {
            rule.rule_name: rule.get_typed_value()
            for rule in insight.rules
        }
        
        try:
            # Step 1: Discover correlations
            logger.info(f"[{insight.name}] Discovering correlations between tables...")
            correlations = await self._discover_correlations(
                goal=insight.goal,
                tables=tables
            )
            logger.info(f"[{insight.name}] Found {len(correlations)} correlations")
            
            # Step 2: Generate SQL with rules
            logger.info(f"[{insight.name}] Generating SQL query...")
            generated = await self._generate_query(
                goal=insight.goal,
                tables=tables,
                correlations=correlations,
                rules=insight.rules,
                severities=insight.severities,
                employee_context=employee_context
            )
            logger.info(f"[{insight.name}] SQL generated ({len(generated['sql'])} chars)")
            
            # Step 3: Validate SQL
            logger.info(f"[{insight.name}] Validating SQL...")
            logger.info(f"[{insight.name}] Generated SQL:\n{generated['sql']}")
            validated_sql = self._validate_sql(generated['sql'])
            logger.info(f"[{insight.name}] SQL validation passed")
            
            # Step 4: Execute with retry on syntax errors
            max_retries = 2
            last_error = None
            results_df = None
            
            for attempt in range(max_retries + 1):
                try:
                    logger.info(f"[{insight.name}] Executing query (attempt {attempt + 1})...")
                    results_df = self._execute_sql(validated_sql)
                    logger.info(f"[{insight.name}] Query returned {len(results_df)} rows")
                    break
                except InsightExecutionError as e:
                    last_error = e
                    error_str = str(e).lower()
                    
                    # Only retry on syntax/parser errors
                    if attempt < max_retries and ('syntax' in error_str or 'parser' in error_str):
                        logger.warning(f"[{insight.name}] SQL execution failed, regenerating (attempt {attempt + 1}): {e}")
                        
                        # Regenerate SQL with the error context
                        regenerate_prompt = f"Previous SQL failed with error: {e}. Please fix the syntax and regenerate."
                        generated = await self._generate_query(
                            goal=insight.goal + f"\n\nPREVIOUS ERROR: {e}",
                            tables=tables,
                            correlations=correlations,
                            rules=insight.rules,
                            severities=insight.severities,
                            employee_context=employee_context
                        )
                        validated_sql = self._validate_sql(generated['sql'])
                        logger.info(f"[{insight.name}] Regenerated SQL:\n{validated_sql}")
                    else:
                        raise
            
            if results_df is None and last_error:
                raise last_error
            
            # Step 5: Extract flags
            flags = self._extract_flags(results_df)
            summary = self._summarize_flags(flags)
            logger.info(f"[{insight.name}] Flagged {len(flags)} records: {summary}")
            
            execution_time = (time.time() - start_time) * 1000
            
            return ConfigurableInsightResult(
                insight_name=insight.name,
                correlations_found=correlations,
                generated_sql=validated_sql,
                ai_explanation=generated.get('explanation', ''),
                total_records=len(results_df),
                flagged_records=flags,
                summary=summary,
                rules_used=rules_snapshot,
                execution_time_ms=execution_time,
                success=True,
            )
            
        except SQLValidationError as e:
            logger.error(f"[{insight.name}] SQL validation failed: {e}")
            return ConfigurableInsightResult(
                insight_name=insight.name,
                correlations_found=[],
                generated_sql="",
                ai_explanation="",
                total_records=0,
                flagged_records=[],
                summary={},
                rules_used=rules_snapshot,
                execution_time_ms=(time.time() - start_time) * 1000,
                success=False,
                error=f"SQL validation failed: {e}",
            )
            
        except Exception as e:
            logger.error(f"[{insight.name}] Execution failed: {e}", exc_info=True)
            # Include correlations and SQL if we got that far
            return ConfigurableInsightResult(
                insight_name=insight.name,
                correlations_found=correlations if 'correlations' in dir() else [],
                generated_sql=generated.get('sql', '') if 'generated' in dir() else '',
                ai_explanation=generated.get('explanation', '') if 'generated' in dir() else '',
                total_records=0,
                flagged_records=[],
                summary={},
                rules_used=rules_snapshot,
                execution_time_ms=(time.time() - start_time) * 1000,
                success=False,
                error=str(e),
            )
    
    async def _discover_correlations(
        self,
        goal: str,
        tables: Dict[str, TableInfo],
    ) -> List[Correlation]:
        """
        Use AI to discover how tables correlate.
        
        Args:
            goal: Natural language goal
            tables: Available tables with schemas
            
        Returns:
            List of discovered correlations
        """
        # Format tables for prompt
        tables_str = format_tables_for_correlation_prompt(tables)
        
        # Build prompt
        prompt = CORRELATION_PROMPT.format(
            goal=goal,
            tables=tables_str
        )
        
        # Call LLM
        client = self._get_client()
        
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert data analyst. Analyze tables and discover correlations."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            
            correlations = []
            for c in result.get("correlations", []):
                correlations.append(Correlation(
                    correlation_type=c.get("correlation_type", "unknown"),
                    from_table=c.get("from_table", ""),
                    from_columns=c.get("from_columns", []),
                    to_table=c.get("to_table", ""),
                    to_columns=c.get("to_columns", []),
                    description=c.get("description", ""),
                    sql_hint=c.get("sql_hint"),
                    confidence=c.get("confidence", 0.5)
                ))
            
            return correlations
            
        except Exception as e:
            logger.error(f"Correlation discovery failed: {e}")
            return []
    
    async def _generate_query(
        self,
        goal: str,
        tables: Dict[str, TableInfo],
        correlations: List[Correlation],
        rules: list,
        severities: list,
        employee_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, str]:
        """
        Use AI to generate SQL query with rules applied.
        
        Args:
            goal: Natural language goal
            tables: Available tables with schemas
            correlations: Discovered correlations
            rules: InsightRule objects with descriptions
            severities: SeverityMapping objects with conditions
            employee_context: Optional employee metadata for role-based filtering
            
        Returns:
            Dict with 'sql' and 'explanation'
        """
        # Format tables
        tables_str = format_tables_for_correlation_prompt(tables)
        
        # Format other components
        correlations_str = format_correlations_for_prompt(correlations)
        rules_str = format_rules_for_prompt(rules)
        severities_str = format_severities_for_prompt(severities)
        
        # Add employee context if available
        employee_context_str = ""
        if employee_context:
            employee_context_str = self._format_employee_context(employee_context)
        
        # Build prompt
        prompt = QUERY_GENERATION_PROMPT.format(
            goal=goal,
            tables=tables_str,
            correlations=correlations_str,
            rules=rules_str,
            severities=severities_str
        )
        
        # Append employee context if available
        if employee_context_str:
            prompt += f"\n\n## EMPLOYEE CONTEXT (use for validation and role-based filtering)\n\n{employee_context_str}"
        
        # Call LLM
        client = self._get_client()
        
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert SQL developer. Generate DuckDB-compatible queries."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            
            return {
                "sql": result.get("sql", ""),
                "explanation": result.get("explanation", "")
            }
            
        except Exception as e:
            logger.error(f"Query generation failed: {e}")
            raise InsightExecutionError(f"Failed to generate SQL: {e}")
    
    def _format_employee_context(self, context: Dict[str, Any]) -> str:
        """
        Format employee context for inclusion in AI prompt.
        
        Args:
            context: Employee context from enricher
            
        Returns:
            Formatted string for prompt
        """
        parts = []
        
        # Role-to-procedure mappings
        if "role_to_procedures" in context:
            parts.append("### Role-to-Procedure Mappings")
            parts.append("Use these to filter medical procedures based on employee role:")
            for role, procedures in context["role_to_procedures"].items():
                parts.append(f"  - {role}: {procedures}")
            parts.append("")
        
        # Cities
        if "cities" in context:
            parts.append("### Known Cities (for location matching)")
            parts.append(f"  {context['cities']}")
            parts.append("")
        
        # Employee list
        if "employees" in context and context["employees"]:
            parts.append("### Employees from Metadata")
            parts.append("Use this to validate employees and get their roles:")
            for emp in context["employees"][:20]:  # Limit to 20
                parts.append(f"  - {emp['name']}: role={emp['role']}, city={emp.get('city', 'unknown')}")
            if len(context["employees"]) > 20:
                parts.append(f"  ... and {len(context['employees']) - 20} more")
            parts.append("")
        
        # SQL hints
        if "sql_hints" in context:
            parts.append("### SQL Patterns for Employee Enrichment")
            for hint_name, hint_sql in context["sql_hints"].items():
                parts.append(f"  - {hint_name}: {hint_sql}")
            parts.append("")
        
        parts.append("### Employee Validation Rules")
        parts.append("1. Employee MUST exist in the metadata table to be considered valid")
        parts.append("2. Use employee's role to determine which procedures are valid for them")
        parts.append("3. A 'גסטרו' nurse should only be matched with גסטרו procedures")
        parts.append("4. Match employee location (city) to procedure location")
        
        return "\n".join(parts)
    
    def _cleanup_sql(self, sql: str) -> str:
        """
        Clean up and fix common issues in AI-generated SQL.
        
        Args:
            sql: Raw SQL from AI
            
        Returns:
            Cleaned SQL
        """
        if not sql:
            return sql
        
        # Strip whitespace
        sql = sql.strip()
        
        # Remove trailing semicolons (DuckDB doesn't need them)
        sql = sql.rstrip(';').strip()
        
        # Remove markdown code block markers if present
        if sql.startswith('```'):
            lines = sql.split('\n')
            # Remove first line (```sql or ```)
            lines = lines[1:]
            # Remove last line if it's ```
            if lines and lines[-1].strip() == '```':
                lines = lines[:-1]
            sql = '\n'.join(lines)
        
        # Fix double spaces that might cause parsing issues
        sql = re.sub(r'  +', ' ', sql)
        
        # Fix missing spaces after commas in some cases
        sql = re.sub(r',([^\s])', r', \1', sql)
        
        # Fix potential issues with NULLS LAST/FIRST (ensure proper spacing)
        sql = re.sub(r'NULLS\s+LAST', 'NULLS LAST', sql, flags=re.IGNORECASE)
        sql = re.sub(r'NULLS\s+FIRST', 'NULLS FIRST', sql, flags=re.IGNORECASE)
        
        # Ensure proper newlines aren't breaking SQL
        sql = re.sub(r'\n\s*\n', '\n', sql)
        
        return sql
    
    def _validate_sql(self, sql: str) -> str:
        """
        Validate generated SQL for safety.
        
        Args:
            sql: Generated SQL query
            
        Returns:
            Validated SQL
            
        Raises:
            SQLValidationError: If SQL contains dangerous operations
        """
        # First clean up the SQL
        sql = self._cleanup_sql(sql)
        
        if not sql or not sql.strip():
            raise SQLValidationError("Empty SQL query")
        
        sql_upper = sql.upper()
        
        # Check for dangerous keywords
        for keyword in self.DANGEROUS_KEYWORDS:
            # Match as whole word
            pattern = r'\b' + keyword + r'\b'
            if re.search(pattern, sql_upper):
                raise SQLValidationError(f"Dangerous SQL keyword detected: {keyword}")
        
        # Must start with SELECT or WITH (for CTEs)
        sql_stripped = sql.strip().upper()
        if not (sql_stripped.startswith('SELECT') or sql_stripped.startswith('WITH')):
            raise SQLValidationError("SQL must start with SELECT or WITH")
        
        # Check for multiple statements (semicolon followed by another statement)
        if re.search(r';\s*(SELECT|WITH|DROP|DELETE|INSERT|UPDATE)', sql_upper):
            raise SQLValidationError("Multiple SQL statements detected")
        
        return sql
    
    def _execute_sql(self, sql: str) -> pd.DataFrame:
        """
        Execute SQL against DuckDB.
        
        Args:
            sql: Validated SQL query
            
        Returns:
            Results as DataFrame
        """
        try:
            result = self.conn.execute(sql).fetchdf()
            return result
        except Exception as e:
            raise InsightExecutionError(f"SQL execution failed: {e}")
    
    def _extract_flags(self, df: pd.DataFrame) -> List[InsightFlag]:
        """
        Extract flagged records from results DataFrame.
        
        Looks for 'severity' and 'evidence' columns.
        
        Args:
            df: Results DataFrame
            
        Returns:
            List of InsightFlag objects
        """
        flags = []
        
        if df.empty:
            return flags
        
        # Check if severity column exists
        severity_col = None
        for col in df.columns:
            if col.lower() == 'severity':
                severity_col = col
                break
        
        if severity_col is None:
            return flags
        
        # Find evidence column
        evidence_col = None
        for col in df.columns:
            if col.lower() == 'evidence':
                evidence_col = col
                break
        
        # Extract flagged rows
        for idx, row in df.iterrows():
            severity_value = row.get(severity_col)
            
            # Skip rows without severity
            if pd.isna(severity_value) or severity_value is None:
                continue
            
            severity_str = str(severity_value).upper()
            
            # Validate severity
            try:
                severity = Severity(severity_str)
            except ValueError:
                logger.warning(f"Invalid severity value: {severity_str}")
                continue
            
            # Get evidence
            evidence = ""
            if evidence_col:
                evidence = str(row.get(evidence_col, ""))
            
            # Build record data (exclude severity and evidence columns)
            record_data = {}
            for col in df.columns:
                if col.lower() not in ('severity', 'evidence'):
                    val = row.get(col)
                    # Handle non-JSON-serializable types
                    if pd.isna(val):
                        record_data[col] = None
                    elif hasattr(val, 'isoformat'):
                        record_data[col] = val.isoformat()
                    else:
                        record_data[col] = val
            
            # Determine condition from evidence or severity
            condition = self._infer_condition(severity, evidence)
            
            flags.append(InsightFlag(
                severity=severity,
                condition=condition,
                record_data=record_data,
                evidence=evidence
            ))
        
        return flags
    
    def _infer_condition(self, severity: Severity, evidence: str) -> str:
        """
        Infer the condition name from severity and evidence.
        
        Args:
            severity: Severity level
            evidence: Evidence text
            
        Returns:
            Condition name string
        """
        evidence_lower = evidence.lower()
        
        # Common patterns
        if 'zero' in evidence_lower or 'no activity' in evidence_lower or 'no procedure' in evidence_lower:
            return 'no_activity'
        elif 'low' in evidence_lower or 'below' in evidence_lower or 'fewer' in evidence_lower:
            return 'low_activity'
        elif 'location' in evidence_lower and ('not' in evidence_lower or 'mismatch' in evidence_lower):
            return 'location_not_matched'
        elif 'outside' in evidence_lower or 'tolerance' in evidence_lower:
            return 'activity_outside_tolerance'
        else:
            # Default based on severity
            if severity == Severity.CRITICAL:
                return 'critical_issue'
            elif severity == Severity.WARNING:
                return 'warning_issue'
            else:
                return 'info_issue'
    
    def _summarize_flags(self, flags: List[InsightFlag]) -> Dict[str, int]:
        """
        Create summary of flags by severity.
        
        Args:
            flags: List of InsightFlag objects
            
        Returns:
            Dict like {"CRITICAL": 1, "WARNING": 2, "INFO": 0}
        """
        summary = {
            "CRITICAL": 0,
            "WARNING": 0,
            "INFO": 0
        }
        
        for flag in flags:
            key = flag.severity.value if hasattr(flag.severity, 'value') else str(flag.severity)
            if key in summary:
                summary[key] += 1
        
        return summary
    
    def get_table_info(self, table_name: str, sample_rows: int = 10) -> Optional[TableInfo]:
        """
        Get TableInfo for a table in DuckDB.
        
        Args:
            table_name: Name of the table
            sample_rows: Number of sample rows to include
            
        Returns:
            TableInfo object or None if table doesn't exist
        """
        try:
            # Get column info
            schema_df = self.conn.execute(f"DESCRIBE {table_name}").fetchdf()
            columns = [
                {"name": row['column_name'], "type": row['column_type']}
                for _, row in schema_df.iterrows()
            ]
            
            # Get row count
            count_result = self.conn.execute(f"SELECT COUNT(*) as cnt FROM {table_name}").fetchone()
            row_count = count_result[0] if count_result else 0
            
            # Get sample data
            sample_df = self.conn.execute(f"SELECT * FROM {table_name} LIMIT {sample_rows}").fetchdf()
            sample_data = sample_df.to_dict('records')
            
            return TableInfo(
                table_name=table_name,
                document_type=table_name,  # Use table name as document type
                columns=columns,
                sample_data=sample_data,
                row_count=row_count
            )
            
        except Exception as e:
            logger.error(f"Failed to get table info for {table_name}: {e}")
            return None

