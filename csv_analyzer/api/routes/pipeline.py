"""
Pipeline Routes - End-to-end file processing with multi-file support.

This module provides the unified API for:
1. Uploading files (CSV, XLSX)
2. Smart analysis with LLM
3. Multi-table relationship detection
4. Natural language queries across all tables
"""

import uuid
import io
import time
import json
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict

import pandas as pd
import duckdb
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, status, Form
from pydantic import BaseModel

from csv_analyzer.core.config import get_settings
from csv_analyzer.storage.s3.client import get_s3_client

logger = logging.getLogger(__name__)
router = APIRouter()


# ============================================================================
# Data Models
# ============================================================================

class AnalyzedTable(BaseModel):
    """A table that has been analyzed."""
    table_id: str
    original_filename: str
    table_name: str  # DuckDB table name
    document_type: str
    description: str
    confidence: float
    row_count: int
    columns: List[Dict[str, Any]]
    suggested_insights: List[str]


class DetectedRelationship(BaseModel):
    """A detected relationship between tables."""
    from_table: str
    from_column: str
    to_table: str
    to_column: str
    relationship_type: str  # one_to_one, one_to_many, many_to_many
    confidence: float
    join_sql: str


class SessionState(BaseModel):
    """State of an analysis session."""
    session_id: str
    vertical: str
    tables: List[AnalyzedTable]
    relationships: List[DetectedRelationship]
    created_at: str
    status: str


class QueryRequest(BaseModel):
    """Request to run a natural language query."""
    question: str
    tables: Optional[List[str]] = None  # Specific tables to query, or all if None


class QueryResponse(BaseModel):
    """Response from a query."""
    sql: str
    explanation: str
    columns: List[str]
    data: List[Dict[str, Any]]
    row_count: int
    execution_time_ms: float


class InsightSuggestion(BaseModel):
    """A suggested insight across tables."""
    name: str
    description: str
    tables_involved: List[str]
    sql: str


# ============================================================================
# In-Memory Session Store (for demo - use Redis/PostgreSQL in production)
# ============================================================================

@dataclass
class Session:
    """An analysis session with loaded data."""
    session_id: str
    vertical: str
    duckdb_conn: Any  # DuckDB connection
    tables: Dict[str, AnalyzedTable] = field(default_factory=dict)
    relationships: List[DetectedRelationship] = field(default_factory=list)
    schemas: Dict[str, Any] = field(default_factory=dict)  # Inferred schemas
    created_at: float = field(default_factory=time.time)
    
    def to_state(self) -> SessionState:
        from datetime import datetime
        return SessionState(
            session_id=self.session_id,
            vertical=self.vertical,
            tables=list(self.tables.values()),
            relationships=self.relationships,
            created_at=datetime.fromtimestamp(self.created_at).isoformat(),
            status="active"
        )


# Session storage
_sessions: Dict[str, Session] = {}


def get_session(session_id: str) -> Session:
    """Get a session by ID."""
    if session_id not in _sessions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found"
        )
    return _sessions[session_id]


# ============================================================================
# Helper Functions
# ============================================================================

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean a DataFrame by removing invalid rows using STRUCTURAL analysis.
    
    Language-agnostic - works with any language by detecting:
    - Rows with Excel formulas (starting with =)
    - Rows that are mostly empty (>70% empty)
    - Rows where first 2 columns are empty (missing key identifiers)
    - Rows that look like outliers (very different from majority)
    """
    
    def has_formula(row):
        """Check if any cell contains a formula."""
        for val in row:
            if pd.isna(val):
                continue
            if str(val).strip().startswith('='):
                return True
        return False
    
    def is_mostly_empty(row, threshold=0.7):
        """Check if a row is mostly empty."""
        empty_count = sum(1 for v in row if pd.isna(v) or str(v).strip() == '')
        return empty_count / len(row) > threshold
    
    def has_key_values(row):
        """Check if first 2 columns have values (likely key identifiers)."""
        key_cols = list(row)[:2]
        non_empty = sum(1 for val in key_cols if not pd.isna(val) and str(val).strip() != '')
        return non_empty >= 2
    
    def count_filled(row):
        """Count non-empty cells."""
        return sum(1 for v in row if not pd.isna(v) and str(v).strip() != '')
    
    # Get original row count
    original_count = len(df)
    
    if original_count == 0:
        return df
    
    # Calculate average filled cells per row (for outlier detection)
    fill_counts = df.apply(count_filled, axis=1)
    avg_filled = fill_counts.mean()
    std_filled = fill_counts.std() if len(df) > 1 else 0
    
    def is_outlier_row(row):
        """Check if row has significantly different structure."""
        filled = count_filled(row)
        # Row is outlier if it has much fewer cells than average
        if std_filled > 0 and filled < avg_filled - 2 * std_filled:
            return True
        # Or if it has very few cells overall
        if filled < 2 and avg_filled > 5:
            return True
        return False
    
    # Filter rows
    mask = df.apply(
        lambda row: (
            not has_formula(row) and 
            not is_mostly_empty(row) and 
            has_key_values(row) and
            not is_outlier_row(row)
        ), 
        axis=1
    )
    cleaned_df = df[mask].copy()
    
    # Reset index
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    removed_count = original_count - len(cleaned_df)
    if removed_count > 0:
        logger.info(f"Cleaned DataFrame: removed {removed_count} invalid rows ({len(cleaned_df)} remaining)")
    
    return cleaned_df


async def analyze_dataframe(df: pd.DataFrame, vertical: str = None) -> Dict[str, Any]:
    """Analyze a DataFrame using the smart analyzer."""
    from csv_analyzer.intelligence.smart_analyzer import SmartAnalyzer
    
    # Clean the DataFrame first
    cleaned_df = clean_dataframe(df)
    
    analyzer = SmartAnalyzer()
    result = await analyzer.analyze(cleaned_df, vertical=vertical)
    
    return {
        "schema": result.inferred_schema,
        "processing_time_ms": result.processing_time_ms,
        "cleaned_df": cleaned_df  # Return cleaned DataFrame for table creation
    }


def create_table_in_duckdb(
    conn: duckdb.DuckDBPyConnection,
    df: pd.DataFrame,
    table_name: str,
    schema: Any
) -> None:
    """Create a table in DuckDB with normalized column names and proper types."""
    # Create a view with normalized names
    conn.register(f"_raw_{table_name}", df)
    
    # Build column renames with type casting
    rename_cols = []
    for col in schema.columns:
        safe_orig = f'"{col.original_name}"'
        safe_new = col.inferred_name.replace(" ", "_").lower()
        
        # Cast based on inferred data type
        if col.data_type in ("float", "currency", "number"):
            # Try to cast to DOUBLE, handling mixed types
            cast_expr = f'TRY_CAST({safe_orig} AS DOUBLE)'
        elif col.data_type == "integer":
            cast_expr = f'TRY_CAST({safe_orig} AS BIGINT)'
        elif col.data_type in ("date", "datetime"):
            cast_expr = f'TRY_CAST({safe_orig} AS TIMESTAMP)'
        elif col.data_type == "boolean":
            cast_expr = f'TRY_CAST({safe_orig} AS BOOLEAN)'
        else:
            cast_expr = f'CAST({safe_orig} AS VARCHAR)'
        
        rename_cols.append(f'{cast_expr} AS "{safe_new}"')
    
    # Create the table with normalized names and types
    create_sql = f"""
        CREATE OR REPLACE TABLE {table_name} AS 
        SELECT {', '.join(rename_cols)} 
        FROM _raw_{table_name}
    """
    conn.execute(create_sql)
    conn.execute(f"DROP VIEW IF EXISTS _raw_{table_name}")
    
    logger.info(f"Created table {table_name} with {len(df)} rows")


async def detect_relationships(
    conn: duckdb.DuckDBPyConnection,
    tables: Dict[str, AnalyzedTable],
    schemas: Dict[str, Any]
) -> List[DetectedRelationship]:
    """Detect relationships between tables using LLM."""
    from csv_analyzer.intelligence.smart_analyzer import SmartAnalyzer
    
    if len(tables) < 2:
        return []
    
    # Build schema descriptions for LLM
    table_descriptions = []
    for table_name, table_info in tables.items():
        schema = schemas.get(table_name)
        if schema:
            cols = [f"  - {c.inferred_name} ({c.semantic_type})" for c in schema.columns]
            table_descriptions.append(
                f"Table: {table_name}\n"
                f"Type: {table_info.document_type}\n"
                f"Columns:\n" + "\n".join(cols)
            )
    
    # Use LLM to detect relationships
    settings = get_settings()
    try:
        from openai import OpenAI
        client = OpenAI(api_key=settings.openai_api_key)
        
        system_prompt = """You are a database expert that identifies relationships between tables.
Given multiple table schemas, identify potential foreign key relationships.

Look for:
1. Columns with similar names (employee_id, user_id, etc.)
2. Columns with matching semantic types
3. Common patterns (one table has IDs that appear in another)

Respond in JSON:
{
    "relationships": [
        {
            "from_table": "table1",
            "from_column": "column1",
            "to_table": "table2", 
            "to_column": "column2",
            "relationship_type": "one_to_many",
            "confidence": 0.9,
            "reasoning": "Why these are related"
        }
    ]
}"""

        user_prompt = f"""Analyze these tables and identify relationships:

{chr(10).join(table_descriptions)}

Identify any foreign key relationships between these tables."""

        response = client.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        
        relationships = []
        for rel in result.get("relationships", []):
            # Generate join SQL
            join_sql = (
                f"SELECT * FROM {rel['from_table']} "
                f"JOIN {rel['to_table']} "
                f"ON {rel['from_table']}.{rel['from_column']} = "
                f"{rel['to_table']}.{rel['to_column']}"
            )
            
            relationships.append(DetectedRelationship(
                from_table=rel["from_table"],
                from_column=rel["from_column"],
                to_table=rel["to_table"],
                to_column=rel["to_column"],
                relationship_type=rel.get("relationship_type", "unknown"),
                confidence=rel.get("confidence", 0.5),
                join_sql=join_sql
            ))
        
        return relationships
        
    except Exception as e:
        logger.error(f"Relationship detection failed: {e}")
        return []


def get_duckdb_column_types(conn: duckdb.DuckDBPyConnection, table_name: str) -> Dict[str, str]:
    """Get actual column types from DuckDB."""
    try:
        result = conn.execute(f"DESCRIBE {table_name}").fetchall()
        return {row[0]: row[1] for row in result}
    except Exception:
        return {}


async def generate_cross_table_sql(
    session: Session,
    question: str,
    specific_tables: List[str] = None
) -> Dict[str, Any]:
    """Generate SQL that may span multiple tables."""
    from csv_analyzer.intelligence.smart_analyzer import SmartAnalyzer
    
    settings = get_settings()
    
    # Build schema context
    tables_to_use = specific_tables or list(session.tables.keys())
    
    schema_descriptions = []
    for table_name in tables_to_use:
        if table_name not in session.schemas:
            continue
        schema = session.schemas[table_name]
        
        # Get actual DuckDB types
        duckdb_types = get_duckdb_column_types(session.duckdb_conn, table_name)
        
        cols = []
        for c in schema.columns:
            col_name = c.inferred_name.replace(" ", "_").lower()
            actual_type = duckdb_types.get(col_name, c.data_type)
            cols.append(f"  - {col_name} (DuckDB type: {actual_type}): {c.description}")
        
        schema_descriptions.append(
            f"Table: {table_name}\n"
            f"Description: {schema.document_description}\n"
            f"Columns:\n" + "\n".join(cols)
        )
    
    # Add relationship context
    relationship_context = ""
    if session.relationships:
        rel_descriptions = []
        for rel in session.relationships:
            rel_descriptions.append(
                f"- {rel.from_table}.{rel.from_column} -> "
                f"{rel.to_table}.{rel.to_column} ({rel.relationship_type})"
            )
        relationship_context = (
            "\nKnown relationships:\n" + "\n".join(rel_descriptions)
        )
    
    try:
        from openai import OpenAI
        client = OpenAI(api_key=settings.openai_api_key)
        
        system_prompt = """You are a SQL expert that generates DuckDB-compatible SQL queries.
Given multiple table schemas and their relationships, generate SQL to answer the user's question.

CRITICAL RULES:
1. Use JOINs when data from multiple tables is needed
2. Use the relationships provided to construct correct JOIN conditions
3. Handle NULL values with COALESCE where appropriate
4. Use clear column aliases
5. If aggregating, include appropriate GROUP BY

TYPE CASTING (VERY IMPORTANT):
- Pay attention to the DuckDB type shown for each column
- If a column is VARCHAR but you need to aggregate it (SUM, AVG, etc.), use TRY_CAST:
  - TRY_CAST(column_name AS DOUBLE) for numeric operations
  - TRY_CAST returns NULL if cast fails, which is safe
- Example: AVG(TRY_CAST(salary AS DOUBLE)) instead of AVG(salary)
- Always check the DuckDB type and cast VARCHAR columns before numeric operations

Respond in JSON:
{
    "sql": "SELECT ...",
    "explanation": "What this query does",
    "tables_used": ["table1", "table2"],
    "confidence": 0.9
}"""

        user_prompt = f"""Available tables:

{chr(10).join(schema_descriptions)}
{relationship_context}

Question: {question}

Generate SQL to answer this question."""

        response = client.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        
        return json.loads(response.choices[0].message.content)
        
    except Exception as e:
        return {
            "sql": None,
            "explanation": f"Failed to generate SQL: {e}",
            "confidence": 0.0
        }


# ============================================================================
# API Routes
# ============================================================================

@router.post("/pipeline/sessions", response_model=SessionState)
async def create_pipeline_session(
    vertical: str = Form(default="general")
):
    """
    Create a new analysis session.
    
    The session will hold all uploaded files and their relationships.
    """
    session_id = str(uuid.uuid4())
    
    # Create DuckDB connection for this session
    conn = duckdb.connect(":memory:")
    
    session = Session(
        session_id=session_id,
        vertical=vertical,
        duckdb_conn=conn
    )
    
    _sessions[session_id] = session
    
    logger.info(f"Created session {session_id} for vertical '{vertical}'")
    return session.to_state()


@router.get("/pipeline/sessions/{session_id}", response_model=SessionState)
async def get_pipeline_session(session_id: str):
    """Get the current state of a session."""
    session = get_session(session_id)
    return session.to_state()


@router.post("/pipeline/sessions/{session_id}/upload", response_model=AnalyzedTable)
async def upload_and_analyze(
    session_id: str,
    file: UploadFile = File(...),
    table_name: Optional[str] = Form(default=None),
    start_row: Optional[int] = Form(default=None),
    end_row: Optional[int] = Form(default=None),
    sheet_name: Optional[str] = Form(default=None)
):
    """
    Upload a file, analyze it with LLM, and add to the session.
    
    Supports CSV and XLSX files. For XLSX:
    - Use start_row/end_row to extract a specific row range
    - Use sheet_name to specify which sheet to read
    - Without row range, uses smart detection (may include summary rows)
    """
    session = get_session(session_id)
    
    # Read file content
    content = await file.read()
    filename = file.filename or "uploaded_file"
    
    # Determine file type and load DataFrame
    if filename.endswith(".xlsx") or filename.endswith(".xls"):
        # Save to temp file (openpyxl/pandas need a file path)
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        
        try:
            if start_row is not None and end_row is not None:
                # User specified exact row range - use pandas directly
                start = int(start_row)
                end = int(end_row)
                df = pd.read_excel(
                    tmp_path,
                    sheet_name=sheet_name or 0,
                    skiprows=start - 1,
                    nrows=end - start,
                    header=0
                )
                logger.info(f"Extracted rows {start}-{end} ({len(df)} data rows)")
            else:
                # Use smart auto-detection
                from csv_analyzer.xlsx_smart_extractor import detect_main_table, extract_table
                
                detected = detect_main_table(tmp_path, sheet_name=sheet_name)
                
                if detected:
                    logger.info(f"Auto-detected table: rows {detected.start_row}-{detected.end_row} ({detected.data_row_count} rows)")
                    df = pd.read_excel(
                        tmp_path,
                        sheet_name=sheet_name or 0,
                        skiprows=detected.start_row - 1,
                        nrows=detected.data_row_count,
                        header=0
                    )
                else:
                    # Fallback to reading entire sheet
                    logger.warning("Could not auto-detect table, reading entire sheet")
                    df = pd.read_excel(tmp_path, sheet_name=sheet_name or 0)
        finally:
            import os
            os.unlink(tmp_path)
    
    elif filename.endswith(".csv"):
        df = pd.read_csv(io.BytesIO(content))
    
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type: {filename}"
        )
    
    # Generate table name if not provided
    if not table_name:
        base_name = filename.rsplit(".", 1)[0]
        # Sanitize for SQL
        table_name = "".join(c if c.isalnum() else "_" for c in base_name).lower()
        # Ensure unique
        counter = 1
        original_name = table_name
        while table_name in session.tables:
            table_name = f"{original_name}_{counter}"
            counter += 1
    
    # Analyze with LLM (includes data cleaning)
    logger.info(f"Analyzing {filename}...")
    analysis = await analyze_dataframe(df, vertical=session.vertical)
    schema = analysis["schema"]
    cleaned_df = analysis.get("cleaned_df", df)  # Use cleaned DataFrame
    
    # Create table in DuckDB with cleaned data
    create_table_in_duckdb(session.duckdb_conn, cleaned_df, table_name, schema)
    
    # Store schema
    session.schemas[table_name] = schema
    
    # Create analyzed table record
    analyzed_table = AnalyzedTable(
        table_id=str(uuid.uuid4()),
        original_filename=filename,
        table_name=table_name,
        document_type=schema.document_type,
        description=schema.document_description,
        confidence=schema.confidence,
        row_count=len(cleaned_df),  # Use cleaned row count
        columns=[
            {
                "original_name": c.original_name,
                "inferred_name": c.inferred_name,
                "data_type": c.data_type,
                "semantic_type": c.semantic_type,
                "description": c.description
            }
            for c in schema.columns
        ],
        suggested_insights=schema.suggested_insights
    )
    
    session.tables[table_name] = analyzed_table
    
    # Re-detect relationships if we have multiple tables
    if len(session.tables) > 1:
        logger.info("Detecting relationships between tables...")
        session.relationships = await detect_relationships(
            session.duckdb_conn,
            session.tables,
            session.schemas
        )
    
    # Upload to S3 for persistence
    try:
        s3 = get_s3_client()
        s3_key = f"raw/{session_id}/{filename}"
        s3.upload_bytes(content, s3_key)
        logger.info(f"Uploaded to S3: {s3_key}")
    except Exception as e:
        logger.warning(f"S3 upload failed (continuing): {e}")
    
    return analyzed_table


@router.post("/pipeline/sessions/{session_id}/query", response_model=QueryResponse)
async def query_session(
    session_id: str,
    request: QueryRequest
):
    """
    Ask a natural language question across all tables in the session.
    
    The LLM will generate SQL that may JOIN multiple tables if needed.
    """
    session = get_session(session_id)
    
    if not session.tables:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No tables uploaded to this session yet"
        )
    
    # Generate SQL
    logger.info(f"Generating SQL for: {request.question}")
    sql_result = await generate_cross_table_sql(
        session,
        request.question,
        request.tables
    )
    
    if not sql_result.get("sql"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not generate SQL: {sql_result.get('explanation')}"
        )
    
    # Execute SQL
    start_time = time.time()
    try:
        result_df = session.duckdb_conn.execute(sql_result["sql"]).fetchdf()
        execution_time = (time.time() - start_time) * 1000
        
        # Convert to response
        return QueryResponse(
            sql=sql_result["sql"],
            explanation=sql_result.get("explanation", ""),
            columns=list(result_df.columns),
            data=result_df.to_dict(orient="records"),
            row_count=len(result_df),
            execution_time_ms=execution_time
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"SQL execution failed: {e}\nSQL: {sql_result['sql']}"
        )


@router.post("/pipeline/sessions/{session_id}/sql")
async def execute_raw_sql(
    session_id: str,
    sql: str = Form(...)
):
    """
    Execute raw SQL directly on the session's data.
    
    Use this for custom queries or to refine LLM-generated SQL.
    """
    session = get_session(session_id)
    
    start_time = time.time()
    try:
        result_df = session.duckdb_conn.execute(sql).fetchdf()
        execution_time = (time.time() - start_time) * 1000
        
        return {
            "columns": list(result_df.columns),
            "data": result_df.to_dict(orient="records"),
            "row_count": len(result_df),
            "execution_time_ms": execution_time
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"SQL execution failed: {e}"
        )


@router.get("/pipeline/sessions/{session_id}/relationships")
async def get_relationships(session_id: str):
    """Get detected relationships between tables."""
    session = get_session(session_id)
    return {
        "relationships": [r.dict() for r in session.relationships]
    }


@router.get("/pipeline/sessions/{session_id}/insights")
async def suggest_insights(session_id: str):
    """Get suggested insights across all tables."""
    session = get_session(session_id)
    
    if not session.tables:
        return {"insights": []}
    
    # Collect per-table insights
    insights = []
    for table_name, table_info in session.tables.items():
        for insight in table_info.suggested_insights:
            insights.append({
                "description": insight,
                "tables_involved": [table_name],
                "type": "single_table"
            })
    
    # Generate cross-table insights if we have relationships
    if session.relationships:
        settings = get_settings()
        try:
            from openai import OpenAI
            client = OpenAI(api_key=settings.openai_api_key)
            
            # Build context
            tables_desc = []
            for t in session.tables.values():
                tables_desc.append(f"- {t.table_name}: {t.description}")
            
            rels_desc = []
            for r in session.relationships:
                rels_desc.append(
                    f"- {r.from_table}.{r.from_column} -> "
                    f"{r.to_table}.{r.to_column}"
                )
            
            response = client.chat.completions.create(
                model=settings.llm_model,
                messages=[
                    {"role": "system", "content": """Suggest cross-table analytical insights.
Given tables and their relationships, suggest 3-5 insights that JOIN multiple tables.

Respond in JSON:
{
    "insights": [
        {"description": "...", "tables": ["t1", "t2"]}
    ]
}"""},
                    {"role": "user", "content": f"""Tables:
{chr(10).join(tables_desc)}

Relationships:
{chr(10).join(rels_desc)}

Suggest cross-table insights."""}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            for insight in result.get("insights", []):
                insights.append({
                    "description": insight["description"],
                    "tables_involved": insight.get("tables", []),
                    "type": "cross_table"
                })
                
        except Exception as e:
            logger.warning(f"Cross-table insight generation failed: {e}")
    
    return {"insights": insights}


@router.get("/pipeline/sessions/{session_id}/schema")
async def get_session_schema(session_id: str):
    """Get the full schema for all tables in the session."""
    session = get_session(session_id)
    
    schemas = {}
    for table_name, table_info in session.tables.items():
        schemas[table_name] = {
            "document_type": table_info.document_type,
            "description": table_info.description,
            "row_count": table_info.row_count,
            "columns": table_info.columns
        }
    
    return {"schemas": schemas}


@router.get("/pipeline/sessions/{session_id}/predefined-insights")
async def list_predefined_insights(session_id: str):
    """
    List all predefined insights and whether they can run on current data.
    
    Predefined insights are YAML files in csv_analyzer/insights/definitions/
    """
    from pathlib import Path
    import yaml
    
    session = get_session(session_id)
    
    # Load all insight definitions
    definitions_dir = Path(__file__).parent.parent.parent / "insights" / "definitions"
    
    insights = []
    for yaml_file in definitions_dir.glob("*.yaml"):
        try:
            with open(yaml_file, 'r', encoding='utf-8') as f:
                definition = yaml.safe_load(f)
            
            required_types = definition.get("requires", [])
            loaded_types = [t.document_type for t in session.tables.values()]
            
            # Check if at least ONE required type is loaded (any match = can run)
            # This allows insights to define multiple acceptable document types
            matching = [r for r in required_types if r in loaded_types]
            missing = [r for r in required_types if r not in loaded_types]
            can_run = len(matching) > 0
            
            insights.append({
                "name": definition.get("name"),
                "description": definition.get("description"),
                "category": definition.get("category"),
                "requires": required_types,
                "parameters": definition.get("parameters", []),
                "can_run": can_run,
                "missing_types": missing,
                "file": yaml_file.name
            })
        except Exception as e:
            logger.warning(f"Failed to load insight {yaml_file}: {e}")
    
    return {
        "loaded_document_types": list(set(t.document_type for t in session.tables.values())),
        "insights": insights
    }


@router.post("/pipeline/sessions/{session_id}/predefined-insights/{insight_name}/run")
async def run_predefined_insight(
    session_id: str,
    insight_name: str,
    parameters: Dict[str, Any] = {}
):
    """
    Run a predefined insight on the session data.
    
    The insight SQL will be adapted to match the actual column names in DuckDB.
    """
    from pathlib import Path
    import yaml
    import time
    
    session = get_session(session_id)
    
    # Find the insight definition
    definitions_dir = Path(__file__).parent.parent.parent / "insights" / "definitions"
    insight_file = definitions_dir / f"{insight_name}.yaml"
    
    if not insight_file.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Insight '{insight_name}' not found"
        )
    
    with open(insight_file, 'r', encoding='utf-8') as f:
        definition = yaml.safe_load(f)
    
    # Get current table schemas (include ALL columns, not just first 10)
    schemas = []
    for table_name, table_info in session.tables.items():
        col_info = ", ".join([f"{c['inferred_name']} ({c['data_type']})" for c in table_info.columns])
        schemas.append(f"Table '{table_name}': {col_info}")
    
    # Use LLM to adapt the SQL to actual schema
    from openai import OpenAI
    settings = get_settings()
    client = OpenAI(api_key=settings.openai_api_key)
    
    response = client.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": """You are a SQL expert. Adapt the given insight SQL to work with the actual table schema.

CRITICAL RULES:
1. Use EXACTLY the table names provided (e.g., if table is 'employees', use FROM employees)
2. Use EXACTLY the column names provided in the schema
3. Keep the same logic and structure
4. Use COALESCE for null safety
5. If a column doesn't exist, skip it or use 0

Return ONLY valid DuckDB SQL, no explanations or markdown."""},
            {"role": "user", "content": f"""Insight: {definition.get('description')}

Original SQL template:
{definition.get('sql')}

Actual tables in DuckDB:
{chr(10).join(schemas)}

Parameters provided: {json.dumps(parameters)}
Default parameter values: {json.dumps({p['name']: p.get('default') for p in definition.get('parameters', []) if p.get('default')})}

IMPORTANT: 
1. Replace all {{{{param}}}} placeholders with actual values (use defaults if not provided)
2. Use EXACTLY the table and column names from the actual schema above
3. Output ONLY executable SQL, no placeholders remaining"""}
        ],
        temperature=0.1
    )
    
    adapted_sql = response.choices[0].message.content.strip()
    
    # Clean up SQL (remove markdown code blocks if present)
    if adapted_sql.startswith("```"):
        adapted_sql = adapted_sql.split("\n", 1)[1]
        if adapted_sql.endswith("```"):
            adapted_sql = adapted_sql.rsplit("```", 1)[0]
    
    # Execute
    start_time = time.time()
    try:
        result_df = session.duckdb_conn.execute(adapted_sql).fetchdf()
        execution_time = (time.time() - start_time) * 1000
        
        return {
            "insight_name": insight_name,
            "description": definition.get("description"),
            "adapted_sql": adapted_sql,
            "columns": result_df.columns.tolist(),
            "data": result_df.to_dict(orient="records"),
            "row_count": len(result_df),
            "execution_time_ms": round(execution_time, 2)
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"SQL execution failed: {str(e)}\n\nSQL:\n{adapted_sql}"
        )


@router.delete("/pipeline/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session and clean up resources."""
    session = get_session(session_id)
    
    # Close DuckDB connection
    session.duckdb_conn.close()
    
    # Remove from storage
    del _sessions[session_id]
    
    return {"status": "deleted", "session_id": session_id}

