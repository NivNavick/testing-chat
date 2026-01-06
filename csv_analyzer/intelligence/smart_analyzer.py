"""
Smart Analyzer - LLM-powered dynamic file understanding without ground truth.

This module provides zero-shot classification and schema inference using LLMs.
It's the fallback when no ground truth examples exist, or for novel file types.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum

import pandas as pd

from csv_analyzer.core.config import get_settings

logger = logging.getLogger(__name__)


class AnalysisMode(Enum):
    """How the analysis was performed."""
    ZERO_SHOT = "zero_shot"  # Pure LLM, no ground truth
    HYBRID = "hybrid"  # LLM + embeddings
    GROUND_TRUTH = "ground_truth"  # Pure embedding match


@dataclass
class InferredColumn:
    """Schema information inferred for a column."""
    original_name: str
    inferred_name: str  # Normalized/English name
    data_type: str  # string, number, date, boolean, etc.
    semantic_type: str  # employee_id, date, amount, name, etc.
    description: str
    sample_values: List[Any]
    nullable: bool
    is_key: bool = False  # Potential primary/foreign key
    related_to: Optional[str] = None  # If FK, what it might link to


@dataclass
class InferredSchema:
    """Complete schema inferred from data."""
    document_type: str
    document_description: str
    columns: List[InferredColumn]
    confidence: float
    relationships: List[Dict[str, str]]  # Potential relationships with other tables
    suggested_insights: List[str]  # What questions can we answer with this data


@dataclass
class SmartAnalysisResult:
    """Complete result from smart analysis."""
    mode: AnalysisMode
    inferred_schema: InferredSchema
    raw_llm_response: Optional[str] = None
    processing_time_ms: float = 0.0
    warnings: List[str] = field(default_factory=list)


class SmartAnalyzer:
    """
    LLM-powered analyzer for unknown files.
    
    Works without ground truth by using LLMs to:
    1. Understand the file structure and content
    2. Infer document types and column meanings
    3. Suggest relationships and insights
    
    Usage:
        analyzer = SmartAnalyzer()
        result = await analyzer.analyze(df)
        print(result.inferred_schema.document_type)
        print(result.inferred_schema.columns)
    """
    
    def __init__(
        self,
        model: str = None,
        api_key: str = None,
        temperature: float = 0.1
    ):
        self.settings = get_settings()
        self.model = model or self.settings.llm_model
        self.api_key = api_key or self.settings.openai_api_key
        self.temperature = temperature
        self._client = None
    
    def _get_client(self):
        """Lazy load OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError("openai package not installed. Run: pip install openai")
        return self._client
    
    def _profile_dataframe(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Create a profile of the DataFrame for LLM analysis."""
        profile = {
            "row_count": len(df),
            "column_count": len(df.columns),
            "columns": []
        }
        
        for col in df.columns:
            col_data = df[col]
            non_null = col_data.dropna()
            
            # Get sample values (up to 5, unique)
            samples = non_null.head(20).unique().tolist()[:5]
            # Convert to strings for serialization
            samples = [str(s)[:100] for s in samples]
            
            col_profile = {
                "name": str(col),
                "dtype": str(col_data.dtype),
                "non_null_count": len(non_null),
                "null_count": col_data.isna().sum(),
                "unique_count": col_data.nunique(),
                "sample_values": samples
            }
            
            # Add numeric stats if applicable
            if pd.api.types.is_numeric_dtype(col_data) and len(non_null) > 0:
                try:
                    col_profile["min"] = float(non_null.min())
                    col_profile["max"] = float(non_null.max())
                    col_profile["mean"] = float(non_null.mean())
                except (TypeError, ValueError):
                    pass  # Skip stats for columns that can't be converted
            
            profile["columns"].append(col_profile)
        
        return profile
    
    def _build_analysis_prompt(
        self,
        profile: Dict[str, Any],
        context: Optional[str] = None,
        vertical: Optional[str] = None
    ) -> Tuple[str, str]:
        """Build the system and user prompts for analysis."""
        
        system_prompt = """You are an expert data analyst that understands tabular data.
Your task is to analyze a dataset and infer its schema, document type, and meaning.

You will receive:
1. Column profiles with names, types, and sample values
2. Optional context about the business domain

Your job is to:
1. Identify what kind of document/data this is (e.g., employee_shifts, sales_orders, patient_records)
2. Understand what each column represents semantically
3. Suggest normalized English names for columns (especially for non-English column names)
4. Identify potential key columns and relationships
5. Suggest useful insights/questions that could be answered with this data

IMPORTANT:
- Column names may be in Hebrew, Arabic, or other languages - translate/normalize them to English
- Infer semantic meaning from both column names AND sample values
- Be specific about data types (string, integer, float, date, datetime, boolean, currency)
- Identify semantic types (employee_id, date, amount, name, phone, email, etc.)

COLUMN NAMING RULES:
- If a column name is a number like "12.25", "1.25", etc., it's likely a MONTH.YEAR format (Dec 2025, Jan 2025)
  → Name it descriptively like "salary_dec_25", "payment_jan_25", or "hours_dec_25" based on context
- If column looks like a rate/multiplier (e.g., "1.25", "2.25" in context of payroll), it might be shift rate multipliers
  → Name it like "rate_125_percent" or "overtime_rate"
- Always make column names descriptive - avoid "unknown_column" or generic names
- Use snake_case for all inferred names

Respond in JSON format with this exact structure:
{
    "document_type": "lowercase_snake_case_name",
    "document_description": "One sentence describing what this data represents",
    "confidence": 0.0-1.0,
    "columns": [
        {
            "original_name": "original column name",
            "inferred_name": "normalized_english_name",
            "data_type": "string|integer|float|date|datetime|boolean|currency",
            "semantic_type": "employee_id|name|date|amount|description|etc",
            "description": "What this column represents",
            "is_key": true/false,
            "related_to": "table_name.column if this might be a foreign key, null otherwise"
        }
    ],
    "relationships": [
        {"description": "This table might relate to X table via Y column"}
    ],
    "suggested_insights": [
        "What is the total X per Y?",
        "How does Z change over time?"
    ]
}"""

        # Build the user prompt with the data profile
        columns_text = []
        for col in profile["columns"]:
            col_text = f"""
Column: "{col['name']}"
  - Type: {col['dtype']}
  - Non-null: {col['non_null_count']}/{profile['row_count']} ({100*col['non_null_count']/max(profile['row_count'],1):.0f}%)
  - Unique values: {col['unique_count']}
  - Samples: {col['sample_values']}"""
            if "min" in col and col.get("mean") is not None:
                col_text += f"\n  - Range: {col['min']} to {col['max']} (mean: {col['mean']:.2f})"
            columns_text.append(col_text)
        
        # Detect if columns look like month.year patterns
        date_like_cols = [col['name'] for col in profile["columns"] 
                         if isinstance(col['name'], (int, float)) or 
                         (isinstance(col['name'], str) and 
                          any(c.isdigit() for c in col['name']) and 
                          '.' in str(col['name']))]
        
        date_hint = ""
        if date_like_cols:
            date_hint = f"""
NOTE: Columns {date_like_cols[:5]}{'...' if len(date_like_cols) > 5 else ''} look like month.year patterns (e.g., 12.25 = December 2025, 1.25 = January 2025).
Name these columns descriptively based on what the numeric values represent (salary, hours, payments, etc.)."""

        user_prompt = f"""Analyze this dataset:

Total rows: {profile['row_count']}
Total columns: {profile['column_count']}

{"Business domain: " + vertical if vertical else ""}
{"Additional context: " + context if context else ""}
{date_hint}

COLUMNS:
{''.join(columns_text)}

Analyze this data and provide the schema inference in JSON format."""

        return system_prompt, user_prompt
    
    async def analyze(
        self,
        df: pd.DataFrame,
        context: Optional[str] = None,
        vertical: Optional[str] = None
    ) -> SmartAnalysisResult:
        """
        Analyze a DataFrame and infer its schema using LLM.
        
        Args:
            df: DataFrame to analyze
            context: Optional context about what this data might be
            vertical: Optional business vertical (medical, finance, etc.)
            
        Returns:
            SmartAnalysisResult with inferred schema
        """
        import time
        start_time = time.time()
        
        # Profile the data
        profile = self._profile_dataframe(df)
        logger.info(f"Profiled DataFrame: {profile['row_count']} rows, {profile['column_count']} columns")
        
        # Build prompts
        system_prompt, user_prompt = self._build_analysis_prompt(
            profile, context, vertical
        )
        
        # Call LLM
        client = self._get_client()
        
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                response_format={"type": "json_object"}
            )
            
            raw_response = response.choices[0].message.content
            result = json.loads(raw_response)
            
            # Parse the response into our data structures
            columns = []
            for col in result.get("columns", []):
                # Find sample values from profile
                profile_col = next(
                    (c for c in profile["columns"] if c["name"] == col.get("original_name")),
                    {}
                )
                
                columns.append(InferredColumn(
                    original_name=col.get("original_name", ""),
                    inferred_name=col.get("inferred_name", col.get("original_name", "")),
                    data_type=col.get("data_type", "string"),
                    semantic_type=col.get("semantic_type", "unknown"),
                    description=col.get("description", ""),
                    sample_values=profile_col.get("sample_values", []),
                    nullable=profile_col.get("null_count", 0) > 0,
                    is_key=col.get("is_key", False),
                    related_to=col.get("related_to")
                ))
            
            schema = InferredSchema(
                document_type=result.get("document_type", "unknown"),
                document_description=result.get("document_description", ""),
                columns=columns,
                confidence=result.get("confidence", 0.5),
                relationships=result.get("relationships", []),
                suggested_insights=result.get("suggested_insights", [])
            )
            
            processing_time = (time.time() - start_time) * 1000
            
            return SmartAnalysisResult(
                mode=AnalysisMode.ZERO_SHOT,
                inferred_schema=schema,
                raw_llm_response=raw_response,
                processing_time_ms=processing_time
            )
            
        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            
            # Return a basic fallback schema
            columns = [
                InferredColumn(
                    original_name=str(col),
                    inferred_name=str(col),
                    data_type="string",
                    semantic_type="unknown",
                    description="Unable to infer",
                    sample_values=[],
                    nullable=True
                )
                for col in df.columns
            ]
            
            return SmartAnalysisResult(
                mode=AnalysisMode.ZERO_SHOT,
                inferred_schema=InferredSchema(
                    document_type="unknown",
                    document_description="Analysis failed",
                    columns=columns,
                    confidence=0.0,
                    relationships=[],
                    suggested_insights=[]
                ),
                warnings=[str(e)],
                processing_time_ms=(time.time() - start_time) * 1000
            )
    
    async def generate_sql_for_insight(
        self,
        table_name: str,
        schema: InferredSchema,
        question: str
    ) -> Dict[str, Any]:
        """
        Generate SQL to answer a question about the data.
        
        Args:
            table_name: Name of the table in DuckDB
            schema: Inferred schema of the table
            question: Natural language question
            
        Returns:
            Dict with sql, explanation, and confidence
        """
        client = self._get_client()
        
        # Build schema description
        columns_desc = []
        for col in schema.columns:
            columns_desc.append(
                f"  - {col.inferred_name} ({col.data_type}): {col.description}"
            )
        
        system_prompt = """You are a SQL expert. Generate DuckDB-compatible SQL queries.
Given a table schema and a question, generate SQL to answer it.

Rules:
1. Use the column names exactly as provided
2. Handle NULL values appropriately
3. Use standard SQL aggregations
4. Add clear column aliases

Respond in JSON:
{
    "sql": "SELECT ...",
    "explanation": "This query...",
    "confidence": 0.0-1.0
}"""

        user_prompt = f"""Table: {table_name}
Document type: {schema.document_type}
Description: {schema.document_description}

Columns:
{chr(10).join(columns_desc)}

Question: {question}

Generate SQL to answer this question."""

        try:
            response = client.chat.completions.create(
                model=self.model,
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
    
    async def suggest_column_mappings(
        self,
        source_schema: InferredSchema,
        target_schema: Dict[str, Any]
    ) -> Dict[str, str]:
        """
        Suggest mappings from source columns to a target schema.
        
        Args:
            source_schema: Inferred schema from the data
            target_schema: Target schema definition
            
        Returns:
            Dict mapping source column names to target field names
        """
        client = self._get_client()
        
        source_cols = [
            f"{c.inferred_name} ({c.semantic_type}): {c.description}"
            for c in source_schema.columns
        ]
        
        target_fields = []
        for field in target_schema.get("fields", []):
            target_fields.append(
                f"{field['name']} ({field.get('type', 'string')}): {field.get('description', '')}"
            )
        
        system_prompt = """You are a data mapping expert.
Given source columns and target schema fields, suggest the best mappings.

Only map columns that have a clear semantic match.
If no good match exists, don't map the column.

Respond in JSON:
{
    "mappings": {
        "source_column_name": "target_field_name",
        ...
    },
    "unmapped_source": ["col1", "col2"],
    "reasoning": "Brief explanation of mappings"
}"""

        user_prompt = f"""Source columns:
{chr(10).join(source_cols)}

Target schema fields:
{chr(10).join(target_fields)}

Suggest the best column mappings."""

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            return result.get("mappings", {})
            
        except Exception as e:
            logger.error(f"Column mapping failed: {e}")
            return {}


# Convenience function for quick analysis
async def analyze_dataframe(
    df: pd.DataFrame,
    vertical: str = None
) -> SmartAnalysisResult:
    """
    Quick analysis of a DataFrame.
    
    Args:
        df: DataFrame to analyze
        vertical: Optional business vertical
        
    Returns:
        SmartAnalysisResult
    """
    analyzer = SmartAnalyzer()
    return await analyzer.analyze(df, vertical=vertical)

