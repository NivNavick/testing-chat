"""
Dynamic Insight Generator - Uses LLM to generate SQL from natural language.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from csv_analyzer.core.config import get_settings
from csv_analyzer.storage.duckdb.session_db import SessionDatabase


@dataclass
class TableSchema:
    """Schema information for a table."""
    name: str
    document_type: Optional[str]
    columns: List[Dict[str, str]]  # [{"name": ..., "type": ...}]
    row_count: int


@dataclass
class GeneratedInsight:
    """A dynamically generated insight."""
    name: str
    description: str
    sql: str
    explanation: str
    requires_tables: List[str]
    confidence: float


class DynamicInsightGenerator:
    """
    Generates insights using LLM based on natural language descriptions.
    """
    
    def __init__(self, db: SessionDatabase = None):
        self.db = db
        self.settings = get_settings()
        self._client = None
    
    def _get_openai_client(self):
        """Lazy load OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.settings.openai_api_key)
            except ImportError:
                return None
        return self._client
    
    def _get_table_schemas(self) -> List[TableSchema]:
        """Get schemas for all loaded tables."""
        schemas = []
        
        if self.db is None:
            return schemas
        
        for table_name, table_info in self.db.loaded_tables.items():
            try:
                col_info = self.db.get_table_schema(table_name)
                schemas.append(TableSchema(
                    name=table_name,
                    document_type=table_info.document_type,
                    columns=col_info,
                    row_count=table_info.row_count
                ))
            except Exception:
                continue
        
        return schemas
    
    def _build_schema_prompt(self, schemas: List[TableSchema]) -> str:
        """Build schema description for the prompt."""
        lines = ["Available tables and their schemas:\n"]
        
        for schema in schemas:
            lines.append(f"\nTable: {schema.name}")
            if schema.document_type:
                lines.append(f"  Type: {schema.document_type}")
            lines.append(f"  Rows: {schema.row_count}")
            lines.append("  Columns:")
            
            for col in schema.columns:
                lines.append(f"    - {col['name']} ({col['type']})")
        
        return "\n".join(lines)
    
    async def generate_insight(
        self,
        description: str,
        table_ids: List[str] = None
    ) -> GeneratedInsight:
        """
        Generate an insight from natural language description.
        
        Args:
            description: Natural language description of what to analyze
            table_ids: Optional list of specific tables to use
            
        Returns:
            GeneratedInsight with SQL and metadata
        """
        client = self._get_openai_client()
        if client is None:
            raise ValueError("OpenAI client not available")
        
        schemas = self._get_table_schemas()
        if not schemas:
            raise ValueError("No tables loaded")
        
        # Filter to specific tables if provided
        if table_ids:
            schemas = [s for s in schemas if s.name in table_ids]
        
        schema_prompt = self._build_schema_prompt(schemas)
        
        system_prompt = """You are a SQL expert that generates DuckDB-compatible SQL queries.
Given table schemas and a user's analysis request, generate a SQL query that answers their question.

Rules:
1. Only use tables and columns that exist in the provided schemas
2. Use standard SQL syntax compatible with DuckDB
3. Include appropriate aggregations and groupings
4. Use clear column aliases
5. Handle NULL values appropriately

Respond in JSON format:
{
    "name": "short_name_for_insight",
    "description": "one-line description",
    "sql": "SELECT ...",
    "explanation": "brief explanation of what the query does",
    "tables_used": ["table1", "table2"]
}"""

        user_prompt = f"""{schema_prompt}

User's analysis request: {description}

Generate a SQL query to answer this request."""

        try:
            response = client.chat.completions.create(
                model=self.settings.llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,
                response_format={"type": "json_object"}
            )
            
            import json
            result = json.loads(response.choices[0].message.content)
            
            return GeneratedInsight(
                name=result.get("name", "generated_insight"),
                description=result.get("description", description),
                sql=result.get("sql", ""),
                explanation=result.get("explanation", ""),
                requires_tables=result.get("tables_used", []),
                confidence=0.8
            )
            
        except Exception as e:
            raise ValueError(f"Failed to generate insight: {e}")
    
    async def suggest_insights(
        self,
        max_suggestions: int = 5
    ) -> List[GeneratedInsight]:
        """
        Suggest interesting insights based on the loaded data.
        
        Args:
            max_suggestions: Maximum number of suggestions
            
        Returns:
            List of suggested insights
        """
        client = self._get_openai_client()
        if client is None:
            return []
        
        schemas = self._get_table_schemas()
        if not schemas:
            return []
        
        schema_prompt = self._build_schema_prompt(schemas)
        
        system_prompt = f"""You are a data analyst expert.
Given table schemas, suggest {max_suggestions} interesting analytical queries 
that would provide business value.

Focus on:
1. Aggregations and summaries
2. Trends and patterns
3. Outliers and anomalies
4. Comparisons between groups
5. Key metrics and KPIs

Respond in JSON format:
{{
    "suggestions": [
        {{
            "name": "short_name",
            "description": "one-line description",
            "sql": "SELECT ...",
            "explanation": "why this is useful"
        }}
    ]
}}"""

        user_prompt = f"""{schema_prompt}

Suggest {max_suggestions} interesting analytical queries for this data."""

        try:
            response = client.chat.completions.create(
                model=self.settings.llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.5,
                response_format={"type": "json_object"}
            )
            
            import json
            result = json.loads(response.choices[0].message.content)
            
            suggestions = []
            for s in result.get("suggestions", []):
                suggestions.append(GeneratedInsight(
                    name=s.get("name", "suggestion"),
                    description=s.get("description", ""),
                    sql=s.get("sql", ""),
                    explanation=s.get("explanation", ""),
                    requires_tables=[],
                    confidence=0.6
                ))
            
            return suggestions
            
        except Exception:
            return []
    
    def validate_sql(self, sql: str) -> tuple:
        """
        Validate SQL syntax using DuckDB.
        
        Args:
            sql: SQL query to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if self.db is None:
            return False, "No database connection"
        
        try:
            # Use EXPLAIN to validate without executing
            self.db.connection.execute(f"EXPLAIN {sql}")
            return True, None
        except Exception as e:
            return False, str(e)

