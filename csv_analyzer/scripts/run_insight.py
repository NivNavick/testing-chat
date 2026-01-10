#!/usr/bin/env python3
"""
Insight Runner Script

Processes CSV files, canonizes columns to schema field names, and runs insights.

Uses:
- SchemaEmbeddingsService (ChromaDB) for column-to-schema matching
- OpenAI for document type verification
- Schema YAML for canonization (Hebrew → English canonical names)

This ensures insights work regardless of customer's column naming (Hebrew, English, etc.)

Usage:
    python run_insight.py --insight early_arrival --files shifts.csv actions.csv --output report.csv
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import pandas as pd
import yaml

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from csv_analyzer.openai_client import get_openai_client
from csv_analyzer.preprocessing import PreprocessingPipeline
from csv_analyzer.columns_analyzer import profile_dataframe
from csv_analyzer.core.schema_registry import get_schema_registry
from csv_analyzer.core.schema_embeddings import SchemaEmbeddingsService
from csv_analyzer.multilingual_embeddings_client import get_multilingual_embeddings_client

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class InsightParameter:
    """Represents a parameter for an insight."""
    def __init__(self, data: Dict[str, Any]):
        self.name = data.get("name", "")
        self.type = data.get("type", "string")
        self.required = data.get("required", False)
        self.default = data.get("default")
        self.description = data.get("description", "")


class InsightDefinition:
    """Represents an insight loaded from YAML."""
    
    def __init__(self, data: Dict[str, Any]):
        self.name = data.get("name", "unknown")
        self.display_name = data.get("display_name", self.name)
        self.description = data.get("description", "")
        self.requires = data.get("requires", [])
        self.type = data.get("type", "sql")  # 'sql' or 'code'
        self.sql = data.get("sql")
        self.handler = data.get("handler")  # For code insights
        self.output_columns = data.get("output_columns", [])
        # Parse parameters as InsightParameter objects
        self.parameters = [
            InsightParameter(p) if isinstance(p, dict) else p
            for p in data.get("parameters", [])
        ]
    
    @classmethod
    def load(cls, insight_name: str, definitions_dir: Optional[Path] = None) -> "InsightDefinition":
        """Load insight definition from YAML file."""
        if definitions_dir is None:
            definitions_dir = Path(__file__).parent.parent / "insights" / "definitions"
        
        yaml_path = definitions_dir / f"{insight_name}.yaml"
        
        if not yaml_path.exists():
            raise FileNotFoundError(f"Insight definition not found: {yaml_path}")
        
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        return cls(data)


class SchemaCanonizer:
    """
    Canonizes CSV columns to schema field names using embeddings.
    
    This ensures that regardless of customer's column naming (Hebrew, English, Spanish...),
    columns are mapped to canonical English schema field names.
    
    Example:
        Customer column: כניסה (Hebrew for "entry")
        Schema field: shift_start
        Result: Column renamed to shift_start
    """
    
    def __init__(
        self,
        vertical: str = "medical",
        openai_client=None,
    ):
        self.vertical = vertical
        self.openai_client = openai_client
        
        # Initialize embeddings client
        logger.info("Initializing multilingual embeddings client...")
        self.embeddings_client = get_multilingual_embeddings_client()
        
        # Initialize schema registry
        logger.info("Loading schema registry...")
        self.schema_registry = get_schema_registry()
        
        # Initialize schema embeddings service (ChromaDB)
        logger.info("Initializing schema embeddings service...")
        self.schema_service = SchemaEmbeddingsService(
            embeddings_client=self.embeddings_client,
            schema_registry=self.schema_registry,
        )
        
        # Index all schemas (builds embeddings in ChromaDB)
        logger.info(f"Indexing schema embeddings for vertical: {vertical}")
        self.schema_service.index_all_schemas(force_reindex=True)
    
    def classify_and_canonize(
        self,
        df: pd.DataFrame,
        threshold: float = 0.5,
    ) -> Tuple[Optional[str], pd.DataFrame, Dict[str, str]]:
        """
        Classify document type and canonize column names to schema fields.
        
        Args:
            df: DataFrame to classify and canonize
            threshold: Minimum similarity threshold for column mapping
            
        Returns:
            Tuple of (document_type, canonized_df, column_mappings)
        """
        # Profile columns
        column_profiles = profile_dataframe(df)
        logger.info(f"Profiled {len(column_profiles)} columns")
        
        # Score columns against all schemas
        result = self.schema_service.score_columns_against_schemas(
            columns=column_profiles,
            vertical=self.vertical,
        )
        
        doc_type_scores = result["document_type_scores"]
        column_matches = result["column_matches"]
        
        if not doc_type_scores:
            logger.warning("No document type scores returned")
            return None, df, {}
        
        # Log all scores
        logger.info("Document type scores (from embeddings):")
        sorted_scores = sorted(
            doc_type_scores.items(),
            key=lambda x: x[1].get("score", 0),
            reverse=True
        )
        for dt, scores in sorted_scores[:3]:
            logger.info(f"  {dt}: {scores.get('score', 0):.3f}")
        
        # Get all candidates for OpenAI verification
        all_candidates = [(dt, info.get("score", 0)) for dt, info in sorted_scores]
        
        # Use OpenAI to verify document type (more accurate than embeddings alone)
        doc_type = self._verify_document_type(df, all_candidates, column_profiles)
        
        logger.info(f"Final document type: {doc_type}")
        
        # Build column mappings for the detected document type
        mappings = {}
        for col_name, matches in column_matches.items():
            if matches:
                # Filter matches for this document type
                doc_matches = [m for m in matches if m.get("document_type") == doc_type]
                if doc_matches:
                    best_match = doc_matches[0]
                    if best_match["similarity"] >= threshold:
                        canonical_name = best_match["field_name"]
                        mappings[col_name] = canonical_name
        
        # Canonize: rename columns to schema field names
        df_canonized = df.copy()
        for orig_col, canonical_col in mappings.items():
            if orig_col in df_canonized.columns and orig_col != canonical_col:
                # Avoid duplicate column names
                if canonical_col not in df_canonized.columns:
                    df_canonized = df_canonized.rename(columns={orig_col: canonical_col})
                    logger.debug(f"  Canonized: {orig_col} → {canonical_col}")
        
        return doc_type, df_canonized, mappings
    
    def _verify_document_type(
        self,
        df: pd.DataFrame,
        candidates: List[Tuple[str, float]],
        column_profiles: List[Dict],
    ) -> str:
        """Use OpenAI to verify document type based on column names and values."""
        if not self.openai_client or not candidates:
            return candidates[0][0] if candidates else None
        
        # Build column summary
        col_summaries = []
        for profile in column_profiles[:15]:
            col_name = profile.get("column_name", "unknown")
            col_type = profile.get("detected_type", "unknown")
            samples = profile.get("sample_values", [])[:3]
            col_summaries.append(f"- {col_name} ({col_type}): {samples}")
        
        columns_text = "\n".join(col_summaries)
        candidates_text = ", ".join([f"{dt} ({score:.2f})" for dt, score in candidates])
        
        # Get document type descriptions from schema
        doc_descriptions = []
        all_valid_types = []
        for schema in self.schema_registry.get_schemas_by_vertical(self.vertical):
            all_valid_types.append(schema.name)
            desc = schema.description or f"Document type: {schema.name}"
            # Include Hebrew aliases for better matching
            hebrew_hints = []
            for field in schema.fields[:5]:
                hebrew_aliases = [a for a in (field.aliases or []) if any(c in a for c in 'אבגדהוזחטיכלמנסעפצקרשת')]
                if hebrew_aliases:
                    hebrew_hints.extend(hebrew_aliases[:2])
            hints_text = f" (Hebrew hints: {', '.join(hebrew_hints[:6])})" if hebrew_hints else ""
            doc_descriptions.append(f"- {schema.name}: {desc}{hints_text}")
        
        descriptions_text = "\n".join(doc_descriptions)
        
        prompt = f"""Given a CSV file with these columns and sample values:

{columns_text}

Which document type best describes this file?

Embedding scores: {candidates_text}

Available document types:
{descriptions_text}

Respond with ONLY the document type name. Choose based on column names and data patterns."""

        try:
            response = self.openai_client.client.chat.completions.create(
                model=self.openai_client.model,
                messages=[
                    {"role": "system", "content": "You are a document classification expert. Respond with only the document type name."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=50,
            )
            result = response.choices[0].message.content.strip().lower()
            
            if result in all_valid_types:
                logger.info(f"OpenAI verified document type: {result}")
                return result
            
            # Check for partial match
            for dt in all_valid_types:
                if dt in result or result in dt:
                    logger.info(f"OpenAI verified document type (partial): {dt}")
                    return dt
            
            logger.warning(f"OpenAI returned '{result}', using top embedding candidate")
            return candidates[0][0]
            
        except Exception as e:
            logger.warning(f"OpenAI verification failed: {e}")
            return candidates[0][0]


class InsightRunner:
    """Runs insights on CSV files with schema-canonized columns."""
    
    def __init__(
        self,
        vertical: str = "medical",
        context_path: Optional[str] = None,
        openai_api_key: Optional[str] = None,
    ):
        self.vertical = vertical
        self.context_path = context_path or str(
            Path(__file__).parent.parent / "contexts" / "medical.yaml"
        )
        
        # Initialize OpenAI client
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
        self.openai_client = get_openai_client()
        
        # Initialize preprocessing pipeline
        self.pipeline = PreprocessingPipeline(
            openai_client=self.openai_client,
            context_path=self.context_path,
        )
        
        # Initialize schema canonizer
        logger.info("Initializing schema canonizer...")
        self.canonizer = SchemaCanonizer(
            vertical=vertical,
            openai_client=self.openai_client,
        )
        
        # DuckDB connection for SQL execution
        import duckdb
        self.connection = duckdb.connect(":memory:")
        
        # Track loaded tables
        self.loaded_tables: Dict[str, pd.DataFrame] = {}
        self.column_mappings: Dict[str, Dict[str, str]] = {}
    
    def load_file(self, file_path: str) -> Optional[str]:
        """
        Load CSV, preprocess, classify, and canonize to schema field names.
        
        If multiple files have the same document type, they are MERGED (concatenated).
        
        Returns:
            Detected document type name
        """
        logger.info(f"Loading file: {file_path}")
        
        # Preprocess (structure detection, metadata extraction, transforms)
        result = self.pipeline.process_with_auto_transforms(file_path)
        df = result.df
        
        # Classify and canonize columns to schema names
        logger.info("Classifying and canonizing columns...")
        doc_type, df_canonized, mappings = self.canonizer.classify_and_canonize(df)
        
        if doc_type is None:
            logger.error(f"Could not detect document type for: {file_path}")
            return None
        
        logger.info(f"Detected: {doc_type}")
        logger.info(f"Canonized {len(mappings)} columns to schema field names")
        
        # Check if this document type already exists - MERGE if so
        if doc_type in self.loaded_tables:
            existing_df = self.loaded_tables[doc_type]
            logger.info(f"  Merging with existing '{doc_type}' table ({len(existing_df)} rows)")
            
            # Concatenate, aligning columns (fill missing with NaN)
            df_canonized = pd.concat([existing_df, df_canonized], ignore_index=True, sort=False)
            logger.info(f"  After merge: {len(df_canonized)} rows")
        
        # Register in DuckDB with canonical column names
        self.connection.register(doc_type, df_canonized)
        self.loaded_tables[doc_type] = df_canonized
        self.column_mappings[doc_type] = mappings
        
        logger.info(f"  Loaded as '{doc_type}': {len(df_canonized)} rows, {len(df_canonized.columns)} columns")
        
        return doc_type
    
    def load_files(self, file_paths: List[str]) -> Dict[str, str]:
        """Load multiple files."""
        results = {}
        for path in file_paths:
            doc_type = self.load_file(path)
            if doc_type:
                results[path] = doc_type
        return results
    
    def run_insight(self, insight: InsightDefinition) -> pd.DataFrame:
        """Run an insight (SQL or code-based) on canonized tables."""
        # Check required tables
        missing = [t for t in insight.requires if t not in self.loaded_tables]
        if missing:
            raise ValueError(
                f"Insight '{insight.name}' requires: {insight.requires}. "
                f"Missing: {missing}. Loaded: {list(self.loaded_tables.keys())}"
            )
        
        logger.info(f"Running insight: {insight.display_name}")
        
        # Check if this is a code insight
        if hasattr(insight, 'type') and insight.type == 'code':
            # Run code insight handler
            from csv_analyzer.insights.code_insights import CodeInsightsRegistry
            
            # Create a minimal engine-like interface for the handler
            loaded_tables_copy = self.loaded_tables
            
            class DataStoreAdapter:
                def list_tables(self):
                    return list(loaded_tables_copy.keys())
            
            class EngineAdapter:
                def __init__(self, connection):
                    self.connection = connection
                    self.data_store = DataStoreAdapter()
                
                def execute_sql(self, sql):
                    return self.connection.execute(sql).fetchdf()
            
            adapter = EngineAdapter(self.connection)
            params = {}
            for param in insight.parameters:
                params[param.name] = param.default
            
            result = CodeInsightsRegistry.run(insight.handler, adapter, params)
        else:
            # Run SQL insight
            result = self.connection.execute(insight.sql).fetchdf()
        
        logger.info(f"  Result: {len(result)} rows")
        
        return result
    
    def get_canonization_summary(self) -> str:
        """Get a summary of column canonization for all tables."""
        lines = []
        for doc_type, mappings in self.column_mappings.items():
            lines.append(f"\n{doc_type}:")
            lines.append("  Original Column → Canonical Schema Field")
            lines.append("  " + "-" * 50)
            for orig, canonical in sorted(mappings.items()):
                lines.append(f"  {orig:<25} → {canonical}")
        return "\n".join(lines)
    
    def get_final_columns(self, doc_type: str) -> List[str]:
        """Get the final canonized column names for a table."""
        if doc_type in self.loaded_tables:
            return list(self.loaded_tables[doc_type].columns)
        return []
    
    def close(self):
        """Clean up resources."""
        self.connection.close()


def main():
    parser = argparse.ArgumentParser(
        description="Run insights on CSV files with schema-canonized columns",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_insight.py --insight early_arrival --files shifts.csv actions.csv --output report.csv
    python run_insight.py --list
        """
    )
    
    parser.add_argument("--insight", "-i", help="Name of the insight to run")
    parser.add_argument("--files", "-f", nargs="+", help="CSV files to process")
    parser.add_argument("--output", "-o", help="Output CSV file path")
    parser.add_argument("--vertical", default="medical", help="Vertical/domain (default: medical)")
    parser.add_argument("--context", help="Path to context YAML")
    parser.add_argument("--api-key", help="OpenAI API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--list", action="store_true", help="List available insights")
    parser.add_argument("--show-canonization", action="store_true", help="Show column canonization details")
    parser.add_argument("--show-columns", action="store_true", help="Show final canonized columns")
    
    args = parser.parse_args()
    
    # List insights
    if args.list:
        definitions_dir = Path(__file__).parent.parent / "insights" / "definitions"
        print("\nAvailable Insights:")
        print("=" * 50)
        for yaml_file in sorted(definitions_dir.glob("*.yaml")):
            try:
                insight = InsightDefinition.load(yaml_file.stem, definitions_dir)
                print(f"\n{insight.name}")
                print(f"  Name: {insight.display_name}")
                print(f"  Description: {insight.description}")
                print(f"  Requires: {', '.join(insight.requires)}")
            except Exception as e:
                print(f"\n{yaml_file.stem}: Error - {e}")
        print()
        return
    
    # Validate args
    if not args.insight:
        parser.error("--insight is required")
    if not args.files:
        parser.error("--files is required")
    if not args.output:
        parser.error("--output is required")
    
    # Load insight
    try:
        insight = InsightDefinition.load(args.insight)
    except FileNotFoundError as e:
        parser.error(str(e))
    
    print()
    print("=" * 60)
    print(f"Running Insight: {insight.display_name}")
    print("=" * 60)
    print(f"Description: {insight.description}")
    print(f"Requires: {', '.join(insight.requires)}")
    print()
    
    # Initialize runner
    runner = InsightRunner(
        vertical=args.vertical,
        context_path=args.context,
        openai_api_key=args.api_key,
    )
    
    try:
        # Load files
        print("Loading and canonizing files...")
        loaded = runner.load_files(args.files)
        
        if not loaded:
            print("ERROR: No files could be loaded")
            sys.exit(1)
        
        print()
        print("Loaded documents:")
        for path, doc_type in loaded.items():
            print(f"  {Path(path).name} → {doc_type}")
        
        # Show canonization details
        if args.show_canonization:
            print()
            print("Column Canonization (Original → Schema Field):")
            print(runner.get_canonization_summary())
        
        # Show final columns
        if args.show_columns:
            print()
            print("Final Canonized Columns:")
            for doc_type in loaded.values():
                print(f"\n  {doc_type}:")
                for col in runner.get_final_columns(doc_type):
                    print(f"    - {col}")
        
        print()
        
        # Run insight
        result = runner.run_insight(insight)
        
        # Save output
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_path, index=False, encoding="utf-8-sig")
        
        print()
        print("=" * 60)
        print("Result")
        print("=" * 60)
        print(result.to_string())
        print()
        print(f"Saved to: {output_path}")
        print(f"Rows: {len(result)}")
        
        if "status" in result.columns:
            alerts = len(result[result["status"] == "ALERT"])
            print(f"Alerts: {alerts}")
        
    finally:
        runner.close()


if __name__ == "__main__":
    main()
