#!/usr/bin/env python3
"""
Insight Runner Script

Processes CSV files (auto-detecting document type using embeddings) and runs an insight query.

Usage:
    python run_insight.py --insight early_arrival --files shifts.csv actions.csv --output report.csv
    
Examples:
    # Run early arrival insight on two files
    python run_insight.py \\
        --insight early_arrival \\
        --files "/path/to/shifts.csv" "/path/to/actions.csv" \\
        --output ./early_arrival_report.csv
    
    # With custom context
    python run_insight.py \\
        --insight early_arrival \\
        --files *.csv \\
        --output ./report.csv \\
        --context csv_analyzer/contexts/medical.yaml
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
from csv_analyzer.services.dspy_service import DSPyClassificationService, create_dspy_service

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class InsightDefinition:
    """Represents an insight loaded from YAML."""
    
    def __init__(self, data: Dict[str, Any]):
        self.name = data.get("name", "unknown")
        self.display_name = data.get("display_name", self.name)
        self.description = data.get("description", "")
        self.requires = data.get("requires", [])
        self.sql = data.get("sql", "")
        self.output_columns = data.get("output_columns", [])
        self.parameters = data.get("parameters", [])
    
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


class EmbeddingsClassifier:
    """
    Classifies documents using embeddings and schema matching.
    
    NOTE: The full ClassificationEngine in csv_analyzer/engines/ requires PostgreSQL 
    for ground truth lookup. This simplified classifier uses only:
    - SchemaEmbeddingsService (ChromaDB) for column-to-field matching
    - SchemaRegistry for loading schema definitions from YAML
    - MultilingualEmbeddingsClient for generating embeddings
    - DSPyClassificationService (optional) for column verification
    - OpenAI for document type verification when embeddings are ambiguous
    """
    
    def __init__(
        self,
        vertical: str = "medical",
        openai_client=None,
        use_dspy: bool = True,
    ):
        self.vertical = vertical
        self.openai_client = openai_client
        
        # Initialize embeddings client
        logger.info("Initializing multilingual embeddings client...")
        self.embeddings_client = get_multilingual_embeddings_client()
        
        # Initialize schema registry
        logger.info("Loading schema registry...")
        self.schema_registry = get_schema_registry()
        
        # Initialize schema embeddings service (ChromaDB - ephemeral/in-memory)
        logger.info("Initializing schema embeddings service (ChromaDB - ephemeral)...")
        self.schema_service = SchemaEmbeddingsService(
            embeddings_client=self.embeddings_client,
            schema_registry=self.schema_registry,
            persist_directory=None,  # Use ephemeral/in-memory ChromaDB
        )
        
        # Initialize DSPy service for column verification (uses existing service)
        self.dspy_service = None
        if use_dspy:
            try:
                self.dspy_service = create_dspy_service()
                if self.dspy_service and self.dspy_service.is_available:
                    logger.info("DSPy service initialized for column verification")
                else:
                    logger.warning("DSPy service not available, falling back to embeddings-only")
                    self.dspy_service = None
            except Exception as e:
                logger.warning(f"Failed to initialize DSPy service: {e}")
        
        # Index all schemas (builds embeddings in ChromaDB)
        logger.info(f"Indexing schema embeddings for vertical: {vertical}")
        self.schema_service.index_all_schemas(force_reindex=True)
    
    def _verify_document_type_with_openai(
        self,
        df: pd.DataFrame,
        all_candidates: List[Tuple[str, float]],  # All candidates, not just top
        column_profiles: List[Dict],
    ) -> str:
        """Use OpenAI to verify document type when embeddings are ambiguous."""
        if not self.openai_client:
            return all_candidates[0][0]  # Return top candidate
        
        # Build a summary of columns and sample values
        col_summaries = []
        for profile in column_profiles[:15]:  # Limit to 15 columns
            col_name = profile.get("column_name", "unknown")
            col_type = profile.get("detected_type", "unknown")
            samples = profile.get("sample_values", [])[:3]
            col_summaries.append(f"- {col_name} ({col_type}): {samples}")
        
        columns_text = "\n".join(col_summaries)
        candidates_text = ", ".join([f"{dt} ({score:.2f})" for dt, score in all_candidates])
        
        # Get document type descriptions dynamically from schema registry
        doc_type_descriptions = []
        all_valid_types = []
        for schema in self.schema_registry.get_schemas_by_vertical(self.vertical):
            doc_type = schema.name
            all_valid_types.append(doc_type)
            description = schema.description or f"Document type: {doc_type}"
            # Get some field hints
            field_names = [f.name for f in schema.fields[:5]]
            doc_type_descriptions.append(f"- {doc_type}: {description} (fields: {', '.join(field_names)})")
        
        descriptions_text = "\n".join(doc_type_descriptions)
        
        prompt = f"""Given a CSV file with these columns and sample values:

{columns_text}

Which document type best describes this file?

Candidate types (with embedding scores): {candidates_text}

Available document types:
{descriptions_text}

Respond with ONLY the document type name (e.g., "{all_valid_types[0] if all_valid_types else 'unknown'}"). Choose based on the column names and data patterns, not just the embedding scores."""

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
            result = response.choices[0].message.content.strip().lower().replace("_", "_")
            
            if result in all_valid_types:
                logger.info(f"OpenAI classified document type: {result}")
                return result
            else:
                # Check for partial match
                for dt in all_valid_types:
                    if dt in result or result in dt:
                        logger.info(f"OpenAI classified document type (partial match): {dt}")
                        return dt
                
                logger.warning(f"OpenAI returned invalid type '{result}', using top candidate")
                return all_candidates[0][0]
                
        except Exception as e:
            logger.warning(f"OpenAI verification failed: {e}, using top candidate")
            return all_candidates[0][0]
    
    def classify(
        self,
        df: pd.DataFrame,
        threshold: float = 0.5,
    ) -> Tuple[Optional[str], Dict[str, str], float]:
        """
        Classify a DataFrame by matching columns to schema fields.
        
        Uses embeddings similarity to find the best matching document type.
        
        Args:
            df: DataFrame to classify
            threshold: Minimum similarity threshold
            
        Returns:
            Tuple of (document_type, column_mappings, confidence)
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
            return None, {}, 0.0
        
        # Log all scores for debugging
        logger.info("Document type classification scores:")
        sorted_scores = sorted(
            doc_type_scores.items(),
            key=lambda x: x[1].get("score", 0),
            reverse=True
        )
        for dt, scores in sorted_scores:
            logger.info(f"  {dt}: {scores.get('score', 0):.3f} (matched {scores.get('matched_columns', 0)}/{scores.get('total_columns', 0)})")
        
        # Get all candidates with scores
        all_candidates = [(dt, info.get("score", 0)) for dt, info in sorted_scores]
        
        # Always use OpenAI to verify document type (embeddings alone are not reliable enough)
        if self.openai_client and all_candidates:
            logger.info("Using OpenAI to verify document type...")
            doc_type = self._verify_document_type_with_openai(df, all_candidates, column_profiles)
            confidence = dict(all_candidates).get(doc_type, all_candidates[0][1] if all_candidates else 0.0)
        elif all_candidates:
            doc_type = all_candidates[0][0]
            confidence = all_candidates[0][1]
        else:
            doc_type = None
            confidence = 0.0
        
        logger.info(f"Best match: {doc_type} (confidence: {confidence:.3f})")
        
        # Build column mappings for the best document type
        mappings = {}
        for col_name, matches in column_matches.items():
            if matches:
                # Filter matches for this document type
                doc_matches = [m for m in matches if m.get("document_type") == doc_type]
                if doc_matches:
                    best_match = doc_matches[0]
                    if best_match["similarity"] >= threshold:
                        mappings[col_name] = best_match["field_name"]
                        logger.debug(
                            f"  {col_name} → {best_match['field_name']} "
                            f"(sim: {best_match['similarity']:.3f})"
                        )
        
        # Optional: Verify mappings with OpenAI
        if self.openai_client and mappings:
            mappings = self._verify_mappings_with_openai(
                df, mappings, doc_type, column_profiles
            )
        
        return doc_type, mappings, confidence
    
    def _verify_mappings_with_openai(
        self,
        df: pd.DataFrame,
        mappings: Dict[str, str],
        doc_type: str,
        column_profiles: List[Dict],
    ) -> Dict[str, str]:
        """Verify mappings using OpenAI (optional enhancement)."""
        # For now, skip OpenAI verification - embeddings are doing well
        # This could be enhanced later for low-confidence mappings
        return mappings


class InsightRunner:
    """Runs insights on CSV files using embeddings-based classification."""
    
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
        
        # Initialize embeddings-based classifier
        logger.info("Initializing embeddings-based classifier...")
        self.classifier = EmbeddingsClassifier(
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
        Load and preprocess a CSV file, auto-detecting document type.
        
        Uses embeddings to match columns to schema fields.
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            Detected document type name, or None if detection failed
        """
        logger.info(f"Loading file: {file_path}")
        
        # Preprocess the file (structure detection, metadata extraction, etc.)
        result = self.pipeline.process_with_auto_transforms(file_path)
        df = result.df
        
        # Classify using embeddings
        logger.info("Classifying document type using embeddings...")
        doc_type, mappings, confidence = self.classifier.classify(df)
        
        if doc_type is None:
            logger.error(f"Could not detect document type for: {file_path}")
            return None
        
        logger.info(f"Detected: {doc_type} (confidence: {confidence:.3f})")
        logger.info(f"Column mappings: {len(mappings)} fields mapped")
        
        # Apply column mappings (rename columns)
        df_mapped = df.copy()
        for orig_col, mapped_col in mappings.items():
            if orig_col in df_mapped.columns:
                df_mapped = df_mapped.rename(columns={orig_col: mapped_col})
        
        # Register in DuckDB
        self.connection.register(doc_type, df_mapped)
        self.loaded_tables[doc_type] = df_mapped
        self.column_mappings[doc_type] = mappings
        
        logger.info(f"  Loaded as '{doc_type}': {len(df_mapped)} rows, {len(df_mapped.columns)} columns")
        
        return doc_type
    
    def load_files(self, file_paths: List[str]) -> Dict[str, str]:
        """
        Load multiple files.
        
        Returns:
            Dict mapping file path to detected document type
        """
        results = {}
        for path in file_paths:
            doc_type = self.load_file(path)
            if doc_type:
                results[path] = doc_type
        return results
    
    def run_insight(self, insight: InsightDefinition) -> pd.DataFrame:
        """
        Run an insight query.
        
        Args:
            insight: Loaded insight definition
            
        Returns:
            Result DataFrame
        """
        # Check required tables
        missing = [t for t in insight.requires if t not in self.loaded_tables]
        if missing:
            raise ValueError(
                f"Insight '{insight.name}' requires tables: {insight.requires}. "
                f"Missing: {missing}. Loaded: {list(self.loaded_tables.keys())}"
            )
        
        logger.info(f"Running insight: {insight.display_name}")
        
        # Execute SQL
        result = self.connection.execute(insight.sql).fetchdf()
        
        logger.info(f"  Result: {len(result)} rows")
        
        return result
    
    def get_mappings_summary(self) -> str:
        """Get a summary of column mappings for all loaded tables."""
        lines = []
        for doc_type, mappings in self.column_mappings.items():
            lines.append(f"\n{doc_type}:")
            for orig, mapped in mappings.items():
                lines.append(f"  {orig} → {mapped}")
        return "\n".join(lines)
    
    def close(self):
        """Clean up resources."""
        self.connection.close()


def main():
    parser = argparse.ArgumentParser(
        description="Run an insight on CSV files (auto-detects document types using embeddings)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run early arrival insight
    python run_insight.py \\
        --insight early_arrival \\
        --files shifts.csv actions.csv \\
        --output report.csv
    
    # List available insights
    python run_insight.py --list
        """
    )
    
    parser.add_argument(
        "--insight", "-i",
        help="Name of the insight to run (loads from insights/definitions/{name}.yaml)"
    )
    parser.add_argument(
        "--files", "-f",
        nargs="+",
        help="CSV files to process (document type auto-detected using embeddings)"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output CSV file path"
    )
    parser.add_argument(
        "--vertical",
        default="medical",
        help="Vertical/domain for schema matching (default: medical)"
    )
    parser.add_argument(
        "--context",
        help="Path to context YAML (default: contexts/medical.yaml)"
    )
    parser.add_argument(
        "--api-key",
        help="OpenAI API key (or set OPENAI_API_KEY env var)"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available insights and exit"
    )
    parser.add_argument(
        "--show-mappings",
        action="store_true",
        help="Show column mappings after classification"
    )
    
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
                print(f"\n{yaml_file.stem}: Error loading - {e}")
        
        print()
        return
    
    # Validate required args
    if not args.insight:
        parser.error("--insight is required")
    if not args.files:
        parser.error("--files is required")
    if not args.output:
        parser.error("--output is required")
    
    # Load insight definition
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
        print("Loading files (using embeddings for classification)...")
        loaded = runner.load_files(args.files)
        
        if not loaded:
            print("ERROR: No files could be loaded")
            sys.exit(1)
        
        print()
        print("Loaded documents:")
        for path, doc_type in loaded.items():
            print(f"  {Path(path).name} → {doc_type}")
        
        # Show mappings if requested
        if args.show_mappings:
            print()
            print("Column Mappings (Hebrew → Canonical):")
            print(runner.get_mappings_summary())
        
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
        
        # Count alerts if applicable
        if "status" in result.columns:
            alerts = len(result[result["status"] == "ALERT"])
            print(f"Alerts: {alerts}")
        
    finally:
        runner.close()


if __name__ == "__main__":
    main()
