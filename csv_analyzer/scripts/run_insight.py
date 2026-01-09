#!/usr/bin/env python3
"""
Insight Runner Script

Processes CSV files (auto-detecting document type) and runs an insight query.

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
from typing import Dict, List, Optional, Any

import pandas as pd
import yaml

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from csv_analyzer.openai_client import get_openai_client
from csv_analyzer.preprocessing import PreprocessingPipeline

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


class DocumentTypeDetector:
    """Detects document type from preprocessed DataFrame."""
    
    # Known document type patterns based on columns
    TYPE_PATTERNS = {
        "employee_shifts": {
            "required_columns": ["תאריך", "כניסה", "יציאה"],
            "metadata_hints": ["employee_name", "employee_id"],
        },
        "medical_actions": {
            "required_columns": ["תאריך טיפול", "שם טיפול"],
            "metadata_hints": [],
        },
    }
    
    @classmethod
    def detect(cls, df: pd.DataFrame, metadata: Optional[Dict] = None) -> Optional[str]:
        """
        Detect document type from DataFrame columns and metadata.
        
        Args:
            df: Preprocessed DataFrame
            metadata: Extracted metadata from preprocessing
            
        Returns:
            Document type name or None if unknown
        """
        columns = set(df.columns)
        
        for doc_type, patterns in cls.TYPE_PATTERNS.items():
            required = set(patterns["required_columns"])
            
            # Check if required columns are present
            if required.issubset(columns):
                logger.info(f"Detected document type: {doc_type}")
                return doc_type
            
            # Check metadata hints
            if metadata:
                hints = patterns.get("metadata_hints", [])
                if hints and all(h in metadata for h in hints):
                    logger.info(f"Detected document type from metadata: {doc_type}")
                    return doc_type
        
        logger.warning(f"Could not detect document type. Columns: {list(columns)[:10]}...")
        return None


class InsightRunner:
    """Runs insights on CSV files."""
    
    def __init__(
        self,
        context_path: Optional[str] = None,
        openai_api_key: Optional[str] = None,
    ):
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
        
        # DuckDB connection for SQL execution
        import duckdb
        self.connection = duckdb.connect(":memory:")
        
        # Track loaded tables
        self.loaded_tables: Dict[str, pd.DataFrame] = {}
    
    def load_file(self, file_path: str) -> Optional[str]:
        """
        Load and preprocess a CSV file, auto-detecting document type.
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            Detected document type name, or None if detection failed
        """
        logger.info(f"Loading file: {file_path}")
        
        # Preprocess the file
        result = self.pipeline.process_with_auto_transforms(file_path)
        df = result.df
        
        # Get metadata for detection
        metadata = {}
        if result.extracted_metadata:
            metadata = result.extracted_metadata.to_dict()
        
        # Detect document type
        doc_type = DocumentTypeDetector.detect(df, metadata)
        
        if doc_type is None:
            logger.error(f"Could not detect document type for: {file_path}")
            return None
        
        # Register in DuckDB
        self.connection.register(doc_type, df)
        self.loaded_tables[doc_type] = df
        
        logger.info(f"  Loaded as '{doc_type}': {len(df)} rows, {len(df.columns)} columns")
        
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
    
    def close(self):
        """Clean up resources."""
        self.connection.close()


def main():
    parser = argparse.ArgumentParser(
        description="Run an insight on CSV files (auto-detects document types)",
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
        help="CSV files to process (document type auto-detected)"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output CSV file path"
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
        context_path=args.context,
        openai_api_key=args.api_key,
    )
    
    try:
        # Load files
        print("Loading files...")
        loaded = runner.load_files(args.files)
        
        if not loaded:
            print("ERROR: No files could be loaded")
            sys.exit(1)
        
        print()
        print("Loaded documents:")
        for path, doc_type in loaded.items():
            print(f"  {Path(path).name} → {doc_type}")
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

