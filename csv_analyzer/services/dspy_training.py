"""
DSPy Training Data Loader.

Loads ground truth examples from the ground_truth/ directory and converts
them into DSPy training examples for optimizing column classification.

The ground truth structure:
    ground_truth/
        medical/
            employee_shifts/
                hebrew_shifts.csv
                standard_english.csv
            staff_clinical_procedures/
                ...

Each CSV has a corresponding expected mapping derived from the schema.
"""

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

# Check if dspy is available
try:
    import dspy
    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False


@dataclass
class ColumnExample:
    """A single column classification example."""
    # Input
    column_name: str
    column_type: str
    sample_values: List[str]
    document_type: str
    vertical: str
    
    # Expected output
    target_field: str
    is_correct_mapping: bool = True  # For verification examples
    
    # Metadata
    source_file: str = ""


@dataclass
class MappingExample:
    """A full CSV mapping example (all columns at once)."""
    columns: List[Dict[str, Any]]
    document_type: str
    vertical: str
    expected_mappings: Dict[str, str]
    source_file: str = ""


@dataclass
class TrainingData:
    """Complete training data for DSPy optimization."""
    # Individual column examples (for ClassifyColumn)
    column_examples: List[ColumnExample] = field(default_factory=list)
    
    # Verification examples (for VerifyMapping)
    verify_examples: List[Tuple[ColumnExample, bool]] = field(default_factory=list)
    
    # Full mapping examples (for ClassifyColumnSet)
    mapping_examples: List[MappingExample] = field(default_factory=list)
    
    def __len__(self):
        return len(self.column_examples)
    
    def to_dspy_examples(self) -> Dict[str, List]:
        """Convert to DSPy Example format."""
        if not DSPY_AVAILABLE:
            raise ImportError("dspy-ai not installed")
        
        classify_examples = []
        verify_examples = []
        
        for ex in self.column_examples:
            classify_examples.append(dspy.Example(
                column_name=ex.column_name,
                column_type=ex.column_type,
                sample_values=", ".join(str(v) for v in ex.sample_values[:5]),
                document_type=ex.document_type,
                target_field=ex.target_field,
            ).with_inputs("column_name", "column_type", "sample_values", "document_type"))
        
        for ex, is_correct in self.verify_examples:
            verify_examples.append(dspy.Example(
                column_name=ex.column_name,
                column_type=ex.column_type,
                sample_values=", ".join(str(v) for v in ex.sample_values[:5]),
                proposed_field=ex.target_field,
                document_type=ex.document_type,
                is_correct=is_correct,
            ).with_inputs("column_name", "column_type", "sample_values", "proposed_field", "document_type"))
        
        return {
            "classify": classify_examples,
            "verify": verify_examples,
        }


class GroundTruthLoader:
    """
    Loads ground truth from the filesystem and generates training examples.
    
    Automatically infers column mappings by matching column names against
    schema aliases.
    """
    
    def __init__(
        self,
        ground_truth_dir: Path = None,
        schemas_dir: Path = None,
    ):
        """
        Initialize the loader.
        
        Args:
            ground_truth_dir: Path to ground_truth/ directory
            schemas_dir: Path to schemas/ directory
        """
        base_dir = Path(__file__).parent.parent
        self.ground_truth_dir = ground_truth_dir or base_dir / "ground_truth"
        self.schemas_dir = schemas_dir or base_dir / "schemas"
        
        # Load all schemas
        self.schemas = self._load_all_schemas()
        logger.info(f"Loaded {len(self.schemas)} schemas")
    
    def _load_all_schemas(self) -> Dict[str, Dict]:
        """Load all schema YAML files."""
        schemas = {}
        
        for vertical_dir in self.schemas_dir.iterdir():
            if not vertical_dir.is_dir():
                continue
            
            vertical = vertical_dir.name
            
            for schema_file in vertical_dir.glob("*.yaml"):
                try:
                    with open(schema_file) as f:
                        schema = yaml.safe_load(f)
                    
                    doc_type = schema.get("name", schema_file.stem)
                    schemas[f"{vertical}/{doc_type}"] = schema
                    
                except Exception as e:
                    logger.warning(f"Failed to load schema {schema_file}: {e}")
        
        return schemas
    
    def _get_schema(self, vertical: str, document_type: str) -> Optional[Dict]:
        """Get schema by vertical and document type."""
        return self.schemas.get(f"{vertical}/{document_type}")
    
    def _normalize_column_name(self, name: str) -> str:
        """Normalize column name for matching."""
        # Remove common separators and lowercase
        normalized = name.lower()
        normalized = re.sub(r"[_\-\.\s]+", "_", normalized)
        normalized = normalized.strip("_")
        return normalized
    
    def _find_field_for_column(
        self,
        column_name: str,
        schema: Dict,
    ) -> Optional[str]:
        """
        Find the schema field that matches a column name.
        
        Checks against field names and aliases.
        """
        normalized_col = self._normalize_column_name(column_name)
        
        for field in schema.get("fields", []):
            field_name = field.get("name", "")
            
            # Check field name
            if self._normalize_column_name(field_name) == normalized_col:
                return field_name
            
            # Check aliases
            aliases = field.get("aliases", [])
            for alias in aliases:
                if self._normalize_column_name(alias) == normalized_col:
                    return field_name
        
        return None
    
    def _detect_column_type(self, series: pd.Series) -> str:
        """Simple column type detection."""
        # Get non-null values
        non_null = series.dropna()
        if len(non_null) == 0:
            return "empty"
        
        sample = non_null.iloc[0]
        
        # Check if it looks like a date/time
        if isinstance(sample, str):
            if re.match(r"\d{4}-\d{2}-\d{2}", sample):
                return "date"
            if re.match(r"\d{1,2}:\d{2}", sample):
                return "time"
        
        # Check if numeric
        try:
            pd.to_numeric(non_null)
            return "numeric"
        except:
            pass
        
        return "text"
    
    def load_training_data(
        self,
        verticals: List[str] = None,
        document_types: List[str] = None,
        include_negatives: bool = True,
        negative_ratio: float = 0.3,
    ) -> TrainingData:
        """
        Load training data from ground truth directory.
        
        Args:
            verticals: List of verticals to include (None = all)
            document_types: List of document types to include (None = all)
            include_negatives: Whether to generate negative examples
            negative_ratio: Ratio of negative to positive examples
            
        Returns:
            TrainingData with examples for DSPy training
        """
        training = TrainingData()
        
        # Iterate through ground truth directory
        for vertical_dir in self.ground_truth_dir.iterdir():
            if not vertical_dir.is_dir():
                continue
            
            vertical = vertical_dir.name
            if verticals and vertical not in verticals:
                continue
            
            for doc_type_dir in vertical_dir.iterdir():
                if not doc_type_dir.is_dir():
                    continue
                
                document_type = doc_type_dir.name
                if document_types and document_type not in document_types:
                    continue
                
                # Get schema
                schema = self._get_schema(vertical, document_type)
                if not schema:
                    logger.warning(f"No schema for {vertical}/{document_type}")
                    continue
                
                # Process CSV files
                for csv_file in doc_type_dir.glob("*.csv"):
                    examples = self._process_csv(
                        csv_file=csv_file,
                        vertical=vertical,
                        document_type=document_type,
                        schema=schema,
                    )
                    training.column_examples.extend(examples)
        
        logger.info(f"Loaded {len(training.column_examples)} column examples")
        
        # Generate verification examples
        training.verify_examples = self._generate_verify_examples(
            training.column_examples,
            include_negatives=include_negatives,
            negative_ratio=negative_ratio,
        )
        logger.info(f"Generated {len(training.verify_examples)} verification examples")
        
        return training
    
    def _process_csv(
        self,
        csv_file: Path,
        vertical: str,
        document_type: str,
        schema: Dict,
    ) -> List[ColumnExample]:
        """Process a single CSV file into training examples."""
        examples = []
        
        try:
            df = pd.read_csv(csv_file)
        except Exception as e:
            logger.warning(f"Failed to read {csv_file}: {e}")
            return examples
        
        for col in df.columns:
            # Find the target field
            target_field = self._find_field_for_column(col, schema)
            
            if not target_field:
                logger.debug(f"No mapping for column '{col}' in {csv_file.name}")
                continue
            
            # Get sample values
            sample_values = df[col].dropna().head(10).tolist()
            sample_values = [str(v) for v in sample_values]
            
            # Detect type
            col_type = self._detect_column_type(df[col])
            
            examples.append(ColumnExample(
                column_name=col,
                column_type=col_type,
                sample_values=sample_values,
                document_type=document_type,
                vertical=vertical,
                target_field=target_field,
                source_file=str(csv_file),
            ))
        
        logger.info(f"Extracted {len(examples)} examples from {csv_file.name}")
        return examples
    
    def _generate_verify_examples(
        self,
        column_examples: List[ColumnExample],
        include_negatives: bool,
        negative_ratio: float,
    ) -> List[Tuple[ColumnExample, bool]]:
        """
        Generate verification examples (positive and negative).
        
        Positive: Column correctly maps to its target field
        Negative: Column incorrectly maps to a different field
        """
        import random
        
        verify_examples = []
        
        # All positive examples
        for ex in column_examples:
            verify_examples.append((ex, True))
        
        if not include_negatives:
            return verify_examples
        
        # Generate negative examples by swapping targets
        num_negatives = int(len(column_examples) * negative_ratio)
        
        for _ in range(num_negatives):
            # Pick a random example
            ex = random.choice(column_examples)
            
            # Pick a different target from the same schema
            schema = self._get_schema(ex.vertical, ex.document_type)
            if not schema:
                continue
            
            all_fields = [f.get("name") for f in schema.get("fields", [])]
            wrong_targets = [f for f in all_fields if f != ex.target_field]
            
            if not wrong_targets:
                continue
            
            wrong_target = random.choice(wrong_targets)
            
            # Create negative example
            neg_ex = ColumnExample(
                column_name=ex.column_name,
                column_type=ex.column_type,
                sample_values=ex.sample_values,
                document_type=ex.document_type,
                vertical=ex.vertical,
                target_field=wrong_target,  # Wrong target!
                source_file=ex.source_file,
            )
            verify_examples.append((neg_ex, False))
        
        random.shuffle(verify_examples)
        return verify_examples


def load_training_data(
    ground_truth_dir: Path = None,
    schemas_dir: Path = None,
) -> TrainingData:
    """
    Convenience function to load training data.
    
    Args:
        ground_truth_dir: Path to ground_truth/ directory
        schemas_dir: Path to schemas/ directory
        
    Returns:
        TrainingData ready for DSPy optimization
    """
    loader = GroundTruthLoader(
        ground_truth_dir=ground_truth_dir,
        schemas_dir=schemas_dir,
    )
    return loader.load_training_data()

