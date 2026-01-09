"""
CSV Preprocessing Pipeline.

Provides:
- AI-based structure detection for multi-row headers
- Value transformers for column normalization
- Row filtering via SQL conditions
- Preprocessing pipeline orchestrator
"""

from csv_analyzer.preprocessing.structure_detector import (
    StructureDetector,
    CSVStructure,
    MetadataRow,
)
from csv_analyzer.preprocessing.transformers import (
    ValueTransformer,
    TransformResult,
)
from csv_analyzer.preprocessing.row_filter import RowFilter
from csv_analyzer.preprocessing.pipeline import (
    PreprocessingPipeline,
    ProcessedCSV,
)

__all__ = [
    "StructureDetector",
    "CSVStructure",
    "MetadataRow",
    "ValueTransformer",
    "TransformResult",
    "RowFilter",
    "PreprocessingPipeline",
    "ProcessedCSV",
]

