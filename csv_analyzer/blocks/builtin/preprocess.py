"""
Preprocess Block - Transform raw CSVs.

Applies preprocessing transformations:
- Split time ranges
- Normalize Hebrew dates
- Clean time prefixes
- Detect multi-row headers
- Extract metadata to _meta columns
"""

import logging
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from csv_analyzer.workflows.block import BlockRegistry
from csv_analyzer.workflows.ontology import DataType
from csv_analyzer.workflows.base_block import BaseBlock, BlockContext
from csv_analyzer.preprocessing.pipeline import PreprocessingPipeline

logger = logging.getLogger(__name__)


class PreprocessBlock(BaseBlock):
    """
    Preprocess CSV files.
    
    Wraps the existing PreprocessingPipeline to:
    - Detect and handle multi-row headers
    - Extract metadata (employee_name, date_range, location)
    - Apply value transformations (split ranges, normalize dates)
    - Inject _meta columns for SQL access
    """
    
    def run(self) -> Dict[str, str]:
        """
        Process input files through preprocessing pipeline.
        
        Returns:
            Dict with 'processed_files' key containing S3 URI of manifest
        """
        # Load input files
        files_manifest = self.load_input("files")
        
        # Handle both list of paths and manifest dict
        if isinstance(files_manifest, list):
            file_paths = files_manifest
        elif isinstance(files_manifest, pd.DataFrame):
            # If it's a DataFrame, extract the values
            file_paths = files_manifest.iloc[:, 0].tolist()
        else:
            file_paths = list(files_manifest.values()) if isinstance(files_manifest, dict) else [files_manifest]
        
        # Get parameters
        detect_time_ranges = self.get_param("detect_time_ranges", True)
        detect_hebrew_dates = self.get_param("detect_hebrew_dates", True)
        clean_time_prefixes = self.get_param("clean_time_prefixes", True)
        context_path = self.get_param("context_path")
        
        # Initialize pipeline
        pipeline = PreprocessingPipeline(
            context_path=context_path,
        )
        
        processed_files = []
        for i, file_path in enumerate(file_paths):
            self.logger.info(f"Preprocessing file {i+1}/{len(file_paths)}: {file_path}")
            
            # Check if it's an S3 URI or local path
            if isinstance(file_path, str) and file_path.startswith("s3://"):
                # Download from S3 to temp file
                with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
                    local_path = tmp.name
                    self.download_from_s3(file_path, local_path)
            else:
                local_path = file_path
            
            try:
                # Process with auto-detection
                result = pipeline.process_with_auto_transforms(
                    local_path,
                    detect_time_ranges=detect_time_ranges,
                    detect_hebrew_dates=detect_hebrew_dates,
                    clean_time_prefixes=clean_time_prefixes,
                )
                
                # Log processing stats
                self.logger.info(
                    f"  Processed: {result.original_rows} → {result.final_rows} rows, "
                    f"metadata columns: {result.metadata_columns_added}"
                )
                
                # Save processed DataFrame to S3
                original_name = Path(file_path).stem if file_path else f"file_{i}"
                s3_uri = self.save_to_s3(f"processed/{original_name}", result.df)
                processed_files.append(s3_uri)
                
            finally:
                # Clean up temp file if we created one
                if file_path.startswith("s3://") and os.path.exists(local_path):
                    os.unlink(local_path)
        
        # Save manifest of processed files
        result_uri = self.save_to_s3("processed_files", processed_files)
        
        return {"processed_files": result_uri}


# Register the block
@BlockRegistry.register(
    name="preprocess_csvs",
    inputs=[{"name": "files", "ontology": DataType.S3_REFERENCES}],
    outputs=[{"name": "processed_files", "ontology": DataType.S3_REFERENCES}],
    parameters=[
        {"name": "detect_time_ranges", "type": "boolean", "default": True, "description": "Auto-detect and split time range columns"},
        {"name": "detect_hebrew_dates", "type": "boolean", "default": True, "description": "Auto-detect and normalize Hebrew dates"},
        {"name": "clean_time_prefixes", "type": "boolean", "default": True, "description": "Auto-clean time prefixes"},
        {"name": "context_path", "type": "string", "default": None, "description": "Path to context YAML for location normalization"},
    ],
    block_class=PreprocessBlock,
    description="Preprocess CSVs: detect structure, extract metadata, apply transformations",
)
def preprocess_csvs(ctx: BlockContext) -> Dict[str, str]:
    """Preprocess CSV files."""
    return PreprocessBlock(ctx).run()

