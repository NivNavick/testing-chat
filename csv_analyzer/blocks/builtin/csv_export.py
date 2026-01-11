"""
CSV Export Block - Aggregate multiple inputs into a CSV file.

This is a "sink" block that accepts multiple field connections
and combines them into a single CSV output file.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

import pandas as pd

from csv_analyzer.workflows.block import BlockRegistry
from csv_analyzer.workflows.ontology import DataType
from csv_analyzer.workflows.base_block import BaseBlock, BlockContext

logger = logging.getLogger(__name__)


class CSVExportBlock(BaseBlock):
    """
    Export data to CSV file.
    
    This block accepts multiple inputs and combines them into a single CSV.
    Each input becomes columns in the output CSV.
    
    Inputs can be:
    - DataFrames (all columns included)
    - Single fields/columns (added as a column)
    - Dicts (keys become columns)
    
    The block has NO outputs - it's a terminal/sink block that writes to disk.
    """
    
    def run(self) -> Dict[str, str]:
        """
        Combine all inputs into a CSV file.
        
        Returns:
            Dict with 'csv_path' pointing to the generated file
        """
        # Get parameters
        output_filename = self.get_param("filename", "output.csv")
        output_dir = self.get_param("output_dir", None)
        include_index = self.get_param("include_index", False)
        
        # Collect all input data
        self.logger.info(f"Collecting {len(self.ctx.inputs)} inputs for CSV export...")
        
        dataframes = []
        metadata = {}
        
        for input_name, input_uri in self.ctx.inputs.items():
            self.logger.info(f"  Loading input: {input_name}")
            
            try:
                data = self._load_input_data(input_uri)
                
                if isinstance(data, pd.DataFrame):
                    # Add source column to identify origin
                    df = data.copy()
                    df["_source"] = input_name
                    dataframes.append(df)
                    self.logger.info(f"    → DataFrame with {len(df)} rows, {len(df.columns)} columns")
                    
                elif isinstance(data, list):
                    if data and isinstance(data[0], dict):
                        # List of dicts → DataFrame
                        df = pd.DataFrame(data)
                        df["_source"] = input_name
                        dataframes.append(df)
                        self.logger.info(f"    → List of {len(data)} records → DataFrame")
                    else:
                        # List of values → single column
                        df = pd.DataFrame({input_name: data})
                        dataframes.append(df)
                        self.logger.info(f"    → List of {len(data)} values → column '{input_name}'")
                        
                elif isinstance(data, dict):
                    # Dict → metadata or single row
                    if all(isinstance(v, (str, int, float, bool, type(None))) for v in data.values()):
                        # Simple dict → metadata row
                        metadata.update(data)
                        self.logger.info(f"    → Dict with {len(data)} fields → metadata")
                    else:
                        # Complex dict (might have lists/nested) → try DataFrame
                        df = pd.DataFrame([data])
                        df["_source"] = input_name
                        dataframes.append(df)
                        self.logger.info(f"    → Dict → single row DataFrame")
                else:
                    self.logger.warning(f"    → Unknown type {type(data)}, skipping")
                    
            except Exception as e:
                self.logger.error(f"    → Error loading {input_name}: {e}")
                continue
        
        # Combine all DataFrames
        if not dataframes:
            self.logger.warning("No data to export!")
            return {"csv_path": ""}
        
        # Strategy: concatenate all DataFrames (union of columns)
        self.logger.info(f"Combining {len(dataframes)} DataFrames...")
        
        combined_df = pd.concat(dataframes, ignore_index=True, sort=False)
        
        # Add metadata as columns if any
        for key, value in metadata.items():
            combined_df[f"_meta_{key}"] = value
        
        # Remove internal columns if not wanted
        if not self.get_param("include_source_column", True):
            if "_source" in combined_df.columns:
                combined_df = combined_df.drop(columns=["_source"])
        
        # Apply filter if specified
        filter_column = self.get_param("filter_column", None)
        filter_value = self.get_param("filter_value", None)
        if filter_column and filter_value and filter_column in combined_df.columns:
            before_count = len(combined_df)
            combined_df = combined_df[combined_df[filter_column] == filter_value]
            self.logger.info(f"Filtered by {filter_column}='{filter_value}': {before_count} → {len(combined_df)} rows")
        
        self.logger.info(f"Combined DataFrame: {len(combined_df)} rows, {len(combined_df.columns)} columns")
        
        # Determine output path
        if output_dir:
            output_path = Path(output_dir) / output_filename
        elif self.ctx.local_output_dir:
            output_path = self.ctx.local_output_dir / self.ctx.workflow_run_id / self.ctx.block_id / output_filename
        else:
            output_path = Path(".") / output_filename
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write CSV
        combined_df.to_csv(output_path, index=include_index, encoding="utf-8-sig")
        self.logger.info(f"✅ CSV exported to: {output_path}")
        self.logger.info(f"   Rows: {len(combined_df)}, Columns: {len(combined_df.columns)}")
        
        # Return metadata (this is a sink block - no outputs to other blocks)
        return {
            "csv_path": str(output_path),
            "row_count": len(combined_df),
            "column_count": len(combined_df.columns),
        }
    
    def _load_input_data(self, uri: str) -> Any:
        """Load data from URI (S3 or local)."""
        # load_from_s3 handles both S3 URIs and local paths
        return self.load_from_s3(uri)


# Register the block
@BlockRegistry.register(
    name="export_csv",
    inputs=[
        # Accept any input - the block handles dynamic input names
        {"name": "*", "ontology": DataType.INSIGHT_RESULT, "required": False, "dynamic": True,
         "description": "Connect any data outputs here - they'll be combined into CSV columns"},
    ],
    outputs=[],  # No outputs - this is a sink block
    parameters=[
        {"name": "filename", "type": "string", "default": "output.csv", 
         "description": "Output CSV filename"},
        {"name": "output_dir", "type": "string", "default": None,
         "description": "Output directory (defaults to workflow output dir)"},
        {"name": "include_index", "type": "boolean", "default": False,
         "description": "Include row index in CSV"},
        {"name": "include_source_column", "type": "boolean", "default": False,
         "description": "Add _source column showing which input each row came from"},
        {"name": "filter_column", "type": "string", "default": None,
         "description": "Column name to filter by"},
        {"name": "filter_value", "type": "string", "default": None,
         "description": "Value to filter for (rows matching this value are kept)"},
    ],
    block_class=CSVExportBlock,
    description="Export combined data to CSV file (sink block - accepts any inputs)",
)
def export_csv(ctx: BlockContext) -> Dict[str, str]:
    """Export data to CSV."""
    return CSVExportBlock(ctx).run()

