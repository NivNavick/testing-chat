"""
Simple Categorize Block - Categorize files by filename pattern.

A lightweight alternative to classification that doesn't require embeddings.
Uses filename patterns to determine document type.
"""

import logging
import re
from pathlib import Path
from typing import Dict, List

import pandas as pd

from csv_analyzer.workflows.block import BlockRegistry
from csv_analyzer.workflows.ontology import DataType
from csv_analyzer.workflows.base_block import BaseBlock, BlockContext

logger = logging.getLogger(__name__)


class SimpleCategorizeBlock(BaseBlock):
    """
    Categorize files by filename pattern.
    
    Rules:
    - Files with "shift" in name → employee_shifts
    - Files with "action" or "medical" in name → medical_actions
    - Files with "lab" in name → lab_results
    - Files with "appointment" in name → patient_appointments
    """
    
    # Filename patterns for each document type
    PATTERNS = {
        "employee_shifts": [r"shift", r"shifts", r"attendance", r"clock"],
        "medical_actions": [r"action", r"medical", r"procedure", r"treatment"],
        "lab_results": [r"lab", r"test", r"result"],
        "patient_appointments": [r"appointment", r"schedule", r"booking"],
    }
    
    def run(self) -> Dict[str, str]:
        """
        Categorize files by filename pattern.
        
        Returns:
            Dict with 'classified_data' key containing manifest URI
        """
        # Load input files
        files_manifest = self.load_input("files")
        
        # Handle input format
        if isinstance(files_manifest, list):
            file_uris = files_manifest
        elif isinstance(files_manifest, pd.DataFrame):
            file_uris = files_manifest.iloc[:, 0].tolist()
        else:
            file_uris = list(files_manifest.values()) if isinstance(files_manifest, dict) else [files_manifest]
        
        self.logger.info(f"Categorizing {len(file_uris)} files by filename pattern")
        
        # Categorize each file
        classified: Dict[str, pd.DataFrame] = {}
        
        for file_uri in file_uris:
            # Get filename from path
            filename = Path(file_uri).stem.lower()
            
            # Determine document type
            doc_type = self._detect_doc_type(filename)
            
            self.logger.info(f"  {Path(file_uri).name} → {doc_type}")
            
            # Load the file
            if isinstance(file_uri, str) and file_uri.startswith("s3://"):
                df = self.load_from_s3(file_uri)
            elif Path(file_uri).exists():
                df = self.load_from_s3(file_uri)  # Works for local JSON files too
            else:
                df = pd.read_csv(file_uri)
            
            # Merge with existing data of same type
            if doc_type in classified:
                classified[doc_type] = pd.concat(
                    [classified[doc_type], df],
                    ignore_index=True,
                    sort=False,
                )
                self.logger.info(f"    Merged: now {len(classified[doc_type])} rows")
            else:
                classified[doc_type] = df
        
        # Log summary
        for doc_type, df in classified.items():
            self.logger.info(f"  {doc_type}: {len(df)} rows, {len(df.columns)} columns")
        
        # Save classified data
        manifest_uri = self.save_classified_data(classified)
        
        return {"classified_data": manifest_uri}
    
    def _detect_doc_type(self, filename: str) -> str:
        """Detect document type from filename."""
        for doc_type, patterns in self.PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, filename, re.IGNORECASE):
                    return doc_type
        
        # Default to generic type
        return "unknown_document"


# Register the block
@BlockRegistry.register(
    name="simple_categorize",
    inputs=[{"name": "files", "ontology": DataType.S3_REFERENCES}],
    outputs=[{"name": "classified_data", "ontology": DataType.CLASSIFIED_DATA}],
    parameters=[],
    block_class=SimpleCategorizeBlock,
    description="Categorize files by filename pattern (no embeddings needed)",
)
def simple_categorize(ctx: BlockContext) -> Dict[str, str]:
    """Categorize files by filename pattern."""
    return SimpleCategorizeBlock(ctx).run()

