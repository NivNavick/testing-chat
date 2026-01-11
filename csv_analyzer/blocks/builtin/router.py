"""
Router Block - Route data by document type.

Takes CLASSIFIED_DATA (dict of doc_type -> DataFrame) and routes each
document type to a separate output, enabling conditional downstream execution.
"""

import logging
from typing import Dict, List

import pandas as pd

from csv_analyzer.workflows.block import BlockRegistry
from csv_analyzer.workflows.ontology import DataType
from csv_analyzer.workflows.base_block import BaseBlock, BlockContext

logger = logging.getLogger(__name__)


class RouterBlock(BaseBlock):
    """
    Route classified data by document type.
    
    For each document type in the input, creates a separate output:
    - employee_shifts → router.employee_shifts
    - medical_actions → router.medical_actions
    - etc.
    
    Downstream blocks can then:
    1. Connect to specific outputs: source: router.employee_shifts
    2. Use conditions: condition: router.employee_shifts AND router.medical_actions
    """
    
    def run(self) -> Dict[str, str]:
        """
        Route classified data by document type.
        
        Returns:
            Dict mapping doc_type to S3 URI for each detected document type
        """
        # Load classified data (manifest -> individual DataFrames)
        classified = self.load_classified_data("classified_data")
        
        outputs = {}
        for doc_type, df in classified.items():
            # Save each document type as a separate output
            s3_uri = self.save_to_s3(doc_type, df)
            outputs[doc_type] = s3_uri
            self.logger.info(f"Routed {doc_type}: {len(df)} rows → {s3_uri}")
        
        # Log available outputs for downstream blocks
        self.logger.info(f"Available outputs: {list(outputs.keys())}")
        
        return outputs


# Register the block with dynamic outputs
@BlockRegistry.register(
    name="route_by_doc_type",
    inputs=[{"name": "classified_data", "ontology": DataType.CLASSIFIED_DATA}],
    outputs=[
        # Define known document type outputs as optional
        {"name": "employee_shifts", "ontology": DataType.EMPLOYEE_SHIFTS, "optional": True},
        {"name": "medical_actions", "ontology": DataType.MEDICAL_ACTIONS, "optional": True},
        {"name": "lab_results", "ontology": DataType.LAB_RESULTS, "optional": True},
        {"name": "patient_appointments", "ontology": DataType.PATIENT_APPOINTMENTS, "optional": True},
    ],
    block_class=RouterBlock,
    description="Route classified data into separate outputs by document type",
)
def route_by_doc_type(ctx: BlockContext) -> Dict[str, str]:
    """Route classified data by document type."""
    return RouterBlock(ctx).run()

