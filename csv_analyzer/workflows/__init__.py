"""
Workflow Engine Package.

Provides DAG-based workflow execution with:
- Block definitions and registry
- Ontology-based type checking
- S3-based data passing
- Parallel execution

Usage:
    from csv_analyzer.workflows import WorkflowEngine, BlockRegistry, DataType
    
    # Run a workflow
    engine = WorkflowEngine.from_yaml("workflows/definitions/medical_early_arrival.yaml")
    results = engine.run({"files": ["shifts.csv", "actions.csv"]})
"""

from csv_analyzer.workflows.ontology import DataType, validate_connection
from csv_analyzer.workflows.block import BlockRegistry, BlockDefinition, Port, Parameter
from csv_analyzer.workflows.base_block import BaseBlock, BlockContext
from csv_analyzer.workflows.engine import WorkflowEngine, Workflow, run_workflow
from csv_analyzer.workflows.executor import BlockExecutor

__all__ = [
    # Ontology
    "DataType",
    "validate_connection",
    # Block system
    "BlockRegistry",
    "BlockDefinition",
    "Port",
    "Parameter",
    # Base classes
    "BaseBlock",
    "BlockContext",
    # Engine
    "WorkflowEngine",
    "Workflow",
    "run_workflow",
    "BlockExecutor",
]

