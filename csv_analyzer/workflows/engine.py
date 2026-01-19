"""
Workflow Engine.

Orchestrates DAG-based workflow execution:
- Loads workflow definitions from YAML
- Builds and validates DAG
- Executes blocks with topological ordering
- Supports parallel execution at each level
"""

import json
import logging
import os
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
import yaml

from csv_analyzer.workflows.block import BlockRegistry, BlockDefinition, Port
from csv_analyzer.workflows.ontology import DataType, validate_connection
from csv_analyzer.workflows.base_block import BlockContext
from csv_analyzer.workflows.executor import BlockExecutor

logger = logging.getLogger(__name__)


@dataclass
class WorkflowBlock:
    """Block instance in a workflow."""
    id: str
    handler: str
    inputs: List[Dict[str, str]] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    condition: Optional[str] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkflowBlock":
        """Create from YAML dict."""
        return cls(
            id=data["id"],
            handler=data["handler"],
            inputs=data.get("inputs", []),
            parameters=data.get("parameters", {}),
            condition=data.get("condition"),
        )


@dataclass
class Workflow:
    """Workflow definition loaded from YAML."""
    name: str
    description: str = ""
    version: str = "1.0"
    parameters: Dict[str, Any] = field(default_factory=dict)
    blocks: List[WorkflowBlock] = field(default_factory=list)
    storage: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Workflow":
        """Create from YAML dict."""
        blocks = [WorkflowBlock.from_dict(b) for b in data.get("blocks", [])]
        return cls(
            name=data.get("name", "unnamed"),
            description=data.get("description", ""),
            version=data.get("version", "1.0"),
            parameters=data.get("parameters", {}),
            blocks=blocks,
            storage=data.get("storage", {}),
        )
    
    @classmethod
    def from_yaml(cls, path: str) -> "Workflow":
        """Load workflow from YAML file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)


class WorkflowEngine:
    """
    DAG-based workflow execution engine.
    
    Features:
    - Topological sorting for dependency resolution
    - Level-based parallel execution
    - Ontology validation at DAG build time
    - S3-based data passing between blocks
    - Conditional block execution
    """
    
    def __init__(
        self,
        workflow: Workflow,
        bucket: Optional[str] = None,
        max_workers: int = 4,
        local_storage_path: Optional[str] = None,
    ):
        """
        Initialize the workflow engine.
        
        Args:
            workflow: Workflow definition
            bucket: S3 bucket for data storage (optional if local_storage_path set)
            max_workers: Maximum parallel workers
            local_storage_path: Local directory for data storage (alternative to S3)
        """
        self.workflow = workflow
        self.bucket = bucket or os.environ.get("S3_BUCKET", "")
        self.max_workers = max_workers
        self.run_id = str(uuid.uuid4())[:8]
        self.local_storage_path = local_storage_path or ""
        
        # Initialize block registry
        BlockRegistry._ensure_initialized()
        
        # Build block lookup
        self.blocks_by_id = {b.id: b for b in workflow.blocks}
        
        # Validate and build DAG
        self._dag: Dict[str, Set[str]] = {}
        self._levels: List[List[str]] = []
        self._build_and_validate_dag()
    
    def _build_and_validate_dag(self) -> None:
        """Build DAG and validate ontology connections."""
        logger.info(f"Building DAG for workflow: {self.workflow.name}")
        
        # Build dependency graph
        for block in self.workflow.blocks:
            deps = set()
            
            for input_spec in block.inputs:
                # Get input name (supports both 'name' and 'field' syntax)
                input_name = input_spec.get("name") or input_spec.get("field")
                
                # Support both single "source" and multiple "sources"
                sources = list(input_spec.get("sources", []))
                if input_spec.get("source"):
                    sources.append(input_spec["source"])
                
                for source in sources:
                    if source:
                        # Parse source: "block_id.output_name" or "block_id.output_name.field"
                        parts = source.split(".")
                        if len(parts) >= 2:
                            source_block_id = parts[0]
                            deps.add(source_block_id)
                            
                            # Validate ontology (skip for field-level sources)
                            if len(parts) == 2:  # block.output format
                                self._validate_connection(source_block_id, parts[1], block.id, input_name)
                            # For block.output.field format, validation is more complex (skip for now)
            
            self._dag[block.id] = deps
        
        # Compute execution levels
        self._levels = self._compute_levels()
        
        logger.info(f"DAG built: {len(self.workflow.blocks)} blocks, {len(self._levels)} levels")
        for i, level in enumerate(self._levels):
            logger.info(f"  Level {i}: {level}")
    
    def _validate_connection(
        self,
        source_block_id: str,
        source_output: str,
        target_block_id: str,
        target_input: str,
    ) -> None:
        """Validate ontology connection between blocks."""
        source_block = self.blocks_by_id.get(source_block_id)
        target_block = self.blocks_by_id.get(target_block_id)
        
        if not source_block:
            raise ValueError(f"Source block not found: {source_block_id}")
        if not target_block:
            raise ValueError(f"Target block not found: {target_block_id}")
        
        # Get block definitions
        source_def = BlockRegistry.get_definition(source_block.handler)
        target_def = BlockRegistry.get_definition(target_block.handler)
        
        if not source_def:
            raise ValueError(f"Block definition not found: {source_block.handler}")
        if not target_def:
            raise ValueError(f"Block definition not found: {target_block.handler}")
        
        # Find output and input ports
        source_port = source_def.get_output(source_output)
        target_port = target_def.get_input(target_input)
        
        if not source_port:
            # Check if it's a dynamic output (router can create outputs on the fly)
            logger.debug(f"Output '{source_output}' not in definition, assuming dynamic")
            return
        
        if not target_port:
            # Check if block accepts dynamic inputs (name="*")
            dynamic_port = target_def.get_input("*")
            if dynamic_port and dynamic_port.dynamic:
                logger.debug(f"Block '{target_block.handler}' accepts dynamic input '{target_input}'")
                return
            raise ValueError(f"Input '{target_input}' not found in block '{target_block.handler}'")
        
        # Validate ontology match
        if not validate_connection(source_port.ontology, target_port.ontology):
            raise ValueError(
                f"Ontology mismatch: {source_block_id}.{source_output} ({source_port.ontology}) "
                f"→ {target_block_id}.{target_input} ({target_port.ontology})"
            )
    
    def _compute_levels(self) -> List[List[str]]:
        """
        Compute execution levels via topological sort.
        
        Blocks at the same level have no inter-dependencies
        and can be executed in parallel.
        """
        # Calculate in-degree for each node
        in_degree = {node: len(deps) for node, deps in self._dag.items()}
        
        levels = []
        remaining = set(self._dag.keys())
        
        while remaining:
            # Find all nodes with in_degree 0 (no remaining dependencies)
            level = [n for n in remaining if in_degree.get(n, 0) == 0]
            
            if not level:
                raise ValueError("Cycle detected in workflow DAG")
            
            levels.append(level)
            
            # Remove this level's nodes and update in_degrees
            for node in level:
                remaining.remove(node)
                # Decrease in_degree for nodes that depend on this one
                for other_node, deps in self._dag.items():
                    if node in deps:
                        in_degree[other_node] -= 1
        
        return levels
    
    def run(
        self,
        runtime_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Dict[str, str]]:
        """
        Execute the workflow.
        
        Args:
            runtime_params: Parameters passed at runtime (highest priority)
            
        Returns:
            Dict mapping block_id to outputs (output_name -> S3 URI)
        """
        runtime_params = runtime_params or {}
        
        logger.info(f"Starting workflow: {self.workflow.name} (run_id={self.run_id})")
        
        # Merge parameters: runtime > env > workflow defaults
        params = self._merge_parameters(runtime_params)
        
        # Results store
        results: Dict[str, Dict[str, str]] = {}
        
        # Execute level by level
        executor = BlockExecutor()
        
        try:
            with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                for level_num, block_ids in enumerate(self._levels):
                    logger.info(f"Executing level {level_num}: {block_ids}")
                    
                    # Submit all blocks at this level
                    futures = {}
                    
                    for block_id in block_ids:
                        block = self.blocks_by_id[block_id]
                        
                        # Check condition
                        if not self._should_run_block(block, results):
                            logger.info(f"Skipping block {block_id}: condition not met")
                            continue
                        
                        # Get block definition
                        block_def = BlockRegistry.get_definition(block.handler)
                        if not block_def:
                            raise ValueError(f"Block definition not found: {block.handler}")
                        
                        # Gather inputs from previous results
                        inputs = self._gather_inputs(block, results)
                        
                        # Merge block parameters
                        block_params = self._get_block_params(block, params)
                        
                        # Create context
                        ctx = BlockContext(
                            inputs=inputs,
                            params=block_params,
                            workflow_run_id=self.run_id,
                            block_id=block_id,
                            bucket=self.bucket,
                            local_storage_path=self.local_storage_path,
                        )
                        
                        # Submit to executor
                        future = pool.submit(executor.execute, block_def, ctx)
                        futures[future] = block_id
                    
                    # Wait for all blocks at this level
                    for future in as_completed(futures):
                        block_id = futures[future]
                        try:
                            result = future.result()
                            results[block_id] = result
                            logger.info(f"Block {block_id} completed: {list(result.keys())}")
                        except Exception as e:
                            logger.error(f"Block {block_id} failed: {e}")
                            raise
            
        finally:
            executor.close()
        
        logger.info(f"Workflow completed: {self.workflow.name}")
        
        return results
    
    def _merge_parameters(self, runtime_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge parameters with priority: runtime > env > workflow defaults.
        """
        params = dict(self.workflow.parameters)  # Start with workflow defaults
        
        # Override with environment variables
        for key in params:
            env_key = key.upper()
            if env_key in os.environ:
                params[key] = os.environ[env_key]
        
        # Override with runtime params (highest priority)
        params.update(runtime_params)
        
        return params
    
    def _get_block_params(
        self,
        block: WorkflowBlock,
        global_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Get merged parameters for a block."""
        # Start with global params
        params = dict(global_params)
        
        # Apply block-specific params
        for key, value in block.parameters.items():
            if isinstance(value, str) and value.startswith("{{") and value.endswith("}}"):
                # Template variable: {{param_name}}
                param_name = value[2:-2].strip()
                if param_name in global_params:
                    params[key] = global_params[param_name]
            else:
                params[key] = value
        
        return params
    
    def _gather_inputs(
        self,
        block: WorkflowBlock,
        results: Dict[str, Dict[str, str]],
    ) -> Dict[str, Any]:
        """
        Gather inputs for a block from previous results.
        
        Supports:
        - Legacy format: name + source → inputs[name] = uri
        - Field format: field + source → field-level correlation
        - Multiple sources: field + sources → correlation across blocks
        
        Field-level correlation:
        When multiple fields reference the same block, rows are correlated.
        Example:
            - field: status, source: early_arrival.status
            - field: name, source: early_arrival.name
        Result: DataFrame with correlated rows {status: X, name: Y}
        """
        inputs = {}
        
        # Separate field-level inputs from legacy inputs
        field_inputs = []
        legacy_inputs = []
        
        for input_spec in block.inputs:
            if "field" in input_spec:
                field_inputs.append(input_spec)
            else:
                legacy_inputs.append(input_spec)
        
        # Handle field-level inputs with correlation
        if field_inputs:
            correlated_data = self._correlate_field_inputs(field_inputs, results)
            inputs["_correlated_data"] = correlated_data
        
        # Handle legacy inputs (backwards compatible)
        for input_spec in legacy_inputs:
            input_name = input_spec["name"]
            
            # Support both single "source" and multiple "sources"
            sources = list(input_spec.get("sources", []))
            if input_spec.get("source"):
                sources.append(input_spec["source"])
            
            if not sources:
                continue
            
            # Collect URIs from all sources
            uris = []
            for source in sources:
                # Parse source: "block_id.output_name"
                parts = source.split(".")
                if len(parts) >= 2:
                    source_block_id = parts[0]
                    source_output = parts[1]
                    
                    if source_block_id not in results:
                        raise ValueError(f"Source block not executed: {source_block_id}")
                    
                    block_results = results[source_block_id]
                    if source_output not in block_results:
                        raise ValueError(
                            f"Output '{source_output}' not found in block '{source_block_id}'. "
                            f"Available: {list(block_results.keys())}"
                        )
                    
                    uris.append(block_results[source_output])
            
            # Single source → single URI; multiple sources → list of URIs
            if len(uris) == 1:
                inputs[input_name] = uris[0]
            else:
                inputs[input_name] = uris
        
        return inputs
    
    def _correlate_field_inputs(
        self,
        field_inputs: List[Dict[str, Any]],
        results: Dict[str, Dict[str, str]],
    ) -> pd.DataFrame:
        """
        Correlate field-level inputs from multiple sources.
        
        Groups fields by source block, loads each block's data once,
        extracts requested fields while preserving row correlation,
        then concatenates across blocks.
        
        Args:
            field_inputs: List of {field: name, source/sources: ...}
            results: Block execution results (block_id → {output → uri})
            
        Returns:
            DataFrame with correlated rows
        """
        # Group fields by source block.output
        # Structure: {(block_id, output_name): [(field_name, column_name), ...]}
        source_groups: Dict[Tuple[str, str], List[Tuple[str, str]]] = defaultdict(list)
        
        for input_spec in field_inputs:
            field_name = input_spec["field"]
            
            # Get sources (single or multiple)
            sources = list(input_spec.get("sources", []))
            if input_spec.get("source"):
                sources.append(input_spec["source"])
            
            for source in sources:
                # Parse: "block_id.column" or "block_id.output.column"
                parts = source.split(".")
                if len(parts) == 2:
                    # Format: block_id.column (default output)
                    block_id, column_name = parts
                    output_name = self._get_default_output(block_id, results)
                elif len(parts) >= 3:
                    # Format: block_id.output.column
                    block_id = parts[0]
                    output_name = parts[1]
                    column_name = ".".join(parts[2:])  # Handle columns with dots
                else:
                    raise ValueError(f"Invalid field source format: {source}")
                
                source_groups[(block_id, output_name)].append((field_name, column_name))
        
        # Load data from each source and extract fields
        all_rows: List[Dict[str, Any]] = []
        
        for (block_id, output_name), field_mappings in source_groups.items():
            # Get the data URI
            if block_id not in results:
                raise ValueError(f"Source block not executed: {block_id}")
            
            block_results = results[block_id]
            if output_name not in block_results:
                raise ValueError(
                    f"Output '{output_name}' not found in block '{block_id}'. "
                    f"Available: {list(block_results.keys())}"
                )
            
            data_uri = block_results[output_name]
            
            # Load the data
            df = self._load_data_from_uri(data_uri)
            
            if not isinstance(df, pd.DataFrame):
                # Convert list of dicts to DataFrame
                if isinstance(df, list):
                    df = pd.DataFrame(df)
                else:
                    raise ValueError(f"Cannot correlate non-tabular data from {block_id}.{output_name}")
            
            # Extract requested columns
            for i in range(len(df)):
                row = {"_source_block": block_id}
                for field_name, column_name in field_mappings:
                    if column_name in df.columns:
                        row[field_name] = df.iloc[i][column_name]
                    else:
                        logger.warning(f"Column '{column_name}' not found in {block_id}.{output_name}")
                        row[field_name] = None
                all_rows.append(row)
        
        if not all_rows:
            return pd.DataFrame()
        
        return pd.DataFrame(all_rows)
    
    def _get_default_output(self, block_id: str, results: Dict[str, Dict[str, str]]) -> str:
        """Get the default/first output name for a block."""
        if block_id not in results:
            raise ValueError(f"Block not executed: {block_id}")
        
        block_results = results[block_id]
        if not block_results:
            raise ValueError(f"Block '{block_id}' has no outputs")
        
        # Return the first output (commonly "result" or similar)
        return next(iter(block_results.keys()))
    
    def _load_data_from_uri(self, uri: str) -> Any:
        """Load data from S3 URI or local path."""
        # Check if this is a Parquet file
        is_parquet = uri.endswith(".parquet")
        
        if uri.startswith("s3://"):
            # S3 loading
            import boto3
            parts = uri[5:].split("/", 1)
            bucket = parts[0]
            key = parts[1]
            
            if is_parquet:
                # Use pandas to read Parquet directly from S3
                return pd.read_parquet(uri)
            else:
                s3 = boto3.client("s3")
                response = s3.get_object(Bucket=bucket, Key=key)
                data = json.loads(response["Body"].read().decode("utf-8"))
        else:
            # Local file
            if is_parquet:
                return pd.read_parquet(uri)
            else:
                with open(uri, 'r', encoding='utf-8') as f:
                    data = json.load(f)
        
        # Convert to DataFrame if it's a list of dicts
        if isinstance(data, list) and data and isinstance(data[0], dict):
            return pd.DataFrame(data)
        
        return data
    
    def _should_run_block(
        self,
        block: WorkflowBlock,
        results: Dict[str, Dict[str, str]],
    ) -> bool:
        """Check if block should run based on condition."""
        if not block.condition:
            return True
        
        # Parse condition: "block1.output1 AND block2.output2"
        condition = block.condition
        
        # Replace AND/OR with Python operators for evaluation
        parts = condition.replace(" AND ", " and ").replace(" OR ", " or ").split()
        
        for part in parts:
            if part in ("and", "or"):
                continue
            
            # Parse: "block_id.output_name"
            if "." in part:
                block_id, output_name = part.split(".", 1)
                
                # Check if output exists
                if block_id not in results:
                    return False
                if output_name not in results[block_id]:
                    return False
        
        return True
    
    @classmethod
    def from_yaml(
        cls,
        workflow_path: str,
        bucket: Optional[str] = None,
        max_workers: int = 4,
        local_storage_path: Optional[str] = None,
    ) -> "WorkflowEngine":
        """
        Create engine from workflow YAML file.
        
        Args:
            workflow_path: Path to workflow YAML
            bucket: S3 bucket for data storage
            max_workers: Maximum parallel workers
            local_storage_path: Local directory for data storage
            
        Returns:
            WorkflowEngine instance
        """
        workflow = Workflow.from_yaml(workflow_path)
        return cls(
            workflow,
            bucket=bucket,
            max_workers=max_workers,
            local_storage_path=local_storage_path,
        )


def run_workflow(
    workflow_name: str,
    runtime_params: Optional[Dict[str, Any]] = None,
    definitions_dir: Optional[str] = None,
    **engine_kwargs,
) -> Dict[str, Dict[str, str]]:
    """
    Convenience function to run a workflow by name.
    
    Args:
        workflow_name: Name of the workflow (without .yaml extension)
        runtime_params: Runtime parameters
        definitions_dir: Directory containing workflow YAMLs
        **engine_kwargs: Additional engine options
        
    Returns:
        Workflow results
    """
    if definitions_dir is None:
        definitions_dir = str(Path(__file__).parent / "definitions")
    
    workflow_path = Path(definitions_dir) / f"{workflow_name}.yaml"
    
    if not workflow_path.exists():
        raise FileNotFoundError(f"Workflow not found: {workflow_path}")
    
    engine = WorkflowEngine.from_yaml(str(workflow_path), **engine_kwargs)
    return engine.run(runtime_params)

