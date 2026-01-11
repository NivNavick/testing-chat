"""
Workflow Engine.

Orchestrates DAG-based workflow execution:
- Loads workflow definitions from YAML
- Builds and validates DAG
- Executes blocks with topological ordering
- Supports parallel execution at each level
"""

import logging
import os
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

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
                source = input_spec.get("source", "")
                if source:
                    # Parse source: "block_id.output_name" or "block_id.output_name.field"
                    parts = source.split(".")
                    if len(parts) >= 2:
                        source_block_id = parts[0]
                        deps.add(source_block_id)
                        
                        # Validate ontology
                        self._validate_connection(source_block_id, parts[1], block.id, input_spec["name"])
            
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
    ) -> Dict[str, str]:
        """Gather inputs for a block from previous results."""
        inputs = {}
        
        for input_spec in block.inputs:
            input_name = input_spec["name"]
            source = input_spec.get("source", "")
            
            if not source:
                continue
            
            # Parse source: "block_id.output_name" or "block_id.output_name.field"
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
                
                inputs[input_name] = block_results[source_output]
        
        return inputs
    
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

