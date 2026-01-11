"""
Block Definition and Registry.

Provides:
- Port: Input/output port definition with ontology type
- Parameter: Block parameter definition
- BlockDefinition: Complete block specification
- BlockRegistry: Unified registry for all blocks
"""

import importlib
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Type

from csv_analyzer.workflows.ontology import DataType

logger = logging.getLogger(__name__)


@dataclass
class Port:
    """
    Input or output port of a block.
    
    Ports define the data flowing in/out of blocks with type information
    for validation at DAG build time.
    """
    name: str
    ontology: DataType
    required: bool = True
    optional: bool = False
    description: str = ""
    source: Optional[str] = None  # For inputs: "block_id.output_name"
    dynamic: bool = False  # If True, block accepts any input name (for sink blocks)


@dataclass
class Parameter:
    """
    Block parameter definition.
    
    Parameters are configuration values that can be set at:
    - Block definition (defaults)
    - Workflow YAML
    - Environment variables
    - Runtime (CLI args)
    """
    name: str
    type: Literal["string", "integer", "float", "boolean", "file_list"]
    required: bool = False
    default: Any = None
    description: str = ""


@dataclass
class BlockDefinition:
    """
    Complete block definition.
    
    Contains all metadata about a block including:
    - inputs/outputs with ontology types
    - parameters with defaults
    - handler function reference
    """
    name: str
    type: Literal["sql", "code", "builtin"] = "code"
    inputs: List[Port] = field(default_factory=list)
    outputs: List[Port] = field(default_factory=list)
    parameters: List[Parameter] = field(default_factory=list)
    handler: Optional[str] = None      # Python function name for code/builtin
    block_class: Optional[Type] = None # Class-based block
    sql: Optional[str] = None          # SQL template for sql type
    description: str = ""
    
    def get_input(self, name: str) -> Optional[Port]:
        """Get input port by name."""
        for port in self.inputs:
            if port.name == name:
                return port
        return None
    
    def get_output(self, name: str) -> Optional[Port]:
        """Get output port by name."""
        for port in self.outputs:
            if port.name == name:
                return port
        return None
    
    def get_parameter(self, name: str) -> Optional[Parameter]:
        """Get parameter by name."""
        for param in self.parameters:
            if param.name == name:
                return param
        return None


class BlockRegistry:
    """
    Unified registry for all block types: builtin, insights, and custom.
    
    Replaces the old CodeInsightsRegistry with a single source of truth.
    Blocks can be registered via:
    - @BlockRegistry.register() decorator for functions
    - BlockRegistry.register_class() for class-based blocks
    
    Usage:
        @BlockRegistry.register(
            name="my_block",
            inputs=[{"name": "data", "ontology": DataType.CLASSIFIED_DATA}],
            outputs=[{"name": "result", "ontology": DataType.INSIGHT_RESULT}],
            parameters=[{"name": "threshold", "type": "integer", "default": 30}]
        )
        def my_block(ctx: BlockContext) -> dict:
            ...
    """
    
    _handlers: Dict[str, Callable] = {}
    _definitions: Dict[str, BlockDefinition] = {}
    _block_classes: Dict[str, Type] = {}
    _initialized: bool = False
    
    @classmethod
    def register(
        cls,
        name: str,
        inputs: List[dict] = None,
        outputs: List[dict] = None,
        parameters: List[dict] = None,
        block_type: Literal["sql", "code", "builtin"] = "code",
        block_class: Type = None,
        description: str = "",
    ):
        """
        Decorator to register a block handler.
        
        Args:
            name: Unique block name
            inputs: List of input port definitions
            outputs: List of output port definitions
            parameters: List of parameter definitions
            block_type: Type of block (sql, code, builtin)
            block_class: Optional class implementing BaseBlock
            description: Human-readable description
            
        Usage:
            @BlockRegistry.register(
                name="early_arrival_matcher",
                inputs=[{"name": "data", "ontology": DataType.CLASSIFIED_DATA}],
                outputs=[{"name": "result", "ontology": DataType.INSIGHT_RESULT}],
                parameters=[{"name": "max_early_minutes", "type": "integer", "default": 30}]
            )
            def early_arrival_matcher(ctx: BlockContext) -> dict:
                ...
        """
        inputs = inputs or []
        outputs = outputs or []
        parameters = parameters or []
        
        def decorator(func):
            # Convert dict inputs to Port objects
            input_ports = []
            for i in inputs:
                ontology = i.get("ontology")
                if isinstance(ontology, str):
                    ontology = DataType(ontology)
                input_ports.append(Port(
                    name=i["name"],
                    ontology=ontology,
                    required=i.get("required", True),
                    optional=i.get("optional", False),
                    description=i.get("description", ""),
                    dynamic=i.get("dynamic", False),
                ))
            
            # Convert dict outputs to Port objects
            output_ports = []
            for o in outputs:
                ontology = o.get("ontology")
                if isinstance(ontology, str):
                    ontology = DataType(ontology)
                output_ports.append(Port(
                    name=o["name"],
                    ontology=ontology,
                    required=o.get("required", True),
                    optional=o.get("optional", False),
                    description=o.get("description", ""),
                ))
            
            # Convert dict parameters to Parameter objects
            param_objs = []
            for p in parameters:
                param_objs.append(Parameter(
                    name=p["name"],
                    type=p.get("type", "string"),
                    required=p.get("required", False),
                    default=p.get("default"),
                    description=p.get("description", ""),
                ))
            
            # Store handler and definition
            cls._handlers[name] = func
            cls._definitions[name] = BlockDefinition(
                name=name,
                type=block_type,
                inputs=input_ports,
                outputs=output_ports,
                parameters=param_objs,
                handler=name,
                block_class=block_class,
                description=description,
            )
            
            if block_class:
                cls._block_classes[name] = block_class
            
            logger.debug(f"Registered block: {name}")
            return func
        
        return decorator
    
    @classmethod
    def register_class(
        cls,
        name: str,
        block_class: Type,
        inputs: List[dict] = None,
        outputs: List[dict] = None,
        parameters: List[dict] = None,
        description: str = "",
    ):
        """
        Register a class-based block.
        
        Args:
            name: Unique block name
            block_class: Class implementing BaseBlock
            inputs: List of input port definitions
            outputs: List of output port definitions
            parameters: List of parameter definitions
            description: Human-readable description
        """
        inputs = inputs or []
        outputs = outputs or []
        parameters = parameters or []
        
        # Convert dicts to proper objects
        input_ports = [
            Port(
                name=i["name"],
                ontology=DataType(i["ontology"]) if isinstance(i.get("ontology"), str) else i.get("ontology"),
                required=i.get("required", True),
                optional=i.get("optional", False),
                description=i.get("description", ""),
            )
            for i in inputs
        ]
        
        output_ports = [
            Port(
                name=o["name"],
                ontology=DataType(o["ontology"]) if isinstance(o.get("ontology"), str) else o.get("ontology"),
                required=o.get("required", True),
                optional=o.get("optional", False),
                description=o.get("description", ""),
            )
            for o in outputs
        ]
        
        param_objs = [
            Parameter(
                name=p["name"],
                type=p.get("type", "string"),
                required=p.get("required", False),
                default=p.get("default"),
                description=p.get("description", ""),
            )
            for p in parameters
        ]
        
        # Store class and definition
        cls._block_classes[name] = block_class
        cls._definitions[name] = BlockDefinition(
            name=name,
            type="code",
            inputs=input_ports,
            outputs=output_ports,
            parameters=param_objs,
            handler=name,
            block_class=block_class,
            description=description,
        )
        
        logger.debug(f"Registered block class: {name}")
    
    @classmethod
    def get(cls, name: str) -> Optional[Callable]:
        """Get a block handler by name."""
        cls._ensure_initialized()
        return cls._handlers.get(name)
    
    @classmethod
    def get_class(cls, name: str) -> Optional[Type]:
        """Get a block class by name."""
        cls._ensure_initialized()
        return cls._block_classes.get(name)
    
    @classmethod
    def get_definition(cls, name: str) -> Optional[BlockDefinition]:
        """Get a block definition by name."""
        cls._ensure_initialized()
        return cls._definitions.get(name)
    
    @classmethod
    def list_blocks(cls) -> List[str]:
        """List all registered block names."""
        cls._ensure_initialized()
        return list(cls._definitions.keys())
    
    @classmethod
    def has_block(cls, name: str) -> bool:
        """Check if a block is registered."""
        cls._ensure_initialized()
        return name in cls._definitions
    
    @classmethod
    def _ensure_initialized(cls):
        """Auto-discover and import all block modules."""
        if cls._initialized:
            return
        cls._initialized = True
        cls._discover_blocks()
    
    @classmethod
    def _discover_blocks(cls):
        """
        Import all modules in blocks/builtin/ and blocks/insights/.
        This triggers the @register decorators.
        """
        base_path = Path(__file__).parent.parent / "blocks"
        
        if not base_path.exists():
            logger.warning(f"Blocks directory not found: {base_path}")
            return
        
        # Discover modules in subdirectories
        for subdir in ["builtin", "insights"]:
            subdir_path = base_path / subdir
            if not subdir_path.exists():
                continue
            
            for file_path in subdir_path.glob("*.py"):
                if file_path.name.startswith("_"):
                    continue
                
                module_name = f"csv_analyzer.blocks.{subdir}.{file_path.stem}"
                try:
                    importlib.import_module(module_name)
                    logger.debug(f"Discovered block module: {module_name}")
                except Exception as e:
                    logger.warning(f"Failed to import block module {module_name}: {e}")
    
    @classmethod
    def reset(cls):
        """Reset the registry (for testing)."""
        cls._handlers.clear()
        cls._definitions.clear()
        cls._block_classes.clear()
        cls._initialized = False

