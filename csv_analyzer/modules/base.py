"""
Base module interface for all processing modules.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, TypeVar
from dataclasses import dataclass
from enum import Enum


class ProcessingMode(str, Enum):
    """Processing mode for the pipeline."""
    AUTO = "AUTO"           # Default, best-effort processing
    GUIDED = "GUIDED"       # Interactive, asks for confirmations
    STRICT = "STRICT"       # Production, only known types
    DISCOVERY = "DISCOVERY"  # Exploration, maximum intelligence


@dataclass
class ModuleContext:
    """Context passed to all modules during processing."""
    session_id: str
    mode: ProcessingMode = ProcessingMode.AUTO
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


class BaseModule(ABC, Generic[InputT, OutputT]):
    """
    Base class for all processing modules.
    
    Each module in the pipeline should inherit from this class
    and implement the process method.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Module name for logging and identification."""
        pass
    
    @abstractmethod
    async def process(self, input_data: InputT, context: ModuleContext) -> OutputT:
        """
        Process the input data and return the output.
        
        Args:
            input_data: The input to process
            context: Processing context with session info and metadata
            
        Returns:
            Processed output
        """
        pass
    
    async def validate_input(self, input_data: InputT, context: ModuleContext) -> bool:
        """
        Validate the input before processing.
        Override in subclasses for custom validation.
        
        Returns:
            True if input is valid, False otherwise
        """
        return True
    
    async def on_error(self, error: Exception, context: ModuleContext) -> None:
        """
        Handle errors during processing.
        Override in subclasses for custom error handling.
        """
        pass


class ModuleResult:
    """Result wrapper for module processing."""
    
    def __init__(
        self,
        success: bool,
        data: Any = None,
        error: str = None,
        metadata: Dict[str, Any] = None
    ):
        self.success = success
        self.data = data
        self.error = error
        self.metadata = metadata or {}
    
    @classmethod
    def ok(cls, data: Any, metadata: Dict[str, Any] = None) -> "ModuleResult":
        """Create a successful result."""
        return cls(success=True, data=data, metadata=metadata)
    
    @classmethod
    def fail(cls, error: str, metadata: Dict[str, Any] = None) -> "ModuleResult":
        """Create a failed result."""
        return cls(success=False, error=error, metadata=metadata)

