"""
Code Insights Package

Provides support for Python-based insights that implement complex logic
beyond what SQL can express.

Usage:
    from csv_analyzer.insights.code_insights import CodeInsightsRegistry
    
    @CodeInsightsRegistry.register("my_insight")
    def my_insight_handler(engine, params):
        # Access loaded tables via engine.execute_sql()
        df = engine.execute_sql("SELECT * FROM employee_shifts")
        # Process and return result DataFrame
        return result_df
"""

import importlib
import logging
import pkgutil
from pathlib import Path
from typing import Any, Callable, Dict, Optional, TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from csv_analyzer.insights.engine import InsightsEngine

logger = logging.getLogger(__name__)

# Type alias for handler functions
HandlerFunc = Callable[["InsightsEngine", Dict[str, Any]], pd.DataFrame]


class CodeInsightsRegistry:
    """
    Registry for Python-based insight handlers.
    
    Handlers are registered using the @register decorator and can be
    executed by name through the run() method.
    
    Example:
        @CodeInsightsRegistry.register("my_handler")
        def my_handler(engine: InsightsEngine, params: Dict) -> pd.DataFrame:
            shifts = engine.execute_sql("SELECT * FROM employee_shifts")
            # ... process data ...
            return result_df
    """
    
    _handlers: Dict[str, HandlerFunc] = {}
    _initialized: bool = False
    
    @classmethod
    def register(cls, name: str) -> Callable[[HandlerFunc], HandlerFunc]:
        """
        Decorator to register a code insight handler.
        
        Args:
            name: Unique identifier for the handler (matches 'handler' field in YAML)
            
        Returns:
            Decorator function
            
        Example:
            @CodeInsightsRegistry.register("early_arrival_matcher")
            def early_arrival_matcher(engine, params):
                ...
        """
        def decorator(func: HandlerFunc) -> HandlerFunc:
            if name in cls._handlers:
                logger.warning(f"Handler '{name}' already registered, overwriting")
            cls._handlers[name] = func
            logger.debug(f"Registered code insight handler: {name}")
            return func
        return decorator
    
    @classmethod
    def run(
        cls,
        name: str,
        engine: "InsightsEngine",
        params: Dict[str, Any],
    ) -> pd.DataFrame:
        """
        Execute a registered handler.
        
        Args:
            name: Handler name (as registered)
            engine: InsightsEngine instance with loaded tables
            params: Validated parameters from insight definition
            
        Returns:
            Result DataFrame from the handler
            
        Raises:
            ValueError: If handler not found
        """
        # Ensure handlers are discovered
        cls._ensure_initialized()
        
        if name not in cls._handlers:
            available = list(cls._handlers.keys())
            raise ValueError(
                f"Code insight handler '{name}' not found. "
                f"Available handlers: {available}"
            )
        
        handler = cls._handlers[name]
        logger.info(f"Running code insight handler: {name}")
        
        return handler(engine, params)
    
    @classmethod
    def get_handler(cls, name: str) -> Optional[HandlerFunc]:
        """Get a handler by name without executing it."""
        cls._ensure_initialized()
        return cls._handlers.get(name)
    
    @classmethod
    def list_handlers(cls) -> list:
        """List all registered handler names."""
        cls._ensure_initialized()
        return list(cls._handlers.keys())
    
    @classmethod
    def _ensure_initialized(cls) -> None:
        """Ensure all handler modules have been imported."""
        if cls._initialized:
            return
        
        cls._initialized = True
        cls._discover_handlers()
    
    @classmethod
    def _discover_handlers(cls) -> None:
        """
        Auto-discover and import all handler modules in this package.
        
        Looks for Python files in the code_insights directory and imports them,
        which triggers the @register decorators.
        """
        package_dir = Path(__file__).parent
        
        for module_info in pkgutil.iter_modules([str(package_dir)]):
            if module_info.name.startswith("_"):
                continue  # Skip __init__ and private modules
            
            try:
                module_name = f"csv_analyzer.insights.code_insights.{module_info.name}"
                importlib.import_module(module_name)
                logger.debug(f"Discovered code insight module: {module_info.name}")
            except Exception as e:
                logger.warning(f"Failed to import code insight module {module_info.name}: {e}")


# Export for convenience
__all__ = ["CodeInsightsRegistry"]

