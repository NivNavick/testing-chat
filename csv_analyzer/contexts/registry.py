"""
Vertical Context Registry.

Loads and manages per-vertical domain knowledge for enhanced column matching.
Provides semantic descriptions and terminology for schema fields.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)


@dataclass
class FieldContext:
    """Context information for a single schema field."""
    semantic_description: str
    examples: List[str] = field(default_factory=list)
    
    def get_embedding_expansion(self) -> str:
        """Get text to expand field embeddings with."""
        # Clean up the description (remove extra whitespace)
        desc = " ".join(self.semantic_description.split())
        return desc


@dataclass
class VerticalContext:
    """
    Domain context for a vertical (e.g., medical, finance).
    
    Contains:
    - Rich semantic descriptions for schema fields
    - Domain-specific terminology for OpenAI prompts
    """
    name: str
    description: str
    field_contexts: Dict[str, FieldContext]
    terminology: Dict[str, str]
    
    def get_context_for_field(self, field_name: str) -> Optional[FieldContext]:
        """Get context for a specific field."""
        return self.field_contexts.get(field_name)
    
    def expand_embedding_text(self, field_name: str, base_text: str) -> str:
        """
        Expand a field's embedding text with semantic context.
        
        Args:
            field_name: Name of the schema field
            base_text: Original embedding_text from schema
            
        Returns:
            Expanded text for better semantic matching
        """
        context = self.get_context_for_field(field_name)
        if not context:
            return base_text
        
        expansion = context.get_embedding_expansion()
        
        # Combine: original text + semantic description
        # This helps the embedding model understand the broader concept
        return f"{base_text} {expansion}"
    
    def get_openai_context(self) -> str:
        """
        Generate context string for OpenAI prompts.
        
        Returns a formatted string with domain description and terminology.
        """
        lines = [
            f"DOMAIN: {self.name.upper()}",
            self.description.strip(),
            "",
            "TERMINOLOGY:",
        ]
        
        for term, definition in self.terminology.items():
            lines.append(f"- {term}: {definition}")
        
        return "\n".join(lines)
    
    def get_terminology_for_prompt(self) -> str:
        """Get terminology section formatted for OpenAI prompts."""
        if not self.terminology:
            return ""
        
        lines = ["Domain-specific terminology:"]
        for term, definition in self.terminology.items():
            lines.append(f"- {term}: {definition}")
        
        return "\n".join(lines)


class VerticalContextRegistry:
    """
    Singleton registry for loading and accessing vertical contexts.
    
    Contexts are loaded from YAML files in the contexts/ directory.
    
    Usage:
        registry = get_context_registry()
        context = registry.get_context("medical")
        
        # Expand embedding text with context
        expanded = context.expand_embedding_text("department_code", "department unit ward")
        
        # Get OpenAI prompt context
        openai_context = context.get_openai_context()
    """
    
    def __init__(self, contexts_dir: Optional[Path] = None):
        """
        Initialize the registry.
        
        Args:
            contexts_dir: Directory containing context YAML files.
                         Defaults to the 'contexts' directory next to this file.
        """
        if contexts_dir is None:
            contexts_dir = Path(__file__).parent
        
        self.contexts_dir = Path(contexts_dir)
        self._contexts: Dict[str, VerticalContext] = {}
        
        logger.debug(f"VerticalContextRegistry initialized with dir: {self.contexts_dir}")
    
    def load_context(self, vertical: str) -> VerticalContext:
        """
        Load a vertical context from YAML file.
        
        Args:
            vertical: Name of the vertical (e.g., "medical")
            
        Returns:
            Loaded VerticalContext
            
        Raises:
            FileNotFoundError: If context file doesn't exist
            ValueError: If context file is invalid
        """
        yaml_path = self.contexts_dir / f"{vertical}.yaml"
        
        if not yaml_path.exists():
            raise FileNotFoundError(
                f"Context file not found: {yaml_path}. "
                f"Available contexts: {self.list_verticals()}"
            )
        
        logger.info(f"Loading vertical context from: {yaml_path}")
        
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        
        if not data:
            raise ValueError(f"Empty or invalid context file: {yaml_path}")
        
        # Parse field contexts
        field_contexts = {}
        for field_name, field_data in data.get("field_contexts", {}).items():
            field_contexts[field_name] = FieldContext(
                semantic_description=field_data.get("semantic_description", ""),
                examples=field_data.get("examples", []),
            )
        
        context = VerticalContext(
            name=data.get("name", vertical),
            description=data.get("description", ""),
            field_contexts=field_contexts,
            terminology=data.get("terminology", {}),
        )
        
        # Cache it
        self._contexts[vertical] = context
        
        logger.info(
            f"Loaded context '{vertical}': "
            f"{len(field_contexts)} field contexts, "
            f"{len(context.terminology)} terminology entries"
        )
        
        return context
    
    def get_context(self, vertical: str) -> Optional[VerticalContext]:
        """
        Get a vertical context, loading it if necessary.
        
        Args:
            vertical: Name of the vertical
            
        Returns:
            VerticalContext or None if not found
        """
        # Return cached if available
        if vertical in self._contexts:
            return self._contexts[vertical]
        
        # Try to load
        try:
            return self.load_context(vertical)
        except FileNotFoundError:
            logger.warning(f"No context file found for vertical: {vertical}")
            return None
        except Exception as e:
            logger.error(f"Failed to load context for vertical '{vertical}': {e}")
            return None
    
    def list_verticals(self) -> List[str]:
        """List all available vertical context files."""
        if not self.contexts_dir.exists():
            return []
        
        return [
            p.stem for p in self.contexts_dir.glob("*.yaml")
            if p.stem != "__init__"
        ]
    
    def clear_cache(self):
        """Clear the context cache (useful for testing or reloading)."""
        self._contexts.clear()


# Global registry instance
_registry: Optional[VerticalContextRegistry] = None


def get_context_registry() -> VerticalContextRegistry:
    """Get the global vertical context registry instance."""
    global _registry
    if _registry is None:
        _registry = VerticalContextRegistry()
    return _registry


def get_vertical_context(vertical: str) -> Optional[VerticalContext]:
    """
    Convenience function to get a vertical context.
    
    Args:
        vertical: Name of the vertical (e.g., "medical")
        
    Returns:
        VerticalContext or None if not found
    """
    return get_context_registry().get_context(vertical)

