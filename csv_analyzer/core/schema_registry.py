"""
Schema Registry - loads and manages target schema definitions.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)


class SchemaField:
    """Represents a single field in a target schema."""
    
    def __init__(self, data: Dict[str, Any]):
        self.name = data["name"]
        self.type = data["type"]
        self.required = data.get("required", False)
        self.description = data.get("description", "")
        # All aliases in any language go in the same list
        self.aliases = data.get("aliases", [])
        self.embedding_text = data.get("embedding_text", "")
        
        # Unit/format transformation support
        self.target_unit = data.get("target_unit")  # e.g., "minutes"
        self.accepts_units = data.get("accepts_units", [])  # e.g., ["minutes", "hours", "seconds"]
        self.target_format = data.get("target_format")  # e.g., "YYYY-MM-DD"
        self.accepts_formats = data.get("accepts_formats", [])  # e.g., ["DD/MM/YYYY", "MM/DD/YYYY"]
    
    def get_embedding_text(self) -> str:
        """
        Get the full text to embed for this field.
        Combines name, description, aliases (all languages), and embedding hints.
        Deduplicates tokens on-the-fly to avoid redundancy.
        The multilingual embedding model handles all languages automatically.
        """
        # Collect all tokens from various sources
        all_tokens = []
        
        # 1. Field name (highest priority)
        all_tokens.append(self.name)
        
        # 2. Add all aliases
        all_tokens.extend(self.aliases)
        
        # 3. Add tokens from description
        if self.description:
            all_tokens.extend(self.description.split())
        
        # 4. Add tokens from embedding_text hints
        if self.embedding_text:
            all_tokens.extend(self.embedding_text.split())
        
        # Deduplicate while preserving order (case-insensitive comparison)
        seen = set()
        unique_tokens = []
        for token in all_tokens:
            token_lower = token.lower()
            if token_lower not in seen:
                seen.add(token_lower)
                unique_tokens.append(token)
        
        return " ".join(unique_tokens)
    
    def __repr__(self):
        return f"SchemaField({self.name}, {self.type}, required={self.required})"


class TargetSchema:
    """Represents a target schema (e.g., employee_shifts)."""
    
    def __init__(self, data: Dict[str, Any]):
        self.name = data["name"]
        self.vertical = data["vertical"]
        self.version = data.get("version", "1.0")
        self.description = data.get("description", "")
        self.fields: List[SchemaField] = [
            SchemaField(f) for f in data.get("fields", [])
        ]
    
    def get_field(self, name: str) -> Optional[SchemaField]:
        """Get a field by name."""
        for field in self.fields:
            if field.name == name:
                return field
        return None
    
    def get_required_fields(self) -> List[SchemaField]:
        """Get all required fields."""
        return [f for f in self.fields if f.required]
    
    def __repr__(self):
        return f"TargetSchema({self.vertical}/{self.name}, {len(self.fields)} fields)"


class SchemaRegistry:
    """
    Registry for all target schemas.
    Loads schemas from YAML files on initialization.
    """
    
    def __init__(self, schemas_dir: Optional[str] = None):
        """
        Initialize the schema registry.
        
        Args:
            schemas_dir: Path to schemas directory. 
                        Defaults to csv_analyzer/schemas/
        """
        if schemas_dir is None:
            schemas_dir = Path(__file__).parent.parent / "schemas"
        
        self.schemas_dir = Path(schemas_dir)
        self.schemas: Dict[str, Dict[str, TargetSchema]] = {}  # vertical -> name -> schema
        
        self._load_all_schemas()
    
    def _load_all_schemas(self):
        """Load all schemas from the schemas directory."""
        if not self.schemas_dir.exists():
            logger.warning(f"Schemas directory not found: {self.schemas_dir}")
            return
        
        for vertical_dir in self.schemas_dir.iterdir():
            if not vertical_dir.is_dir():
                continue
            
            vertical_name = vertical_dir.name
            self.schemas[vertical_name] = {}
            
            for schema_file in vertical_dir.glob("*.yaml"):
                try:
                    with open(schema_file) as f:
                        data = yaml.safe_load(f)
                    
                    schema = TargetSchema(data)
                    self.schemas[vertical_name][schema.name] = schema
                    logger.info(f"Loaded schema: {vertical_name}/{schema.name}")
                except Exception as e:
                    logger.error(f"Failed to load schema {schema_file}: {e}")
    
    def get_schema(self, vertical: str, name: str) -> Optional[TargetSchema]:
        """Get a specific schema."""
        return self.schemas.get(vertical, {}).get(name)
    
    def get_all_schemas(self) -> List[TargetSchema]:
        """Get all schemas."""
        all_schemas = []
        for vertical_schemas in self.schemas.values():
            all_schemas.extend(vertical_schemas.values())
        return all_schemas
    
    def get_schemas_by_vertical(self, vertical: str) -> List[TargetSchema]:
        """Get all schemas for a vertical."""
        return list(self.schemas.get(vertical, {}).values())
    
    def get_all_fields(self) -> List[tuple]:
        """
        Get all fields from all schemas.
        
        Returns:
            List of (vertical, document_type, field) tuples
        """
        fields = []
        for vertical, schemas in self.schemas.items():
            for doc_type, schema in schemas.items():
                for field in schema.fields:
                    fields.append((vertical, doc_type, field))
        return fields
    
    @staticmethod
    def _normalize_for_alias_match(name: str) -> str:
        """
        Normalize a name for alias matching.
        
        Removes common separators (underscores, dashes, spaces) and lowercases.
        This allows 'EmpID' to match 'emp_id', 'Worker_Code' to match 'worker_code', etc.
        """
        import re
        # Remove underscores, dashes, spaces; lowercase
        return re.sub(r'[_\-\s]+', '', name.lower().strip())
    
    def build_alias_lookup(
        self,
        vertical: Optional[str] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Build a lookup table mapping aliases to their target fields.
        
        This enables O(1) exact-match lookups for column names that match
        known aliases, which should take priority over embedding similarity.
        
        Args:
            vertical: Optional vertical filter (e.g., "medical")
            
        Returns:
            Dict mapping normalized alias -> {
                "vertical": str,
                "document_type": str,
                "field_name": str,
                "field_type": str,
                "required": bool,
                "description": str,
            }
            
        Note:
            - Keys are normalized (lowercase, no separators)
            - Field names themselves are also added as aliases
            - If an alias appears in multiple schemas, first occurrence wins
            - Both original form and separator-stripped form are indexed
        """
        lookup = {}
        
        verticals_to_check = (
            [vertical] if vertical and vertical in self.schemas
            else self.schemas.keys()
        )
        
        for vert in verticals_to_check:
            for doc_type, schema in self.schemas.get(vert, {}).items():
                for field in schema.fields:
                    # Add field name itself as a key
                    all_aliases = [field.name] + field.aliases
                    
                    field_info = {
                        "vertical": vert,
                        "document_type": doc_type,
                        "field_name": field.name,
                        "field_type": field.type,
                        "required": field.required,
                        "description": field.description,
                    }
                    
                    for alias in all_aliases:
                        # Skip empty aliases
                        if not alias or not alias.strip():
                            continue
                        
                        # Add both normalized forms:
                        # 1. Simple lowercase (for exact matches like "worker_code")
                        simple_normalized = alias.lower().strip()
                        if simple_normalized and simple_normalized not in lookup:
                            lookup[simple_normalized] = field_info
                        
                        # 2. Separator-stripped (for matches like "EmpID" -> "emp_id")
                        stripped_normalized = self._normalize_for_alias_match(alias)
                        if stripped_normalized and stripped_normalized not in lookup:
                            lookup[stripped_normalized] = field_info
        
        logger.debug(f"Built alias lookup with {len(lookup)} entries")
        return lookup


# Global registry instance
_registry: Optional[SchemaRegistry] = None


def get_schema_registry(schemas_dir: Optional[str] = None) -> SchemaRegistry:
    """Get the global schema registry instance."""
    global _registry
    if _registry is None:
        _registry = SchemaRegistry(schemas_dir)
    return _registry
