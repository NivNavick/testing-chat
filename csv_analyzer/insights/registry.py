"""
Insights Registry - Discovers and loads insight definitions from YAML files.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from csv_analyzer.insights.models import InsightDefinition

logger = logging.getLogger(__name__)

# Default path for insight definitions
DEFAULT_DEFINITIONS_PATH = Path(__file__).parent / "definitions"


class InsightsRegistry:
    """
    Registry for insight definitions.
    
    Discovers and loads insight definitions from YAML files,
    providing lookup and validation capabilities.
    
    Usage:
        registry = InsightsRegistry()
        registry.load_definitions()
        
        insight = registry.get("cost_per_shift")
        print(insight.sql)
    """
    
    def __init__(self, definitions_path: Optional[Path] = None):
        """
        Initialize the registry.
        
        Args:
            definitions_path: Path to directory containing YAML definitions.
                            If None, uses the default definitions/ directory.
        """
        self.definitions_path = definitions_path or DEFAULT_DEFINITIONS_PATH
        self._insights: Dict[str, InsightDefinition] = {}
        self._loaded = False
    
    def load_definitions(self, force_reload: bool = False) -> int:
        """
        Load all insight definitions from YAML files.
        
        Args:
            force_reload: If True, reload even if already loaded.
            
        Returns:
            Number of insights loaded.
        """
        if self._loaded and not force_reload:
            return len(self._insights)
        
        self._insights.clear()
        
        if not self.definitions_path.exists():
            logger.warning(f"Definitions path does not exist: {self.definitions_path}")
            return 0
        
        # Load all YAML files
        yaml_files = list(self.definitions_path.glob("*.yaml")) + \
                     list(self.definitions_path.glob("*.yml"))
        
        for yaml_file in yaml_files:
            try:
                self._load_file(yaml_file)
            except Exception as e:
                logger.error(f"Failed to load {yaml_file}: {e}")
        
        self._loaded = True
        logger.info(f"Loaded {len(self._insights)} insight definitions")
        return len(self._insights)
    
    def _load_file(self, path: Path) -> None:
        """Load insight definitions from a single YAML file."""
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        
        if not data:
            logger.warning(f"Empty YAML file: {path}")
            return
        
        # Support single insight or list of insights
        if isinstance(data, list):
            for item in data:
                self._register_insight(item, path)
        else:
            self._register_insight(data, path)
    
    def _register_insight(self, data: dict, source_file: Path) -> None:
        """Register a single insight definition."""
        try:
            insight = InsightDefinition.from_dict(data, source_file)
            
            if insight.name in self._insights:
                logger.warning(
                    f"Duplicate insight name '{insight.name}' - "
                    f"overwriting with definition from {source_file}"
                )
            
            self._insights[insight.name] = insight
            logger.debug(f"Registered insight: {insight.name}")
            
        except KeyError as e:
            logger.error(f"Missing required field in {source_file}: {e}")
        except Exception as e:
            logger.error(f"Failed to parse insight in {source_file}: {e}")
    
    def get(self, name: str) -> Optional[InsightDefinition]:
        """
        Get an insight definition by name.
        
        Args:
            name: The insight name (e.g., "cost_per_shift")
            
        Returns:
            InsightDefinition if found, None otherwise.
        """
        if not self._loaded:
            self.load_definitions()
        return self._insights.get(name)
    
    def list_all(self) -> List[InsightDefinition]:
        """Get all registered insights."""
        if not self._loaded:
            self.load_definitions()
        return list(self._insights.values())
    
    def list_names(self) -> List[str]:
        """Get names of all registered insights."""
        if not self._loaded:
            self.load_definitions()
        return list(self._insights.keys())
    
    def list_by_category(self, category: str) -> List[InsightDefinition]:
        """Get insights filtered by category."""
        if not self._loaded:
            self.load_definitions()
        return [i for i in self._insights.values() if i.category == category]
    
    def list_by_tag(self, tag: str) -> List[InsightDefinition]:
        """Get insights that have a specific tag."""
        if not self._loaded:
            self.load_definitions()
        return [i for i in self._insights.values() if tag in i.tags]
    
    def list_by_requirements(self, available_tables: List[str]) -> List[InsightDefinition]:
        """
        Get insights that can run with the available tables.
        
        Args:
            available_tables: List of document types currently loaded.
            
        Returns:
            List of insights whose requirements are satisfied.
        """
        if not self._loaded:
            self.load_definitions()
        
        available_set = set(available_tables)
        return [
            i for i in self._insights.values()
            if set(i.requires).issubset(available_set)
        ]
    
    def get_required_tables(self, insight_name: str) -> List[str]:
        """Get the tables required by an insight."""
        insight = self.get(insight_name)
        return insight.requires if insight else []
    
    def validate_insight(self, name: str, available_tables: List[str]) -> Dict:
        """
        Validate whether an insight can run.
        
        Args:
            name: Insight name
            available_tables: Tables currently loaded
            
        Returns:
            Dict with 'valid' boolean and 'missing_tables' list
        """
        insight = self.get(name)
        if not insight:
            return {"valid": False, "error": f"Insight '{name}' not found"}
        
        available_set = set(available_tables)
        required_set = set(insight.requires)
        missing = required_set - available_set
        
        return {
            "valid": len(missing) == 0,
            "missing_tables": list(missing),
            "insight": insight.name,
        }
    
    def register(self, insight: InsightDefinition) -> None:
        """
        Programmatically register an insight definition.
        
        Useful for dynamically created insights.
        """
        self._insights[insight.name] = insight
        logger.debug(f"Programmatically registered insight: {insight.name}")
    
    def unregister(self, name: str) -> bool:
        """Remove an insight from the registry."""
        if name in self._insights:
            del self._insights[name]
            return True
        return False
    
    def summary(self) -> str:
        """Get a summary of registered insights."""
        if not self._loaded:
            self.load_definitions()
        
        lines = [f"Insights Registry: {len(self._insights)} insights"]
        for name, insight in sorted(self._insights.items()):
            lines.append(f"  - {name}: {insight.description[:50]}...")
            lines.append(f"    Requires: {', '.join(insight.requires)}")
        
        return "\n".join(lines)

