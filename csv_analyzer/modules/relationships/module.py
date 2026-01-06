"""
Relationship Detection Module - Detects relationships between tables.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import pandas as pd

from csv_analyzer.modules.base import BaseModule, ModuleContext, ModuleResult


@dataclass
class DetectedRelationship:
    """A detected relationship between tables."""
    from_table_id: str
    from_column: str
    to_table_id: str
    to_column: str
    relationship_type: str  # ONE_TO_ONE, ONE_TO_MANY, MANY_TO_MANY
    confidence: float
    method: str  # NAME_MATCH, VALUE_MATCH, LLM
    transform_sql: Optional[str] = None  # SQL to transform keys if needed


@dataclass
class RelationshipInput:
    """Input for relationship detection."""
    tables: Dict[str, pd.DataFrame]  # table_id -> DataFrame
    table_types: Dict[str, str]  # table_id -> document_type


@dataclass
class RelationshipOutput:
    """Output from relationship detection."""
    relationships: List[DetectedRelationship]


class RelationshipDetectionModule(BaseModule[RelationshipInput, RelationshipOutput]):
    """
    Module for detecting relationships between tables.
    
    Detection methods:
    1. Column name matching (employee_id <-> employee_id)
    2. Value overlap analysis
    3. LLM-based inference for complex relationships
    """
    
    @property
    def name(self) -> str:
        return "relationship_detection"
    
    async def process(
        self,
        input_data: RelationshipInput,
        context: ModuleContext
    ) -> ModuleResult:
        """
        Detect relationships between tables.
        
        Args:
            input_data: Tables and their types
            context: Processing context
            
        Returns:
            ModuleResult with detected relationships
        """
        try:
            relationships = []
            
            table_ids = list(input_data.tables.keys())
            
            # Compare each pair of tables
            for i, table_id_1 in enumerate(table_ids):
                for table_id_2 in table_ids[i + 1:]:
                    df1 = input_data.tables[table_id_1]
                    df2 = input_data.tables[table_id_2]
                    
                    # Find potential key columns
                    rels = self._detect_relationships_between_tables(
                        table_id_1=table_id_1,
                        df1=df1,
                        table_id_2=table_id_2,
                        df2=df2
                    )
                    relationships.extend(rels)
            
            output = RelationshipOutput(relationships=relationships)
            
            return ModuleResult.ok(
                data=output,
                metadata={"relationship_count": len(relationships)}
            )
            
        except Exception as e:
            return ModuleResult.fail(
                error=str(e)
            )
    
    def _detect_relationships_between_tables(
        self,
        table_id_1: str,
        df1: pd.DataFrame,
        table_id_2: str,
        df2: pd.DataFrame
    ) -> List[DetectedRelationship]:
        """Detect relationships between two tables."""
        relationships = []
        
        cols1 = list(df1.columns)
        cols2 = list(df2.columns)
        
        for col1 in cols1:
            for col2 in cols2:
                # Check for name-based match
                if self._columns_match_by_name(col1, col2):
                    confidence, rel_type = self._analyze_value_overlap(
                        df1[col1], df2[col2]
                    )
                    
                    if confidence > 0.5:
                        relationships.append(DetectedRelationship(
                            from_table_id=table_id_1,
                            from_column=col1,
                            to_table_id=table_id_2,
                            to_column=col2,
                            relationship_type=rel_type,
                            confidence=confidence,
                            method="NAME_MATCH"
                        ))
                    continue
                
                # Check for value-based match on ID-like columns
                if self._is_potential_key(col1) and self._is_potential_key(col2):
                    confidence, rel_type = self._analyze_value_overlap(
                        df1[col1], df2[col2]
                    )
                    
                    if confidence > 0.7:
                        relationships.append(DetectedRelationship(
                            from_table_id=table_id_1,
                            from_column=col1,
                            to_table_id=table_id_2,
                            to_column=col2,
                            relationship_type=rel_type,
                            confidence=confidence,
                            method="VALUE_MATCH"
                        ))
        
        return relationships
    
    def _columns_match_by_name(self, col1: str, col2: str) -> bool:
        """Check if column names suggest a relationship."""
        col1_lower = str(col1).lower().strip()
        col2_lower = str(col2).lower().strip()
        
        # Exact match
        if col1_lower == col2_lower:
            return True
        
        # Common suffixes
        key_suffixes = ["_id", "id", "_key", "_code", "_no", "_number"]
        
        for suffix in key_suffixes:
            col1_base = col1_lower.replace(suffix, "")
            col2_base = col2_lower.replace(suffix, "")
            
            if col1_base and col2_base and col1_base == col2_base:
                return True
        
        return False
    
    def _is_potential_key(self, col_name: str) -> bool:
        """Check if column name suggests it could be a key."""
        name_lower = str(col_name).lower()
        
        key_indicators = [
            "id", "key", "code", "no", "number", "num",
            "employee", "staff", "patient", "customer", "order"
        ]
        
        return any(ind in name_lower for ind in key_indicators)
    
    def _analyze_value_overlap(
        self,
        series1: pd.Series,
        series2: pd.Series
    ) -> tuple:
        """
        Analyze value overlap between two columns.
        
        Returns:
            Tuple of (confidence, relationship_type)
        """
        # Get unique values
        values1 = set(series1.dropna().astype(str))
        values2 = set(series2.dropna().astype(str))
        
        if not values1 or not values2:
            return 0.0, "UNKNOWN"
        
        # Calculate overlap
        overlap = values1.intersection(values2)
        overlap_ratio1 = len(overlap) / len(values1) if values1 else 0
        overlap_ratio2 = len(overlap) / len(values2) if values2 else 0
        
        # Determine relationship type
        if overlap_ratio1 > 0.9 and overlap_ratio2 > 0.9:
            rel_type = "ONE_TO_ONE"
            confidence = min(overlap_ratio1, overlap_ratio2)
        elif overlap_ratio1 > 0.8:
            rel_type = "ONE_TO_MANY"  # Table 1 has fewer unique, likely parent
            confidence = overlap_ratio1
        elif overlap_ratio2 > 0.8:
            rel_type = "MANY_TO_ONE"  # Table 2 has fewer unique
            confidence = overlap_ratio2
        else:
            rel_type = "MANY_TO_MANY"
            confidence = max(overlap_ratio1, overlap_ratio2) * 0.8
        
        return confidence, rel_type

