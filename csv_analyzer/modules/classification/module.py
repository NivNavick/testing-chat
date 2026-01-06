"""
Classification Module - Document type and column classification.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass

import pandas as pd

from csv_analyzer.modules.base import BaseModule, ModuleContext, ModuleResult


@dataclass
class ClassificationInput:
    """Input for classification processing."""
    table_id: str
    table_name: str
    df: pd.DataFrame
    session_vertical: str  # e.g., "medical"


@dataclass
class ColumnMapping:
    """Mapping of a source column to a target schema column."""
    source_column: str
    target_column: str
    confidence: float
    method: str  # EMBEDDING, LLM, EXACT


@dataclass
class ClassificationOutput:
    """Output from classification processing."""
    document_type: str
    confidence: float
    method: str  # EMBEDDING, LLM, HYBRID
    column_mappings: Dict[str, ColumnMapping]
    unmapped_columns: list
    llm_reasoning: Optional[str] = None


class ClassificationModule(BaseModule[ClassificationInput, ClassificationOutput]):
    """
    Module for classifying tables and mapping columns.
    
    Uses the existing classification engine but with persistence.
    """
    
    def __init__(self):
        self._engine = None
    
    @property
    def name(self) -> str:
        return "classification"
    
    def _get_engine(self):
        """Lazy load the classification engine."""
        if self._engine is None:
            try:
                from csv_analyzer.engines.classification_engine import ClassificationEngine
                self._engine = ClassificationEngine()
            except ImportError:
                return None
        return self._engine
    
    async def process(
        self,
        input_data: ClassificationInput,
        context: ModuleContext
    ) -> ModuleResult:
        """
        Classify a table and map columns.
        
        Args:
            input_data: Table data to classify
            context: Processing context
            
        Returns:
            ModuleResult with classification results
        """
        engine = self._get_engine()
        
        if engine is None:
            return ModuleResult.fail(
                error="Classification engine not available",
                metadata={"table_id": input_data.table_id}
            )
        
        try:
            # Use existing classification engine
            result = engine.classify(
                df=input_data.df,
                context=input_data.session_vertical
            )
            
            # Parse engine result
            if result.get("classification"):
                doc_type = result["classification"].get("document_type", "unknown")
                confidence = result["classification"].get("confidence", 0.0)
                method = result["classification"].get("method", "EMBEDDING")
            else:
                doc_type = "unknown"
                confidence = 0.0
                method = "NONE"
            
            # Parse column mappings
            column_mappings = {}
            unmapped = []
            
            if result.get("column_mappings"):
                for source_col, mapping in result["column_mappings"].items():
                    if mapping and mapping.get("target"):
                        column_mappings[source_col] = ColumnMapping(
                            source_column=source_col,
                            target_column=mapping["target"],
                            confidence=mapping.get("confidence", 0.0),
                            method=mapping.get("method", "EMBEDDING")
                        )
                    else:
                        unmapped.append(source_col)
            
            output = ClassificationOutput(
                document_type=doc_type,
                confidence=confidence,
                method=method,
                column_mappings=column_mappings,
                unmapped_columns=unmapped,
                llm_reasoning=result.get("llm_reasoning")
            )
            
            return ModuleResult.ok(
                data=output,
                metadata={
                    "table_id": input_data.table_id,
                    "document_type": doc_type,
                    "confidence": confidence
                }
            )
            
        except Exception as e:
            return ModuleResult.fail(
                error=str(e),
                metadata={"table_id": input_data.table_id}
            )
    
    async def classify_simple(
        self,
        df: pd.DataFrame,
        vertical: str = "medical"
    ) -> ClassificationOutput:
        """
        Simple classification without context.
        
        Args:
            df: DataFrame to classify
            vertical: Business vertical
            
        Returns:
            ClassificationOutput
        """
        input_data = ClassificationInput(
            table_id="temp",
            table_name="temp",
            df=df,
            session_vertical=vertical
        )
        
        context = ModuleContext(session_id="temp")
        result = await self.process(input_data, context)
        
        if result.success:
            return result.data
        else:
            # Return empty classification on error
            return ClassificationOutput(
                document_type="unknown",
                confidence=0.0,
                method="ERROR",
                column_mappings={},
                unmapped_columns=list(df.columns),
                llm_reasoning=result.error
            )

