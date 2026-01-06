"""
Business Rule Detection Module - Infers business rules from data.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import pandas as pd
import numpy as np

from csv_analyzer.modules.base import BaseModule, ModuleContext, ModuleResult


@dataclass
class DetectedRule:
    """A detected business rule."""
    rule_type: str  # FORMULA, CONDITIONAL, LOOKUP, CONSTRAINT
    description: str
    applies_to_tables: List[str]
    condition_sql: Optional[str] = None
    formula_sql: Optional[str] = None
    confidence: float = 0.0
    source: str = "DETECTED"  # DETECTED, USER_PROVIDED, LLM_INFERRED
    evidence: Optional[str] = None


@dataclass
class RuleDetectionInput:
    """Input for rule detection."""
    tables: Dict[str, pd.DataFrame]  # table_id -> DataFrame
    table_types: Dict[str, str]  # table_id -> document_type


@dataclass
class RuleDetectionOutput:
    """Output from rule detection."""
    rules: List[DetectedRule]


class BusinessRuleDetectionModule(BaseModule[RuleDetectionInput, RuleDetectionOutput]):
    """
    Module for detecting business rules from data patterns.
    
    Detection methods:
    1. Formula detection (e.g., total = quantity * price)
    2. Conditional rules (e.g., discount applied when quantity > 10)
    3. Lookup relationships (e.g., status codes)
    4. Constraints (e.g., values must be positive)
    """
    
    @property
    def name(self) -> str:
        return "business_rule_detection"
    
    async def process(
        self,
        input_data: RuleDetectionInput,
        context: ModuleContext
    ) -> ModuleResult:
        """
        Detect business rules in the data.
        
        Args:
            input_data: Tables to analyze
            context: Processing context
            
        Returns:
            ModuleResult with detected rules
        """
        try:
            rules = []
            
            for table_id, df in input_data.tables.items():
                # Detect formulas
                formula_rules = self._detect_formulas(table_id, df)
                rules.extend(formula_rules)
                
                # Detect constraints
                constraint_rules = self._detect_constraints(table_id, df)
                rules.extend(constraint_rules)
                
                # Detect conditional patterns
                conditional_rules = self._detect_conditionals(table_id, df)
                rules.extend(conditional_rules)
            
            output = RuleDetectionOutput(rules=rules)
            
            return ModuleResult.ok(
                data=output,
                metadata={"rule_count": len(rules)}
            )
            
        except Exception as e:
            return ModuleResult.fail(error=str(e))
    
    def _detect_formulas(
        self,
        table_id: str,
        df: pd.DataFrame
    ) -> List[DetectedRule]:
        """Detect formula relationships between numeric columns."""
        rules = []
        
        # Get numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            return rules
        
        # Check for multiplication relationships (total = a * b)
        for i, col_a in enumerate(numeric_cols):
            for j, col_b in enumerate(numeric_cols):
                if i >= j:
                    continue
                
                for col_result in numeric_cols:
                    if col_result in [col_a, col_b]:
                        continue
                    
                    # Check if result = a * b
                    try:
                        product = df[col_a] * df[col_b]
                        correlation = product.corr(df[col_result])
                        
                        if correlation > 0.99:
                            # Check for exact match
                            if np.allclose(product.dropna(), df[col_result].dropna(), rtol=0.01):
                                rules.append(DetectedRule(
                                    rule_type="FORMULA",
                                    description=f"{col_result} = {col_a} × {col_b}",
                                    applies_to_tables=[table_id],
                                    formula_sql=f"{col_result} = {col_a} * {col_b}",
                                    confidence=0.95,
                                    source="DETECTED",
                                    evidence=f"Correlation: {correlation:.4f}"
                                ))
                    except Exception:
                        continue
        
        # Check for sum relationships (total = a + b)
        for i, col_a in enumerate(numeric_cols):
            for j, col_b in enumerate(numeric_cols):
                if i >= j:
                    continue
                
                for col_result in numeric_cols:
                    if col_result in [col_a, col_b]:
                        continue
                    
                    try:
                        total = df[col_a] + df[col_b]
                        if np.allclose(total.dropna(), df[col_result].dropna(), rtol=0.01):
                            rules.append(DetectedRule(
                                rule_type="FORMULA",
                                description=f"{col_result} = {col_a} + {col_b}",
                                applies_to_tables=[table_id],
                                formula_sql=f"{col_result} = {col_a} + {col_b}",
                                confidence=0.95,
                                source="DETECTED"
                            ))
                    except Exception:
                        continue
        
        return rules
    
    def _detect_constraints(
        self,
        table_id: str,
        df: pd.DataFrame
    ) -> List[DetectedRule]:
        """Detect value constraints in columns."""
        rules = []
        
        for col in df.columns:
            # Check for non-negative constraint
            if pd.api.types.is_numeric_dtype(df[col]):
                if (df[col].dropna() >= 0).all():
                    # Check if column name suggests it should be positive
                    name_lower = str(col).lower()
                    if any(term in name_lower for term in ["amount", "price", "cost", "hours", "count", "quantity"]):
                        rules.append(DetectedRule(
                            rule_type="CONSTRAINT",
                            description=f"{col} must be non-negative",
                            applies_to_tables=[table_id],
                            condition_sql=f"{col} >= 0",
                            confidence=0.8,
                            source="DETECTED"
                        ))
            
            # Check for categorical constraints (limited set of values)
            if df[col].nunique() < 10 and df[col].nunique() >= 2:
                unique_values = df[col].dropna().unique().tolist()
                if all(isinstance(v, str) for v in unique_values):
                    rules.append(DetectedRule(
                        rule_type="CONSTRAINT",
                        description=f"{col} must be one of: {', '.join(map(str, unique_values))}",
                        applies_to_tables=[table_id],
                        condition_sql=f"{col} IN ({', '.join(repr(v) for v in unique_values)})",
                        confidence=0.7,
                        source="DETECTED"
                    ))
        
        return rules
    
    def _detect_conditionals(
        self,
        table_id: str,
        df: pd.DataFrame
    ) -> List[DetectedRule]:
        """Detect conditional business rules."""
        rules = []
        
        # Look for columns that might have conditional logic
        # e.g., discount applied based on quantity
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in numeric_cols:
            # Check for threshold patterns
            col_lower = str(col).lower()
            
            if "discount" in col_lower or "bonus" in col_lower:
                # Find potential condition columns
                for cond_col in numeric_cols:
                    if cond_col == col:
                        continue
                    
                    # Check if the discount/bonus is only applied above a threshold
                    non_zero_mask = df[col] > 0
                    if non_zero_mask.sum() > 0 and non_zero_mask.sum() < len(df):
                        cond_values = df.loc[non_zero_mask, cond_col]
                        if len(cond_values) > 0:
                            threshold = cond_values.min()
                            if threshold > df[cond_col].min():
                                rules.append(DetectedRule(
                                    rule_type="CONDITIONAL",
                                    description=f"{col} applied when {cond_col} >= {threshold}",
                                    applies_to_tables=[table_id],
                                    condition_sql=f"CASE WHEN {cond_col} >= {threshold} THEN {col} ELSE 0 END",
                                    confidence=0.6,
                                    source="DETECTED",
                                    evidence=f"Threshold detected: {threshold}"
                                ))
        
        return rules

