"""
Transformation Detector.

Uses semantic analysis to detect what unit/format source data is in,
and determines if a transformation is needed to match the target schema.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from csv_analyzer.transformations.registry import (
    get_conversion,
    normalize_unit,
    UNIT_ALIASES,
    TransformationInfo,
)

logger = logging.getLogger(__name__)


@dataclass
class TransformationResult:
    """Result of transformation detection."""
    needs_transformation: bool
    source_unit: Optional[str]
    target_unit: Optional[str]
    transformation: Optional[TransformationInfo]
    confidence: str  # "high", "medium", "low"
    reason: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "needs_transformation": self.needs_transformation,
            "source_unit": self.source_unit,
            "target_unit": self.target_unit,
            "transform_function": self.transformation.name if self.transformation else None,
            "formula": self.transformation.formula_description if self.transformation else None,
            "confidence": self.confidence,
            "reason": self.reason,
        }


def detect_source_unit(
    column_name: str,
    sample_values: List[Any],
    target_field: str,
    target_unit: str,
    accepts_units: List[str],
    openai_client: Optional[Any] = None,
) -> Optional[str]:
    """
    Detect what unit the source data is in.
    
    Uses a combination of:
    1. Column name analysis (e.g., "dur_hrs" suggests hours)
    2. Sample value range analysis (e.g., 1-12 suggests hours, 60-480 suggests minutes)
    3. OpenAI semantic analysis (if available)
    
    Args:
        column_name: Name of the source column
        sample_values: Sample values from the column
        target_field: Name of the target field
        target_unit: What unit the target expects
        accepts_units: List of units the target can accept (with conversion)
        openai_client: Optional OpenAI client for semantic analysis
        
    Returns:
        Detected unit or None if unable to detect
    """
    # Step 1: Try to detect from column name
    detected = _detect_unit_from_name(column_name, accepts_units)
    if detected:
        logger.debug(f"Detected unit '{detected}' from column name '{column_name}'")
        return detected
    
    # Step 2: Try to detect from sample values
    detected = _detect_unit_from_values(sample_values, target_unit, accepts_units)
    if detected:
        logger.debug(f"Detected unit '{detected}' from sample values")
        return detected
    
    # Step 3: Use OpenAI if available
    if openai_client is not None:
        detected = _detect_unit_with_openai(
            openai_client,
            column_name,
            sample_values,
            target_field,
            target_unit,
            accepts_units,
        )
        if detected:
            logger.debug(f"Detected unit '{detected}' using OpenAI")
            return detected
    
    return None


def _detect_unit_from_name(column_name: str, accepts_units: List[str]) -> Optional[str]:
    """
    Try to detect unit from column name.
    
    Looks for unit indicators in the column name like:
    - "dur_hrs" → hours
    - "tiempo_horas" → hours
    - "weight_kg" → kg
    """
    name_lower = column_name.lower()
    
    # Check each accepted unit and its aliases
    for unit in accepts_units:
        unit_normalized = normalize_unit(unit)
        if unit_normalized is None:
            continue
        
        # Check if unit name or any alias appears in column name
        aliases = UNIT_ALIASES.get(unit_normalized, [])
        all_terms = [unit_normalized] + aliases
        
        for term in all_terms:
            term_lower = term.lower()
            # Check for term in column name (with word boundaries)
            if (
                f"_{term_lower}" in name_lower or
                f"{term_lower}_" in name_lower or
                name_lower.endswith(term_lower) or
                name_lower.startswith(term_lower) or
                name_lower == term_lower
            ):
                return unit_normalized
    
    return None


def _detect_unit_from_values(
    sample_values: List[Any],
    target_unit: str,
    accepts_units: List[str],
) -> Optional[str]:
    """
    Try to detect unit from sample value ranges.
    
    Uses heuristics based on value ranges:
    - If values are 0-24 and target is minutes → likely hours
    - If values are 30-600 and target is hours → likely minutes
    """
    try:
        numeric_values = [float(v) for v in sample_values if _is_numeric(v)]
        if not numeric_values:
            return None
        
        min_val = min(numeric_values)
        max_val = max(numeric_values)
        avg_val = sum(numeric_values) / len(numeric_values)
        
        # Time-based heuristics
        if target_unit == "minutes":
            # If values are small (0-24 range), likely hours
            if max_val <= 24 and min_val >= 0 and avg_val < 15:
                if "hours" in accepts_units:
                    return "hours"
            # If values are very small decimals, might be hours
            if max_val < 15 and all(v % 1 != 0 for v in numeric_values):
                if "hours" in accepts_units:
                    return "hours"
        
        elif target_unit == "hours":
            # If values are large (30+ range), likely minutes
            if min_val >= 30 and max_val <= 600:
                if "minutes" in accepts_units:
                    return "minutes"
        
        elif target_unit == "seconds":
            # If values are small (0-60), might be minutes
            if max_val <= 60 and min_val >= 0:
                if "minutes" in accepts_units:
                    return "minutes"
        
        # Temperature heuristics
        if target_unit == "celsius":
            # Fahrenheit typically has values 32-212
            if min_val >= 32 and max_val <= 212:
                if "fahrenheit" in accepts_units:
                    return "fahrenheit"
        
        elif target_unit == "fahrenheit":
            # Celsius typically has values -40 to 100
            if min_val >= -40 and max_val <= 100 and max_val < 50:
                if "celsius" in accepts_units:
                    return "celsius"
        
        # Weight heuristics
        if target_unit == "kg":
            # If values are suspiciously large, might be grams
            if min_val >= 100 and max_val >= 1000:
                if "g" in accepts_units:
                    return "g"
            # If values are suspiciously large with decimals, might be lbs
            if 100 <= avg_val <= 400:
                if "lbs" in accepts_units:
                    return "lbs"
        
    except (ValueError, TypeError):
        pass
    
    return None


def _is_numeric(value: Any) -> bool:
    """Check if a value is numeric."""
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return False


def _detect_unit_with_openai(
    openai_client: Any,
    column_name: str,
    sample_values: List[Any],
    target_field: str,
    target_unit: str,
    accepts_units: List[str],
) -> Optional[str]:
    """
    Use OpenAI to detect the source unit.
    """
    import json
    
    samples_str = ", ".join(str(v) for v in sample_values[:5])
    units_str = ", ".join(accepts_units)
    
    prompt = f"""Analyze this column and determine what unit the data is in.

COLUMN:
- Name: {column_name}
- Sample values: {samples_str}

TARGET FIELD: {target_field}
- Expected unit: {target_unit}
- Also accepts: {units_str}

Based on the column name and sample values, what unit is the source data in?
Consider the column name might be in any language.

Respond with JSON:
{{
    "detected_unit": one of [{units_str}] or null if can't determine,
    "confidence": "high" | "medium" | "low",
    "reason": "brief explanation"
}}"""

    try:
        response = openai_client.client.chat.completions.create(
            model=openai_client.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a data analysis expert. Detect units from column names and values."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        
        result_text = response.choices[0].message.content
        result = json.loads(result_text)
        
        detected = result.get("detected_unit")
        confidence = result.get("confidence", "low")
        reason = result.get("reason", "")
        
        if detected and detected in accepts_units:
            logger.info(
                f"OpenAI detected unit '{detected}' for '{column_name}' "
                f"({confidence} confidence): {reason}"
            )
            return detected
        
    except Exception as e:
        logger.warning(f"OpenAI unit detection failed for '{column_name}': {e}")
    
    return None


def detect_transformation(
    column_name: str,
    sample_values: List[Any],
    target_field: str,
    target_unit: Optional[str],
    accepts_units: Optional[List[str]],
    openai_client: Optional[Any] = None,
) -> TransformationResult:
    """
    Detect if a transformation is needed between source column and target field.
    
    Args:
        column_name: Name of the source column
        sample_values: Sample values from the column
        target_field: Name of the target field
        target_unit: What unit the target expects (e.g., "minutes")
        accepts_units: List of units the target can accept (e.g., ["minutes", "hours"])
        openai_client: Optional OpenAI client for semantic analysis
        
    Returns:
        TransformationResult with transformation details
    """
    # No transformation needed if target doesn't specify units
    if not target_unit or not accepts_units:
        return TransformationResult(
            needs_transformation=False,
            source_unit=None,
            target_unit=None,
            transformation=None,
            confidence="high",
            reason="Target field does not specify unit requirements",
        )
    
    # Detect source unit
    source_unit = detect_source_unit(
        column_name=column_name,
        sample_values=sample_values,
        target_field=target_field,
        target_unit=target_unit,
        accepts_units=accepts_units,
        openai_client=openai_client,
    )
    
    if source_unit is None:
        return TransformationResult(
            needs_transformation=False,
            source_unit=None,
            target_unit=target_unit,
            transformation=None,
            confidence="low",
            reason="Could not determine source unit",
        )
    
    # If source matches target, no transformation needed
    if source_unit == target_unit:
        return TransformationResult(
            needs_transformation=False,
            source_unit=source_unit,
            target_unit=target_unit,
            transformation=None,
            confidence="high",
            reason=f"Source unit '{source_unit}' matches target",
        )
    
    # Check if conversion exists
    conversion = get_conversion(source_unit, target_unit)
    if conversion:
        return TransformationResult(
            needs_transformation=True,
            source_unit=source_unit,
            target_unit=target_unit,
            transformation=conversion,
            confidence="high",
            reason=f"Convert {source_unit} → {target_unit} ({conversion.formula_description})",
        )
    
    # No conversion available
    return TransformationResult(
        needs_transformation=False,
        source_unit=source_unit,
        target_unit=target_unit,
        transformation=None,
        confidence="medium",
        reason=f"No conversion available from '{source_unit}' to '{target_unit}'",
    )
