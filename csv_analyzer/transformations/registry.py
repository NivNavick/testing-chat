"""
Unit Conversion Registry.

Centralized knowledge base of all supported unit conversions.
The system uses this to determine how to transform source data
to match target schema expectations.

Usage in data post-processor:
    from csv_analyzer.transformations import get_transform_function
    
    func = get_transform_function("hours_to_minutes")
    transformed_value = func(8.5)  # Returns 510
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple


@dataclass
class TransformationInfo:
    """Information about a transformation."""
    name: str  # Unique function name for invoking (e.g., "hours_to_minutes")
    from_unit: str
    to_unit: str
    formula_description: str
    transform_func: Callable[[Any], Any]
    inverse_func: Optional[Callable[[Any], Any]] = None
    category: str = "unit"  # unit, format, type


# ============================================================================
# Unit Conversions
# ============================================================================

UNIT_CONVERSIONS: Dict[Tuple[str, str], TransformationInfo] = {
    # Time conversions
    ("hours", "minutes"): TransformationInfo(
        name="hours_to_minutes",
        from_unit="hours",
        to_unit="minutes",
        formula_description="value × 60",
        transform_func=lambda v: v * 60,
        inverse_func=lambda v: v / 60,
        category="unit",
    ),
    ("minutes", "hours"): TransformationInfo(
        name="minutes_to_hours",
        from_unit="minutes",
        to_unit="hours",
        formula_description="value ÷ 60",
        transform_func=lambda v: v / 60,
        inverse_func=lambda v: v * 60,
        category="unit",
    ),
    ("seconds", "minutes"): TransformationInfo(
        name="seconds_to_minutes",
        from_unit="seconds",
        to_unit="minutes",
        formula_description="value ÷ 60",
        transform_func=lambda v: v / 60,
        inverse_func=lambda v: v * 60,
        category="unit",
    ),
    ("minutes", "seconds"): TransformationInfo(
        name="minutes_to_seconds",
        from_unit="minutes",
        to_unit="seconds",
        formula_description="value × 60",
        transform_func=lambda v: v * 60,
        inverse_func=lambda v: v / 60,
        category="unit",
    ),
    ("hours", "seconds"): TransformationInfo(
        name="hours_to_seconds",
        from_unit="hours",
        to_unit="seconds",
        formula_description="value × 3600",
        transform_func=lambda v: v * 3600,
        inverse_func=lambda v: v / 3600,
        category="unit",
    ),
    ("seconds", "hours"): TransformationInfo(
        name="seconds_to_hours",
        from_unit="seconds",
        to_unit="hours",
        formula_description="value ÷ 3600",
        transform_func=lambda v: v / 3600,
        inverse_func=lambda v: v * 3600,
        category="unit",
    ),
    
    # Weight conversions
    ("kg", "lbs"): TransformationInfo(
        name="kg_to_lbs",
        from_unit="kg",
        to_unit="lbs",
        formula_description="value × 2.20462",
        transform_func=lambda v: v * 2.20462,
        inverse_func=lambda v: v / 2.20462,
        category="unit",
    ),
    ("lbs", "kg"): TransformationInfo(
        name="lbs_to_kg",
        from_unit="lbs",
        to_unit="kg",
        formula_description="value ÷ 2.20462",
        transform_func=lambda v: v / 2.20462,
        inverse_func=lambda v: v * 2.20462,
        category="unit",
    ),
    ("g", "kg"): TransformationInfo(
        name="g_to_kg",
        from_unit="g",
        to_unit="kg",
        formula_description="value ÷ 1000",
        transform_func=lambda v: v / 1000,
        inverse_func=lambda v: v * 1000,
        category="unit",
    ),
    ("kg", "g"): TransformationInfo(
        name="kg_to_g",
        from_unit="kg",
        to_unit="g",
        formula_description="value × 1000",
        transform_func=lambda v: v * 1000,
        inverse_func=lambda v: v / 1000,
        category="unit",
    ),
    
    # Temperature conversions
    ("celsius", "fahrenheit"): TransformationInfo(
        name="celsius_to_fahrenheit",
        from_unit="celsius",
        to_unit="fahrenheit",
        formula_description="(value × 9/5) + 32",
        transform_func=lambda v: (v * 9 / 5) + 32,
        inverse_func=lambda v: (v - 32) * 5 / 9,
        category="unit",
    ),
    ("fahrenheit", "celsius"): TransformationInfo(
        name="fahrenheit_to_celsius",
        from_unit="fahrenheit",
        to_unit="celsius",
        formula_description="(value - 32) × 5/9",
        transform_func=lambda v: (v - 32) * 5 / 9,
        inverse_func=lambda v: (v * 9 / 5) + 32,
        category="unit",
    ),
    
    # Length conversions
    ("cm", "inches"): TransformationInfo(
        name="cm_to_inches",
        from_unit="cm",
        to_unit="inches",
        formula_description="value ÷ 2.54",
        transform_func=lambda v: v / 2.54,
        inverse_func=lambda v: v * 2.54,
        category="unit",
    ),
    ("inches", "cm"): TransformationInfo(
        name="inches_to_cm",
        from_unit="inches",
        to_unit="cm",
        formula_description="value × 2.54",
        transform_func=lambda v: v * 2.54,
        inverse_func=lambda v: v / 2.54,
        category="unit",
    ),
    ("m", "feet"): TransformationInfo(
        name="m_to_feet",
        from_unit="m",
        to_unit="feet",
        formula_description="value × 3.28084",
        transform_func=lambda v: v * 3.28084,
        inverse_func=lambda v: v / 3.28084,
        category="unit",
    ),
    ("feet", "m"): TransformationInfo(
        name="feet_to_m",
        from_unit="feet",
        to_unit="m",
        formula_description="value ÷ 3.28084",
        transform_func=lambda v: v / 3.28084,
        inverse_func=lambda v: v * 3.28084,
        category="unit",
    ),
    
    # Percentage conversions
    ("decimal", "percentage"): TransformationInfo(
        name="decimal_to_percentage",
        from_unit="decimal",
        to_unit="percentage",
        formula_description="value × 100",
        transform_func=lambda v: v * 100,
        inverse_func=lambda v: v / 100,
        category="unit",
    ),
    ("percentage", "decimal"): TransformationInfo(
        name="percentage_to_decimal",
        from_unit="percentage",
        to_unit="decimal",
        formula_description="value ÷ 100",
        transform_func=lambda v: v / 100,
        inverse_func=lambda v: v * 100,
        category="unit",
    ),
}

# ============================================================================
# Format Conversions (dates, strings, etc.)
# ============================================================================

FORMAT_CONVERSIONS: Dict[Tuple[str, str], TransformationInfo] = {
    # Date formats
    ("DD/MM/YYYY", "YYYY-MM-DD"): TransformationInfo(
        name="date_dmy_slash_to_iso",
        from_unit="DD/MM/YYYY",
        to_unit="YYYY-MM-DD",
        formula_description="reformat date",
        transform_func=lambda v: _convert_date(v, "%d/%m/%Y", "%Y-%m-%d"),
        category="format",
    ),
    ("MM/DD/YYYY", "YYYY-MM-DD"): TransformationInfo(
        name="date_mdy_slash_to_iso",
        from_unit="MM/DD/YYYY",
        to_unit="YYYY-MM-DD",
        formula_description="reformat date",
        transform_func=lambda v: _convert_date(v, "%m/%d/%Y", "%Y-%m-%d"),
        category="format",
    ),
    ("DD-MM-YYYY", "YYYY-MM-DD"): TransformationInfo(
        name="date_dmy_dash_to_iso",
        from_unit="DD-MM-YYYY",
        to_unit="YYYY-MM-DD",
        formula_description="reformat date",
        transform_func=lambda v: _convert_date(v, "%d-%m-%Y", "%Y-%m-%d"),
        category="format",
    ),
    
    # Boolean representations
    ("yes_no", "boolean"): TransformationInfo(
        name="yesno_to_boolean",
        from_unit="yes_no",
        to_unit="boolean",
        formula_description="Yes/No → true/false",
        transform_func=lambda v: str(v).lower() in ('yes', 'y', '1', 'true', 'כן', 'sí', 'да'),
        category="type",
    ),
}


def _convert_date(value: str, from_format: str, to_format: str) -> str:
    """Convert date string between formats."""
    from datetime import datetime
    try:
        dt = datetime.strptime(value, from_format)
        return dt.strftime(to_format)
    except (ValueError, TypeError):
        return value  # Return as-is if conversion fails


# ============================================================================
# Lookup Functions
# ============================================================================

def get_conversion(from_unit: str, to_unit: str) -> Optional[TransformationInfo]:
    """
    Get the conversion info between two units.
    
    Args:
        from_unit: Source unit
        to_unit: Target unit
        
    Returns:
        TransformationInfo or None if no conversion exists
    """
    # Check unit conversions first
    key = (from_unit.lower(), to_unit.lower())
    if key in UNIT_CONVERSIONS:
        return UNIT_CONVERSIONS[key]
    
    # Check format conversions
    if key in FORMAT_CONVERSIONS:
        return FORMAT_CONVERSIONS[key]
    
    return None


def get_conversion_info(from_unit: str, to_unit: str) -> Optional[Dict[str, Any]]:
    """
    Get conversion info as a dictionary (for JSON serialization).
    """
    info = get_conversion(from_unit, to_unit)
    if info is None:
        return None
    
    return {
        "transform_function": info.name,  # Function name to invoke
        "from_unit": info.from_unit,
        "to_unit": info.to_unit,
        "formula": info.formula_description,
        "category": info.category,
    }


def can_convert(from_unit: str, to_unit: str) -> bool:
    """Check if a conversion exists between two units."""
    return get_conversion(from_unit, to_unit) is not None


# Build a name-based lookup for easy function invocation
_TRANSFORM_BY_NAME: Dict[str, TransformationInfo] = {}


def _build_name_lookup():
    """Build the name-based lookup table."""
    global _TRANSFORM_BY_NAME
    for info in UNIT_CONVERSIONS.values():
        _TRANSFORM_BY_NAME[info.name] = info
    for info in FORMAT_CONVERSIONS.values():
        _TRANSFORM_BY_NAME[info.name] = info


def get_transform_function(name: str) -> Optional[Callable[[Any], Any]]:
    """
    Get a transformation function by name.
    
    Usage in data post-processor:
        from csv_analyzer.transformations import get_transform_function
        
        func = get_transform_function("hours_to_minutes")
        transformed_value = func(8.5)  # Returns 510.0
    
    Args:
        name: Transformation function name (e.g., "hours_to_minutes")
        
    Returns:
        The transformation function, or None if not found
    """
    if not _TRANSFORM_BY_NAME:
        _build_name_lookup()
    
    info = _TRANSFORM_BY_NAME.get(name)
    return info.transform_func if info else None


def get_transform_info_by_name(name: str) -> Optional[TransformationInfo]:
    """
    Get full transformation info by name.
    
    Args:
        name: Transformation function name
        
    Returns:
        TransformationInfo or None if not found
    """
    if not _TRANSFORM_BY_NAME:
        _build_name_lookup()
    
    return _TRANSFORM_BY_NAME.get(name)


def list_available_transforms() -> List[str]:
    """
    List all available transformation function names.
    
    Returns:
        List of transformation names
    """
    if not _TRANSFORM_BY_NAME:
        _build_name_lookup()
    
    return list(_TRANSFORM_BY_NAME.keys())


def get_accepted_units(target_unit: str) -> List[str]:
    """
    Get all units that can be converted TO the target unit.
    
    Args:
        target_unit: The target unit (what the schema expects)
        
    Returns:
        List of units that can be converted to target_unit
    """
    accepted = [target_unit]  # Always accept the target unit itself
    
    for (from_u, to_u), _ in UNIT_CONVERSIONS.items():
        if to_u.lower() == target_unit.lower():
            accepted.append(from_u)
    
    for (from_u, to_u), _ in FORMAT_CONVERSIONS.items():
        if to_u.lower() == target_unit.lower():
            accepted.append(from_u)
    
    return list(set(accepted))


# Unit aliases for semantic matching
UNIT_ALIASES: Dict[str, List[str]] = {
    "hours": ["hrs", "hr", "hour", "h", "horas", "שעות", "heures", "часов", "ساعات"],
    "minutes": ["mins", "min", "m", "minutos", "דקות", "minutes", "минут", "دقائق"],
    "seconds": ["secs", "sec", "s", "segundos", "שניות", "secondes", "секунд", "ثواني"],
    "kg": ["kilograms", "kilogram", "kilo", "קילו", "קילוגרם"],
    "lbs": ["pounds", "pound", "lb", "ליברות"],
    "celsius": ["c", "centigrade", "צלזיוס"],
    "fahrenheit": ["f", "פרנהייט"],
    "percentage": ["%", "percent", "pct", "אחוז"],
    "decimal": ["dec", "ratio", "יחס"],
}


def normalize_unit(unit: str) -> Optional[str]:
    """
    Normalize a unit string to its canonical form.
    
    Args:
        unit: Unit string (e.g., "hrs", "hr", "hours" all → "hours")
        
    Returns:
        Canonical unit name or None if not recognized
    """
    unit_lower = unit.lower().strip()
    
    # Check if it's already a canonical unit
    if unit_lower in UNIT_ALIASES:
        return unit_lower
    
    # Check aliases
    for canonical, aliases in UNIT_ALIASES.items():
        if unit_lower in [a.lower() for a in aliases]:
            return canonical
    
    return None
