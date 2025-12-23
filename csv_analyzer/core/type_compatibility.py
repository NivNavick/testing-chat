"""
Type Compatibility Module.

Defines compatibility rules between detected column types and schema field types.
Returns a multiplier (0.0 to 1.0) that adjusts the similarity score.
"""

from typing import Dict, Tuple

# Type compatibility matrix
# Maps (source_type, target_type) -> compatibility_score
# Higher score = more compatible
# 1.0 = perfect match
# 0.85-0.95 = compatible with minor conversion
# 0.4 = incompatible (harsh penalty but not full rejection)

TYPE_COMPATIBILITY: Dict[Tuple[str, str], float] = {
    # Exact matches
    ("datetime", "datetime"): 1.0,
    ("date", "date"): 1.0,
    ("time_of_day", "datetime"): 0.95,  # Time can be part of datetime
    ("integer", "integer"): 1.0,
    ("float", "float"): 1.0,
    ("boolean", "boolean"): 1.0,
    ("text", "string"): 1.0,
    ("string", "string"): 1.0,
    ("id_like", "string"): 1.0,
    ("categorical", "string"): 1.0,
    ("free_text", "string"): 1.0,
    ("duration", "integer"): 0.95,  # Duration stored as minutes/seconds
    ("duration", "float"): 0.95,
    
    # Datetime family - compatible with each other
    ("datetime", "date"): 0.85,  # Can extract date from datetime
    ("datetime", "string"): 0.85,  # Can format as string
    ("date", "datetime"): 0.85,  # Can expand to datetime (with 00:00:00)
    ("date", "string"): 0.85,
    ("time_of_day", "string"): 0.85,
    ("time_of_day", "time"): 1.0,  # Exact match for time
    
    # Numeric family - compatible with each other
    ("integer", "float"): 0.95,  # Integer can be float
    ("float", "integer"): 0.80,  # Float to int loses precision
    ("integer", "string"): 0.75,  # Can convert but not ideal
    ("float", "string"): 0.75,
    ("duration", "string"): 0.70,
    
    # Boolean conversions
    ("boolean", "string"): 0.80,
    ("boolean", "integer"): 0.75,  # 0/1 encoding
    
    # ID-like is flexible
    ("id_like", "integer"): 0.85,  # Numeric IDs
    ("id_like", "float"): 0.70,  # Less common
    
    # Text/string is the most flexible target
    ("text", "text"): 1.0,
    ("categorical", "text"): 0.90,
    ("free_text", "text"): 1.0,
}

# Default penalty for incompatible types (not in the matrix)
INCOMPATIBLE_PENALTY = 0.4


def get_type_compatibility(source_type: str, target_type: str) -> float:
    """
    Get the compatibility score between a detected column type and a schema field type.
    
    Args:
        source_type: Detected type from column profiler (datetime, date, integer, etc.)
        target_type: Schema field type (string, datetime, integer, etc.)
        
    Returns:
        Compatibility score from 0.0 to 1.0
        - 1.0: Perfect match
        - 0.85-0.95: Compatible with minor conversion
        - 0.4: Incompatible (harsh penalty)
    """
    # Normalize types to lowercase
    source = source_type.lower().strip() if source_type else "unknown"
    target = target_type.lower().strip() if target_type else "unknown"
    
    # Handle unknown types gracefully
    if source == "unknown" or target == "unknown":
        return 0.7  # Neutral - don't penalize too much if type is unknown
    
    # Check for empty type
    if source == "empty":
        return 0.5  # Empty columns get moderate penalty
    
    # Direct lookup
    key = (source, target)
    if key in TYPE_COMPATIBILITY:
        return TYPE_COMPATIBILITY[key]
    
    # Check reverse (some mappings are symmetric)
    reverse_key = (target, source)
    if reverse_key in TYPE_COMPATIBILITY:
        return TYPE_COMPATIBILITY[reverse_key]
    
    # Special case: any string-like source to string target
    string_like_sources = {"text", "string", "id_like", "categorical", "free_text"}
    if source in string_like_sources and target == "string":
        return 1.0
    
    # Special case: any numeric source to numeric target
    numeric_sources = {"integer", "float", "duration"}
    numeric_targets = {"integer", "float", "number"}
    if source in numeric_sources and target in numeric_targets:
        return 0.85
    
    # Special case: datetime family
    datetime_sources = {"datetime", "date", "time_of_day"}
    datetime_targets = {"datetime", "date", "time"}
    if source in datetime_sources and target in datetime_targets:
        return 0.80
    
    # No match found - apply incompatibility penalty
    return INCOMPATIBLE_PENALTY


def is_type_compatible(source_type: str, target_type: str, threshold: float = 0.6) -> bool:
    """
    Check if two types are compatible (above threshold).
    
    Args:
        source_type: Detected column type
        target_type: Schema field type
        threshold: Minimum compatibility score to consider compatible
        
    Returns:
        True if types are compatible
    """
    return get_type_compatibility(source_type, target_type) >= threshold


def normalize_detected_type(detected_type: str) -> str:
    """
    Normalize detected type names for consistency.
    
    The column profiler uses some specific names that need mapping
    to more general type categories.
    """
    type_map = {
        "time_of_day": "time_of_day",
        "datetime": "datetime",
        "date": "date",
        "integer": "integer",
        "float": "float",
        "boolean": "boolean",
        "text": "text",
        "string": "string",
        "id_like": "id_like",
        "categorical": "categorical",
        "free_text": "free_text",
        "duration": "duration",
        "empty": "empty",
    }
    return type_map.get(detected_type.lower(), detected_type.lower())

