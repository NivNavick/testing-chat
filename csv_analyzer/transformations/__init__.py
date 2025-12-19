"""
Transformations module for unit conversions and data transformations.

Usage in data post-processor:
    from csv_analyzer.transformations import get_transform_function
    
    # Get function by name from mapping result
    func = get_transform_function("hours_to_minutes")
    transformed_value = func(8.5)  # Returns 510.0
    
    # List all available transformations
    from csv_analyzer.transformations import list_available_transforms
    print(list_available_transforms())
    # ['hours_to_minutes', 'minutes_to_hours', 'kg_to_lbs', ...]
"""

from csv_analyzer.transformations.registry import (
    UNIT_CONVERSIONS,
    FORMAT_CONVERSIONS,
    get_conversion,
    get_conversion_info,
    can_convert,
    get_transform_function,
    get_transform_info_by_name,
    list_available_transforms,
    TransformationInfo,
)
from csv_analyzer.transformations.detector import (
    detect_source_unit,
    detect_transformation,
    TransformationResult,
)

__all__ = [
    # Registry
    "UNIT_CONVERSIONS",
    "FORMAT_CONVERSIONS",
    "get_conversion",
    "get_conversion_info",
    "can_convert",
    "get_transform_function",
    "get_transform_info_by_name",
    "list_available_transforms",
    "TransformationInfo",
    # Detector
    "detect_source_unit",
    "detect_transformation",
    "TransformationResult",
]
