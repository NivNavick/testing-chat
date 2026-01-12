"""
Ontology System for Workflow Blocks.

Defines data types for block input/output ports, ensuring type-safe connections
between blocks in the DAG.
"""

from enum import Enum
from typing import Set


class DataType(Enum):
    """
    Data types for block input/output ports.
    
    Container types represent data structures passed between blocks.
    Document types represent specific classified document schemas.
    Field types enable field-level connections to specific columns.
    """
    
    # Container types
    RAW_FILES = "raw_files"                  # List of local file paths
    S3_REFERENCES = "s3_references"          # List of S3 URIs after upload
    CLASSIFIED_DATA = "classified_data"      # Dict[doc_type, DataFrame]
    INSIGHT_RESULT = "insight_result"        # DataFrame with insight output
    JSON_DATA = "json_data"                  # Generic JSON-serializable dict
    
    # Document types (outputs from router)
    EMPLOYEE_SHIFTS = "employee_shifts"
    MEDICAL_ACTIONS = "medical_actions"
    LAB_RESULTS = "lab_results"
    PATIENT_APPOINTMENTS = "patient_appointments"
    EMPLOYEE_COMPENSATION = "employee_compensation"
    EMPLOYEE_MONTHLY_SALARY = "employee_monthly_salary"
    
    # Field types (for field-level connections)
    DATETIME_FIELD = "datetime_field"
    STRING_FIELD = "string_field"
    NUMBER_FIELD = "number_field"
    BOOLEAN_FIELD = "boolean_field"


# Document types that can be routed from CLASSIFIED_DATA
DOCUMENT_TYPES: Set[DataType] = {
    DataType.EMPLOYEE_SHIFTS,
    DataType.MEDICAL_ACTIONS,
    DataType.LAB_RESULTS,
    DataType.PATIENT_APPOINTMENTS,
    DataType.EMPLOYEE_COMPENSATION,
    DataType.EMPLOYEE_MONTHLY_SALARY,
}

# Field types for column-level connections
FIELD_TYPES: Set[DataType] = {
    DataType.DATETIME_FIELD,
    DataType.STRING_FIELD,
    DataType.NUMBER_FIELD,
    DataType.BOOLEAN_FIELD,
}


def validate_connection(source_type: DataType, target_type: DataType) -> bool:
    """
    Check if source output can connect to target input.
    
    Args:
        source_type: DataType of the source output port
        target_type: DataType of the target input port
        
    Returns:
        True if connection is valid, False otherwise
    """
    # Exact match is always valid
    if source_type == target_type:
        return True
    
    # Document types can connect to INSIGHT_RESULT inputs (for processing)
    if source_type in DOCUMENT_TYPES and target_type == DataType.INSIGHT_RESULT:
        return True
    
    return False


def get_document_type_name(doc_type: str) -> DataType:
    """
    Convert a document type string to DataType enum.
    
    Args:
        doc_type: String name like "employee_shifts"
        
    Returns:
        Corresponding DataType enum value
        
    Raises:
        ValueError: If document type is not recognized
    """
    try:
        return DataType(doc_type)
    except ValueError:
        raise ValueError(
            f"Unknown document type: {doc_type}. "
            f"Valid types: {[dt.value for dt in DOCUMENT_TYPES]}"
        )


def is_document_type(data_type: DataType) -> bool:
    """Check if a DataType is a document type."""
    return data_type in DOCUMENT_TYPES


def is_field_type(data_type: DataType) -> bool:
    """Check if a DataType is a field type."""
    return data_type in FIELD_TYPES

