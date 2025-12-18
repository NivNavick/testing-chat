"""
Text representation generator for CSV profiles.

Converts structured column profiles into rich text descriptions
that can be embedded for semantic similarity search.
"""

from typing import Any, Dict, List


def column_to_text(column_profile: Dict[str, Any]) -> str:
    """
    Convert a single column profile to a text description.
    
    Args:
        column_profile: Column profile dict from column_profiler
        
    Returns:
        Text description of the column
    """
    parts = []
    
    # Column name (most important signal!)
    col_name = column_profile.get("column_name", "unknown")
    col_type = column_profile.get("detected_type", "unknown")
    
    parts.append(f"{col_name} ({col_type})")
    
    # Add cardinality hint
    unique_ratio = column_profile.get("unique_ratio", 0)
    if unique_ratio > 0.9:
        parts.append("high cardinality")
    elif unique_ratio < 0.1:
        parts.append("low cardinality")
    elif unique_ratio < 0.3:
        parts.append("categorical")
    
    # Add unique count for categorical-like columns
    unique_count = column_profile.get("unique_count", 0)
    if unique_count > 0 and unique_ratio < 0.3:
        parts.append(f"{unique_count} unique values")
    
    # Add sample values (crucial for semantic matching!)
    samples = column_profile.get("sample_values", [])[:7]  # Limit to 7 samples
    if samples:
        # Clean and truncate sample values
        clean_samples = []
        for s in samples:
            s_str = str(s)
            if len(s_str) > 30:
                s_str = s_str[:27] + "..."
            clean_samples.append(s_str)
        parts.append(f"examples: {', '.join(clean_samples)}")
    
    # Add range for numeric columns
    num_min = column_profile.get("numeric_min")
    num_max = column_profile.get("numeric_max")
    if num_min is not None and num_max is not None:
        parts.append(f"range {num_min}-{num_max}")
    
    # Add range for datetime columns
    dt_min = column_profile.get("datetime_min")
    dt_max = column_profile.get("datetime_max")
    if dt_min and dt_max:
        parts.append(f"range {dt_min} to {dt_max}")
    
    return "; ".join(parts)


def csv_to_text_representation(column_profiles: List[Dict[str, Any]]) -> str:
    """
    Convert column profiles to a rich text description for embedding.
    
    This text representation is what gets embedded and stored in the vector database.
    The quality of this representation directly impacts classification accuracy.
    
    Args:
        column_profiles: List of column profile dicts from profile_dataframe()
        
    Returns:
        Single text string describing the CSV structure
        
    Example output:
        "Table with 5 columns: emp_id (text; high cardinality; examples: E001, E002, E003) | 
        work_date (date; range 2024-01-01 to 2024-12-31) | 
        clock_in (time_of_day; examples: 08:00, 09:00) | ..."
    """
    if not column_profiles:
        return "Empty table with no columns"
    
    column_texts = []
    for col in column_profiles:
        column_texts.append(column_to_text(col))
    
    return f"Table with {len(column_profiles)} columns: " + " | ".join(column_texts)


def column_to_embedding_text(column_profile: Dict[str, Any]) -> str:
    """
    Convert a single column profile to text optimized for column-level embedding.
    
    This is used for the column_mappings_kb table where we embed individual columns
    to learn column name → target field associations.
    
    Args:
        column_profile: Column profile dict
        
    Returns:
        Text representation for embedding
    """
    col_name = column_profile.get("column_name", "unknown")
    col_type = column_profile.get("detected_type", "unknown")
    samples = column_profile.get("sample_values", [])[:5]
    
    parts = [
        f"column named {col_name}",
        f"type {col_type}",
    ]
    
    if samples:
        parts.append(f"sample values: {', '.join(str(s) for s in samples)}")
    
    # Add cardinality context
    unique_ratio = column_profile.get("unique_ratio", 0)
    if unique_ratio > 0.9:
        parts.append("unique identifier field")
    elif unique_ratio < 0.1:
        parts.append("categorical field with few values")
    
    return "; ".join(parts)
