"""
Schema-driven data canonicalizer.

Applies transformation rules defined in schema YAML files to convert
raw CSV data into canonical format with standardized column names and values.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from csv_analyzer.core.schema_registry import TargetSchema, TransformationRule

logger = logging.getLogger(__name__)


# Hebrew day-of-week letters mapping
HEBREW_DAYS = {
    "א": 1,  # Sunday
    "ב": 2,  # Monday
    "ג": 3,  # Tuesday
    "ד": 4,  # Wednesday
    "ה": 5,  # Thursday
    "ו": 6,  # Friday
    "ש": 7,  # Saturday (Shabbat)
}


class SchemaTransformer:
    """
    Applies transformations defined in schema YAML to a DataFrame.
    
    Usage:
        transformer = SchemaTransformer()
        canonical_df = transformer.transform(df, schema, column_mappings)
    """
    
    def __init__(self):
        self.stats = {
            "transforms_applied": 0,
            "rows_filtered": 0,
            "columns_created": 0,
            "errors": [],
        }
    
    def transform(
        self,
        df: pd.DataFrame,
        schema: TargetSchema,
        column_mappings: Dict[str, str],
    ) -> pd.DataFrame:
        """
        Apply all transformations from schema to DataFrame.
        
        Args:
            df: Input DataFrame with original column names
            schema: Target schema with transformation rules
            column_mappings: Map of original column names -> schema field names
                            e.g., {"שעת_כניסה": "shift_start", "תאריך": "shift_date"}
        
        Returns:
            Transformed DataFrame with canonical column names and values
        """
        # Reset stats
        self.stats = {
            "transforms_applied": 0,
            "rows_filtered": 0,
            "columns_created": 0,
            "errors": [],
        }
        
        # Create reverse mapping: schema field name -> original column name
        field_to_original = {v: k for k, v in column_mappings.items()}
        
        logger.info(f"Applying {len(schema.transformations)} transformations for schema {schema.name}")
        
        # Apply each transformation rule
        for transform in schema.transformations:
            try:
                df = self._apply_transform(df, transform, field_to_original, column_mappings)
                self.stats["transforms_applied"] += 1
            except Exception as e:
                error_msg = f"Transform {transform.type} failed: {str(e)}"
                logger.warning(error_msg)
                self.stats["errors"].append(error_msg)
        
        logger.info(
            f"Transformation complete: {self.stats['transforms_applied']} applied, "
            f"{self.stats['rows_filtered']} rows filtered, "
            f"{self.stats['columns_created']} columns created"
        )
        
        return df
    
    def _apply_transform(
        self,
        df: pd.DataFrame,
        transform: TransformationRule,
        field_to_original: Dict[str, str],
        column_mappings: Dict[str, str],
    ) -> pd.DataFrame:
        """Apply a single transformation rule."""
        
        if transform.type == "split_column":
            return self._split_column(df, transform, field_to_original)
        elif transform.type == "filter_rows":
            return self._filter_rows(df, transform, field_to_original)
        elif transform.type == "parse_hebrew_date":
            return self._parse_hebrew_date(df, transform, field_to_original)
        elif transform.type == "extract_pattern":
            return self._extract_pattern(df, transform, field_to_original)
        elif transform.type == "clean_time":
            return self._clean_time(df, transform, field_to_original)
        else:
            logger.warning(f"Unknown transform type: {transform.type}")
            return df
    
    def _get_source_column(
        self,
        df: pd.DataFrame,
        transform: TransformationRule,
        field_to_original: Dict[str, str],
    ) -> Optional[str]:
        """
        Get the actual column name to use as source.
        Tries the mapped column first, then the original field name directly.
        """
        source_field = transform.source
        
        # Try mapped column name first
        if source_field in field_to_original:
            original_col = field_to_original[source_field]
            if original_col in df.columns:
                return original_col
        
        # Try schema field name directly (might already be renamed)
        if source_field in df.columns:
            return source_field
        
        # Try fallback source if specified
        if transform.fallback_source:
            if transform.fallback_source in field_to_original:
                fallback_col = field_to_original[transform.fallback_source]
                if fallback_col in df.columns:
                    return fallback_col
            if transform.fallback_source in df.columns:
                return transform.fallback_source
        
        logger.debug(f"Source column not found for {source_field}, available: {list(df.columns)[:10]}...")
        return None
    
    def _split_column(
        self,
        df: pd.DataFrame,
        transform: TransformationRule,
        field_to_original: Dict[str, str],
    ) -> pd.DataFrame:
        """
        Split a column containing a range into two columns.
        
        Example: "06:30 - 14:00" -> shift_start="06:30", shift_end="14:00"
        """
        source_col = self._get_source_column(df, transform, field_to_original)
        if source_col is None:
            return df
        
        delimiter = transform.params.get("delimiter", " - ")
        output_fields = transform.params.get("output_fields", [])
        
        if len(output_fields) < 2:
            logger.warning(f"split_column requires at least 2 output_fields")
            return df
        
        # Split the column
        def safe_split(value):
            if pd.isna(value) or not isinstance(value, str):
                return [None, None]
            parts = str(value).split(delimiter, 1)
            if len(parts) == 2:
                return [p.strip() for p in parts]
            return [value, None]
        
        split_result = df[source_col].apply(safe_split)
        
        # Create new columns
        df[output_fields[0]] = split_result.apply(lambda x: x[0] if x else None)
        df[output_fields[1]] = split_result.apply(lambda x: x[1] if x else None)
        
        self.stats["columns_created"] += 2
        logger.debug(f"Split {source_col} into {output_fields}")
        
        return df
    
    def _filter_rows(
        self,
        df: pd.DataFrame,
        transform: TransformationRule,
        field_to_original: Dict[str, str],
    ) -> pd.DataFrame:
        """
        Filter out rows based on field values.
        
        Example: Exclude rows where billing_code == "התחשבנויות"
        """
        exclude_where = transform.params.get("exclude_where", [])
        
        initial_count = len(df)
        
        for exclusion in exclude_where:
            field_name = exclusion.get("field")
            values = exclusion.get("values", [])
            match_type = exclusion.get("match_type", "equals")
            
            # Find actual column name
            actual_col = None
            if field_name in field_to_original:
                actual_col = field_to_original[field_name]
            if actual_col is None or actual_col not in df.columns:
                if field_name in df.columns:
                    actual_col = field_name
                else:
                    continue
            
            if match_type == "equals":
                # Exact match
                df = df[~df[actual_col].isin(values)]
            elif match_type == "contains":
                # Contains match
                for val in values:
                    df = df[~df[actual_col].astype(str).str.contains(val, na=False)]
        
        rows_filtered = initial_count - len(df)
        self.stats["rows_filtered"] += rows_filtered
        
        if rows_filtered > 0:
            logger.debug(f"Filtered {rows_filtered} rows")
        
        return df
    
    def _parse_hebrew_date(
        self,
        df: pd.DataFrame,
        transform: TransformationRule,
        field_to_original: Dict[str, str],
    ) -> pd.DataFrame:
        """
        Parse Hebrew date format and extract day of month.
        
        Example: "ה - 04" -> day_of_month=4
        """
        source_col = self._get_source_column(df, transform, field_to_original)
        if source_col is None:
            return df
        
        output_field = transform.params.get("output_field", "day_of_month")
        
        def extract_day_number(value):
            if pd.isna(value):
                return None
            value = str(value).strip()
            
            # Pattern: "X - DD" where X is Hebrew letter and DD is day number
            # Also handle variations like "04", "4", "א - 1"
            
            # Try to find a number in the string
            numbers = re.findall(r'\d+', value)
            if numbers:
                return int(numbers[-1])  # Take the last number found
            
            return None
        
        df[output_field] = df[source_col].apply(extract_day_number)
        self.stats["columns_created"] += 1
        
        logger.debug(f"Parsed Hebrew date from {source_col} to {output_field}")
        
        return df
    
    def _extract_pattern(
        self,
        df: pd.DataFrame,
        transform: TransformationRule,
        field_to_original: Dict[str, str],
    ) -> pd.DataFrame:
        """
        Extract and normalize patterns from a column.
        
        Example: "מחלקת גסטרו בת ים" -> city_code="bat_yam"
        """
        source_col = self._get_source_column(df, transform, field_to_original)
        if source_col is None:
            return df
        
        patterns = transform.params.get("patterns", [])
        output_field = transform.params.get("output_field")
        match_type = transform.params.get("match_type", "contains")
        
        if not output_field:
            logger.warning("extract_pattern requires output_field parameter")
            return df
        
        def extract(value):
            if pd.isna(value):
                return None
            value_str = str(value)
            
            for pattern in patterns:
                match_str = pattern.get("match", "")
                output_val = pattern.get("output", "")
                
                if match_type == "contains":
                    if match_str in value_str:
                        return output_val
                elif match_type == "equals":
                    if value_str.strip() == match_str:
                        return output_val
                elif match_type == "regex":
                    if re.search(match_str, value_str):
                        return output_val
            
            return None
        
        df[output_field] = df[source_col].apply(extract)
        self.stats["columns_created"] += 1
        
        logger.debug(f"Extracted pattern from {source_col} to {output_field}")
        
        return df
    
    def _clean_time(
        self,
        df: pd.DataFrame,
        transform: TransformationRule,
        field_to_original: Dict[str, str],
    ) -> pd.DataFrame:
        """
        Clean time values by stripping prefixes and special characters.
        
        Example: "* 06:30" -> "06:30"
        """
        source_col = self._get_source_column(df, transform, field_to_original)
        if source_col is None:
            return df
        
        strip_prefixes = transform.params.get("strip_prefixes", [])
        strip_chars = transform.params.get("strip_chars", [])
        
        def clean(value):
            if pd.isna(value):
                return value
            value_str = str(value)
            
            # Strip specific prefixes
            for prefix in strip_prefixes:
                if value_str.startswith(prefix):
                    value_str = value_str[len(prefix):]
            
            # Strip specific characters
            for char in strip_chars:
                value_str = value_str.replace(char, "")
            
            return value_str.strip()
        
        df[source_col] = df[source_col].apply(clean)
        
        logger.debug(f"Cleaned time values in {source_col}")
        
        return df


def apply_schema_transformations(
    df: pd.DataFrame,
    schema: TargetSchema,
    column_mappings: Dict[str, str],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Convenience function to apply schema transformations.
    
    Args:
        df: Input DataFrame
        schema: Target schema with transformations
        column_mappings: Column name mappings (original -> schema field)
    
    Returns:
        Tuple of (transformed DataFrame, transformation stats)
    """
    transformer = SchemaTransformer()
    transformed_df = transformer.transform(df, schema, column_mappings)
    return transformed_df, transformer.stats

