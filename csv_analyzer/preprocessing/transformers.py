"""
Value Transformers for CSV preprocessing.

Provides transformations for:
- Splitting range values (e.g., "19:00 - 19:15" → two columns)
- Normalizing Hebrew dates (e.g., "ב - 01" → ISO date)
- Cleaning prefixes/suffixes (e.g., "* 14:30" → "14:30")
- Custom regex extractions
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class TransformResult:
    """Result of applying transformations to a DataFrame."""
    df: pd.DataFrame
    transformations_applied: int
    rows_affected: int
    columns_added: List[str] = field(default_factory=list)
    columns_modified: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


@dataclass
class TransformConfig:
    """Configuration for a column transformation."""
    source_column: str
    transform_type: str
    pattern: Optional[str] = None
    replacement: Optional[str] = None
    output_columns: Optional[List[Dict[str, Any]]] = None
    context_date_column: Optional[str] = None
    options: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TransformConfig":
        """Create from dictionary (from YAML)."""
        return cls(
            source_column=data["source_column"],
            transform_type=data["transform_type"],
            pattern=data.get("pattern"),
            replacement=data.get("replacement", ""),
            output_columns=data.get("output_columns"),
            context_date_column=data.get("context_date_column"),
            options=data.get("options", {}),
        )


class ValueTransformer:
    """
    Applies value transformations to DataFrame columns.
    
    Supported transform types:
    - split_range: Split "19:00 - 19:15" into two columns
    - hebrew_date_normalize: Convert "ב - 01" to ISO date
    - clean_prefix: Remove prefix pattern (e.g., "* " from "* 14:30")
    - clean_suffix: Remove suffix pattern
    - extract_pattern: Extract values using regex groups
    - replace_pattern: Replace matching pattern
    
    Usage:
        transformer = ValueTransformer()
        
        configs = [
            TransformConfig(
                source_column="שעות טיפול",
                transform_type="split_range",
                pattern=r"^(\\d{2}:\\d{2})\\s*-\\s*(\\d{2}:\\d{2})$",
                output_columns=[
                    {"name": "start_time", "group": 1},
                    {"name": "end_time", "group": 2},
                ],
            ),
        ]
        
        result = transformer.apply_all(df, configs)
    """
    
    # Hebrew day letters to day index (Sunday=0)
    HEBREW_DAY_MAP = {
        'א': 0,  # Sunday (Aleph)
        'ב': 1,  # Monday (Bet)
        'ג': 2,  # Tuesday (Gimel)
        'ד': 3,  # Wednesday (Dalet)
        'ה': 4,  # Thursday (He)
        'ו': 5,  # Friday (Vav)
        'ש': 6,  # Saturday (Shin)
    }
    
    # Reverse map: day index to Hebrew letter
    DAY_INDEX_TO_HEBREW = {v: k for k, v in HEBREW_DAY_MAP.items()}
    
    def __init__(self, context: Optional[Dict[str, Any]] = None):
        """
        Initialize the transformer.
        
        Args:
            context: Optional context with date_range, default values, etc.
        """
        self.context = context or {}
        
        # Register transform handlers
        self._handlers: Dict[str, Callable] = {
            "split_range": self._transform_split_range,
            "hebrew_date_normalize": self._transform_hebrew_date,
            "clean_prefix": self._transform_clean_prefix,
            "clean_suffix": self._transform_clean_suffix,
            "extract_pattern": self._transform_extract_pattern,
            "replace_pattern": self._transform_replace_pattern,
            "time_normalize": self._transform_time_normalize,
        }
    
    def set_context(self, key: str, value: Any) -> None:
        """Set a context value."""
        self.context[key] = value
    
    def apply_all(
        self,
        df: pd.DataFrame,
        configs: List[Union[TransformConfig, Dict[str, Any]]],
    ) -> TransformResult:
        """
        Apply all transformations to the DataFrame.
        
        Args:
            df: Input DataFrame
            configs: List of transformation configurations
            
        Returns:
            TransformResult with the modified DataFrame
        """
        result_df = df.copy()
        total_applied = 0
        total_rows = 0
        columns_added = []
        columns_modified = []
        errors = []
        
        for config in configs:
            if isinstance(config, dict):
                config = TransformConfig.from_dict(config)
            
            if config.source_column not in result_df.columns:
                errors.append(f"Source column '{config.source_column}' not found")
                continue
            
            handler = self._handlers.get(config.transform_type)
            if not handler:
                errors.append(f"Unknown transform type: {config.transform_type}")
                continue
            
            try:
                result_df, rows_affected, new_cols, mod_cols = handler(result_df, config)
                total_applied += 1
                total_rows += rows_affected
                columns_added.extend(new_cols)
                columns_modified.extend(mod_cols)
                
                logger.info(
                    f"Applied '{config.transform_type}' to '{config.source_column}': "
                    f"{rows_affected} rows affected"
                )
                
            except Exception as e:
                error_msg = f"Error applying {config.transform_type} to {config.source_column}: {e}"
                errors.append(error_msg)
                logger.warning(error_msg)
        
        return TransformResult(
            df=result_df,
            transformations_applied=total_applied,
            rows_affected=total_rows,
            columns_added=columns_added,
            columns_modified=columns_modified,
            errors=errors,
        )
    
    def _transform_split_range(
        self,
        df: pd.DataFrame,
        config: TransformConfig,
    ) -> Tuple[pd.DataFrame, int, List[str], List[str]]:
        """
        Split a range column into multiple columns.
        
        Example: "19:00 - 19:15" → start_time="19:00", end_time="19:15"
        """
        col = config.source_column
        pattern = config.pattern or r'^(.+?)\s*-\s*(.+)$'
        output_cols = config.output_columns or []
        
        if not output_cols:
            # Default output columns
            output_cols = [
                {"name": f"{col}_start", "group": 1},
                {"name": f"{col}_end", "group": 2},
            ]
        
        regex = re.compile(pattern)
        new_columns = {}
        rows_affected = 0
        
        # Initialize new columns
        for out_col in output_cols:
            new_columns[out_col["name"]] = [None] * len(df)
        
        # Process each row
        for idx, value in enumerate(df[col]):
            if pd.isna(value):
                continue
            
            match = regex.match(str(value))
            if match:
                rows_affected += 1
                for out_col in output_cols:
                    group_num = out_col.get("group", 1)
                    try:
                        extracted = match.group(group_num)
                        new_columns[out_col["name"]][idx] = extracted.strip()
                    except IndexError:
                        pass
        
        # Add new columns to DataFrame
        columns_added = []
        for col_name, values in new_columns.items():
            df[col_name] = values
            columns_added.append(col_name)
        
        return df, rows_affected, columns_added, []
    
    def _transform_hebrew_date(
        self,
        df: pd.DataFrame,
        config: TransformConfig,
    ) -> Tuple[pd.DataFrame, int, List[str], List[str]]:
        """
        Normalize Hebrew date format to ISO date.
        
        Example: "ב - 01" (Monday, 1st) → "2025-12-01"
        
        Requires date_range context to determine month/year.
        """
        col = config.source_column
        pattern = config.pattern or r'^([אבגדהוש])\s*-\s*(\d{1,2})$'
        
        # Get date range from context
        date_range_start = self.context.get("date_range_start")
        date_range_end = self.context.get("date_range_end")
        
        if not date_range_start:
            logger.warning("No date_range_start in context for Hebrew date normalization")
            # Try to use current month
            now = datetime.now()
            date_range_start = now.replace(day=1).strftime("%d.%m.%Y")
        
        # Parse the reference date
        try:
            if '.' in str(date_range_start):
                ref_date = datetime.strptime(str(date_range_start), "%d.%m.%Y")
            else:
                ref_date = datetime.fromisoformat(str(date_range_start))
        except Exception:
            ref_date = datetime.now().replace(day=1)
        
        regex = re.compile(pattern)
        new_values = []
        rows_affected = 0
        
        for value in df[col]:
            if pd.isna(value):
                new_values.append(None)
                continue
            
            str_value = str(value).strip()
            match = regex.match(str_value)
            
            if match:
                hebrew_day = match.group(1)
                day_num = int(match.group(2))
                
                # Calculate the actual date
                try:
                    # Create date with the day number in the reference month
                    actual_date = ref_date.replace(day=day_num)
                    new_values.append(actual_date.strftime("%Y-%m-%d"))
                    rows_affected += 1
                except ValueError:
                    # Invalid day for month
                    new_values.append(str_value)
            else:
                # Try to parse as existing date
                new_values.append(str_value)
        
        df[col] = new_values
        return df, rows_affected, [], [col]
    
    def _transform_clean_prefix(
        self,
        df: pd.DataFrame,
        config: TransformConfig,
    ) -> Tuple[pd.DataFrame, int, List[str], List[str]]:
        """
        Remove prefix pattern from values.
        
        Example: "* 14:30" → "14:30"
        """
        col = config.source_column
        pattern = config.pattern or r'^\*\s*'
        replacement = config.replacement or ""
        
        regex = re.compile(pattern)
        rows_affected = 0
        
        def clean_value(value):
            nonlocal rows_affected
            if pd.isna(value):
                return value
            str_val = str(value)
            if regex.search(str_val):
                rows_affected += 1
                return regex.sub(replacement, str_val)
            return str_val
        
        df[col] = df[col].apply(clean_value)
        return df, rows_affected, [], [col]
    
    def _transform_clean_suffix(
        self,
        df: pd.DataFrame,
        config: TransformConfig,
    ) -> Tuple[pd.DataFrame, int, List[str], List[str]]:
        """Remove suffix pattern from values."""
        col = config.source_column
        pattern = config.pattern or r'\s*$'
        replacement = config.replacement or ""
        
        # Ensure pattern matches at end
        if not pattern.endswith('$'):
            pattern = pattern + '$'
        
        regex = re.compile(pattern)
        rows_affected = 0
        
        def clean_value(value):
            nonlocal rows_affected
            if pd.isna(value):
                return value
            str_val = str(value)
            if regex.search(str_val):
                rows_affected += 1
                return regex.sub(replacement, str_val)
            return str_val
        
        df[col] = df[col].apply(clean_value)
        return df, rows_affected, [], [col]
    
    def _transform_extract_pattern(
        self,
        df: pd.DataFrame,
        config: TransformConfig,
    ) -> Tuple[pd.DataFrame, int, List[str], List[str]]:
        """
        Extract values using regex groups into new columns.
        
        Similar to split_range but more generic.
        """
        return self._transform_split_range(df, config)
    
    def _transform_replace_pattern(
        self,
        df: pd.DataFrame,
        config: TransformConfig,
    ) -> Tuple[pd.DataFrame, int, List[str], List[str]]:
        """
        Replace matching pattern with replacement string.
        """
        col = config.source_column
        pattern = config.pattern
        replacement = config.replacement or ""
        
        if not pattern:
            return df, 0, [], []
        
        regex = re.compile(pattern)
        rows_affected = 0
        
        def replace_value(value):
            nonlocal rows_affected
            if pd.isna(value):
                return value
            str_val = str(value)
            if regex.search(str_val):
                rows_affected += 1
                return regex.sub(replacement, str_val)
            return str_val
        
        df[col] = df[col].apply(replace_value)
        return df, rows_affected, [], [col]
    
    def _transform_time_normalize(
        self,
        df: pd.DataFrame,
        config: TransformConfig,
    ) -> Tuple[pd.DataFrame, int, List[str], List[str]]:
        """
        Normalize time values to consistent format (HH:MM).
        
        Handles:
        - "* 14:30" → "14:30"
        - "9:30" → "09:30"
        - "14.30" → "14:30"
        """
        col = config.source_column
        rows_affected = 0
        
        def normalize_time(value):
            nonlocal rows_affected
            if pd.isna(value):
                return value
            
            str_val = str(value).strip()
            original = str_val
            
            # Remove leading asterisk
            str_val = re.sub(r'^\*\s*', '', str_val)
            
            # Replace . with :
            str_val = str_val.replace('.', ':')
            
            # Pad single-digit hour
            time_match = re.match(r'^(\d{1,2}):(\d{2})$', str_val)
            if time_match:
                hour = int(time_match.group(1))
                minute = time_match.group(2)
                str_val = f"{hour:02d}:{minute}"
            
            if str_val != original:
                rows_affected += 1
            
            return str_val
        
        df[col] = df[col].apply(normalize_time)
        return df, rows_affected, [], [col]


# Convenience function for common transformations
def create_split_range_config(
    source_column: str,
    start_column: str,
    end_column: str,
    separator: str = "-",
) -> TransformConfig:
    """Create a split_range transformation config."""
    pattern = rf'^(.+?)\s*{re.escape(separator)}\s*(.+)$'
    return TransformConfig(
        source_column=source_column,
        transform_type="split_range",
        pattern=pattern,
        output_columns=[
            {"name": start_column, "group": 1},
            {"name": end_column, "group": 2},
        ],
    )


def create_clean_prefix_config(
    source_column: str,
    prefix_pattern: str = r'^\*\s*',
) -> TransformConfig:
    """Create a clean_prefix transformation config."""
    return TransformConfig(
        source_column=source_column,
        transform_type="clean_prefix",
        pattern=prefix_pattern,
        replacement="",
    )


def create_hebrew_date_config(
    source_column: str,
) -> TransformConfig:
    """Create a Hebrew date normalization config."""
    return TransformConfig(
        source_column=source_column,
        transform_type="hebrew_date_normalize",
        pattern=r'^([אבגדהוש])\s*-\s*(\d{1,2})$',
    )

