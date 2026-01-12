"""
Value Transformers for CSV preprocessing.

Provides transformations for:
- Splitting range values (e.g., "19:00 - 19:15" → two columns)
- Normalizing Hebrew dates (e.g., "ב - 01" → ISO date)
- Cleaning prefixes/suffixes (e.g., "* 14:30" → "14:30")
- Location normalization (e.g., "בת ים - גיאפה" → "bat_yam_gastro")
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


@dataclass
class LocationRule:
    """A rule for normalizing locations to canonical form."""
    patterns: List[str]  # Regex patterns to match
    canonical: str  # The canonical location ID
    display_name: Optional[str] = None  # Human-readable name
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LocationRule":
        """Create from dictionary (from YAML)."""
        patterns = data.get("patterns", [])
        if isinstance(patterns, str):
            patterns = [patterns]
        return cls(
            patterns=patterns,
            canonical=data["canonical"],
            display_name=data.get("display_name"),
        )


class LocationNormalizer:
    """
    Normalizes location names to canonical identifiers.
    
    This enables cross-document joins where locations are written differently:
    - Shifts: "בת ים - עצמאיים - גיאפה"
    - Actions: "בסט מדיקל -בת ים/מכון גסטרו בת ים"
    
    Both can be normalized to: "bat_yam_gastro"
    
    YAML Configuration:
    ```yaml
    preprocessing:
      location_normalization:
        source_columns:
          - department
          - _meta_location
        output_column: _normalized_location
        rules:
          - patterns:
              - "בת ים.*גסטרו"
              - "גיאפה"
            canonical: "bat_yam_gastro"
            display_name: "בת ים - גסטרו"
          - patterns:
              - "חדרה"
            canonical: "hadera"
            display_name: "חדרה"
    ```
    """
    
    def __init__(self, rules: Optional[List[LocationRule]] = None):
        """
        Initialize with location rules.
        
        Args:
            rules: List of LocationRule objects
        """
        self.rules = rules or []
        self._compiled_rules: List[Tuple[List[re.Pattern], str, Optional[str]]] = []
        self._compile_rules()
    
    def _compile_rules(self) -> None:
        """Compile regex patterns for efficiency."""
        self._compiled_rules = []
        for rule in self.rules:
            compiled_patterns = []
            for pattern in rule.patterns:
                try:
                    compiled_patterns.append(re.compile(pattern, re.IGNORECASE))
                except re.error as e:
                    logger.warning(f"Invalid location pattern '{pattern}': {e}")
            if compiled_patterns:
                self._compiled_rules.append((compiled_patterns, rule.canonical, rule.display_name))
    
    def add_rule(self, rule: LocationRule) -> None:
        """Add a new rule."""
        self.rules.append(rule)
        compiled_patterns = []
        for pattern in rule.patterns:
            try:
                compiled_patterns.append(re.compile(pattern, re.IGNORECASE))
            except re.error as e:
                logger.warning(f"Invalid location pattern '{pattern}': {e}")
        if compiled_patterns:
            self._compiled_rules.append((compiled_patterns, rule.canonical, rule.display_name))
    
    def normalize(self, location: str) -> Optional[str]:
        """
        Normalize a location string to its canonical form.
        
        Args:
            location: The location string to normalize
            
        Returns:
            Canonical location ID if matched, None otherwise
        """
        if not location or pd.isna(location):
            return None
        
        location_str = str(location)
        
        for compiled_patterns, canonical, _ in self._compiled_rules:
            for pattern in compiled_patterns:
                if pattern.search(location_str):
                    return canonical
        
        return None
    
    def normalize_with_display(self, location: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Normalize a location and return both canonical ID and display name.
        
        Returns:
            Tuple of (canonical_id, display_name) or (None, None)
        """
        if not location or pd.isna(location):
            return None, None
        
        location_str = str(location)
        
        for compiled_patterns, canonical, display_name in self._compiled_rules:
            for pattern in compiled_patterns:
                if pattern.search(location_str):
                    return canonical, display_name
        
        return None, None
    
    def apply_to_dataframe(
        self,
        df: pd.DataFrame,
        source_columns: List[str],
        output_column: str = "_normalized_location",
        include_display: bool = False,
    ) -> Tuple[pd.DataFrame, int]:
        """
        Apply location normalization to a DataFrame.
        
        Checks multiple source columns and uses the first match.
        
        Args:
            df: Input DataFrame
            source_columns: Columns to check for location (in order of priority)
            output_column: Name of the new normalized column
            include_display: If True, also add a display name column
            
        Returns:
            Tuple of (modified DataFrame, rows affected)
        """
        rows_affected = 0
        normalized_values = []
        display_values = [] if include_display else None
        
        for _, row in df.iterrows():
            normalized = None
            display = None
            
            # Try each source column until we get a match
            for col in source_columns:
                if col in df.columns:
                    value = row.get(col)
                    if value and not pd.isna(value):
                        normalized, display = self.normalize_with_display(value)
                        if normalized:
                            break
            
            normalized_values.append(normalized)
            if include_display:
                display_values.append(display)
            
            if normalized:
                rows_affected += 1
        
        df[output_column] = normalized_values
        if include_display:
            df[f"{output_column}_display"] = display_values
        
        return df, rows_affected


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
            "split_delimiter": self._transform_split_delimiter,
            "hebrew_date_normalize": self._transform_hebrew_date,
            "clean_prefix": self._transform_clean_prefix,
            "clean_suffix": self._transform_clean_suffix,
            "clean_number": self._transform_clean_number,
            "extract_pattern": self._transform_extract_pattern,
            "replace_pattern": self._transform_replace_pattern,
            "time_normalize": self._transform_time_normalize,
            "normalize_location": self._transform_normalize_location,
        }
        
        # Location normalizer (set via set_location_normalizer)
        self._location_normalizer: Optional[LocationNormalizer] = None
    
    def set_location_normalizer(self, normalizer: LocationNormalizer) -> None:
        """Set the location normalizer for location transforms."""
        self._location_normalizer = normalizer
    
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
    
    def _transform_split_delimiter(
        self,
        df: pd.DataFrame,
        config: TransformConfig,
    ) -> Tuple[pd.DataFrame, int, List[str], List[str]]:
        """
        Split a column by delimiter into multiple columns.
        
        Handles optional second values (e.g., "73" stays as primary only,
        "73/82" becomes primary=73, secondary=82).
        
        Example: "73/82" → rate_primary="73", rate_secondary="82"
        Example: "73" → rate_primary="73", rate_secondary=None
        
        YAML config:
        ```yaml
        - source_column: "תעריף"
          transform_type: split_delimiter
          pattern: "^(\\d+)(?:/(\\d+))?$"
          output_columns:
            - name: rate_primary
              group: 1
            - name: rate_secondary
              group: 2
        ```
        """
        col = config.source_column
        # Default pattern handles "/" delimiter with optional second value
        pattern = config.pattern or r'^(\d+)(?:/(\d+))?$'
        output_cols = config.output_columns or []
        
        if not output_cols:
            # Default output columns
            output_cols = [
                {"name": f"{col}_primary", "group": 1},
                {"name": f"{col}_secondary", "group": 2},
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
            
            str_value = str(value).strip()
            match = regex.match(str_value)
            
            if match:
                rows_affected += 1
                for out_col in output_cols:
                    group_num = out_col.get("group", 1)
                    try:
                        extracted = match.group(group_num)
                        if extracted is not None:
                            # Try to convert to number if possible
                            try:
                                extracted = float(extracted) if '.' in extracted else int(extracted)
                            except (ValueError, TypeError):
                                extracted = extracted.strip()
                        new_columns[out_col["name"]][idx] = extracted
                    except IndexError:
                        pass
        
        # Add new columns to DataFrame
        columns_added = []
        for col_name, values in new_columns.items():
            df[col_name] = values
            columns_added.append(col_name)
        
        return df, rows_affected, columns_added, []
    
    def _transform_clean_number(
        self,
        df: pd.DataFrame,
        config: TransformConfig,
    ) -> Tuple[pd.DataFrame, int, List[str], List[str]]:
        """
        Clean numeric values by removing formatting characters.
        
        Example: "8,419" → "8419"
        Example: "1,234.56" → "1234.56"
        
        YAML config:
        ```yaml
        - source_column: "amount"
          transform_type: clean_number
          pattern: ","
          replacement: ""
        ```
        """
        col = config.source_column
        pattern = config.pattern or r','
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
                cleaned = regex.sub(replacement, str_val)
                # Try to convert to numeric
                try:
                    return float(cleaned) if '.' in cleaned else int(cleaned)
                except (ValueError, TypeError):
                    return cleaned
            return value
        
        df[col] = df[col].apply(clean_value)
        return df, rows_affected, [], [col]

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
    
    def _transform_normalize_location(
        self,
        df: pd.DataFrame,
        config: TransformConfig,
    ) -> Tuple[pd.DataFrame, int, List[str], List[str]]:
        """
        Normalize location values using configured rules.
        
        Options in config:
        - output_column: Name for the normalized column (default: _normalized_location)
        - include_display: Also add a display name column
        - rules: Inline rules (alternative to using set_location_normalizer)
        
        Example YAML:
        ```yaml
        - source_column: department
          transform_type: normalize_location
          options:
            output_column: _normalized_location
            include_display: true
            rules:
              - patterns: ["בת ים.*גסטרו", "גיאפה"]
                canonical: bat_yam_gastro
                display_name: "בת ים - גסטרו"
        ```
        """
        col = config.source_column
        options = config.options or {}
        output_column = options.get("output_column", "_normalized_location")
        include_display = options.get("include_display", False)
        inline_rules = options.get("rules", [])
        
        # Use inline rules if provided, otherwise use the pre-configured normalizer
        if inline_rules:
            normalizer = LocationNormalizer([
                LocationRule.from_dict(r) for r in inline_rules
            ])
        elif self._location_normalizer:
            normalizer = self._location_normalizer
        else:
            logger.warning("No location rules configured for normalize_location transform")
            return df, 0, [], []
        
        rows_affected = 0
        normalized_values = []
        display_values = [] if include_display else None
        
        for value in df[col]:
            if pd.isna(value):
                normalized_values.append(None)
                if include_display:
                    display_values.append(None)
                continue
            
            canonical, display = normalizer.normalize_with_display(str(value))
            normalized_values.append(canonical)
            if include_display:
                display_values.append(display)
            
            if canonical:
                rows_affected += 1
        
        df[output_column] = normalized_values
        columns_added = [output_column]
        
        if include_display:
            df[f"{output_column}_display"] = display_values
            columns_added.append(f"{output_column}_display")
        
        return df, rows_affected, columns_added, []


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

