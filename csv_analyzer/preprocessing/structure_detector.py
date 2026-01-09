"""
AI-based CSV Structure Detector.

Analyzes raw CSV content to detect:
- Header row position (handles multi-row headers)
- Metadata rows (employee name, date range, location, etc.)
- Data start row
- Extracted metadata values
"""

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class MetadataRow:
    """Information extracted from a metadata row."""
    row_index: int
    raw_content: str
    extracted_values: Dict[str, Any] = field(default_factory=dict)
    row_type: str = "metadata"  # metadata, header, data
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "row_index": self.row_index,
            "raw_content": self.raw_content,
            "extracted_values": self.extracted_values,
            "row_type": self.row_type,
        }


@dataclass
class CSVStructure:
    """
    Detected structure of a CSV file.
    
    Attributes:
        header_row: Index of the actual header row (0-based)
        data_start_row: Index where data begins
        metadata_rows: List of metadata rows with extracted values
        detected_encoding: Detected file encoding
        has_multi_row_header: Whether the CSV has metadata before headers
        confidence: Confidence score for the detection
    """
    header_row: int
    data_start_row: int
    metadata_rows: List[MetadataRow] = field(default_factory=list)
    detected_encoding: str = "utf-8"
    has_multi_row_header: bool = False
    confidence: float = 0.0
    raw_preview: List[str] = field(default_factory=list)
    
    def get_extracted_metadata(self) -> Dict[str, Any]:
        """
        Get all extracted metadata values combined.
        
        Earlier rows take precedence - if a value is already set,
        later rows won't overwrite it.
        """
        result = {}
        for row in self.metadata_rows:
            for key, value in row.extracted_values.items():
                # Only set if not already present (earlier rows take precedence)
                if key not in result and value is not None:
                    result[key] = value
        return result
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "header_row": self.header_row,
            "data_start_row": self.data_start_row,
            "metadata_rows": [r.to_dict() for r in self.metadata_rows],
            "detected_encoding": self.detected_encoding,
            "has_multi_row_header": self.has_multi_row_header,
            "confidence": self.confidence,
        }


class StructureDetector:
    """
    Detects CSV structure using AI and heuristics.
    
    Handles CSVs with:
    - Standard single-row headers
    - Multi-row headers with metadata (employee name, date range, etc.)
    - Varying column counts in metadata rows
    
    Usage:
        detector = StructureDetector(openai_client)
        structure = detector.detect("shifts.csv")
        
        print(structure.header_row)  # 2
        print(structure.metadata_rows[0].extracted_values)  # {"employee_name": "איאד שאהין"}
    """
    
    # Hebrew day name patterns (Sunday=א through Saturday=ש)
    HEBREW_DAY_PATTERN = re.compile(r'^([אבגדהוש])\s*-\s*(\d{1,2})$')
    
    # Date range pattern (e.g., "01.12.2025 - 31.12.2025")
    DATE_RANGE_PATTERN = re.compile(r'(\d{1,2}\.\d{1,2}\.\d{4})\s*-\s*(\d{1,2}\.\d{1,2}\.\d{4})')
    
    # Location pattern (Hebrew with hyphens)
    LOCATION_PATTERN = re.compile(r'[\u0590-\u05FF].*-.*[\u0590-\u05FF]')
    
    def __init__(self, openai_client=None, max_preview_rows: int = 10):
        """
        Initialize the structure detector.
        
        Args:
            openai_client: Optional OpenAI client for AI-based detection
            max_preview_rows: Maximum rows to preview for detection
        """
        self.openai_client = openai_client
        self.max_preview_rows = max_preview_rows
    
    def detect(
        self,
        csv_file: Union[str, Path, pd.DataFrame],
        use_ai: bool = True,
    ) -> CSVStructure:
        """
        Detect the structure of a CSV file.
        
        Args:
            csv_file: Path to CSV file or DataFrame
            use_ai: Whether to use AI for detection (falls back to heuristics if False)
            
        Returns:
            CSVStructure with detected information
        """
        # Read raw lines for analysis
        if isinstance(csv_file, pd.DataFrame):
            # Convert DataFrame back to CSV lines for analysis
            raw_lines = csv_file.to_csv(index=False).split('\n')[:self.max_preview_rows]
        else:
            raw_lines = self._read_raw_lines(csv_file)
        
        if not raw_lines:
            return CSVStructure(header_row=0, data_start_row=1)
        
        # First, try heuristic detection
        structure = self._detect_with_heuristics(raw_lines)
        
        # If we have OpenAI and need more confidence, use AI
        if use_ai and self.openai_client and structure.confidence < 0.8:
            ai_structure = self._detect_with_ai(raw_lines)
            if ai_structure and ai_structure.confidence > structure.confidence:
                structure = ai_structure
        
        structure.raw_preview = raw_lines[:5]
        return structure
    
    def _read_raw_lines(self, csv_file: Union[str, Path]) -> List[str]:
        """Read raw lines from a CSV file."""
        path = Path(csv_file)
        
        # Try different encodings
        encodings = ['utf-8', 'utf-8-sig', 'cp1255', 'iso-8859-8', 'latin-1']
        
        for encoding in encodings:
            try:
                with open(path, 'r', encoding=encoding) as f:
                    lines = []
                    for i, line in enumerate(f):
                        if i >= self.max_preview_rows:
                            break
                        lines.append(line.strip())
                    return lines
            except (UnicodeDecodeError, UnicodeError):
                continue
        
        logger.warning(f"Could not decode {csv_file} with any encoding")
        return []
    
    def _detect_with_heuristics(self, raw_lines: List[str]) -> CSVStructure:
        """
        Detect structure using heuristics.
        
        Heuristics:
        1. Count columns in each row - header row typically has consistent column count with data
        2. Look for metadata patterns in early rows (date ranges, Hebrew text, etc.)
        3. Identify row with most column-like content (headers)
        """
        if not raw_lines:
            return CSVStructure(header_row=0, data_start_row=1, confidence=0.5)
        
        # Parse each line to get column counts
        row_analysis = []
        for i, line in enumerate(raw_lines):
            cols = self._parse_csv_line(line)
            non_empty_cols = [c for c in cols if c.strip()]
            
            row_analysis.append({
                "index": i,
                "line": line,
                "col_count": len(cols),
                "non_empty_count": len(non_empty_cols),
                "cols": cols,
                "is_metadata": self._is_metadata_row(cols),
                "has_date_range": bool(self.DATE_RANGE_PATTERN.search(line)),
                "has_location": bool(self.LOCATION_PATTERN.search(line)),
            })
        
        # Find the header row
        # Header row typically:
        # - Has many non-empty columns
        # - Columns look like headers (text, not numbers/dates)
        # - Is followed by consistent data rows
        
        header_row = 0
        metadata_rows = []
        
        # Check first few rows for metadata patterns
        for i, analysis in enumerate(row_analysis[:5]):
            if analysis["is_metadata"]:
                # Extract metadata from this row using heuristics
                extracted = self._extract_metadata_from_row(analysis["cols"], analysis["line"])
                if extracted:
                    metadata_rows.append(MetadataRow(
                        row_index=i,
                        raw_content=analysis["line"],
                        extracted_values=extracted,
                        row_type="metadata",
                    ))
                    header_row = i + 1
            else:
                # This looks like headers or data
                break
        
        # If we have OpenAI and found metadata rows, use AI for better extraction
        if self.openai_client and metadata_rows:
            ai_metadata = self._extract_metadata_with_ai(raw_lines, len(metadata_rows))
            if ai_metadata:
                # Replace heuristic metadata with AI-extracted metadata
                # Distribute to appropriate rows (first row gets name/location, etc.)
                for row in metadata_rows:
                    row.extracted_values = {}  # Clear heuristic values
                
                # Put all AI metadata in the first metadata row
                if metadata_rows:
                    metadata_rows[0].extracted_values = ai_metadata
        
        # Verify header_row makes sense
        if header_row < len(row_analysis):
            # Data should start after header
            data_start_row = header_row + 1
        else:
            header_row = 0
            data_start_row = 1
        
        has_multi_row = len(metadata_rows) > 0
        confidence = 0.9 if has_multi_row else 0.7
        
        return CSVStructure(
            header_row=header_row,
            data_start_row=data_start_row,
            metadata_rows=metadata_rows,
            has_multi_row_header=has_multi_row,
            confidence=confidence,
        )
    
    def _parse_csv_line(self, line: str) -> List[str]:
        """Parse a CSV line handling quotes."""
        import csv
        import io
        try:
            reader = csv.reader(io.StringIO(line))
            return next(reader, [])
        except Exception:
            return line.split(',')
    
    def _is_metadata_row(self, cols: List[str]) -> bool:
        """
        Check if a row looks like metadata rather than headers/data.
        
        Metadata rows typically:
        - Have few non-empty columns
        - Contain date ranges, names, locations
        - Don't look like typical column headers
        """
        non_empty = [c.strip() for c in cols if c.strip()]
        
        if not non_empty:
            return False
        
        # If most columns are empty, likely metadata
        if len(non_empty) <= 3 and len(cols) > 5:
            return True
        
        # Check first cell for metadata patterns
        first_cell = cols[0].strip() if cols else ""
        
        # Date range in first cells
        full_line = ','.join(cols)
        if self.DATE_RANGE_PATTERN.search(full_line):
            return True
        
        # Location-like pattern
        if self.LOCATION_PATTERN.search(first_cell):
            # But not if it looks like a header
            if len(non_empty) <= 3:
                return True
        
        return False
    
    def _extract_metadata_from_row(
        self,
        cols: List[str],
        raw_line: str,
    ) -> Dict[str, Any]:
        """Extract metadata values from a row using heuristics."""
        result = {}
        
        full_text = ','.join(cols)
        
        # Extract date range
        date_match = self.DATE_RANGE_PATTERN.search(full_text)
        if date_match:
            result["date_range_start"] = date_match.group(1)
            result["date_range_end"] = date_match.group(2)
        
        # Look for employee name (Hebrew text in first column)
        first_col = cols[0].strip() if cols else ""
        if first_col and self._looks_like_name(first_col):
            result["employee_name"] = first_col
        
        # Look for location (Hebrew with hyphens, typically has area info)
        for col in cols:
            col = col.strip()
            if col and self._looks_like_location(col):
                result["location"] = col
                break
        
        # Look for employee ID (numeric, typically 3-4 digits)
        for col in cols:
            col = col.strip()
            if col.isdigit() and 2 <= len(col) <= 6:
                result["employee_id"] = col
                break
        
        return result
    
    def _extract_metadata_with_ai(
        self,
        raw_lines: List[str],
        num_metadata_rows: int,
    ) -> Dict[str, Any]:
        """
        Use OpenAI to extract metadata from the first rows of the CSV.
        
        This is more accurate than heuristics for complex cases.
        """
        if not self.openai_client:
            return {}
        
        # Get the metadata rows
        metadata_preview = '\n'.join(raw_lines[:num_metadata_rows])
        
        prompt = f"""Extract metadata from these CSV header rows. These rows contain information about the document, NOT the actual data.

CSV metadata rows:
```
{metadata_preview}
```

Extract these fields if present:
- employee_name: Person's name (e.g., "איאד שאהין")
- employee_id: Numeric employee ID (e.g., "353")
- location: Physical location/branch/clinic (e.g., "בת ים - עצמאיים - גיאפה")
- department: Department or unit name
- date_range_start: Start date if a date range is present (format: DD.MM.YYYY)
- date_range_end: End date if a date range is present (format: DD.MM.YYYY)
- company_name: Company or organization name
- contract_type: Type of contract/employment (e.g., "חוזה שעתי", "שכיר")

Important:
- "location" should be a physical place (city, clinic, branch), NOT a contract type
- Contract/employment types like "חוזה שעתי - תקן 7 שעות" are NOT locations
- Return null for fields that are not clearly present

Respond with JSON only:
{{
    "employee_name": <string or null>,
    "employee_id": <string or null>,
    "location": <string or null>,
    "department": <string or null>,
    "date_range_start": <string or null>,
    "date_range_end": <string or null>,
    "company_name": <string or null>,
    "contract_type": <string or null>
}}"""

        try:
            response = self.openai_client.client.chat.completions.create(
                model=self.openai_client.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a data extraction expert. Extract metadata from CSV headers accurately. Be precise - only extract values you are confident about."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                response_format={"type": "json_object"},
            )
            
            result = json.loads(response.choices[0].message.content)
            
            # Filter out null values
            extracted = {k: v for k, v in result.items() if v is not None}
            
            logger.info(f"AI extracted metadata: {list(extracted.keys())}")
            return extracted
            
        except Exception as e:
            logger.warning(f"AI metadata extraction failed: {e}")
            return {}
    
    def _looks_like_name(self, text: str) -> bool:
        """Check if text looks like a person's name."""
        # Contains Hebrew characters
        has_hebrew = bool(re.search(r'[\u0590-\u05FF]', text))
        
        # Has reasonable length for a name
        reasonable_length = 3 <= len(text) <= 50
        
        # Doesn't contain typical header words
        header_words = ['תאריך', 'שעה', 'סוג', 'מחלקה', 'קוד', 'סכום', 'שם טיפול']
        not_header = not any(w in text for w in header_words)
        
        # Doesn't look like a location (no hyphens with Hebrew on both sides)
        not_location = not self._looks_like_location(text)
        
        return has_hebrew and reasonable_length and not_header and not_location
    
    def _looks_like_location(self, text: str) -> bool:
        """Check if text looks like a location."""
        # Known non-location patterns to exclude
        non_location_patterns = [
            'חוזה',      # Contract
            'תקן',       # Standard/quota
            'שעתי',      # Hourly
            'שכיר',      # Salaried
            'עצמאי',     # Independent
            'משרה',      # Position
            'התחשבנות',  # Settlement
            'חודשי',     # Monthly
        ]
        
        # If contains non-location keywords, it's not a location
        if any(pattern in text for pattern in non_location_patterns):
            return False
        
        # Known location indicators (cities, areas)
        location_indicators = [
            'בת ים', 'תל אביב', 'חיפה', 'ירושלים', 'באר שבע',
            'חדרה', 'נתניה', 'אשדוד', 'פתח תקווה', 'ראשון לציון',
            'סניף', 'מרפאה', 'מכון', 'בית חולים', 'מרכז',
        ]
        
        # If contains known location indicator, it's a location
        if any(indicator in text for indicator in location_indicators):
            return True
        
        # Location pattern: Hebrew - Hebrew (e.g., "בת ים - עצמאיים - גיאפה")
        # But only if it has at least 3 parts (more specific than contract types)
        parts = text.split('-')
        if len(parts) >= 3:
            # Check if parts contain Hebrew
            hebrew_parts = sum(1 for p in parts if re.search(r'[\u0590-\u05FF]', p.strip()))
            return hebrew_parts >= 3
        
        return False
    
    def _detect_with_ai(self, raw_lines: List[str]) -> Optional[CSVStructure]:
        """Use OpenAI to detect CSV structure."""
        if not self.openai_client:
            return None
        
        preview = '\n'.join(raw_lines[:self.max_preview_rows])
        
        prompt = f"""Analyze this CSV file structure and identify:
1. Which row contains the actual column headers (0-indexed)
2. Which row the data starts (0-indexed)
3. Any metadata rows before the headers (employee name, date range, location, etc.)

CSV Preview (first {len(raw_lines)} rows):
```
{preview}
```

Respond with JSON only:
{{
    "header_row": <int>,
    "data_start_row": <int>,
    "has_metadata": <bool>,
    "metadata": {{
        "employee_name": <string or null>,
        "employee_id": <string or null>,
        "location": <string or null>,
        "date_range_start": <string or null>,
        "date_range_end": <string or null>
    }},
    "confidence": <float 0-1>,
    "reasoning": <string>
}}"""

        try:
            response = self.openai_client.client.chat.completions.create(
                model=self.openai_client.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a data analyst expert at understanding CSV file structures. "
                                   "Analyze the structure and extract metadata. Be precise with row indices."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                response_format={"type": "json_object"},
            )
            
            result = json.loads(response.choices[0].message.content)
            
            # Build metadata rows
            metadata_rows = []
            if result.get("has_metadata"):
                for i in range(result["header_row"]):
                    if i < len(raw_lines):
                        extracted = {}
                        if result.get("metadata"):
                            extracted = {k: v for k, v in result["metadata"].items() if v}
                        
                        metadata_rows.append(MetadataRow(
                            row_index=i,
                            raw_content=raw_lines[i],
                            extracted_values=extracted if i == 0 else {},  # Only first row gets metadata
                            row_type="metadata",
                        ))
            
            return CSVStructure(
                header_row=result["header_row"],
                data_start_row=result["data_start_row"],
                metadata_rows=metadata_rows,
                has_multi_row_header=result.get("has_metadata", False),
                confidence=result.get("confidence", 0.8),
            )
            
        except Exception as e:
            logger.warning(f"AI structure detection failed: {e}")
            return None
    
    def normalize_csv(
        self,
        csv_file: Union[str, Path],
        structure: CSVStructure,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Load CSV using detected structure and return normalized DataFrame.
        
        Args:
            csv_file: Path to CSV file
            structure: Detected CSV structure
            
        Returns:
            Tuple of (DataFrame, extracted_metadata)
        """
        # Read CSV starting from the header row
        # When header=N, pandas uses row N (0-indexed) as the header
        # and automatically skips rows 0 to N-1
        df = pd.read_csv(
            csv_file,
            header=structure.header_row,
        )
        
        # Skip any rows between header and data start if needed
        # (e.g., if there's a blank row between headers and data)
        rows_to_skip = structure.data_start_row - structure.header_row - 1
        if rows_to_skip > 0:
            df = df.iloc[rows_to_skip:]
            df = df.reset_index(drop=True)
        
        # Get extracted metadata
        metadata = structure.get_extracted_metadata()
        
        logger.info(
            f"Normalized CSV: header_row={structure.header_row}, "
            f"data_start={structure.data_start_row}, "
            f"rows={len(df)}, metadata={list(metadata.keys())}"
        )
        
        return df, metadata

