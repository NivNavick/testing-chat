"""
File Ingress Module - Handles file parsing and table extraction.
"""

import io
from typing import List
from dataclasses import dataclass

import pandas as pd

from csv_analyzer.modules.base import BaseModule, ModuleContext, ModuleResult
from csv_analyzer.storage.s3.operations import S3Operations


@dataclass
class ExtractedTable:
    """Represents an extracted table from a file."""
    name: str
    location: str  # e.g., "Sheet1" or "Sheet1:A1:M50"
    df: pd.DataFrame
    structure_type: str  # STANDARD, PIVOT, HIERARCHY, etc.
    transforms_applied: List[str] = None
    
    @property
    def row_count(self) -> int:
        return len(self.df)
    
    @property
    def column_count(self) -> int:
        return len(self.df.columns)
    
    @property
    def columns_info(self) -> List[dict]:
        return [
            {
                "name": str(col),
                "type": str(self.df[col].dtype),
                "sample_values": self.df[col].head(3).tolist()
            }
            for col in self.df.columns
        ]


@dataclass
class FileIngressInput:
    """Input for file ingress processing."""
    s3_key: str
    filename: str
    file_type: str  # csv, xlsx, xls


@dataclass
class FileIngressOutput:
    """Output from file ingress processing."""
    tables: List[ExtractedTable]
    original_structure: str  # Overall file structure type


class FileIngressModule(BaseModule[FileIngressInput, FileIngressOutput]):
    """
    Module for file ingress - parsing files and extracting tables.
    
    Supports:
    - CSV files (single table)
    - XLSX files (multiple sheets/tables)
    - Detection of pivot tables, hierarchies, etc.
    """
    
    def __init__(self, s3_ops: S3Operations = None):
        self.s3_ops = s3_ops or S3Operations()
    
    @property
    def name(self) -> str:
        return "file_ingress"
    
    async def process(
        self,
        input_data: FileIngressInput,
        context: ModuleContext
    ) -> ModuleResult:
        """
        Process a file from S3 and extract tables.
        
        Args:
            input_data: File information including S3 key
            context: Processing context
            
        Returns:
            ModuleResult with extracted tables
        """
        try:
            # Download file from S3
            file_bytes = await self.s3_ops.download_raw_file(input_data.s3_key)
            file_obj = io.BytesIO(file_bytes)
            
            # Extract tables based on file type
            if input_data.file_type == "csv":
                tables = self._extract_csv(file_obj, input_data.filename)
            else:
                tables = self._extract_xlsx(file_obj, input_data.filename)
            
            # Determine overall file structure
            structure = self._determine_file_structure(tables)
            
            output = FileIngressOutput(
                tables=tables,
                original_structure=structure
            )
            
            return ModuleResult.ok(
                data=output,
                metadata={
                    "table_count": len(tables),
                    "total_rows": sum(t.row_count for t in tables)
                }
            )
            
        except Exception as e:
            return ModuleResult.fail(
                error=str(e),
                metadata={"s3_key": input_data.s3_key}
            )
    
    def _extract_csv(self, file_obj: io.BytesIO, filename: str) -> List[ExtractedTable]:
        """Extract table from CSV file."""
        df = pd.read_csv(file_obj)
        
        # Clean column names
        df.columns = [str(col).strip() for col in df.columns]
        
        # Remove completely empty rows and columns
        df = df.dropna(how='all')
        df = df.dropna(axis=1, how='all')
        
        table = ExtractedTable(
            name=filename.rsplit(".", 1)[0],
            location="CSV",
            df=df,
            structure_type="STANDARD",
            transforms_applied=["cleaned"]
        )
        
        return [table]
    
    def _extract_xlsx(self, file_obj: io.BytesIO, filename: str) -> List[ExtractedTable]:
        """Extract tables from XLSX file."""
        tables = []
        
        try:
            xlsx = pd.ExcelFile(file_obj)
            
            for sheet_name in xlsx.sheet_names:
                try:
                    df = pd.read_excel(xlsx, sheet_name=sheet_name, header=None)
                    
                    # Skip empty sheets
                    if df.empty or df.dropna(how='all').empty:
                        continue
                    
                    # Find header row (first row with mostly non-null values)
                    header_row = self._find_header_row(df)
                    
                    if header_row is not None:
                        # Re-read with correct header
                        file_obj.seek(0)
                        df = pd.read_excel(
                            xlsx,
                            sheet_name=sheet_name,
                            header=header_row
                        )
                    else:
                        # Use first row as header
                        df.columns = df.iloc[0]
                        df = df.iloc[1:]
                    
                    # Clean
                    df.columns = [str(col).strip() for col in df.columns]
                    df = df.dropna(how='all')
                    df = df.dropna(axis=1, how='all')
                    
                    if df.empty:
                        continue
                    
                    # Detect structure type
                    structure_type = self._detect_table_structure(df)
                    
                    table = ExtractedTable(
                        name=sheet_name,
                        location=sheet_name,
                        df=df.reset_index(drop=True),
                        structure_type=structure_type,
                        transforms_applied=["cleaned", "header_detected"]
                    )
                    
                    tables.append(table)
                    
                except Exception as e:
                    # Skip problematic sheets but continue
                    print(f"Warning: Could not process sheet {sheet_name}: {e}")
                    continue
                    
        except Exception as e:
            raise ValueError(f"Failed to read XLSX file: {e}")
        
        return tables
    
    def _find_header_row(self, df: pd.DataFrame, max_rows: int = 10) -> int:
        """Find the likely header row in a DataFrame."""
        for i in range(min(max_rows, len(df))):
            row = df.iloc[i]
            non_null = row.notna().sum()
            
            # If more than 50% of values are non-null strings, likely header
            if non_null > len(row) * 0.5:
                string_count = sum(1 for v in row if isinstance(v, str) and v.strip())
                if string_count > non_null * 0.5:
                    return i
        
        return 0
    
    def _detect_table_structure(self, df: pd.DataFrame) -> str:
        """Detect the structure type of a table."""
        # Check for pivot table indicators
        if self._is_likely_pivot(df):
            return "PIVOT"
        
        # Check for hierarchical structure
        if self._is_likely_hierarchy(df):
            return "HIERARCHY"
        
        return "STANDARD"
    
    def _is_likely_pivot(self, df: pd.DataFrame) -> bool:
        """Check if DataFrame looks like a pivot table."""
        if len(df.columns) < 3:
            return False
        
        # Check if first column has repeated values (row headers)
        first_col = df.iloc[:, 0]
        if first_col.nunique() < len(first_col) * 0.5:
            # Check if other columns look like date/category headers
            cols = [str(c) for c in df.columns[1:]]
            date_like = sum(1 for c in cols if any(char.isdigit() for char in c))
            if date_like > len(cols) * 0.5:
                return True
        
        return False
    
    def _is_likely_hierarchy(self, df: pd.DataFrame) -> bool:
        """Check if DataFrame has hierarchical structure."""
        if len(df.columns) < 2:
            return False
        
        # Check if first few columns have many empty values (indentation pattern)
        for i in range(min(3, len(df.columns))):
            null_ratio = df.iloc[:, i].isna().sum() / len(df)
            if null_ratio > 0.3:
                return True
        
        return False
    
    def _determine_file_structure(self, tables: List[ExtractedTable]) -> str:
        """Determine overall file structure based on extracted tables."""
        if len(tables) == 0:
            return "EMPTY"
        
        if len(tables) == 1:
            return tables[0].structure_type
        
        # Multiple tables
        structures = set(t.structure_type for t in tables)
        if len(structures) == 1:
            return f"MULTI_{list(structures)[0]}"
        
        return "MIXED"

