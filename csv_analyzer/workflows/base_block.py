"""
Base Block Class.

Provides shared functionality for all blocks:
- S3 save/load operations
- DataFrame serialization
- Logging helpers
- Input validation
"""

import io
import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import boto3
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class BlockContext:
    """
    Runtime context passed to block execution.
    
    Contains all information needed to execute a block:
    - inputs: S3 URIs or local paths from previous blocks
    - params: Merged parameters
    - S3 client and bucket for data operations (optional for local mode)
    """
    inputs: Dict[str, str] = field(default_factory=dict)  # S3 URIs or local paths
    params: Dict[str, Any] = field(default_factory=dict)  # Merged parameters
    workflow_run_id: str = ""
    block_id: str = ""
    bucket: str = ""
    s3_client: Any = None  # boto3 S3 client
    local_storage_path: str = ""  # Local path for non-S3 storage
    
    @property
    def is_local_mode(self) -> bool:
        """Check if running in local mode (no S3)."""
        return not self.bucket and self.local_storage_path
    
    def __post_init__(self):
        """Initialize S3 client if not provided."""
        if self.s3_client is None and self.bucket:
            self.s3_client = boto3.client(
                "s3",
                region_name=os.environ.get("AWS_REGION", "us-east-1"),
            )


class BaseBlock(ABC):
    """
    Base class for all blocks with shared S3 and utility methods.
    
    Provides:
    - S3 save/load operations
    - DataFrame serialization (JSON format)
    - Logging helpers
    - Input validation
    
    Usage:
        class MyBlock(BaseBlock):
            def run(self) -> Dict[str, str]:
                data = self.load_input("my_input")
                result = process(data)
                return {"output": self.save_to_s3("output", result)}
    """
    
    def __init__(self, ctx: BlockContext):
        """
        Initialize the block with context.
        
        Args:
            ctx: BlockContext with inputs, params, and S3 configuration
        """
        self.ctx = ctx
        self.logger = logging.getLogger(f"block.{ctx.block_id}")
        
        # Initialize S3 client if not provided
        if ctx.s3_client is None and ctx.bucket:
            ctx.s3_client = boto3.client(
                "s3",
                region_name=os.environ.get("AWS_REGION", "us-east-1"),
            )
    
    # ==================== S3 Operations ====================
    
    def save_to_s3(self, name: str, data: Any) -> str:
        """
        Save data to S3 or local storage as JSON, return URI.
        
        Args:
            name: Output name (becomes filename)
            data: DataFrame, dict, list, or any JSON-serializable data
            
        Returns:
            S3 URI or local file path
        """
        # Convert DataFrame to list of records
        if isinstance(data, pd.DataFrame):
            json_data = data.to_dict(orient="records")
        else:
            json_data = data
        
        json_bytes = json.dumps(json_data, ensure_ascii=False, default=str).encode("utf-8")
        
        # Local mode - save to local filesystem
        if self.ctx.is_local_mode:
            from pathlib import Path
            local_dir = Path(self.ctx.local_storage_path) / self.ctx.workflow_run_id / self.ctx.block_id
            local_dir.mkdir(parents=True, exist_ok=True)
            
            # Handle nested paths (e.g., "classified_data/employee_shifts")
            file_path = local_dir / f"{name}.json"
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_path.write_bytes(json_bytes)
            uri = str(file_path)
            self.logger.info(f"Saved {name} → {uri}")
            return uri
        
        # S3 mode
        if not self.ctx.bucket:
            raise ValueError("S3 bucket not configured in BlockContext")
        
        prefix = f"workflows/{self.ctx.workflow_run_id}/{self.ctx.block_id}/"
        s3_key = f"{prefix}{name}.json"
        
        # Upload to S3
        self.ctx.s3_client.put_object(
            Bucket=self.ctx.bucket,
            Key=s3_key,
            Body=json_bytes,
            ContentType="application/json",
        )
        
        s3_uri = f"s3://{self.ctx.bucket}/{s3_key}"
        self.logger.info(f"Saved {name} → {s3_uri}")
        return s3_uri
    
    def load_from_s3(self, uri: str) -> Any:
        """
        Load data from S3 URI or local file path.
        
        Args:
            uri: S3 URI (s3://bucket/key) or local file path
            
        Returns:
            DataFrame if data is list of dicts, otherwise raw JSON data
        """
        from pathlib import Path
        
        # Check if it's a local file path
        if not uri.startswith("s3://"):
            path = Path(uri)
            if path.exists():
                data = json.loads(path.read_text(encoding="utf-8"))
            else:
                raise FileNotFoundError(f"Local file not found: {uri}")
        else:
            # S3 mode
            bucket, key = self._parse_s3_uri(uri)
            response = self.ctx.s3_client.get_object(Bucket=bucket, Key=key)
            data = json.loads(response["Body"].read().decode("utf-8"))
        
        # Auto-convert list of dicts to DataFrame
        if isinstance(data, list) and data and isinstance(data[0], dict):
            return pd.DataFrame(data)
        
        return data
    
    def load_input(self, name: str) -> Any:
        """
        Load an input by name from S3.
        
        Args:
            name: Input port name
            
        Returns:
            Loaded data (DataFrame or raw)
        """
        if name not in self.ctx.inputs:
            raise ValueError(
                f"Input '{name}' not found. Available: {list(self.ctx.inputs.keys())}"
            )
        
        s3_uri = self.ctx.inputs[name]
        self.logger.info(f"Loading input '{name}' from {s3_uri}")
        return self.load_from_s3(s3_uri)
    
    def load_classified_data(self, input_name: str = "data") -> Dict[str, pd.DataFrame]:
        """
        Load CLASSIFIED_DATA input (manifest + individual DataFrames).
        
        Args:
            input_name: Name of the classified_data input
            
        Returns:
            Dict mapping doc_type to DataFrame
        """
        if input_name not in self.ctx.inputs:
            raise ValueError(
                f"Input '{input_name}' not found. Available: {list(self.ctx.inputs.keys())}"
            )
        
        manifest_uri = self.ctx.inputs[input_name]
        manifest = self.load_from_s3(manifest_uri)
        
        # If manifest is already a DataFrame, it's a single document
        if isinstance(manifest, pd.DataFrame):
            return {input_name: manifest}
        
        # Load each document type from manifest
        result = {}
        for doc_type, s3_uri in manifest.items():
            self.logger.info(f"Loading {doc_type} from {s3_uri}")
            result[doc_type] = self.load_from_s3(s3_uri)
        
        return result
    
    def save_classified_data(
        self,
        classified: Dict[str, pd.DataFrame],
        output_name: str = "classified_data",
    ) -> str:
        """
        Save CLASSIFIED_DATA output (individual DataFrames + manifest).
        
        Args:
            classified: Dict mapping doc_type to DataFrame
            output_name: Base name for the output
            
        Returns:
            S3 URI of the manifest file
        """
        s3_refs = {}
        for doc_type, df in classified.items():
            s3_uri = self.save_to_s3(f"{output_name}/{doc_type}", df)
            s3_refs[doc_type] = s3_uri
            self.logger.info(f"Saved {doc_type}: {len(df)} rows → {s3_uri}")
        
        # Save manifest pointing to all files
        manifest_uri = self.save_to_s3(f"{output_name}_manifest", s3_refs)
        return manifest_uri
    
    # ==================== Local File Operations ====================
    
    def load_local_file(self, file_path: str) -> pd.DataFrame:
        """
        Load a local CSV file.
        
        Args:
            file_path: Path to local CSV file
            
        Returns:
            DataFrame
        """
        self.logger.info(f"Loading local file: {file_path}")
        return pd.read_csv(file_path)
    
    def download_from_s3(self, s3_uri: str, local_path: str) -> str:
        """
        Download a file from S3 to local path.
        
        Args:
            s3_uri: S3 URI
            local_path: Local file path to save to
            
        Returns:
            Local file path
        """
        bucket, key = self._parse_s3_uri(s3_uri)
        self.ctx.s3_client.download_file(bucket, key, local_path)
        self.logger.info(f"Downloaded {s3_uri} → {local_path}")
        return local_path
    
    # ==================== Utilities ====================
    
    def get_param(self, name: str, default: Any = None) -> Any:
        """Get a parameter with optional default."""
        return self.ctx.params.get(name, default)
    
    def require_param(self, name: str) -> Any:
        """Get a required parameter, raise if missing."""
        if name not in self.ctx.params:
            raise ValueError(f"Required parameter '{name}' not provided")
        return self.ctx.params[name]
    
    def _parse_s3_uri(self, s3_uri: str) -> tuple:
        """Parse s3://bucket/key into (bucket, key)."""
        if not s3_uri.startswith("s3://"):
            raise ValueError(f"Invalid S3 URI: {s3_uri}")
        parts = s3_uri[5:].split("/", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid S3 URI format: {s3_uri}")
        return parts[0], parts[1]
    
    # ==================== Abstract Method ====================
    
    @abstractmethod
    def run(self) -> Dict[str, str]:
        """
        Execute the block logic.
        
        Returns:
            Dict mapping output_name to S3 URI
        """
        pass

