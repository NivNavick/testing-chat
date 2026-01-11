"""
Upload Block - Upload files to S3.

Takes local file paths and uploads them to S3, returning S3 URIs
for downstream blocks.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List

import boto3

from csv_analyzer.workflows.block import BlockRegistry
from csv_analyzer.workflows.ontology import DataType
from csv_analyzer.workflows.base_block import BaseBlock, BlockContext

logger = logging.getLogger(__name__)


class UploadBlock(BaseBlock):
    """
    Upload local files to S3.
    
    Supports:
    - Multiple file uploads
    - Custom bucket and prefix
    - Skip mode for local development
    """
    
    def run(self) -> Dict[str, str]:
        """
        Upload files to S3.
        
        Returns:
            Dict with 'uploaded_files' key containing S3 URI of manifest
        """
        files = self.require_param("files")
        skip_upload = self.get_param("skip_upload", False)
        
        # For local development - skip S3, return local paths
        if skip_upload:
            self.logger.info(f"Skip upload mode: returning {len(files)} local paths")
            result_uri = self.save_to_s3("uploaded_files", files)
            return {"uploaded_files": result_uri}
        
        # Get bucket from params or environment
        bucket = self.get_param("bucket") or os.environ.get("S3_BUCKET")
        if not bucket:
            raise ValueError("S3 bucket not specified. Set 'bucket' param or S3_BUCKET env var")
        
        # Build prefix with workflow run ID for organization
        prefix = self.get_param("prefix") or f"workflows/{self.ctx.workflow_run_id}/uploads/"
        
        # Initialize boto3 client
        s3_client = boto3.client(
            "s3",
            region_name=os.environ.get("AWS_REGION", "us-east-1"),
        )
        
        s3_uris = []
        for file_path in files:
            path = Path(file_path)
            s3_key = f"{prefix}{path.name}"
            
            # Upload file
            s3_client.upload_file(
                Filename=str(path),
                Bucket=bucket,
                Key=s3_key,
                ExtraArgs={"ContentType": "text/csv"},
            )
            
            s3_uri = f"s3://{bucket}/{s3_key}"
            s3_uris.append(s3_uri)
            self.logger.info(f"Uploaded {path.name} → {s3_uri}")
        
        # Save manifest of uploaded files
        result_uri = self.save_to_s3("uploaded_files", s3_uris)
        
        return {"uploaded_files": result_uri}


# Register the block
@BlockRegistry.register(
    name="upload_to_s3",
    inputs=[],  # Entry point - no inputs
    outputs=[{"name": "uploaded_files", "ontology": DataType.S3_REFERENCES}],
    parameters=[
        {"name": "files", "type": "file_list", "required": True, "description": "Local file paths to upload"},
        {"name": "bucket", "type": "string", "default": None, "description": "S3 bucket name"},
        {"name": "prefix", "type": "string", "default": None, "description": "S3 key prefix"},
        {"name": "skip_upload", "type": "boolean", "default": False, "description": "Skip S3 upload for local dev"},
    ],
    block_class=UploadBlock,
    description="Upload local files to S3",
)
def upload_to_s3(ctx: BlockContext) -> Dict[str, str]:
    """Upload files to S3."""
    return UploadBlock(ctx).run()

