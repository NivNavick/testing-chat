"""
Async S3 client wrapper using aioboto3.

Provides async operations for:
- File upload/download
- File streaming
- Presigned URL generation
- Batch operations
"""

import asyncio
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, AsyncIterator, BinaryIO, Dict, List, Optional, Union

import aioboto3
from botocore.config import Config
from botocore.exceptions import ClientError

from csv_analyzer.api.config import S3PrefixConfig

logger = logging.getLogger(__name__)


class S3ClientError(Exception):
    """S3 client error."""
    pass


class S3Client:
    """
    Async S3 client wrapper.
    
    Provides high-level async operations for S3 with automatic
    error handling, retries, and logging.
    """
    
    def __init__(
        self,
        bucket: str,
        region: str = "us-east-1",
        endpoint_url: Optional[str] = None,
        prefix_config: Optional[S3PrefixConfig] = None,
        max_retries: int = 3,
    ):
        """
        Initialize the S3 client.
        
        Args:
            bucket: S3 bucket name
            region: AWS region
            endpoint_url: Optional endpoint URL (for LocalStack)
            prefix_config: S3 path prefix configuration
            max_retries: Maximum number of retries for failed operations
        """
        self.bucket = bucket
        self.region = region
        self.endpoint_url = endpoint_url
        self.prefix = prefix_config or S3PrefixConfig()
        self.max_retries = max_retries
        
        # Configure boto
        self._config = Config(
            retries={
                "max_attempts": max_retries,
                "mode": "adaptive",
            },
            connect_timeout=5,
            read_timeout=30,
        )
        
        # Session for connection pooling
        self._session = aioboto3.Session()
        self._client = None
    
    async def _get_client(self):
        """Get or create S3 client."""
        if self._client is None:
            self._client = await self._session.client(
                "s3",
                region_name=self.region,
                endpoint_url=self.endpoint_url,
                config=self._config,
            ).__aenter__()
        return self._client
    
    async def close(self) -> None:
        """Close the S3 client."""
        if self._client is not None:
            await self._client.__aexit__(None, None, None)
            self._client = None
    
    def _build_key(self, prefix_type: str, *parts: str) -> str:
        """Build S3 key with prefix."""
        prefix = getattr(self.prefix, prefix_type, "")
        return prefix + "/".join(str(p) for p in parts if p)
    
    def get_uri(self, key: str) -> str:
        """Get S3 URI for a key."""
        return f"s3://{self.bucket}/{key}"
    
    def parse_uri(self, uri: str) -> tuple:
        """Parse S3 URI into bucket and key."""
        if not uri.startswith("s3://"):
            raise ValueError(f"Invalid S3 URI: {uri}")
        
        parts = uri[5:].split("/", 1)
        bucket = parts[0]
        key = parts[1] if len(parts) > 1 else ""
        return bucket, key
    
    # ========================================================================
    # Upload Operations
    # ========================================================================
    
    async def upload_file(
        self,
        local_path: Union[str, Path],
        s3_key: str,
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Upload a local file to S3.
        
        Args:
            local_path: Path to local file
            s3_key: S3 object key
            content_type: Optional content type
            metadata: Optional metadata dict
            
        Returns:
            S3 URI of uploaded file
        """
        client = await self._get_client()
        local_path = Path(local_path)
        
        if not local_path.exists():
            raise FileNotFoundError(f"File not found: {local_path}")
        
        extra_args = {}
        if content_type:
            extra_args["ContentType"] = content_type
        if metadata:
            extra_args["Metadata"] = metadata
        
        try:
            logger.debug(f"Uploading {local_path} to s3://{self.bucket}/{s3_key}")
            
            await client.upload_file(
                str(local_path),
                self.bucket,
                s3_key,
                ExtraArgs=extra_args if extra_args else None,
            )
            
            uri = self.get_uri(s3_key)
            logger.info(f"Uploaded file to {uri}")
            return uri
            
        except ClientError as e:
            logger.error(f"Failed to upload file: {e}")
            raise S3ClientError(f"Upload failed: {e}")
    
    async def upload_fileobj(
        self,
        file_obj: BinaryIO,
        s3_key: str,
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Upload a file object to S3.
        
        Args:
            file_obj: File-like object
            s3_key: S3 object key
            content_type: Optional content type
            metadata: Optional metadata dict
            
        Returns:
            S3 URI of uploaded file
        """
        client = await self._get_client()
        
        extra_args = {}
        if content_type:
            extra_args["ContentType"] = content_type
        if metadata:
            extra_args["Metadata"] = metadata
        
        try:
            logger.debug(f"Uploading file object to s3://{self.bucket}/{s3_key}")
            
            await client.upload_fileobj(
                file_obj,
                self.bucket,
                s3_key,
                ExtraArgs=extra_args if extra_args else None,
            )
            
            uri = self.get_uri(s3_key)
            logger.info(f"Uploaded file object to {uri}")
            return uri
            
        except ClientError as e:
            logger.error(f"Failed to upload file object: {e}")
            raise S3ClientError(f"Upload failed: {e}")
    
    async def upload_bytes(
        self,
        data: bytes,
        s3_key: str,
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Upload bytes to S3.
        
        Args:
            data: Bytes to upload
            s3_key: S3 object key
            content_type: Optional content type
            metadata: Optional metadata dict
            
        Returns:
            S3 URI of uploaded file
        """
        client = await self._get_client()
        
        extra_args = {}
        if content_type:
            extra_args["ContentType"] = content_type
        if metadata:
            extra_args["Metadata"] = metadata
        
        try:
            logger.debug(f"Uploading bytes to s3://{self.bucket}/{s3_key}")
            
            await client.put_object(
                Bucket=self.bucket,
                Key=s3_key,
                Body=data,
                **extra_args,
            )
            
            uri = self.get_uri(s3_key)
            logger.info(f"Uploaded bytes to {uri}")
            return uri
            
        except ClientError as e:
            logger.error(f"Failed to upload bytes: {e}")
            raise S3ClientError(f"Upload failed: {e}")
    
    # ========================================================================
    # Download Operations
    # ========================================================================
    
    async def download_file(
        self,
        s3_key: str,
        local_path: Union[str, Path],
    ) -> Path:
        """
        Download a file from S3.
        
        Args:
            s3_key: S3 object key
            local_path: Path to save file
            
        Returns:
            Path to downloaded file
        """
        client = await self._get_client()
        local_path = Path(local_path)
        
        # Create parent directories
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            logger.debug(f"Downloading s3://{self.bucket}/{s3_key} to {local_path}")
            
            await client.download_file(
                self.bucket,
                s3_key,
                str(local_path),
            )
            
            logger.info(f"Downloaded file to {local_path}")
            return local_path
            
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code == "404" or error_code == "NoSuchKey":
                raise FileNotFoundError(f"S3 object not found: {s3_key}")
            logger.error(f"Failed to download file: {e}")
            raise S3ClientError(f"Download failed: {e}")
    
    async def download_bytes(self, s3_key: str) -> bytes:
        """
        Download bytes from S3.
        
        Args:
            s3_key: S3 object key
            
        Returns:
            Downloaded bytes
        """
        client = await self._get_client()
        
        try:
            logger.debug(f"Downloading bytes from s3://{self.bucket}/{s3_key}")
            
            response = await client.get_object(
                Bucket=self.bucket,
                Key=s3_key,
            )
            
            data = await response["Body"].read()
            logger.info(f"Downloaded {len(data)} bytes from {s3_key}")
            return data
            
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code == "404" or error_code == "NoSuchKey":
                raise FileNotFoundError(f"S3 object not found: {s3_key}")
            logger.error(f"Failed to download bytes: {e}")
            raise S3ClientError(f"Download failed: {e}")
    
    # ========================================================================
    # Delete Operations
    # ========================================================================
    
    async def delete_file(self, s3_key: str) -> bool:
        """
        Delete a single file from S3.
        
        Args:
            s3_key: S3 object key
            
        Returns:
            True if deleted successfully
        """
        client = await self._get_client()
        
        try:
            logger.debug(f"Deleting s3://{self.bucket}/{s3_key}")
            
            await client.delete_object(
                Bucket=self.bucket,
                Key=s3_key,
            )
            
            logger.info(f"Deleted {s3_key}")
            return True
            
        except ClientError as e:
            logger.error(f"Failed to delete file: {e}")
            raise S3ClientError(f"Delete failed: {e}")
    
    async def delete_prefix(self, prefix: str) -> int:
        """
        Delete all files under a prefix.
        
        Args:
            prefix: S3 key prefix
            
        Returns:
            Number of files deleted
        """
        client = await self._get_client()
        deleted_count = 0
        
        try:
            logger.debug(f"Deleting all objects under s3://{self.bucket}/{prefix}")
            
            # List all objects under prefix
            paginator = client.get_paginator("list_objects_v2")
            
            async for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
                objects = page.get("Contents", [])
                
                if not objects:
                    continue
                
                # Delete in batches of 1000 (S3 limit)
                delete_objects = [{"Key": obj["Key"]} for obj in objects]
                
                await client.delete_objects(
                    Bucket=self.bucket,
                    Delete={"Objects": delete_objects},
                )
                
                deleted_count += len(delete_objects)
            
            logger.info(f"Deleted {deleted_count} objects under {prefix}")
            return deleted_count
            
        except ClientError as e:
            logger.error(f"Failed to delete prefix: {e}")
            raise S3ClientError(f"Delete prefix failed: {e}")
    
    # ========================================================================
    # List Operations
    # ========================================================================
    
    async def list_files(
        self,
        prefix: str,
        max_keys: int = 1000,
    ) -> List[Dict[str, Any]]:
        """
        List files under a prefix.
        
        Args:
            prefix: S3 key prefix
            max_keys: Maximum number of keys to return
            
        Returns:
            List of file metadata dicts
        """
        client = await self._get_client()
        files = []
        
        try:
            paginator = client.get_paginator("list_objects_v2")
            
            async for page in paginator.paginate(
                Bucket=self.bucket,
                Prefix=prefix,
                PaginationConfig={"MaxItems": max_keys},
            ):
                for obj in page.get("Contents", []):
                    files.append({
                        "key": obj["Key"],
                        "size": obj["Size"],
                        "last_modified": obj["LastModified"],
                        "etag": obj.get("ETag", "").strip('"'),
                    })
            
            return files
            
        except ClientError as e:
            logger.error(f"Failed to list files: {e}")
            raise S3ClientError(f"List failed: {e}")
    
    async def exists(self, s3_key: str) -> bool:
        """
        Check if an object exists.
        
        Args:
            s3_key: S3 object key
            
        Returns:
            True if object exists
        """
        client = await self._get_client()
        
        try:
            await client.head_object(
                Bucket=self.bucket,
                Key=s3_key,
            )
            return True
            
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code == "404":
                return False
            raise S3ClientError(f"Head object failed: {e}")
    
    async def get_file_metadata(self, s3_key: str) -> Dict[str, Any]:
        """
        Get file metadata.
        
        Args:
            s3_key: S3 object key
            
        Returns:
            Metadata dict with size, content_type, last_modified, etc.
        """
        client = await self._get_client()
        
        try:
            response = await client.head_object(
                Bucket=self.bucket,
                Key=s3_key,
            )
            
            return {
                "key": s3_key,
                "size": response["ContentLength"],
                "content_type": response.get("ContentType"),
                "last_modified": response["LastModified"],
                "etag": response.get("ETag", "").strip('"'),
                "metadata": response.get("Metadata", {}),
            }
            
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code == "404":
                raise FileNotFoundError(f"S3 object not found: {s3_key}")
            raise S3ClientError(f"Head object failed: {e}")
    
    # ========================================================================
    # Presigned URLs
    # ========================================================================
    
    async def generate_presigned_url(
        self,
        s3_key: str,
        expiration: int = 3600,
        http_method: str = "GET",
    ) -> str:
        """
        Generate a presigned URL for an object.
        
        Args:
            s3_key: S3 object key
            expiration: URL expiration in seconds (default: 1 hour)
            http_method: HTTP method (GET or PUT)
            
        Returns:
            Presigned URL
        """
        client = await self._get_client()
        
        try:
            if http_method.upper() == "GET":
                url = await client.generate_presigned_url(
                    "get_object",
                    Params={"Bucket": self.bucket, "Key": s3_key},
                    ExpiresIn=expiration,
                )
            elif http_method.upper() == "PUT":
                url = await client.generate_presigned_url(
                    "put_object",
                    Params={"Bucket": self.bucket, "Key": s3_key},
                    ExpiresIn=expiration,
                )
            else:
                raise ValueError(f"Unsupported HTTP method: {http_method}")
            
            return url
            
        except ClientError as e:
            logger.error(f"Failed to generate presigned URL: {e}")
            raise S3ClientError(f"Presigned URL generation failed: {e}")
    
    # ========================================================================
    # Convenience Methods
    # ========================================================================
    
    async def upload_session_document(
        self,
        session_id: str,
        document_id: str,
        local_path: Union[str, Path],
        content_type: str = "text/csv",
    ) -> str:
        """
        Upload a document to a session's folder.
        
        Args:
            session_id: Session ID
            document_id: Document ID
            local_path: Path to local file
            content_type: Content type
            
        Returns:
            S3 URI
        """
        filename = Path(local_path).name
        s3_key = self._build_key("sessions", session_id, "documents", f"{document_id}_{filename}")
        return await self.upload_file(local_path, s3_key, content_type=content_type)
    
    async def upload_workflow_output(
        self,
        execution_id: str,
        block_id: str,
        output_name: str,
        local_path: Union[str, Path],
        content_type: str = "text/csv",
    ) -> str:
        """
        Upload a workflow output file.
        
        Args:
            execution_id: Workflow execution ID
            block_id: Block ID
            output_name: Output name
            local_path: Path to local file
            content_type: Content type
            
        Returns:
            S3 URI
        """
        filename = Path(local_path).name
        s3_key = self._build_key("workflows", execution_id, block_id, f"{output_name}_{filename}")
        return await self.upload_file(local_path, s3_key, content_type=content_type)
    
    async def delete_session(self, session_id: str) -> int:
        """
        Delete all files for a session.
        
        Args:
            session_id: Session ID
            
        Returns:
            Number of files deleted
        """
        prefix = self._build_key("sessions", session_id)
        return await self.delete_prefix(prefix)
    
    async def delete_workflow_execution(self, execution_id: str) -> int:
        """
        Delete all files for a workflow execution.
        
        Args:
            execution_id: Execution ID
            
        Returns:
            Number of files deleted
        """
        prefix = self._build_key("workflows", execution_id)
        return await self.delete_prefix(prefix)

