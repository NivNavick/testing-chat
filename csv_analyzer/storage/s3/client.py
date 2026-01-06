"""
S3 Client wrapper for file storage operations.
"""

import boto3
from botocore.exceptions import ClientError
from typing import Optional, BinaryIO
from functools import lru_cache

from csv_analyzer.core.config import get_settings


class S3Client:
    """Wrapper around boto3 S3 client with convenience methods."""
    
    def __init__(
        self,
        bucket: str = None,
        region: str = None,
        access_key: str = None,
        secret_key: str = None,
        endpoint_url: str = None
    ):
        settings = get_settings()
        
        self.bucket = bucket or settings.s3_bucket
        self.region = region or settings.s3_region
        
        # Build client config
        client_kwargs = {"region_name": self.region}
        
        if endpoint_url or settings.s3_endpoint_url:
            client_kwargs["endpoint_url"] = endpoint_url or settings.s3_endpoint_url
        
        if access_key or settings.s3_access_key:
            client_kwargs["aws_access_key_id"] = access_key or settings.s3_access_key
            client_kwargs["aws_secret_access_key"] = secret_key or settings.s3_secret_key
        
        self._client = boto3.client("s3", **client_kwargs)
    
    @property
    def client(self):
        """Access the underlying boto3 client."""
        return self._client
    
    def upload_file(
        self,
        file_obj: BinaryIO,
        key: str,
        content_type: str = None,
        metadata: dict = None
    ) -> str:
        """
        Upload a file object to S3.
        
        Args:
            file_obj: File-like object to upload
            key: S3 object key
            content_type: MIME type of the file
            metadata: Additional metadata to store with the object
            
        Returns:
            The S3 key of the uploaded object
        """
        extra_args = {}
        
        if content_type:
            extra_args["ContentType"] = content_type
        
        if metadata:
            extra_args["Metadata"] = metadata
        
        self._client.upload_fileobj(
            file_obj,
            self.bucket,
            key,
            ExtraArgs=extra_args if extra_args else None
        )
        
        return key
    
    def upload_bytes(
        self,
        data: bytes,
        key: str,
        content_type: str = None,
        metadata: dict = None
    ) -> str:
        """
        Upload bytes to S3.
        
        Args:
            data: Bytes to upload
            key: S3 object key
            content_type: MIME type of the data
            metadata: Additional metadata
            
        Returns:
            The S3 key of the uploaded object
        """
        extra_args = {}
        
        if content_type:
            extra_args["ContentType"] = content_type
        
        if metadata:
            extra_args["Metadata"] = metadata
        
        self._client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=data,
            **extra_args
        )
        
        return key
    
    def download_file(self, key: str) -> bytes:
        """
        Download a file from S3.
        
        Args:
            key: S3 object key
            
        Returns:
            File contents as bytes
        """
        response = self._client.get_object(Bucket=self.bucket, Key=key)
        return response["Body"].read()
    
    def download_to_file(self, key: str, file_obj: BinaryIO) -> None:
        """
        Download a file from S3 to a file object.
        
        Args:
            key: S3 object key
            file_obj: File-like object to write to
        """
        self._client.download_fileobj(self.bucket, key, file_obj)
    
    def generate_presigned_url(
        self,
        key: str,
        expiration: int = 3600,
        operation: str = "get_object"
    ) -> str:
        """
        Generate a presigned URL for an S3 object.
        
        Args:
            key: S3 object key
            expiration: URL expiration time in seconds (default 1 hour)
            operation: S3 operation (get_object or put_object)
            
        Returns:
            Presigned URL string
        """
        params = {"Bucket": self.bucket, "Key": key}
        
        return self._client.generate_presigned_url(
            ClientMethod=operation,
            Params=params,
            ExpiresIn=expiration
        )
    
    def delete_file(self, key: str) -> bool:
        """
        Delete a file from S3.
        
        Args:
            key: S3 object key
            
        Returns:
            True if deletion was successful
        """
        try:
            self._client.delete_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError:
            return False
    
    def file_exists(self, key: str) -> bool:
        """
        Check if a file exists in S3.
        
        Args:
            key: S3 object key
            
        Returns:
            True if file exists
        """
        try:
            self._client.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError:
            return False
    
    def list_files(self, prefix: str = "", max_keys: int = 1000) -> list:
        """
        List files in S3 with a given prefix.
        
        Args:
            prefix: Key prefix to filter by
            max_keys: Maximum number of keys to return
            
        Returns:
            List of S3 object keys
        """
        response = self._client.list_objects_v2(
            Bucket=self.bucket,
            Prefix=prefix,
            MaxKeys=max_keys
        )
        
        if "Contents" not in response:
            return []
        
        return [obj["Key"] for obj in response["Contents"]]
    
    def get_file_metadata(self, key: str) -> Optional[dict]:
        """
        Get metadata for an S3 object.
        
        Args:
            key: S3 object key
            
        Returns:
            Object metadata dict or None if not found
        """
        try:
            response = self._client.head_object(Bucket=self.bucket, Key=key)
            return {
                "content_type": response.get("ContentType"),
                "content_length": response.get("ContentLength"),
                "last_modified": response.get("LastModified"),
                "metadata": response.get("Metadata", {})
            }
        except ClientError:
            return None


@lru_cache
def get_s3_client() -> S3Client:
    """Get cached S3 client instance."""
    return S3Client()

