"""
Configuration settings for the Analytics Platform.
Uses pydantic-settings for environment variable loading.
"""

from typing import Optional
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )
    
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_debug: bool = False
    
    # S3 Configuration
    s3_bucket: str = "analytics-platform-775555669663"
    s3_region: str = "us-east-1"
    s3_access_key: Optional[str] = None  # Uses ~/.aws/credentials if not set
    s3_secret_key: Optional[str] = None  # Uses ~/.aws/credentials if not set
    s3_endpoint_url: Optional[str] = None  # For MinIO/LocalStack (leave None for real AWS)
    
    # PostgreSQL Configuration
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str = "postgres"
    postgres_password: str = "postgres"
    postgres_db: str = "analytics"
    
    @property
    def postgres_url(self) -> str:
        """Construct PostgreSQL connection URL."""
        return f"postgresql://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
    
    @property
    def postgres_async_url(self) -> str:
        """Construct async PostgreSQL connection URL."""
        return f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
    
    # DuckDB Configuration
    duckdb_memory_limit: str = "4GB"
    duckdb_threads: int = 4
    
    # LLM Configuration
    openai_api_key: Optional[str] = None
    llm_model: str = "gpt-4o-mini"
    
    # Processing Configuration
    default_mode: str = "AUTO"
    max_file_size_mb: int = 100
    
    # ChromaDB (existing)
    chroma_persist_directory: str = "storage/chroma"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()

