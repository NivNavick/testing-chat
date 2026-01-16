"""
Configuration management for the CSV Analyzer API.

Loads configuration from YAML file with environment variable overrides.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class DatabaseConfig(BaseModel):
    """PostgreSQL database configuration."""
    host: str = "localhost"
    port: int = 5432
    name: str = "csv_mapping"
    user: str = "postgres"
    password: str = "postgres"
    pool_min: int = 2
    pool_max: int = 10
    
    @property
    def dsn(self) -> str:
        """Get PostgreSQL connection string."""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"
    
    @property
    def asyncpg_dsn(self) -> str:
        """Get asyncpg-compatible connection string."""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"


class S3PrefixConfig(BaseModel):
    """S3 path prefix configuration."""
    sessions: str = "sessions/"
    uploads: str = "uploads/"
    workflows: str = "workflows/"
    temp: str = "temp/"


class S3Config(BaseModel):
    """AWS S3 configuration."""
    bucket: str = "csv-analyzer-data"
    region: str = "us-east-1"
    prefix: S3PrefixConfig = Field(default_factory=S3PrefixConfig)
    endpoint_url: Optional[str] = None  # For LocalStack
    
    @property
    def is_localstack(self) -> bool:
        """Check if using LocalStack."""
        return self.endpoint_url is not None


class StorageConfig(BaseModel):
    """Storage configuration."""
    workflows_path: str = "./csv_analyzer/workflows/definitions"
    upload_max_size_mb: int = 100
    temp_files_ttl: int = 3600
    local_temp_dir: str = "/tmp/csv_analyzer"
    
    @property
    def upload_max_size_bytes(self) -> int:
        """Get max upload size in bytes."""
        return self.upload_max_size_mb * 1024 * 1024


class APIConfig(BaseModel):
    """API server configuration."""
    title: str = "CSV Analyzer API"
    version: str = "1.0.0"
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: List[str] = Field(default_factory=lambda: ["http://localhost:3000", "http://localhost:8080"])
    workers: int = 4


class EmbeddingsConfig(BaseModel):
    """Embeddings model configuration."""
    model: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    cache_dir: str = "./models"


class LoggingConfig(BaseModel):
    """Logging configuration."""
    level: str = "INFO"
    format: str = "json"


class Settings(BaseModel):
    """Application settings."""
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    s3: S3Config = Field(default_factory=S3Config)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    
    @classmethod
    def from_yaml(cls, path: str) -> "Settings":
        """Load settings from YAML file."""
        config_path = Path(path)
        
        if not config_path.exists():
            # Return defaults if no config file
            return cls()
        
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        
        # Apply environment variable overrides
        data = cls._apply_env_overrides(data)
        
        return cls(**data)
    
    @classmethod
    def _apply_env_overrides(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment variable overrides to config."""
        # Database overrides
        if "database" not in data:
            data["database"] = {}
        
        if os.environ.get("DB_HOST"):
            data["database"]["host"] = os.environ["DB_HOST"]
        if os.environ.get("DB_PORT"):
            data["database"]["port"] = int(os.environ["DB_PORT"])
        if os.environ.get("DB_NAME"):
            data["database"]["name"] = os.environ["DB_NAME"]
        if os.environ.get("DB_USER"):
            data["database"]["user"] = os.environ["DB_USER"]
        if os.environ.get("DB_PASSWORD"):
            data["database"]["password"] = os.environ["DB_PASSWORD"]
        
        # S3 overrides
        if "s3" not in data:
            data["s3"] = {}
        
        if os.environ.get("S3_BUCKET"):
            data["s3"]["bucket"] = os.environ["S3_BUCKET"]
        if os.environ.get("S3_REGION"):
            data["s3"]["region"] = os.environ["S3_REGION"]
        if os.environ.get("S3_ENDPOINT_URL"):
            data["s3"]["endpoint_url"] = os.environ["S3_ENDPOINT_URL"]
        
        # API overrides
        if "api" not in data:
            data["api"] = {}
        
        if os.environ.get("API_HOST"):
            data["api"]["host"] = os.environ["API_HOST"]
        if os.environ.get("API_PORT"):
            data["api"]["port"] = int(os.environ["API_PORT"])
        
        return data


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get application settings (singleton)."""
    global _settings
    
    if _settings is None:
        # Look for config file in project root
        config_paths = [
            Path("config.yaml"),
            Path("config.yml"),
            Path(__file__).parent.parent.parent / "config.yaml",
            Path(__file__).parent.parent.parent / "config.yml",
        ]
        
        config_path = None
        for path in config_paths:
            if path.exists():
                config_path = str(path)
                break
        
        if config_path:
            _settings = Settings.from_yaml(config_path)
        else:
            _settings = Settings()
    
    return _settings


def reload_settings(config_path: Optional[str] = None) -> Settings:
    """Reload settings from config file."""
    global _settings
    
    if config_path:
        _settings = Settings.from_yaml(config_path)
    else:
        _settings = None
        return get_settings()
    
    return _settings

