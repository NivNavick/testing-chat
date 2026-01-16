"""
Async wrappers for synchronous operations.

Provides async interfaces to existing synchronous engines
by running them in a thread pool executor.
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Thread pool for sync operations
_executor: Optional[ThreadPoolExecutor] = None


def get_executor(max_workers: int = 4) -> ThreadPoolExecutor:
    """Get or create the thread pool executor."""
    global _executor
    if _executor is None:
        _executor = ThreadPoolExecutor(max_workers=max_workers)
    return _executor


async def run_sync(func, *args, **kwargs):
    """
    Run a synchronous function in the thread pool.
    
    Usage:
        result = await run_sync(sync_function, arg1, arg2, kwarg1=value)
    """
    loop = asyncio.get_event_loop()
    executor = get_executor()
    
    if kwargs:
        return await loop.run_in_executor(
            executor,
            lambda: func(*args, **kwargs)
        )
    else:
        return await loop.run_in_executor(
            executor,
            func,
            *args
        )


async def classify_csv_async(
    csv_path: str,
    vertical: Optional[str] = None,
    k: int = 5,
    hybrid: bool = True,
    use_dspy: bool = False,
    force_reindex: bool = False,
) -> Dict[str, Any]:
    """
    Async wrapper for CSV classification.
    
    Args:
        csv_path: Path to CSV file
        vertical: Optional vertical filter
        k: Number of similar examples
        hybrid: Use hybrid scoring
        use_dspy: Use DSPy for column classification
        force_reindex: Force reindex embeddings
        
    Returns:
        Classification result dict
    """
    def _classify():
        from csv_analyzer.db.connection import init_database, Database
        from csv_analyzer.engines.classification_engine import ClassificationEngine
        from csv_analyzer.multilingual_embeddings_client import get_multilingual_embeddings_client
        from csv_analyzer.api.config import get_settings
        
        settings = get_settings()
        
        # Initialize database if needed
        if not Database.is_initialized():
            init_database(
                host=settings.database.host,
                port=settings.database.port,
                database=settings.database.name,
                user=settings.database.user,
                password=settings.database.password,
                run_migrations=False,
            )
        
        # Get embeddings client
        embeddings_client = get_multilingual_embeddings_client()
        
        if not embeddings_client.is_available:
            raise ValueError("Embedding model not available")
        
        # Create DSPy service if requested
        dspy_service = None
        if use_dspy:
            try:
                from csv_analyzer.services.dspy_service import create_dspy_service
                dspy_service = create_dspy_service(enabled=True)
            except Exception as e:
                logger.warning(f"Failed to initialize DSPy: {e}")
        
        # Create engine and classify
        engine = ClassificationEngine(
            embeddings_client,
            dspy_service=dspy_service,
        )
        
        if hybrid and vertical:
            result = engine.classify_hybrid(
                csv_file=csv_path,
                vertical=vertical,
                k=k,
                force_reindex=force_reindex,
            )
            return result.to_dict()
        else:
            result = engine.classify(
                csv_file=csv_path,
                vertical=vertical,
                k=k,
            )
            return result.to_dict()
    
    return await run_sync(_classify)


async def run_workflow_async(
    workflow_path: str,
    parameters: Dict[str, Any],
    bucket: Optional[str] = None,
    max_workers: int = 4,
) -> Dict[str, Dict[str, str]]:
    """
    Async wrapper for workflow execution.
    
    Args:
        workflow_path: Path to workflow YAML file
        parameters: Workflow parameters
        bucket: S3 bucket (optional)
        max_workers: Max parallel workers
        
    Returns:
        Workflow results (block_id -> output_name -> s3_uri)
    """
    def _run():
        from csv_analyzer.workflows.engine import WorkflowEngine
        from csv_analyzer.api.config import get_settings
        
        settings = get_settings()
        
        engine = WorkflowEngine.from_yaml(
            workflow_path,
            bucket=bucket or settings.s3.bucket,
            max_workers=max_workers,
        )
        
        return engine.run(parameters)
    
    return await run_sync(_run)


async def profile_dataframe_async(csv_path: str) -> List[Dict[str, Any]]:
    """
    Async wrapper for DataFrame profiling.
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        List of column profiles
    """
    def _profile():
        import pandas as pd
        from csv_analyzer.columns_analyzer import profile_dataframe
        
        df = pd.read_csv(csv_path)
        profiles = profile_dataframe(df)
        
        return [
            {
                "column_name": p.get("column_name"),
                "detected_type": p.get("detected_type"),
                "sample_values": p.get("sample_values", [])[:5],
                "null_count": p.get("null_count", 0),
                "unique_count": p.get("unique_count", 0),
            }
            for p in profiles
        ]
    
    return await run_sync(_profile)

