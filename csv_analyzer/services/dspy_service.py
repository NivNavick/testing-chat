"""
DSPy Column Classification Service.

Uses DSPy for LLM-based column classification with optimizable prompts.
Can be trained on ground truth for better accuracy.

Modes:
1. Fallback mode: Only classify columns below embedding confidence threshold
2. Verify mode: Verify ALL column mappings (rejects wrong matches)

Supports parallel verification for faster processing.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Check if DSPy is available
try:
    from csv_analyzer.services.dspy_classifier import (
        DSPyColumnClassifier,
        create_dspy_classifier,
        DSPY_AVAILABLE,
    )
except ImportError:
    DSPY_AVAILABLE = False
    logger.warning("DSPy not available. Run: pip install dspy-ai")


class DSPyClassificationService:
    """
    DSPy-based column classification service.
    
    Uses optimizable prompts that can be trained on ground truth.
    
    Modes:
    
    1. Fallback mode (process_unmapped_columns):
       - Only called for columns below confidence threshold
       - Picks best match from embedding candidates
    
    2. Verify mode (verify_and_find_match):
       - Verifies if a proposed match is semantically correct
       - If rejected, tries next candidate
       - Continues until a match is accepted or all exhausted
    """
    
    # Number of candidates to evaluate
    TOP_K_CANDIDATES = 5
    
    # Default number of parallel workers
    DEFAULT_WORKERS = 4
    
    def __init__(
        self,
        enabled: bool = True,
        compiled_path: Optional[Path] = None,
        model: str = "openai/gpt-4o-mini",
    ):
        """
        Initialize the DSPy classification service.
        
        Args:
            enabled: Whether service is enabled
            compiled_path: Path to compiled DSPy model (for optimized prompts)
            model: LLM model to use (default: openai/gpt-4o-mini)
        """
        self.enabled = enabled
        self._classifier = None
        self.model = model
        
        if not self.enabled:
            return
        
        if not DSPY_AVAILABLE:
            logger.error("DSPy not available. Run: pip install dspy-ai")
            self.enabled = False
            return
        
        self._classifier = create_dspy_classifier(
            model=model,
            compiled_path=compiled_path,
        )
        
        if self._classifier is None:
            logger.error("Failed to initialize DSPy classifier")
            self.enabled = False
        else:
            if compiled_path:
                logger.info(f"🚀 DSPy loaded compiled model from {compiled_path}")
            else:
                logger.info(f"🚀 DSPy initialized (zero-shot, model: {model})")
    
    @property
    def is_available(self) -> bool:
        """Check if service is available."""
        return self.enabled and self._classifier is not None and self._classifier.is_available
    
    def verify_and_find_match(
        self,
        column_name: str,
        column_type: str,
        sample_values: List[str],
        candidates: List[Dict[str, Any]],
        document_type: str,
        schema_registry: Optional[Any] = None,
        vertical: Optional[str] = None,
        vertical_context: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        NON-GREEDY: Send ALL candidates to DSPy and let it pick the best one.
        
        This is more accurate than greedy verification because DSPy sees
        ALL options before deciding (e.g., can choose shift_end over actual_end
        when column is "Scheduled_End").
        
        Args:
            column_name: Source column name
            column_type: Detected column type
            sample_values: Sample values from column
            candidates: Candidate fields from embeddings (ordered by similarity)
            document_type: Target document type
            schema_registry: Optional schema registry for transformation info
            vertical: Optional vertical for schema lookup
            vertical_context: Optional VerticalContext for domain terminology
            
        Returns:
            {
                "target_field": str or None,
                "confidence": float,
                "field_type": str or None,
                "required": bool,
                "source": "dspy",
                "attempts": 1,
                "reason": str
            }
        """
        if not self.is_available:
            # Return first candidate without verification
            if candidates:
                first = candidates[0]
                return {
                    "target_field": first.get("field_name"),
                    "confidence": first.get("similarity", 0.8),
                    "field_type": first.get("field_type"),
                    "required": first.get("required", False),
                    "source": "embeddings_unverified",
                    "attempts": 0,
                    "reason": "DSPy not available"
                }
            return self._no_match("No candidates available")
        
        if not candidates:
            return self._no_match("No candidates available")
        
        # Get domain context if available
        domain_context = None
        if vertical_context:
            domain_context = vertical_context.get_openai_context()
        
        # NON-GREEDY: Send ALL candidates to DSPy in one call
        # DSPy sees the full picture and picks the BEST match
        result = self._classifier.classify_column(
            column_name=column_name,
            column_type=column_type,
            sample_values=sample_values,
            candidates=candidates,
            document_type=document_type,
            domain_context=domain_context,
        )
        
        if result.target_field:
            # Find the matching candidate to get field_type and required
            field_type = None
            required = False
            similarity = 0.8
            for candidate in candidates:
                if candidate.get("field_name") == result.target_field:
                    field_type = candidate.get("field_type")
                    required = candidate.get("required", False)
                    similarity = candidate.get("similarity", 0.8)
                    break
            
            # Convert confidence level to numeric
            confidence_map = {"high": 0.95, "medium": 0.80, "low": 0.65}
            numeric_confidence = confidence_map.get(result.confidence, similarity)
            
            logger.info(
                f"✅ '{column_name}' → '{result.target_field}' "
                f"(non-greedy, {result.confidence} confidence)"
            )
            return {
                "target_field": result.target_field,
                "confidence": numeric_confidence,
                "field_type": field_type,
                "required": required,
                "source": "dspy",
                "attempts": 1,  # Single call, non-greedy
                "reason": result.reason,
            }
        
        logger.info(f"⚪ '{column_name}' has no valid match (DSPy returned none)")
        return self._no_match(result.reason or "No suitable match found")
    
    def verify_columns_parallel(
        self,
        columns_to_verify: List[Dict[str, Any]],
        document_type: str,
        schema_registry: Optional[Any] = None,
        vertical: Optional[str] = None,
        vertical_context: Optional[Any] = None,
        max_workers: int = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Verify multiple columns in parallel for faster processing.
        
        Args:
            columns_to_verify: List of column info dicts:
                - column_name: str
                - column_type: str
                - sample_values: List[str]
                - candidates: List[Dict] - embedding candidates
            document_type: Target document type
            schema_registry: Optional schema registry
            vertical: Optional vertical
            vertical_context: Optional domain context
            max_workers: Number of parallel workers (default: 4)
            
        Returns:
            Dict mapping column_name -> verification result
        """
        if not self.is_available:
            logger.debug("DSPy not available for parallel verification")
            return {}
        
        if not columns_to_verify:
            return {}
        
        max_workers = max_workers or self.DEFAULT_WORKERS
        num_columns = len(columns_to_verify)
        
        logger.info(f"🚀 Verifying {num_columns} columns in parallel (max {max_workers} workers)")
        
        results = {}
        
        def verify_single_column(col_info: Dict) -> Tuple[str, Dict[str, Any]]:
            """Verify a single column (runs in thread)."""
            col_name = col_info["column_name"]
            result = self.verify_and_find_match(
                column_name=col_name,
                column_type=col_info.get("column_type", "unknown"),
                sample_values=col_info.get("sample_values", []),
                candidates=col_info.get("candidates", []),
                document_type=document_type,
                schema_registry=schema_registry,
                vertical=vertical,
                vertical_context=vertical_context,
            )
            return col_name, result
        
        # Use ThreadPoolExecutor for parallel I/O-bound operations
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all verification tasks
            future_to_column = {
                executor.submit(verify_single_column, col): col["column_name"]
                for col in columns_to_verify
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_column):
                col_name = future_to_column[future]
                try:
                    name, result = future.result()
                    results[name] = result
                except Exception as e:
                    logger.error(f"Error verifying '{col_name}': {e}")
                    results[col_name] = self._no_match(f"Error: {e}")
        
        logger.info(f"✅ Parallel verification complete: {len(results)} columns processed")
        return results
    
    def _no_match(self, reason: str) -> Dict[str, Any]:
        """Return a 'no match' result."""
        return {
            "target_field": None,
            "confidence": 0.0,
            "field_type": None,
            "required": False,
            "source": "dspy",
            "attempts": 0,
            "reason": reason
        }
    
    def classify_columns(
        self,
        columns: List[Dict[str, Any]],
        document_type: str,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Classify multiple columns using DSPy.
        
        Args:
            columns: List of column info dicts:
                - column_name: str
                - column_type: str
                - sample_values: List[str]
                - candidates: List[Dict] - embedding candidates
            document_type: Target document type
            
        Returns:
            Dict mapping column_name -> classification result
        """
        if not self.is_available:
            logger.debug("DSPy not available")
            return {}
        
        if not columns:
            return {}
        
        logger.info(f"🧠 Classifying {len(columns)} columns with DSPy")
        
        results = {}
        for col in columns:
            col_name = col["column_name"]
            col_type = col.get("column_type", "unknown")
            sample_values = col.get("sample_values", [])
            candidates = col.get("candidates", [])
            
            # Use DSPy to classify
            result = self._classifier.classify_column(
                column_name=col_name,
                column_type=col_type,
                sample_values=sample_values,
                candidates=candidates,
                document_type=document_type,
            )
            
            # Convert confidence to numeric
            confidence_map = {"high": 0.90, "medium": 0.75, "low": 0.60}
            numeric_confidence = confidence_map.get(result.confidence, 0.5)
            
            # Find field type from candidates
            field_type = None
            required = False
            if result.target_field:
                for candidate in candidates:
                    if candidate.get("field_name") == result.target_field:
                        field_type = candidate.get("field_type")
                        required = candidate.get("required", False)
                        break
            
            results[col_name] = {
                "target_field": result.target_field,
                "confidence": numeric_confidence if result.target_field else 0.0,
                "field_type": field_type,
                "required": required,
                "sources": ["dspy"],
                "reason": result.reason,
            }
            
            if result.target_field:
                logger.info(
                    f"✅ '{col_name}' → '{result.target_field}' ({result.confidence})"
                )
            else:
                logger.info(f"⚪ '{col_name}' has no match")
        
        return results
    
    def classify_columns_parallel(
        self,
        columns: List[Dict[str, Any]],
        document_type: str,
        max_workers: int = 4,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Classify multiple columns in PARALLEL using DSPy.
        
        Args:
            columns: List of column info dicts
            document_type: Target document type
            max_workers: Number of parallel workers
            
        Returns:
            Dict mapping column_name -> classification result
        """
        if not self.is_available:
            logger.debug("DSPy not available")
            return {}
        
        if not columns:
            return {}
        
        logger.info(f"🚀 Classifying {len(columns)} columns in parallel (workers={max_workers})")
        
        results = {}
        
        def _classify_single(col: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
            """Classify a single column."""
            col_name = col["column_name"]
            col_type = col.get("column_type", "unknown")
            sample_values = col.get("sample_values", [])
            candidates = col.get("candidates", [])
            
            try:
                result = self._classifier.classify_column(
                    column_name=col_name,
                    column_type=col_type,
                    sample_values=sample_values,
                    candidates=candidates,
                    document_type=document_type,
                )
                
                # Convert confidence to numeric
                confidence_map = {"high": 0.90, "medium": 0.75, "low": 0.60}
                numeric_confidence = confidence_map.get(result.confidence, 0.5)
                
                # Find field type from candidates
                field_type = None
                required = False
                if result.target_field:
                    for candidate in candidates:
                        if candidate.get("field_name") == result.target_field:
                            field_type = candidate.get("field_type")
                            required = candidate.get("required", False)
                            break
                
                return col_name, {
                    "target_field": result.target_field,
                    "confidence": numeric_confidence if result.target_field else 0.0,
                    "field_type": field_type,
                    "required": required,
                    "sources": ["dspy"],
                    "reason": result.reason,
                }
            except Exception as e:
                logger.error(f"Error classifying '{col_name}': {e}")
                return col_name, {
                    "target_field": None,
                    "confidence": 0.0,
                    "field_type": None,
                    "required": False,
                    "sources": ["dspy"],
                    "reason": f"Error: {str(e)}",
                }
        
        # Run in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_classify_single, col): col for col in columns}
            
            for future in as_completed(futures):
                col_name, result = future.result()
                results[col_name] = result
                
                if result.get("target_field"):
                    logger.info(f"✅ '{col_name}' → '{result['target_field']}'")
                else:
                    logger.info(f"⚪ '{col_name}' has no match")
        
        return results


def create_dspy_service(
    model: str = "openai/gpt-4o-mini",
    compiled_path: Optional[Path] = None,
    enabled: bool = True,
) -> DSPyClassificationService:
    """
    Create DSPy classification service.
    
    Args:
        model: LLM model (default: openai/gpt-4o-mini)
        compiled_path: Path to compiled model (optional)
        enabled: Whether to enable the service
        
    Returns:
        Configured DSPyClassificationService
    """
    if not enabled:
        return DSPyClassificationService(enabled=False)
    
    return DSPyClassificationService(
        enabled=True,
        compiled_path=compiled_path,
        model=model,
    )

