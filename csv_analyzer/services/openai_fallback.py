"""
OpenAI Fallback Service for Column Mapping.

Supports two modes:
1. Fallback mode: Only use OpenAI for columns below confidence threshold
2. Verify mode: Use OpenAI to verify ALL matches (even high confidence ones)

Pre-filters candidates using embeddings to keep API calls efficient.
"""

import logging
from typing import Any, Dict, List, Optional

from csv_analyzer.openai_client import OpenAIClient, get_openai_client

logger = logging.getLogger(__name__)


class OpenAIFallbackService:
    """
    Service to handle OpenAI fallback for unmapped columns.
    
    Supports two modes:
    
    1. Fallback mode (process_unmapped_columns):
       - Only called for columns below confidence threshold
       - Asks OpenAI to pick best match from candidates
    
    2. Verify mode (verify_and_find_match):
       - Verifies if a proposed match is semantically correct
       - If rejected, tries next candidate
       - Continues until a match is accepted or all exhausted
    """
    
    # Number of candidates to send to OpenAI
    TOP_K_CANDIDATES = 5
    
    def __init__(
        self,
        openai_client: Optional[OpenAIClient] = None,
        enabled: bool = True,
    ):
        """
        Initialize the fallback service.
        
        Args:
            openai_client: Pre-configured OpenAI client (or None to auto-create)
            enabled: Whether fallback is enabled
        """
        self.enabled = enabled
        self._client = openai_client
        
        if self.enabled and self._client is None:
            self._client = get_openai_client()
            if self._client is None:
                logger.warning("OpenAI fallback disabled - client not available")
                self.enabled = False
    
    @property
    def is_available(self) -> bool:
        """Check if fallback service is available."""
        return self.enabled and self._client is not None
    
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
        Verify candidates one by one until finding a correct match.
        
        Args:
            column_name: Source column name
            column_type: Detected column type
            sample_values: Sample values from column
            candidates: List of candidate fields from embeddings (ordered by similarity)
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
                "source": "openai_verified",
                "attempts": int,  # How many candidates we tried
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
                    "reason": "OpenAI not available for verification"
                }
            return self._no_match("No candidates available")
        
        # Get domain context for OpenAI if available
        openai_context = None
        if vertical_context:
            openai_context = vertical_context.get_openai_context()
        
        # Try each candidate in order
        for attempt, candidate in enumerate(candidates, 1):
            field_name = candidate.get("field_name")
            field_description = candidate.get("description", "")
            
            # Get transformation info from schema if available
            accepts_units = None
            target_unit = None
            if schema_registry and vertical:
                schema = schema_registry.get_schema(vertical, document_type)
                if schema:
                    field = schema.get_field(field_name)
                    if field:
                        accepts_units = getattr(field, 'accepts_units', None)
                        target_unit = getattr(field, 'target_unit', None)
            
            # Verify with OpenAI (including transformation and domain context)
            result = self._client.verify_mapping(
                column_name=column_name,
                column_type=column_type,
                sample_values=sample_values,
                proposed_field=field_name,
                field_description=field_description,
                document_type=document_type,
                accepts_units=accepts_units,
                target_unit=target_unit,
                vertical_context=openai_context,
            )
            
            if result.get("is_correct"):
                # Found a valid match!
                logger.info(
                    f"✅ '{column_name}' → '{field_name}' verified after {attempt} attempt(s)"
                )
                return {
                    "target_field": field_name,
                    "confidence": candidate.get("similarity", 0.8),
                    "field_type": candidate.get("field_type"),
                    "required": candidate.get("required", False),
                    "source": "openai_verified",
                    "attempts": attempt,
                    "reason": result.get("reason", "Verified by OpenAI"),
                    "needs_transformation": result.get("needs_transformation", False),
                }
            else:
                logger.info(
                    f"❌ '{column_name}' → '{field_name}' rejected: {result.get('reason', '')}"
                )
        
        # All candidates rejected
        logger.info(
            f"⚪ '{column_name}' has no valid match after trying {len(candidates)} candidates"
        )
        return self._no_match(f"All {len(candidates)} candidates rejected by OpenAI")
    
    def _no_match(self, reason: str) -> Dict[str, Any]:
        """Return a 'no match' result."""
        return {
            "target_field": None,
            "confidence": 0.0,
            "field_type": None,
            "required": False,
            "source": "openai_verified",
            "attempts": 0,
            "reason": reason
        }
    
    def process_unmapped_columns(
        self,
        unmapped_columns: List[Dict[str, Any]],
        document_type: str,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Process unmapped columns using OpenAI fallback.
        
        Args:
            unmapped_columns: List of column info dicts, each containing:
                - column_name: str
                - column_type: str
                - sample_values: List[str]
                - candidates: List[Dict] - top embedding matches
            document_type: Target document type (e.g., "employee_shifts")
            
        Returns:
            Dict mapping column_name -> {
                "target_field": str or None,
                "confidence": float (0.0-1.0),
                "source": "openai_fallback",
                "openai_confidence": "high" | "medium" | "low",
                "reason": str
            }
        """
        if not self.is_available:
            logger.debug("OpenAI fallback not available, returning empty results")
            return {}
        
        if not unmapped_columns:
            return {}
        
        logger.info(f"Processing {len(unmapped_columns)} unmapped columns with OpenAI fallback")
        
        # Use batch processing for efficiency
        results = self._client.suggest_column_mappings_batch(
            columns=unmapped_columns,
            document_type=document_type,
        )
        
        # Convert OpenAI results to our format
        processed = {}
        for col in unmapped_columns:
            col_name = col["column_name"]
            openai_result = results.get(col_name, {})
            
            mapped_field = openai_result.get("mapped_field")
            openai_confidence = openai_result.get("confidence", "low")
            reason = openai_result.get("reason", "")
            
            # Convert OpenAI confidence to numeric
            confidence_map = {
                "high": 0.90,
                "medium": 0.75,
                "low": 0.60,
                "error": 0.0,
            }
            numeric_confidence = confidence_map.get(openai_confidence, 0.0)
            
            # Find the field type from candidates if we have a match
            field_type = None
            required = False
            if mapped_field:
                for candidate in col.get("candidates", []):
                    if candidate.get("field_name") == mapped_field:
                        field_type = candidate.get("field_type")
                        required = candidate.get("required", False)
                        break
            
            processed[col_name] = {
                "target_field": mapped_field,
                "confidence": numeric_confidence if mapped_field else 0.0,
                "field_type": field_type,
                "required": required,
                "source": "openai_fallback",
                "openai_confidence": openai_confidence,
                "reason": reason,
            }
            
            if mapped_field:
                logger.info(
                    f"OpenAI mapped '{col_name}' → '{mapped_field}' "
                    f"({openai_confidence} confidence: {reason})"
                )
            else:
                logger.info(
                    f"OpenAI confirmed '{col_name}' is unmappable: {reason}"
                )
        
        return processed


def create_fallback_service(
    api_key: Optional[str] = None,
    model: str = "gpt-4o-mini",
    enabled: bool = True,
) -> OpenAIFallbackService:
    """
    Factory function to create fallback service.
    
    Args:
        api_key: OpenAI API key (or use OPENAI_API_KEY env var)
        model: OpenAI model to use
        enabled: Whether to enable the service
        
    Returns:
        Configured OpenAIFallbackService
    """
    if not enabled:
        return OpenAIFallbackService(enabled=False)
    
    client = get_openai_client(api_key=api_key, model=model)
    return OpenAIFallbackService(openai_client=client, enabled=enabled)
