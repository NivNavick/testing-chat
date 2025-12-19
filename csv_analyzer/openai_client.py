"""
OpenAI Client wrapper for CSV Analyzer.

Provides a simple interface to OpenAI API for column mapping fallback.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI package not installed. Fallback mapping will be disabled.")


class OpenAIClient:
    """
    OpenAI client wrapper for column mapping assistance.
    
    Uses GPT to analyze unmapped columns and suggest mappings
    from a pre-filtered list of candidates.
    """
    
    DEFAULT_MODEL = "gpt-4o-mini"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
    ):
        """
        Initialize OpenAI client.
        
        Args:
            api_key: OpenAI API key. If not provided, uses OPENAI_API_KEY env var.
            model: Model to use (default: gpt-4o-mini)
        """
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "OpenAI package not installed. Run: pip install openai"
            )
        
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not provided. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        self.model = model
        self.client = OpenAI(api_key=self.api_key)
        logger.info(f"OpenAI client initialized with model: {self.model}")
    
    def verify_mapping(
        self,
        column_name: str,
        column_type: str,
        sample_values: List[str],
        proposed_field: str,
        field_description: str,
        document_type: str,
    ) -> Dict[str, Any]:
        """
        Verify if a proposed mapping is semantically correct.
        
        Args:
            column_name: Name of the source column
            column_type: Detected type (text, integer, date, etc.)
            sample_values: Sample values from the column
            proposed_field: The field we're proposing to map to
            field_description: Description of the proposed field
            document_type: Target document type
            
        Returns:
            {
                "is_correct": bool,
                "reason": "explanation"
            }
        """
        samples_str = ", ".join(str(v) for v in sample_values[:5])
        
        prompt = f"""Verify if this column mapping is semantically correct:

SOURCE COLUMN:
- Name: {column_name}
- Type: {column_type}
- Sample values: {samples_str}

PROPOSED MAPPING:
- Target field: {proposed_field}
- Field description: {field_description}
- Document type: {document_type}

Is "{column_name}" a correct match for "{proposed_field}"?

Consider:
1. Do the sample values match what the field expects?
2. Does the column NAME semantically match the field meaning?
3. Is there a conceptual mismatch? (e.g., "break_minutes" vs "duration_minutes")

Respond with JSON:
{{
    "is_correct": true or false,
    "reason": "brief explanation"
}}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a data mapping expert. Verify if column mappings are semantically correct. Be strict - reject mappings where the concepts don't match, even if words are similar."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,
                response_format={"type": "json_object"},
            )
            
            result_text = response.choices[0].message.content
            result = json.loads(result_text)
            
            logger.info(
                f"OpenAI verify '{column_name}' → '{proposed_field}': "
                f"{'✅ CORRECT' if result.get('is_correct') else '❌ REJECTED'} - {result.get('reason', '')}"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"OpenAI verify error for '{column_name}': {e}")
            # On error, assume the match is correct (don't block)
            return {
                "is_correct": True,
                "reason": f"Verification error: {e}"
            }

    def suggest_column_mapping(
        self,
        column_name: str,
        column_type: str,
        sample_values: List[str],
        candidates: List[Dict[str, Any]],
        document_type: str,
    ) -> Dict[str, Any]:
        """
        Ask GPT to suggest the best mapping for a column from candidates.
        
        Args:
            column_name: Name of the source column
            column_type: Detected type (text, integer, date, etc.)
            sample_values: Sample values from the column
            candidates: Top N candidate fields from embedding search
                        Each has: field_name, description, field_type, similarity
            document_type: The target document type (e.g., "employee_shifts")
            
        Returns:
            {
                "mapped_field": "field_name" or None,
                "confidence": "high" | "medium" | "low",
                "reason": "explanation"
            }
        """
        # Build the prompt
        prompt = self._build_mapping_prompt(
            column_name=column_name,
            column_type=column_type,
            sample_values=sample_values,
            candidates=candidates,
            document_type=document_type,
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": self._get_system_prompt()
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                temperature=0.1,  # Low temperature for consistent results
                response_format={"type": "json_object"},
            )
            
            result_text = response.choices[0].message.content
            result = json.loads(result_text)
            
            logger.info(
                f"OpenAI mapping for '{column_name}': "
                f"{result.get('mapped_field', 'NONE')} ({result.get('confidence', '?')})"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"OpenAI API error for column '{column_name}': {e}")
            return {
                "mapped_field": None,
                "confidence": "error",
                "reason": str(e)
            }
    
    def suggest_column_mappings_batch(
        self,
        columns: List[Dict[str, Any]],
        document_type: str,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Process multiple unmapped columns in a single API call.
        
        Args:
            columns: List of column info dicts, each containing:
                - column_name
                - column_type
                - sample_values
                - candidates (top N from embeddings)
            document_type: Target document type
            
        Returns:
            Dict mapping column_name -> mapping result
        """
        if not columns:
            return {}
        
        # For efficiency, batch all columns in one prompt
        prompt = self._build_batch_mapping_prompt(columns, document_type)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": self._get_system_prompt()
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,
                response_format={"type": "json_object"},
            )
            
            result_text = response.choices[0].message.content
            result = json.loads(result_text)
            
            # Result should have "mappings" key with dict of column -> mapping
            mappings = result.get("mappings", {})
            
            logger.info(f"OpenAI batch mapping: processed {len(mappings)} columns")
            
            return mappings
            
        except Exception as e:
            logger.error(f"OpenAI batch API error: {e}")
            # Return empty results for all columns
            return {
                col["column_name"]: {
                    "mapped_field": None,
                    "confidence": "error",
                    "reason": str(e)
                }
                for col in columns
            }
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for column mapping."""
        return """You are a data mapping expert. Your task is to analyze CSV columns and determine which target schema field they should map to.

You will be given:
1. A source column with its name, type, and sample values
2. A list of candidate target fields (pre-filtered by semantic similarity)

Your job is to:
1. Analyze the column name and sample values
2. Determine if any candidate field is a good match
3. If yes, return the field name with confidence level
4. If no good match exists, return null

Always respond in JSON format.

Be strict: only map if there's a clear semantic match. It's better to say "no match" than to force a bad mapping."""

    def _build_mapping_prompt(
        self,
        column_name: str,
        column_type: str,
        sample_values: List[str],
        candidates: List[Dict[str, Any]],
        document_type: str,
    ) -> str:
        """Build the prompt for single column mapping."""
        samples_str = ", ".join(str(v) for v in sample_values[:5])
        
        candidates_str = "\n".join(
            f"  {i+1}. {c['field_name']} ({c['field_type']}) - {c.get('description', 'No description')}"
            for i, c in enumerate(candidates)
        )
        
        return f"""Analyze this column and determine the best mapping:

SOURCE COLUMN:
- Name: {column_name}
- Type: {column_type}
- Sample values: {samples_str}

TARGET SCHEMA: {document_type}
CANDIDATE FIELDS:
{candidates_str}

Which candidate field should "{column_name}" map to?

Respond with JSON:
{{
    "mapped_field": "field_name" or null if no good match,
    "confidence": "high" | "medium" | "low",
    "reason": "brief explanation"
}}"""

    def _build_batch_mapping_prompt(
        self,
        columns: List[Dict[str, Any]],
        document_type: str,
    ) -> str:
        """Build prompt for batch column mapping."""
        columns_text = []
        
        for col in columns:
            samples_str = ", ".join(str(v) for v in col.get("sample_values", [])[:5])
            candidates = col.get("candidates", [])
            
            candidates_str = "\n    ".join(
                f"- {c['field_name']} ({c.get('field_type', '?')}): {c.get('description', 'No description')[:50]}"
                for c in candidates
            )
            
            columns_text.append(f"""
COLUMN: {col['column_name']}
  Type: {col.get('column_type', 'unknown')}
  Samples: {samples_str}
  Candidates:
    {candidates_str}
""")
        
        all_columns = "\n".join(columns_text)
        
        return f"""Analyze these columns and determine the best mapping for each:

TARGET SCHEMA: {document_type}

{all_columns}

For EACH column, determine if any candidate is a good match.

Respond with JSON:
{{
    "mappings": {{
        "column_name_1": {{
            "mapped_field": "field_name" or null,
            "confidence": "high" | "medium" | "low",
            "reason": "brief explanation"
        }},
        "column_name_2": {{ ... }},
        ...
    }}
}}"""


def get_openai_client(
    api_key: Optional[str] = None,
    model: str = OpenAIClient.DEFAULT_MODEL,
) -> Optional[OpenAIClient]:
    """
    Factory function to get OpenAI client.
    
    Returns None if OpenAI is not available or not configured.
    """
    if not OPENAI_AVAILABLE:
        logger.warning("OpenAI not available - fallback disabled")
        return None
    
    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OPENAI_API_KEY not set - fallback disabled")
        return None
    
    try:
        return OpenAIClient(api_key=api_key, model=model)
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI client: {e}")
        return None
