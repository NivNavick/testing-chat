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
        accepts_units: Optional[List[str]] = None,
        target_unit: Optional[str] = None,
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
            accepts_units: Optional list of units the target field can accept (with conversion)
            target_unit: Optional target unit the field expects
            
        Returns:
            {
                "is_correct": bool,
                "reason": "explanation",
                "needs_transformation": bool (if unit conversion needed)
            }
        """
        samples_str = ", ".join(str(v) for v in sample_values[:5])
        
        # Build transformation context if available
        transformation_context = ""
        if accepts_units and target_unit:
            other_units = [u for u in accepts_units if u != target_unit]
            if other_units:
                transformation_context = f"""
IMPORTANT - UNIT CONVERSION SUPPORT:
The target field "{proposed_field}" expects data in {target_unit}, but ALSO ACCEPTS:
- {', '.join(other_units)}
If the source column contains data in any of these units ({', '.join(other_units)}), 
it IS a valid match because the system will automatically convert to {target_unit}.
For example: if source is in "hours" and target expects "minutes", this is VALID (will multiply by 60).
"""
        
        prompt = f"""Verify if this column mapping is semantically correct for a {document_type} document.

CONTEXT: This is a {document_type} document. Consider domain-specific terminology:
- In shift schedules: "exit time", "departure time", "clock out" = shift_end
- In shift schedules: "entry time", "arrival time", "clock in" = shift_start  
- In medical records: "performer", "provider", "doctor" = the person who did the action
- Column names may be in any language (Hebrew, Spanish, etc.)
{transformation_context}
SOURCE COLUMN:
- Name: {column_name}
- Type: {column_type}
- Sample values: {samples_str}

PROPOSED MAPPING:
- Target field: {proposed_field}
- Field description: {field_description}

Question: In the context of {document_type}, does "{column_name}" map to "{proposed_field}"?

Be PRACTICAL, not overly strict:
- Accept if the concepts are equivalent in this domain
- Accept translations and synonyms (exit time = shift end)
- Accept if the data represents the same concept but in different units (hours vs minutes for duration)
- Reject only if there's a CLEAR semantic mismatch (e.g., "break_minutes" vs "shift_end_time")

Respond with JSON:
{{
    "is_correct": true or false,
    "reason": "brief explanation",
    "needs_transformation": true or false (set to true if unit conversion is needed)
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
3. The document type context (e.g., employee_shifts, medical_actions)

Your job is to:
1. Analyze the column name and sample values IN CONTEXT
2. Consider domain-specific terminology and translations
3. Accept matches where concepts are equivalent (e.g., "exit time" = "shift end" in schedules)
4. Reject only clear mismatches (e.g., "break time" ≠ "total duration")

IMPORTANT:
- Column names may be in ANY language (Hebrew, Spanish, Arabic, etc.)
- Be PRACTICAL: in real-world data, "departure time" in a shift schedule IS the shift end
- Accept synonyms and translations
- Don't be overly pedantic about wording

Always respond in JSON format."""

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
