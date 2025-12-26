"""
DSPy-based Column Classifier Service.

Provides optimizable LLM-based column classification that can be trained
on ground truth examples to learn optimal prompts.

Two modes:
1. Zero-shot: Use DSPy with default prompts (still better than raw OpenAI)
2. Optimized: Compile with ground truth for best accuracy

Usage:
    # Zero-shot mode
    service = DSPyColumnClassifier()
    result = service.verify_mapping(column_name="שעת_יציאה", ...)

    # Optimized mode (after running optimizer)
    service = DSPyColumnClassifier.load_optimized("path/to/compiled_model")
"""

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Literal

logger = logging.getLogger(__name__)

# Check if dspy is available
try:
    import dspy
    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False
    logger.warning("dspy-ai not installed. Run: pip install dspy-ai")


# =============================================================================
# DSPy Signatures (Define the input/output structure)
# =============================================================================

if DSPY_AVAILABLE:

    class VerifyMappingSignature(dspy.Signature):
        """Verify if a proposed column-to-field mapping is semantically correct.
        
        Consider domain-specific terminology and translations.
        In shift schedules: 'exit time' = 'shift_end', 'entry time' = 'shift_start'.
        Column names may be in any language (Hebrew, Spanish, etc.).
        """
        
        column_name: str = dspy.InputField(desc="Source column name (may be in any language)")
        column_type: str = dspy.InputField(desc="Detected data type (datetime, integer, text, etc.)")
        sample_values: str = dspy.InputField(desc="Comma-separated sample values from the column")
        proposed_field: str = dspy.InputField(desc="The schema field we're proposing to map to")
        field_description: str = dspy.InputField(desc="Description of the proposed field")
        document_type: str = dspy.InputField(desc="Target document type (e.g., employee_shifts)")
        domain_context: str = dspy.InputField(desc="Domain-specific terminology hints")
        
        is_correct: bool = dspy.OutputField(desc="True if the mapping is semantically correct")
        reason: str = dspy.OutputField(desc="Brief explanation for the decision")
        needs_transformation: bool = dspy.OutputField(desc="True if unit/format conversion is needed")


    class ClassifyColumnSignature(dspy.Signature):
        """Select the best target field for a source column from candidates.
        
        Consider:
        - Domain-specific terminology and translations
        - Column names may be in Hebrew, Spanish, or other languages
        - Similarity scores from embeddings (higher = more likely)
        - Data type compatibility
        
        Pick the BEST match or 'none' if no candidate is appropriate.
        """
        
        column_name: str = dspy.InputField(desc="Source column name")
        column_type: str = dspy.InputField(desc="Detected data type")
        sample_values: str = dspy.InputField(desc="Comma-separated sample values")
        candidates: str = dspy.InputField(desc="JSON list of candidate fields with similarity scores")
        document_type: str = dspy.InputField(desc="Target document type")
        domain_context: str = dspy.InputField(desc="Domain terminology hints")
        
        target_field: str = dspy.OutputField(desc="Best matching field name or 'none'")
        confidence: Literal["high", "medium", "low"] = dspy.OutputField(desc="Confidence level")
        reason: str = dspy.OutputField(desc="Brief explanation")


    class ClassifyColumnSetSignature(dspy.Signature):
        """Map all columns in a CSV to schema fields, avoiding duplicates.
        
        This considers the FULL CONTEXT of all columns to avoid conflicts
        where multiple columns would map to the same target field.
        
        For example, if two columns look like 'employee_id', pick the better one.
        """
        
        columns: str = dspy.InputField(desc="JSON list of columns with their candidates")
        document_type: str = dspy.InputField(desc="Target document type")
        schema_fields: str = dspy.InputField(desc="JSON list of available schema fields")
        domain_context: str = dspy.InputField(desc="Domain terminology hints")
        
        mappings: str = dspy.OutputField(desc="JSON object: {column_name: target_field or null}")
        conflicts_resolved: str = dspy.OutputField(desc="Description of any conflicts that were resolved")


# =============================================================================
# DSPy Service Class
# =============================================================================

@dataclass
class VerifyResult:
    """Result of verify_mapping."""
    is_correct: bool
    reason: str
    needs_transformation: bool = False


@dataclass 
class ClassifyResult:
    """Result of classify_column."""
    target_field: Optional[str]
    confidence: str
    reason: str


@dataclass
class ClassifySetResult:
    """Result of classify_column_set."""
    mappings: Dict[str, Optional[str]]
    conflicts_resolved: str


class DSPyColumnClassifier:
    """
    DSPy-based column classifier.
    
    Can operate in two modes:
    1. Zero-shot: Uses DSPy's ChainOfThought with default prompts
    2. Optimized: Uses a compiled model trained on ground truth
    """
    
    DEFAULT_MODEL = "openai/gpt-4o-mini"
    
    def __init__(
        self,
        model: str = None,
        api_key: Optional[str] = None,
        compiled_path: Optional[Path] = None,
    ):
        """
        Initialize the DSPy classifier.
        
        Args:
            model: LLM model to use (default: openai/gpt-4o-mini)
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
            compiled_path: Path to a compiled/optimized model to load
        """
        if not DSPY_AVAILABLE:
            raise ImportError("dspy-ai not installed. Run: pip install dspy-ai")
        
        self.model_name = model or self.DEFAULT_MODEL
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not provided. Set OPENAI_API_KEY or pass api_key."
            )
        
        # Configure DSPy
        self._configure_dspy()
        
        # Initialize predictors (will be replaced if loading compiled model)
        self._init_predictors()
        
        # Load compiled model if provided
        if compiled_path:
            self.load_optimized(compiled_path)
        
        logger.info(f"DSPy classifier initialized with model: {self.model_name}")
    
    def _configure_dspy(self):
        """Configure DSPy with the LLM."""
        lm = dspy.LM(
            self.model_name,
            api_key=self.api_key,
            temperature=0.1,  # Low temperature for consistency
        )
        dspy.configure(lm=lm)
    
    def _init_predictors(self):
        """Initialize the DSPy predictors (modules)."""
        # Use ChainOfThought for better reasoning
        self.verify_predictor = dspy.ChainOfThought(VerifyMappingSignature)
        self.classify_predictor = dspy.ChainOfThought(ClassifyColumnSignature)
        self.classify_set_predictor = dspy.ChainOfThought(ClassifyColumnSetSignature)
    
    @property
    def is_available(self) -> bool:
        """Check if the service is available."""
        return DSPY_AVAILABLE and self.api_key is not None
    
    def verify_mapping(
        self,
        column_name: str,
        column_type: str,
        sample_values: List[str],
        proposed_field: str,
        field_description: str,
        document_type: str,
        domain_context: Optional[str] = None,
        accepts_units: Optional[List[str]] = None,
        target_unit: Optional[str] = None,
    ) -> VerifyResult:
        """
        Verify if a proposed mapping is semantically correct.
        
        Args:
            column_name: Source column name
            column_type: Detected type
            sample_values: Sample values from the column
            proposed_field: Proposed target field
            field_description: Description of the target field
            document_type: Target document type
            domain_context: Optional domain-specific context
            accepts_units: Optional list of acceptable units (for transformation)
            target_unit: Optional target unit
            
        Returns:
            VerifyResult with is_correct, reason, needs_transformation
        """
        # Build domain context
        context = self._build_domain_context(domain_context, accepts_units, target_unit)
        
        try:
            result = self.verify_predictor(
                column_name=column_name,
                column_type=column_type,
                sample_values=", ".join(str(v) for v in sample_values[:5]),
                proposed_field=proposed_field,
                field_description=field_description or "No description",
                document_type=document_type,
                domain_context=context,
            )
            
            logger.info(
                f"DSPy verify '{column_name}' → '{proposed_field}': "
                f"{'✅' if result.is_correct else '❌'} - {result.reason}"
            )
            
            return VerifyResult(
                is_correct=result.is_correct,
                reason=result.reason,
                needs_transformation=result.needs_transformation,
            )
            
        except Exception as e:
            logger.error(f"DSPy verify error: {e}")
            # On error, assume correct (don't block)
            return VerifyResult(
                is_correct=True,
                reason=f"Verification error: {e}",
                needs_transformation=False,
            )
    
    def classify_column(
        self,
        column_name: str,
        column_type: str,
        sample_values: List[str],
        candidates: List[Dict[str, Any]],
        document_type: str,
        domain_context: Optional[str] = None,
    ) -> ClassifyResult:
        """
        Classify a single column by selecting the best candidate.
        
        Args:
            column_name: Source column name
            column_type: Detected type
            sample_values: Sample values
            candidates: List of candidate fields from embeddings
            document_type: Target document type
            domain_context: Optional domain context
            
        Returns:
            ClassifyResult with target_field, confidence, reason
        """
        context = self._build_domain_context(domain_context)
        
        # Format candidates for the prompt
        candidates_json = json.dumps([
            {
                "field_name": c.get("field_name"),
                "field_type": c.get("field_type"),
                "description": c.get("description", "")[:100],
                "similarity": round(c.get("similarity", 0), 3),
            }
            for c in candidates
        ], ensure_ascii=False)
        
        try:
            result = self.classify_predictor(
                column_name=column_name,
                column_type=column_type,
                sample_values=", ".join(str(v) for v in sample_values[:5]),
                candidates=candidates_json,
                document_type=document_type,
                domain_context=context,
            )
            
            target = result.target_field if result.target_field != "none" else None
            
            logger.info(
                f"DSPy classify '{column_name}' → '{target}' "
                f"({result.confidence}): {result.reason}"
            )
            
            return ClassifyResult(
                target_field=target,
                confidence=result.confidence,
                reason=result.reason,
            )
            
        except Exception as e:
            logger.error(f"DSPy classify error: {e}")
            # Return first candidate on error
            if candidates:
                return ClassifyResult(
                    target_field=candidates[0].get("field_name"),
                    confidence="low",
                    reason=f"Error fallback: {e}",
                )
            return ClassifyResult(target_field=None, confidence="low", reason=str(e))
    
    def classify_column_set(
        self,
        columns: List[Dict[str, Any]],
        document_type: str,
        schema_fields: List[Dict[str, Any]],
        domain_context: Optional[str] = None,
    ) -> ClassifySetResult:
        """
        Classify all columns in a CSV, considering conflicts.
        
        This method considers the full context to avoid multiple columns
        mapping to the same target field.
        
        Args:
            columns: List of columns with their info and candidates
            document_type: Target document type
            schema_fields: Available schema fields
            domain_context: Optional domain context
            
        Returns:
            ClassifySetResult with all mappings and conflict resolution info
        """
        context = self._build_domain_context(domain_context)
        
        # Format columns for the prompt
        columns_json = json.dumps([
            {
                "column_name": c.get("column_name"),
                "column_type": c.get("column_type"),
                "sample_values": c.get("sample_values", [])[:3],
                "candidates": [
                    {
                        "field_name": cand.get("field_name"),
                        "similarity": round(cand.get("similarity", 0), 3),
                    }
                    for cand in c.get("candidates", [])[:5]
                ],
            }
            for c in columns
        ], ensure_ascii=False)
        
        fields_json = json.dumps([
            {
                "name": f.get("name"),
                "type": f.get("type"),
                "required": f.get("required", False),
            }
            for f in schema_fields
        ], ensure_ascii=False)
        
        try:
            result = self.classify_set_predictor(
                columns=columns_json,
                document_type=document_type,
                schema_fields=fields_json,
                domain_context=context,
            )
            
            # Parse the mappings JSON
            mappings = json.loads(result.mappings)
            
            logger.info(
                f"DSPy classify set: {len(mappings)} columns mapped. "
                f"Conflicts: {result.conflicts_resolved}"
            )
            
            return ClassifySetResult(
                mappings=mappings,
                conflicts_resolved=result.conflicts_resolved,
            )
            
        except Exception as e:
            logger.error(f"DSPy classify set error: {e}")
            # Return empty mappings on error
            return ClassifySetResult(
                mappings={c.get("column_name"): None for c in columns},
                conflicts_resolved=f"Error: {e}",
            )
    
    def _build_domain_context(
        self,
        custom_context: Optional[str] = None,
        accepts_units: Optional[List[str]] = None,
        target_unit: Optional[str] = None,
    ) -> str:
        """Build the domain context string for prompts."""
        parts = []
        
        # Add custom context
        if custom_context:
            parts.append(custom_context)
        else:
            # Default medical/shift context
            parts.append("""Domain terminology:
- In shift schedules: 'exit time', 'departure' = shift_end; 'entry time', 'arrival' = shift_start
- In medical: 'performer', 'provider' = the person who did the action
- Column names may be in Hebrew, Spanish, or other languages""")
        
        # Add unit transformation context
        if accepts_units and target_unit:
            other_units = [u for u in accepts_units if u != target_unit]
            if other_units:
                parts.append(
                    f"Unit conversion: Target field expects {target_unit}, "
                    f"but accepts {', '.join(other_units)} (will auto-convert)"
                )
        
        return "\n".join(parts)
    
    def save(self, path: Path):
        """Save the compiled model to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        self.verify_predictor.save(path / "verify_predictor.json")
        self.classify_predictor.save(path / "classify_predictor.json")
        self.classify_set_predictor.save(path / "classify_set_predictor.json")
        
        logger.info(f"DSPy model saved to {path}")
    
    def load_optimized(self, path: Path):
        """Load a compiled/optimized model from disk."""
        path = Path(path)
        
        if (path / "verify_predictor.json").exists():
            self.verify_predictor.load(path / "verify_predictor.json")
            logger.info("Loaded optimized verify_predictor")
        
        if (path / "classify_predictor.json").exists():
            self.classify_predictor.load(path / "classify_predictor.json")
            logger.info("Loaded optimized classify_predictor")
        
        if (path / "classify_set_predictor.json").exists():
            self.classify_set_predictor.load(path / "classify_set_predictor.json")
            logger.info("Loaded optimized classify_set_predictor")
    
    @classmethod
    def load(cls, path: Path, model: str = None, api_key: str = None) -> "DSPyColumnClassifier":
        """
        Load a compiled model from disk.
        
        Args:
            path: Path to the saved model directory
            model: LLM model to use
            api_key: OpenAI API key
            
        Returns:
            DSPyColumnClassifier with loaded optimized predictors
        """
        return cls(model=model, api_key=api_key, compiled_path=path)


# =============================================================================
# Factory function
# =============================================================================

def create_dspy_classifier(
    model: str = None,
    api_key: str = None,
    compiled_path: Optional[Path] = None,
) -> Optional[DSPyColumnClassifier]:
    """
    Factory function to create DSPy classifier.
    
    Returns None if DSPy is not available.
    """
    if not DSPY_AVAILABLE:
        logger.warning("DSPy not available")
        return None
    
    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OPENAI_API_KEY not set")
        return None
    
    try:
        return DSPyColumnClassifier(
            model=model,
            api_key=api_key,
            compiled_path=compiled_path,
        )
    except Exception as e:
        logger.error(f"Failed to create DSPy classifier: {e}")
        return None

