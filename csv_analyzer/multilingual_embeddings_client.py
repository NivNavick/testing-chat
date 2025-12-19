"""
Multilingual embeddings client using sentence-transformers.
Supports Hebrew, English, and 100+ other languages with good cross-language performance.
"""

import os
import logging
import time
import warnings
from typing import List, Optional, Tuple
import numpy as np

# Disable tokenizers parallelism warning (must be set before importing transformers)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.getLogger(__name__)

# Suppress the "UNEXPECTED" key warning from transformers model loading
warnings.filterwarnings("ignore", message=".*position_ids.*")


class MultilingualEmbeddingsClient:
    """
    Multilingual embeddings client using intfloat/multilingual-e5-large.
    
    This model provides much better cross-language semantic similarity than OpenAI models:
    - Hebrew ↔ English: ~0.50-0.60 similarity (vs 0.18 with OpenAI)
    - Dimensions: 1024 (vs 1536 for OpenAI)
    - Cost: FREE (self-hosted)
    """
    
    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-large",
        device: str = "cpu",  # Change to "cuda" if GPU available
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """
        Initialize the multilingual embeddings client.
        
        Args:
            model_name: Hugging Face model name
            device: Device to run model on ("cpu" or "cuda")
            max_retries: Maximum number of retry attempts
            retry_delay: Base delay between retries in seconds
        """
        self.model_name = model_name
        self.device = device
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._model = None  # Internal SentenceTransformer instance
        self._dimension = 1024  # multilingual-e5-large dimension
        
        try:
            from sentence_transformers import SentenceTransformer
            import transformers
            
            # Suppress the "UNEXPECTED" key warning during model loading
            transformers.logging.set_verbosity_error()
            
            logger.info(f"Loading multilingual embedding model: {model_name}")
            self._model = SentenceTransformer(model_name, device=device)
            
            # Restore normal logging level
            transformers.logging.set_verbosity_warning()
            
            logger.info(f"✅ Multilingual model loaded successfully on {device}")
        except ImportError:
            logger.error(
                "sentence-transformers not installed! "
                "Run: poetry install"
            )
            self._model = None
        except Exception as e:
            logger.error(f"Failed to load multilingual model: {e}")
            self._model = None
    
    @property
    def is_available(self) -> bool:
        """Check if the embeddings client is available."""
        return self._model is not None
    
    @property
    def model(self) -> str:
        """Get the model name (for API compatibility with OpenAI client)."""
        return self.model_name
    
    def get_embedding_dimension(self) -> int:
        """Get the embedding dimension for the current model."""
        return self._dimension
    
    def create_embedding(self, text: str) -> Optional[List[float]]:
        """
        Create a single embedding for the given text.
        
        For E5 models, we need to add a prefix for queries vs documents:
        - Query: "query: {text}"
        - Document: "passage: {text}"
        
        This method is SYNCHRONOUS and will block. Use create_embedding_async() 
        in async contexts to avoid blocking the event loop.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector or None if failed
        """
        if not self.is_available:
            logger.error("Multilingual embeddings client not available")
            return None
        
        if not text or not text.strip():
            logger.warning("Empty text provided for embedding")
            return None
        
        try:
            # E5 models require prefix for optimal performance
            # For queries, use "query: " prefix
            # For documents/articles, use "passage: " prefix
            # We'll detect based on length (queries are typically shorter)
            if len(text) < 200:
                prefixed_text = f"query: {text}"
            else:
                prefixed_text = f"passage: {text}"
            
            # Generate embedding
            embedding = self._model.encode(
                prefixed_text,
                convert_to_numpy=True,
                normalize_embeddings=True  # Normalize for cosine similarity
            )
            
            # Convert to list
            embedding_list = embedding.tolist()
            
            logger.debug(f"Created embedding with {len(embedding_list)} dimensions")
            return embedding_list
            
        except Exception as e:
            logger.error(f"Error creating embedding: {e}")
            return None
    
    async def create_embedding_async(self, text: str) -> Optional[List[float]]:
        """
        Async wrapper for create_embedding.
        
        NOTE: Currently this just calls the sync version directly because
        asyncio.to_thread() conflicts with asyncpg's greenlets (xd2s error).
        
        Sentence-transformers encoding is CPU-bound (not I/O), so it's safe
        to call synchronously. For production with many concurrent requests,
        consider using a separate worker process.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector or None if failed
        """
        # Just call sync version directly to avoid greenlet conflicts
        return self.create_embedding(text)
    
    def create_embeddings_batch(
        self, 
        texts: List[str],
        batch_size: int = 32,
        is_query: bool = False
    ) -> List[Tuple[str, Optional[List[float]]]]:
        """
        Create embeddings for multiple texts in batches.
        
        Args:
            texts: List of texts to embed
            batch_size: Number of texts to process in each batch
            is_query: If True, use "query: " prefix; otherwise use "passage: "
            
        Returns:
            List of (text, embedding) tuples
        """
        if not self.is_available:
            logger.error("Multilingual embeddings client not available")
            return [(text, None) for text in texts]
        
        results = []
        prefix = "query: " if is_query else "passage: "
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            logger.info(f"Processing embedding batch {i//batch_size + 1} ({len(batch)} items)")
            
            try:
                # Filter out empty texts
                valid_batch = [(idx, text) for idx, text in enumerate(batch) if text and text.strip()]
                
                if not valid_batch:
                    # All texts in batch are empty
                    results.extend([(text, None) for text in batch])
                    continue
                
                # Add prefix and prepare texts
                prefixed_texts = [f"{prefix}{text}" for _, text in valid_batch]
                
                # Generate embeddings
                embeddings = self._model.encode(
                    prefixed_texts,
                    batch_size=batch_size,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False
                )
                
                # Map results back to original order
                batch_results = [None] * len(batch)
                for (original_idx, _), embedding in zip(valid_batch, embeddings):
                    batch_results[original_idx] = embedding.tolist()
                
                # Add to results
                for text, embedding in zip(batch, batch_results):
                    results.append((text, embedding))
                    
            except Exception as e:
                logger.error(f"Batch embedding failed: {e}")
                # Add failed batch with None embeddings
                results.extend([(text, None) for text in batch])
        
        logger.info(f"Created embeddings for {len(results)} texts")
        return results


# Global multilingual embeddings client instance
_multilingual_client: Optional[MultilingualEmbeddingsClient] = None


def get_multilingual_embeddings_client(
    model_name: Optional[str] = None,
    device: str = "cpu",
    force_recreate: bool = False
) -> MultilingualEmbeddingsClient:
    """
    Get the global multilingual embeddings client instance.
    
    Args:
        model_name: Model name to use (default: intfloat/multilingual-e5-large)
        device: Device to run on ("cpu" or "cuda")
        force_recreate: Force recreation of the client
        
    Returns:
        MultilingualEmbeddingsClient instance
    """
    global _multilingual_client
    
    if _multilingual_client is None or force_recreate:
        _multilingual_client = MultilingualEmbeddingsClient(
            model_name=model_name or "intfloat/multilingual-e5-large",
            device=device
        )
    
    return _multilingual_client

