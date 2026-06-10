"""
Query embedding encoder using sentence-transformers.

Wraps the all-MiniLM-L6-v2 model for generating 384-dimensional
L2-normalized embeddings from text queries.
"""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


class QueryEncoder:
    """
    Encodes text queries into dense vector embeddings.

    Uses sentence-transformers with all-MiniLM-L6-v2 (384-dim) by default.
    Embeddings are L2-normalized so that cosine similarity equals dot product
    and can be related to L2 distance: cosine_sim = 1 - L2² / 2.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: Optional[str] = None,
    ):
        """
        Args:
            model_name: Name of the sentence-transformers model.
            device: 'cpu', 'cuda', or None (auto-detect).
        """
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

        logger.info(
            f"Initialized QueryEncoder: model={model_name}, "
            f"dim={self.embedding_dim}, device={self.model.device}"
        )

    def encode(
        self,
        queries: list[str],
        batch_size: int = 64,
        normalize: bool = True,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Encode a list of text queries into embeddings.

        Args:
            queries: List of query strings.
            batch_size: Batch size for encoding (larger = faster on GPU).
            normalize: If True, L2-normalize embeddings (default: True).
            show_progress: Show tqdm progress bar.

        Returns:
            numpy array of shape (len(queries), embedding_dim), dtype float32.
        """
        logger.info(
            f"Encoding {len(queries)} queries (batch_size={batch_size}, "
            f"normalize={normalize})..."
        )

        embeddings = self.model.encode(
            queries,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=normalize,
        )

        embeddings = np.array(embeddings, dtype=np.float32)

        # Re-normalize after the float32 cast. sentence-transformers normalizes
        # internally, but casting (and mixed-precision inference on GPU) leaves
        # residual norm drift that grows with embedding_dim. Explicitly dividing
        # by the float32 norm pulls every vector back to unit length.
        if normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            np.divide(embeddings, norms, out=embeddings, where=norms > 0)

        # Verify shape
        assert embeddings.shape == (len(queries), self.embedding_dim), (
            f"Expected shape ({len(queries)}, {self.embedding_dim}), "
            f"got {embeddings.shape}"
        )

        # Check for NaN (sentence-transformers can produce NaN for malformed inputs)
        nan_count = np.isnan(embeddings).any(axis=1).sum()
        if nan_count > 0:
            logger.warning(f"Found {nan_count} embeddings with NaN values!")

        # Verify normalization. Use a float32-appropriate tolerance: norm error
        # accumulates with embedding_dim, so 1e-5 is too strict for 768-dim models.
        if normalize:
            norms = np.linalg.norm(embeddings, axis=1)
            if not np.allclose(norms, 1.0, atol=1e-4):
                max_dev = np.abs(norms - 1.0).max()
                logger.warning(
                    f"Some embeddings are not properly L2-normalized "
                    f"(max deviation {max_dev:.2e})"
                )

        logger.info(f"Encoding complete: {embeddings.shape}, dtype={embeddings.dtype}")
        return embeddings

    def encode_single(self, query: str, normalize: bool = True) -> np.ndarray:
        """Encode a single query. Returns shape (1, embedding_dim)."""
        return self.encode([query], batch_size=1, normalize=normalize, show_progress=False)
