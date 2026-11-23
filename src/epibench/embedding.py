"""Standalone local embedding service using FastEmbed.

Provides dense embeddings using BAAI/bge-small-en-v1.5 (384 dimensions)
with cached singleton model loading and cosine similarity computation.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence
import numpy as np


class EmbeddingService:
    """Cached FastEmbed wrapper for local dense embeddings."""

    DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
    _instances: Dict[str, "EmbeddingService"] = {}

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self._model: Optional[object] = None

    @classmethod
    def get_instance(cls, model_name: str = DEFAULT_MODEL) -> "EmbeddingService":
        if model_name not in cls._instances:
            cls._instances[model_name] = cls(model_name=model_name)
        return cls._instances[model_name]

    def _ensure_model(self):
        if self._model is None:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(model_name=self.model_name)
        return self._model

    def embed_batch(self, texts: Sequence[str]) -> List[List[float]]:
        """Embed a batch of texts into dense float vectors."""
        if not texts:
            return []
        model = self._ensure_model()
        vectors = list(model.embed(list(texts)))
        return [np.asarray(v, dtype=np.float32).tolist() for v in vectors]

    def embed_text(self, text: str) -> List[float]:
        """Embed a single text string into a dense float vector."""
        vecs = self.embed_batch([text])
        return vecs[0] if vecs else []

    @staticmethod
    def cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
        """Compute cosine similarity between two embedding vectors."""
        a = np.asarray(vec_a, dtype=np.float64)
        b = np.asarray(vec_b, dtype=np.float64)
        norm_a = float(np.linalg.norm(a))
        norm_b = float(np.linalg.norm(b))
        if norm_a <= 1e-12 or norm_b <= 1e-12:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
