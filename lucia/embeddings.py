"""Embedding backends for Lucía's local semantic retrieval.

The default protocol is dependency-free. ``SentenceTransformerEmbedding`` is
an optional real local embedding backend; its model is loaded only when that
backend is instantiated.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from typing import Protocol

_TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)


def normalize_text(text: str) -> str:
    """Normalize case and accents for deterministic local matching."""
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(char for char in text if not unicodedata.combining(char))


def tokenize(text: str) -> list[str]:
    """Return normalized word tokens, excluding one-character tokens."""
    return [token for token in _TOKEN_RE.findall(normalize_text(text)) if len(token) > 1]


def cosine_similarity(left: Counter[str], right: Counter[str]) -> float:
    """Compute cosine similarity between sparse token vectors."""
    if not left or not right:
        return 0.0
    dot = sum(value * right.get(key, 0.0) for key, value in left.items())
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def text_vector(text: str) -> Counter[str]:
    """Build a sparse bag-of-words vector for the dependency-free fallback."""
    return Counter(tokenize(text))


class EmbeddingProvider(Protocol):
    """Contract for interchangeable local embedding models."""

    def embed(self, text: str) -> tuple[float, ...]: ...


class SentenceTransformerEmbedding:
    """Real local embeddings through sentence-transformers.

    The model runs locally after its first download. No text is sent to a
    remote API. A multilingual MiniLM model is the default because Lucía is
    intended to operate across Spanish and English text on modest hardware.
    """

    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is required for real semantic embeddings; "
                "install the semantic extra with: pip install -e '.[semantic]'"
            ) from exc
        self._model = SentenceTransformer(model_name)

    def embed(self, text: str) -> tuple[float, ...]:
        """Encode one text into a normalized dense vector."""
        vector = self._model.encode(text, normalize_embeddings=True)
        return tuple(float(value) for value in vector)


def dense_cosine_similarity(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    """Compute cosine similarity for dense embedding vectors."""
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)
