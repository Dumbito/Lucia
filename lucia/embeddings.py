"""Small local text-vector primitives for semantic memory retrieval.

This module intentionally has no model dependency yet.  It provides a stable
interface for TF-IDF-style lexical vectors so a real local embedding model can
be plugged in later without changing the memory contract.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter

_TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)


def normalize_text(text: str) -> str:
    """Normalize case and accents for language-agnostic local matching."""
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
    """Build a sparse bag-of-words vector from normalized text."""
    return Counter(tokenize(text))
