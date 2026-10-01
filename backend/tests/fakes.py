"""Deterministic, dependency-free encoders for tests.

Bag-of-words hashing gives embeddings where texts sharing words are similar, which is enough
to test filtering, fan-out and ranking behaviour without downloading ONNX models."""

import math
import re
import zlib
from collections import Counter
from collections.abc import Sequence

from qdrant_client import models

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = {"the", "a", "an", "of", "and", "or", "to", "in", "on", "for", "with", "is", "are", "we"}


def _tokens(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


def _bucket(token: str, n: int) -> int:
    return zlib.crc32(token.encode()) % n


class HashingDense:
    def __init__(self, dim: int = 256):
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for tok in _tokens(text):
            vec[_bucket(tok, self.dim)] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class HashingSparse:
    def _embed(self, text: str) -> models.SparseVector:
        counts = Counter(_bucket(t, 2**20) for t in _tokens(text))
        indices = sorted(counts)
        return models.SparseVector(indices=indices, values=[float(counts[i]) for i in indices])

    def embed_documents(self, texts: Sequence[str]) -> list[models.SparseVector]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> models.SparseVector:
        return self._embed(text)


class KeywordReranker:
    """Scores a chunk by how many query words it contains (stand-in for a cross-encoder)."""

    def __init__(self):
        self.calls: list[int] = []  # pool size per call

    def rerank(self, query, hits):
        from dataclasses import replace

        self.calls.append(len(hits))
        words = set(_tokens(query))
        rescored = [
            replace(h, score=float(len(words & set(_tokens(h.payload["text"]))))) for h in hits
        ]
        return sorted(rescored, key=lambda h: h.score, reverse=True)
