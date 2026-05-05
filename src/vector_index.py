"""Vector-search abstraction for face embeddings.

The default NumPy implementation is fast enough for small and medium laptop
deployments. The FAISS hook can be enabled later without changing recognition
code paths.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SearchResult:
    index: int
    distance: float


class NumpyFaceIndex:
    """Brute-force L2 index over normalized Dlib face embeddings."""

    def __init__(self, embeddings: np.ndarray) -> None:
        self.embeddings = np.asarray(embeddings, dtype=np.float32)
        if self.embeddings.ndim == 1 and self.embeddings.size:
            self.embeddings = self.embeddings.reshape(1, -1)

    @property
    def is_empty(self) -> bool:
        return self.embeddings.size == 0

    def search(self, query_embedding: np.ndarray) -> SearchResult | None:
        if self.is_empty:
            return None

        query = np.asarray(query_embedding, dtype=np.float32)
        distances = np.linalg.norm(self.embeddings - query, axis=1)
        best_index = int(np.argmin(distances))
        return SearchResult(index=best_index, distance=float(distances[best_index]))


class FaissFaceIndex:
    """Optional FAISS L2 index for large deployments."""

    def __init__(self, embeddings: np.ndarray) -> None:
        try:
            import faiss  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "FAISS is not installed. Install faiss-cpu to use FaissFaceIndex."
            ) from exc

        self._faiss = faiss
        self.embeddings = np.asarray(embeddings, dtype=np.float32)
        dimension = 128 if self.embeddings.size == 0 else self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        if self.embeddings.size:
            self.index.add(self.embeddings)

    @property
    def is_empty(self) -> bool:
        return self.index.ntotal == 0

    def search(self, query_embedding: np.ndarray) -> SearchResult | None:
        if self.is_empty:
            return None

        query = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
        squared_distances, indices = self.index.search(query, 1)
        distance = float(np.sqrt(squared_distances[0][0]))
        return SearchResult(index=int(indices[0][0]), distance=distance)

