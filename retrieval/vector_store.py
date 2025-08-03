from __future__ import annotations

"""Simple vector store with optional FAISS backend."""
from typing import List
import numpy as np

try:  # pragma: no cover - faiss may be unavailable
    import faiss  # type: ignore
except Exception:  # pragma: no cover
    faiss = None


class VectorStore:
    """Store documents as dense vectors and perform similarity search."""

    def __init__(self, documents: List[str] | None = None, dim: int = 128) -> None:
        self.dim = dim
        self.documents: List[str] = []
        if faiss:
            self.index = faiss.IndexFlatIP(dim)
        else:  # pragma: no cover - fallback path
            self.index = None
            self.vectors: List[np.ndarray] = []
        if documents:
            self.add(documents)

    def _embed(self, text: str) -> np.ndarray:
        vec = np.frombuffer(text.encode("utf-8"), dtype=np.uint8).astype("float32")
        if vec.size < self.dim:
            vec = np.pad(vec, (0, self.dim - vec.size))
        else:
            vec = vec[: self.dim]
        norm = np.linalg.norm(vec) or 1.0
        return vec / norm

    def add(self, docs: List[str]) -> None:
        embeddings = np.vstack([self._embed(d) for d in docs]) if docs else np.empty((0, self.dim), dtype="float32")
        if self.index is not None and embeddings.size:
            self.index.add(embeddings)
        else:  # pragma: no cover - fallback path
            for emb in embeddings:
                self.vectors.append(emb)
        self.documents.extend(docs)

    def search(self, query: str, k: int = 3) -> List[str]:
        if not self.documents:
            return []
        qvec = self._embed(query).reshape(1, -1)
        k = min(k, len(self.documents))
        if self.index is not None:
            _, idxs = self.index.search(qvec, k)
            ids = idxs[0]
        else:  # pragma: no cover - fallback path
            sims = [float(np.dot(qvec.squeeze(), v)) for v in self.vectors]
            ids = np.argsort(sims)[::-1][:k]
        return [self.documents[i] for i in ids]
