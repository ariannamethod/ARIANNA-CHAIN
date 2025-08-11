import sys
import types

import numpy as np
import pytest

import arianna_chain
from arianna_chain import VectorStore


def _patch_sentence_transformer(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide a lightweight stand‑in for SentenceTransformer."""

    class DummySentenceTransformer:
        def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
            self._dim = 2

        def encode(self, texts: list[str]) -> np.ndarray:  # type: ignore[no-untyped-def]
            out: list[np.ndarray] = []
            for t in texts:
                tl = t.lower()
                if "cat" in tl or "feline" in tl:
                    out.append(np.array([1.0, 0.0], dtype=np.float32))
                elif "dog" in tl or "canine" in tl:
                    out.append(np.array([0.0, 1.0], dtype=np.float32))
                else:
                    out.append(np.zeros(2, dtype=np.float32))
            return np.stack(out)

        def get_sentence_embedding_dimension(self) -> int:  # type: ignore[no-untyped-def]
            return self._dim

    module = types.SimpleNamespace(SentenceTransformer=DummySentenceTransformer)
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)


def test_search_with_faiss(monkeypatch: pytest.MonkeyPatch) -> None:
    if arianna_chain.faiss is None:
        pytest.skip("faiss not installed")
    _patch_sentence_transformer(monkeypatch)
    docs = ["a small cat", "a friendly dog"]
    store = VectorStore(docs)
    results = store.search("feline", k=len(docs))
    assert results[0] == docs[0]
    assert store.dim == 2


def test_search_without_faiss(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(arianna_chain, "faiss", None)
    _patch_sentence_transformer(monkeypatch)
    docs = ["a small cat", "a friendly dog"]
    store = VectorStore(docs)
    results = store.search("canine", k=len(docs))
    assert results[0] == docs[1]
    assert store.dim == 2
