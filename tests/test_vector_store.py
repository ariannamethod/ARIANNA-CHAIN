import numpy as np
import pytest
import arianna_chain
from arianna_chain import VectorStore


def _embed(text: str, dim: int) -> np.ndarray:
    vec = np.frombuffer(text.encode("utf-8"), dtype="uint8").astype("float32")
    if vec.size < dim:
        vec = np.pad(vec, (0, dim - vec.size))
    else:
        vec = vec[: dim]
    norm = np.linalg.norm(vec) or 1.0
    return vec / norm


def _expected_order(docs: list[str], query: str, dim: int) -> list[str]:
    qvec = _embed(query, dim)
    sims = [float(np.dot(qvec, _embed(d, dim))) for d in docs]
    ids = np.argsort(sims)[::-1]
    return [docs[i] for i in ids]


def test_search_with_faiss() -> None:
    if arianna_chain.faiss is None:
        pytest.skip("faiss not installed")
    docs = ["foo", "bar", "baz"]
    query = "foo"
    dim = 16
    store = VectorStore(docs, dim=dim)
    expected = _expected_order(docs, query, dim)
    results = store.search(query, k=len(docs))
    assert results == expected


def test_search_without_faiss(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(arianna_chain, "faiss", None)
    docs = ["foo", "bar", "baz"]
    query = "foo"
    dim = 16
    store = VectorStore(docs, dim=dim)
    expected = _expected_order(docs, query, dim)
    results = store.search(query, k=len(docs))
    assert results == expected
