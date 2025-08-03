import os

import numpy as np

from arianna_chain import SelfMonitor


def test_search_exact(tmp_path):
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        monitor = SelfMonitor(db_path=str(tmp_path / "mem.sqlite"))
        monitor.log("hello world", "out1")
        monitor.log("another message", "out2")
        results = monitor.search("hello world")
        assert ("hello world", "out1") in results
    finally:
        os.chdir(cwd)


def test_search_tfidf_limit(tmp_path):
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        monitor = SelfMonitor(db_path=str(tmp_path / "mem.sqlite"))
        for i in range(3):
            monitor.log(f"hello {i}", f"out{i}")
        results = monitor.search("hello", limit=2)
        assert len(results) == 2
        assert all("hello" in p for p, _ in results)
    finally:
        os.chdir(cwd)


def test_embedding_search(tmp_path):
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        monitor = SelfMonitor(db_path=str(tmp_path / "mem.sqlite"))

        class DummyEmbed:
            def encode(self, texts):
                mapping = {
                    "cat": np.array([1.0, 0.0], dtype=np.float32),
                    "dog": np.array([0.0, 1.0], dtype=np.float32),
                    "feline": np.array([1.0, 0.0], dtype=np.float32),
                }
                return np.stack([mapping.get(t, np.zeros(2, dtype=np.float32)) for t in texts])

        monitor.embed_model = DummyEmbed()
        monitor.log("cat", "meow")
        monitor.log("dog", "bark")
        results = monitor.search_embedding("feline", limit=1)
        assert results and results[0][0] == "cat"
    finally:
        os.chdir(cwd)
