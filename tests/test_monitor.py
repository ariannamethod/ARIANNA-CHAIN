import os
import hashlib
import numpy as np

from arianna_chain import TOOLS
from arianna_core import SelfMonitor


def test_search_exact(tmp_path):
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        with SelfMonitor(db_path=str(tmp_path / "mem.sqlite")) as monitor:
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
        with SelfMonitor(db_path=str(tmp_path / "mem.sqlite")) as monitor:
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
        with SelfMonitor(db_path=str(tmp_path / "mem.sqlite")) as monitor:
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


def test_search_combined(tmp_path):
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        with SelfMonitor(db_path=str(tmp_path / "mem.sqlite")) as monitor:
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
            results = monitor.search("feline", limit=1)
            assert results and results[0][0] == "cat"
    finally:
        os.chdir(cwd)


def test_link_and_graph_search(tmp_path):
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        with SelfMonitor() as monitor:
            monitor.log("p1", "o1")
            note_text = "note1"
            monitor.note(note_text)
            p_sha = hashlib.sha256("p1".encode()).hexdigest()
            n_sha = hashlib.sha256(note_text.encode()).hexdigest()
            monitor.link_prompt(p_sha, n_sha, "refers")
            edges = monitor.graph_search(p_sha, depth=1)
            assert (p_sha, n_sha, "refers") in edges
            monitor.log("p2", "o2")
            note2 = "note2"
            monitor.note(note2)
            p2_sha = hashlib.sha256("p2".encode()).hexdigest()
            n2_sha = hashlib.sha256(note2.encode()).hexdigest()
            TOOLS["memory.link"](prompt_sha=p2_sha, note_sha=n2_sha, relation="mentions")
            edges2 = monitor.graph_search(p2_sha, depth=1)
            assert (p2_sha, n2_sha, "mentions") in edges2
    finally:
        os.chdir(cwd)
