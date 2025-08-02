"""Self-monitoring utilities for Arianna-C.

The monitor snapshot the entire repository and logs prompts and
responses for further fine-tuning. All files are stored inside an
SQLite database so the system can reflect on its own source and data.
"""

from __future__ import annotations

import hashlib
import sqlite3
import time
from pathlib import Path


class SelfMonitor:
    """Record code snapshots and generation events."""

    def __init__(self, db_path: str = "arianna_memory.sqlite"):
        self.conn = sqlite3.connect(db_path)
        self._init_db()
        self.snapshot_codebase()

    def _init_db(self) -> None:
        cur = self.conn.cursor()
        cur.execute(
            "CREATE TABLE IF NOT EXISTS files(path TEXT PRIMARY KEY, content BLOB, sha256 TEXT)"
        )
        cur.execute(
            "CREATE TABLE IF NOT EXISTS logs(ts REAL, prompt TEXT, output TEXT, sha256 TEXT)"
        )
        cur.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS prompts_index USING fts5(prompt, output)"
        )
        self.conn.commit()

    def snapshot_codebase(self, root: str | Path = ".") -> None:
        """Store all files in the repository with their hashes."""
        root_path = Path(root)
        for path in root_path.rglob("*"):
            if not path.is_file():
                continue
            if path.name == "arianna_memory.sqlite":
                continue
            data = path.read_bytes()
            sha = hashlib.sha256(data).hexdigest()
            cur = self.conn.cursor()
            cur.execute(
                "INSERT OR REPLACE INTO files(path, content, sha256) VALUES (?,?,?)",
                (str(path), sqlite3.Binary(data), sha),
            )
        self.conn.commit()

    def log(self, prompt: str, output: str) -> None:
        """Log a generation event with timestamp."""
        sha = hashlib.sha256(prompt.encode()).hexdigest()
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO logs(ts, prompt, output, sha256) VALUES (?,?,?,?)",
            (time.time(), prompt, output, sha),
        )
        cur.execute(
            "INSERT INTO prompts_index(prompt, output) VALUES (?,?)",
            (prompt, output),
        )
        self.conn.commit()

    def _search_tfidf(self, query: str, limit: int = 5) -> list[tuple[str, str]]:
        cur = self.conn.cursor()
        cur.execute(
            "SELECT prompt, output FROM prompts_index WHERE prompts_index MATCH ? "
            "ORDER BY bm25(prompts_index) LIMIT ?",
            (query, limit),
        )
        return cur.fetchall()

    def search(self, prompt: str, limit: int = 5) -> list[tuple[str, str]]:
        """Return top-k similar prompt/output pairs.

        Exact SHA-256 matches are preferred; otherwise a TF-IDF lookup is used.
        """

        sha = hashlib.sha256(prompt.encode()).hexdigest()
        cur = self.conn.cursor()
        cur.execute(
            "SELECT prompt, output FROM logs WHERE sha256 = ? LIMIT ?",
            (sha, limit),
        )
        rows = cur.fetchall()
        if rows:
            return rows
        return self._search_tfidf(prompt, limit=limit)

    def search_prompts(self, query: str, limit: int = 5) -> list[tuple[str, str]]:
        """Search previously logged prompts similar to the query."""
        return self._search_tfidf(query, limit=limit)


__all__ = ["SelfMonitor"]
