from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
import math
import threading
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, TYPE_CHECKING, cast

import numpy as np
import torch
import torch.nn as nn

try:  # pragma: no cover - optional dependency
    import faiss  # type: ignore
except Exception:  # pragma: no cover
    faiss = None

if TYPE_CHECKING:  # pragma: no cover - for type checkers only
    from sentence_transformers import SentenceTransformer
    import faiss as faiss_module

from .config import settings
from .tokenizer import tokenizer


# ---- Reasoning utilities -----------------------------------------------------

TAG_RE = re.compile(r"^\s*<think>.*?</think>\s*<answer>.*?</answer>\s*$", re.DOTALL)


def validate_reasoning_tags(text: str) -> bool:
    return bool(TAG_RE.fullmatch(text.strip()))


def format_reward(text: str) -> float:
    think_blocks = re.findall(r"<think>(.*?)</think>", text, flags=re.DOTALL)
    answer_blocks = re.findall(r"<answer>(.*?)</answer>", text, flags=re.DOTALL)
    if len(think_blocks) != 1 or len(answer_blocks) != 1:
        return 0.0
    think_pos = text.find("<think>")
    answer_pos = text.find("<answer>")
    return 1.0 if 0 <= think_pos < answer_pos else 0.0


def reasoning_steps_reward(text: str) -> float:
    match = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL)
    if not match:
        return 0.0
    think_content = match.group(1)
    lines = [line.strip() for line in think_content.splitlines()]
    count = 0
    for line in lines:
        if re.match(r"^(\d+\.\s+|-\s+|\*\s+)", line):
            count += 1
    return 1.0 if count >= 3 else 0.0


@dataclass
class ThoughtLogEntry:
    timestamp: str
    message: str
    tokens: int
    entropy: float
    perplexity: float | None = None
    valid_tags: bool = True
    confidence: float = 0.0


class ThoughtComplexityLogger:
    def __init__(self, log_file: str | Path = "logs/thought_log.jsonl") -> None:
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.logs: List[ThoughtLogEntry] = []

    def log_turn(
        self,
        message: str,
        tokens: int,
        entropy: float,
        perplexity: float | None = None,
        confidence: float = 0.0,
    ) -> ThoughtLogEntry:
        valid = validate_reasoning_tags(message)
        entry = ThoughtLogEntry(
            timestamp=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            message=message,
            tokens=max(0, int(tokens)),
            entropy=float(entropy),
            perplexity=None if perplexity is None else float(perplexity),
            valid_tags=valid,
            confidence=float(confidence),
        )
        self.logs.append(entry)
        with self.log_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry.__dict__, ensure_ascii=False) + "\n")
        return entry

    def recent(self, n: int = 7) -> List[ThoughtLogEntry]:
        return self.logs[-n:]


def estimate_complexity_and_entropy(
    message: str,
    model: Optional[nn.Module] = None,
    *,
    n: int = 2,
) -> tuple[int, float, float | None]:
    tokens_1d = tokenizer.encode(message)
    token_count = int(tokens_1d.shape[-1])
    n = max(1, min(n, token_count))
    arr = tokens_1d.reshape(-1).tolist()
    counts: Counter[tuple[int, ...]] = Counter(
        tuple(arr[i : i + n]) for i in range(max(0, token_count - n + 1))
    )
    total = sum(counts.values())
    if total:
        probs = [c / total for c in counts.values()]
        entropy = -sum(p * math.log2(p) for p in probs)
        max_entropy = math.log2(len(counts)) if counts else 1.0
        entropy = entropy / max_entropy if max_entropy else 0.0
    else:
        entropy = 0.0
    perplexity = None
    if model is not None and token_count > 1:
        model.eval()
        with torch.no_grad():
            inp = tokens_1d
            out = model(inp[:, :-1], inp[:, 1:])
            loss = out[1] if isinstance(out, tuple) else out
            if loss is not None:
                try:
                    loss_t = loss if isinstance(loss, torch.Tensor) else torch.tensor(float(loss))
                    perplexity = float(torch.exp(loss_t))
                    entropy /= max(perplexity, 1e-8)
                except Exception:
                    pass
    return token_count, float(entropy), perplexity


thought_logger = ThoughtComplexityLogger()


# ---- SelfMonitor -------------------------------------------------------------


class SelfMonitor:
    """SQLite-backed monitor for prompts, notes and links.

    The monitor maintains a single SQLite connection. SQLite connections are
    not inherently thread-safe, so operations are synchronised with a
    :class:`threading.Lock`. For heavy concurrent workloads prefer using one
    connection per process or an external connection pool.

    Parameters
    ----------
    db_path:
        Path to the SQLite database file.
    use_embeddings:
        Whether to compute and store text embeddings.
    check_same_thread:
        Passed to :func:`sqlite3.connect`; defaults to ``True`` which restricts
        the connection to the creating thread.
    """

    _snapshotted = False

    def __init__(
        self,
        db_path: str = "arianna_memory.sqlite",
        use_embeddings: bool | None = None,
        *,
        check_same_thread: bool = True,
    ):
        self.lock = threading.Lock()
        self.conn = sqlite3.connect(db_path, check_same_thread=check_same_thread)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")
        env_flag = settings.arianna_disable_embed
        self.use_embeddings = use_embeddings if use_embeddings is not None else not env_flag
        self.embed_model: Optional["SentenceTransformer"] = None
        self.faiss_index: Optional["faiss_module.Index"] = None
        self.faiss_ids: list[str] = []
        self.faiss_dim = 0
        self.index_dir = Path("logs/faiss_index")
        self.index_file = self.index_dir / "index.faiss"
        self.ids_file = self.index_dir / "ids.json"
        self._load_faiss_index()
        self._init_db()
        snapshot_flag = settings.arianna_snapshot_codebase
        if snapshot_flag and not SelfMonitor._snapshotted:
            self.snapshot_codebase()
            SelfMonitor._snapshotted = True

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "SelfMonitor":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # -- database -------------------------------------------------------------
    def _init_db(self) -> None:
        cur = self.conn.cursor()
        cur.execute("CREATE TABLE IF NOT EXISTS files(path TEXT PRIMARY KEY, content BLOB, sha256 TEXT)")
        cur.execute("CREATE TABLE IF NOT EXISTS logs(ts REAL, prompt TEXT, output TEXT, sha256 TEXT)")
        cur.execute("CREATE VIRTUAL TABLE IF NOT EXISTS prompts_index USING fts5(prompt, output)")
        cur.execute("CREATE TABLE IF NOT EXISTS notes(ts REAL, text TEXT, sha256 TEXT)")
        try:
            cur.execute("ALTER TABLE notes ADD COLUMN sha256 TEXT")
        except sqlite3.OperationalError:
            pass
        cur.execute("CREATE VIRTUAL TABLE IF NOT EXISTS notes_index USING fts5(text)")
        cur.execute("CREATE TABLE IF NOT EXISTS links(src_sha TEXT, dst_sha TEXT, relation TEXT)")
        cur.execute("CREATE TABLE IF NOT EXISTS embeddings(sha256 TEXT PRIMARY KEY, embedding BLOB)")
        self.conn.commit()

    def _load_faiss_index(self) -> None:
        if faiss is None:
            return
        self.index_dir.mkdir(parents=True, exist_ok=True)
        if self.index_file.exists() and self.ids_file.exists():
            try:
                index = faiss.read_index(str(self.index_file))
                self.faiss_index = index
                self.faiss_ids = json.loads(self.ids_file.read_text())
                self.faiss_dim = index.d
            except Exception:
                self.faiss_index = None
                self.faiss_ids = []
                self.faiss_dim = 0

    def _add_to_index(self, sha: str, vec: np.ndarray) -> None:
        if faiss is None:
            return
        dim = len(vec)
        if self.faiss_index is None:
            self.faiss_index = faiss.IndexFlatIP(dim)
            self.faiss_dim = dim
        if dim != self.faiss_dim or self.faiss_index is None:
            return
        index = self.faiss_index
        v = vec.astype("float32")
        n = np.linalg.norm(v) + 1e-9
        index.add((v / n).reshape(1, -1))
        self.faiss_ids.append(sha)
        faiss.write_index(index, str(self.index_file))
        self.ids_file.write_text(json.dumps(self.faiss_ids))

    def snapshot_codebase(self, root: str | Path = ".") -> None:
        root_path = Path(root)
        SKIP_DIRS = {".git", "__pycache__", "venv", "env", "logs", "node_modules", ".pytest_cache"}
        SKIP_SUFFIXES = {
            ".sqlite",
            ".db",
            ".pdf",
            ".bin",
            ".pt",
            ".pth",
            ".zip",
            ".tar",
            ".png",
            ".jpg",
            ".jpeg",
            ".env",
            ".toml",
            ".yaml",
            ".yml",
        }
        model = self._ensure_embed_model()
        for path in root_path.rglob("*"):
            if not path.is_file():
                continue
            if any(part in SKIP_DIRS for part in path.parts):
                continue
            if path.suffix.lower() in SKIP_SUFFIXES:
                continue
            try:
                data = path.read_bytes()
            except Exception:
                continue
            if len(data) > 2_000_000:
                continue
            sha = hashlib.sha256(data).hexdigest()
            cur = self.conn.cursor()
            cur.execute(
                "INSERT OR REPLACE INTO files(path, content, sha256) VALUES (?,?,?)",
                (str(path), sqlite3.Binary(data), sha),
            )
            if model:
                text = data.decode("utf-8", errors="ignore")
                vec = model.encode([text])[0].astype("float32")
                cur.execute(
                    "INSERT OR REPLACE INTO embeddings(sha256, embedding) VALUES (?,?)",
                    (sha, sqlite3.Binary(vec.tobytes())),
                )
                self._add_to_index(sha, vec)
        self.conn.commit()

    def _ensure_embed_model(self) -> Optional["SentenceTransformer"]:
        if not self.use_embeddings:
            return None
        if self.embed_model is None:
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore

                self.embed_model = SentenceTransformer(settings.arianna_embed_model)
            except Exception:  # pragma: no cover
                self.embed_model = None
                self.use_embeddings = False
        return self.embed_model

    # -- logging --------------------------------------------------------------
    def log(self, prompt: str, output: str) -> None:
        with self.lock:
            sha = hashlib.sha256(prompt.encode("utf-8", errors="ignore")).hexdigest()
            cur = self.conn.cursor()
            cur.execute(
                "INSERT INTO logs(ts, prompt, output, sha256) VALUES (?,?,?,?)",
                (time.time(), prompt, output, sha),
            )
            cur.execute(
                "INSERT INTO prompts_index(prompt, output) VALUES (?,?)",
                (prompt, output),
            )
            if self._ensure_embed_model():
                assert self.embed_model is not None
                vec = self.embed_model.encode([prompt])[0].astype("float32")
                cur.execute(
                    "INSERT OR REPLACE INTO embeddings(sha256, embedding) VALUES (?,?)",
                    (sha, sqlite3.Binary(vec.tobytes())),
                )
                self._add_to_index(sha, vec)
            self.conn.commit()

    def note(self, text: str) -> None:
        with self.lock:
            sha = hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()
            cur = self.conn.cursor()
            cur.execute(
                "INSERT INTO notes(ts, text, sha256) VALUES (?, ?, ?)",
                (time.time(), text, sha),
            )
            cur.execute("INSERT INTO notes_index(text) VALUES (?)", (text,))
            self.conn.commit()

    def link_prompt(self, prompt_sha: str, note_sha: str, relation: str) -> None:
        with self.lock:
            cur = self.conn.cursor()
            cur.execute(
                "INSERT INTO links(src_sha, dst_sha, relation) VALUES (?,?,?)",
                (prompt_sha, note_sha, relation),
            )
            self.conn.commit()

    def graph_search(self, start_sha: str, depth: int) -> list[tuple[str, str, str]]:
        cur = self.conn.cursor()
        visited = {start_sha}
        frontier = {start_sha}
        edges: set[tuple[str, str, str]] = set()
        for _ in range(depth):
            next_frontier = set()
            for sha in frontier:
                cur.execute(
                    "SELECT src_sha, dst_sha, relation FROM links WHERE src_sha = ? OR dst_sha = ?",
                    (sha, sha),
                )
                for src, dst, rel in cur.fetchall():
                    edges.add((src, dst, rel))
                    neighbor = dst if src == sha else src
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_frontier.add(neighbor)
            frontier = next_frontier
            if not frontier:
                break
        return list(edges)

    # -- search ---------------------------------------------------------------
    def _search_tfidf(self, query: str, limit: int = 5, return_scores: bool = False):
        cur = self.conn.cursor()
        cur.execute(
            (
                "SELECT prompt, output, bm25(prompts_index) as score "
                "FROM prompts_index WHERE prompts_index MATCH ? ORDER BY score LIMIT ?"
            ),
            (query, limit),
        )
        rows = cur.fetchall()
        if return_scores:
            return [(p, o, 1 / (1 + s)) for p, o, s in rows]
        return [(p, o) for p, o, _ in rows]

    def _search_notes(self, query: str, limit: int = 5) -> list[str]:
        cur = self.conn.cursor()
        cur.execute(
            "SELECT text FROM notes_index WHERE notes_index MATCH ? ORDER BY bm25(notes_index) LIMIT ?",
            (query, limit),
        )
        return [r[0] for r in cur.fetchall()]

    def search(self, prompt: str, limit: int = 5) -> list[tuple[str, str]]:
        sha = hashlib.sha256(prompt.encode("utf-8", errors="ignore")).hexdigest()
        cur = self.conn.cursor()
        cur.execute("SELECT prompt, output FROM logs WHERE sha256 = ? LIMIT ?", (sha, limit))
        rows = cur.fetchall()
        if rows:
            return rows
        scored: dict[tuple[str, str], float] = {}
        for p, o, s in self._search_tfidf(prompt, limit=limit * 2, return_scores=True):
            scored[(p, o)] = max(scored.get((p, o), 0.0), s)
        for p, o, s in self.search_embedding(prompt, limit=limit * 2, return_scores=True):
            scored[(p, o)] = max(scored.get((p, o), 0.0), s)
        ordered = sorted(scored.items(), key=lambda kv: kv[1], reverse=True)
        return [pair for pair, _ in ordered[:limit]]

    def search_faiss(self, query: str, limit: int = 5, return_scores: bool = False):
        if not self._ensure_embed_model() or self.faiss_index is None:
            return []
        assert self.embed_model is not None
        index = self.faiss_index
        qv = self.embed_model.encode([query])[0].astype("float32")
        if len(qv) != self.faiss_dim or index is None:
            return []
        n = np.linalg.norm(qv) + 1e-9
        D, indices = index.search((qv / n).reshape(1, -1), limit)
        cur = self.conn.cursor()
        out = []
        for score, idx in zip(D[0], indices[0]):
            if idx == -1 or idx >= len(self.faiss_ids):
                continue
            sha = self.faiss_ids[int(idx)]
            cur.execute("SELECT prompt, output FROM logs WHERE sha256 = ? ORDER BY ts DESC LIMIT 1", (sha,))
            row = cur.fetchone()
            if row:
                if return_scores:
                    out.append((row[0], row[1], float((score + 1) / 2)))
                else:
                    out.append(row)
        return out

    def search_embedding(self, query: str, limit: int = 5, return_scores: bool = False):
        return self.search_faiss(query, limit=limit, return_scores=return_scores)

    def search_prompts_and_notes(self, query: str, limit: int = 5) -> list[str]:
        prs = self.search_faiss(query, limit=limit)
        nts = self._search_notes(query, limit=limit)
        out = []
        for p, o in prs:
            p1 = p.strip().splitlines()[0][:160]
            o1 = o.strip().splitlines()[0][:200]
            out.append(f"Q:{p1} | A:{o1}")
        out.extend(nts)
        return out[:limit]


__all__ = [
    "validate_reasoning_tags",
    "format_reward",
    "reasoning_steps_reward",
    "ThoughtLogEntry",
    "ThoughtComplexityLogger",
    "estimate_complexity_and_entropy",
    "thought_logger",
    "SelfMonitor",
]
