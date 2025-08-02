from __future__ import annotations

import argparse
import hashlib
import json
import math
import sqlite3
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

# --- Tokenizer utilities ---
_tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
_tokenizer.pre_tokenizer = ByteLevel()
trainer = BpeTrainer(special_tokens=["[UNK]"])
CORE_PROMPT = (Path(__file__).resolve().parent / "core_prompt.txt").read_text(encoding="utf-8")
print("core_prompt.txt loaded [OK]")
_tokenizer.train_from_iterator([CORE_PROMPT], trainer)


class TokenizerWrapper:
    """Light wrapper around ``tokenizers.Tokenizer`` providing torch helpers."""

    def __init__(self, tk: Tokenizer):
        self._tk = tk

    @property
    def vocab_size(self) -> int:
        return self._tk.get_vocab_size()

    def encode(self, text: str) -> torch.Tensor:
        """Encode ``text`` into a tensor of token ids."""
        ids = self._tk.encode(text).ids
        return torch.tensor([ids], dtype=torch.long)

    def decode(self, tokens: torch.Tensor) -> str:
        """Decode token ids back into a string."""
        ids = tokens.squeeze().tolist()
        return self._tk.decode(ids)


# Public tokenizer instance

tokenizer = TokenizerWrapper(_tokenizer)


# --- Thought complexity logger ---
@dataclass
class ThoughtLogEntry:
    timestamp: str
    message: str
    complexity: int
    entropy: float


class ThoughtComplexityLogger:
    """Track complexity and entropy of generated thoughts."""

    def __init__(self, log_file: str | Path = "logs/thought_log.jsonl") -> None:
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.logs: List[ThoughtLogEntry] = []

    def log_turn(self, message: str, complexity_scale: int, entropy: float) -> ThoughtLogEntry:
        entry = ThoughtLogEntry(
            timestamp=datetime.utcnow().isoformat() + "Z",
            message=message,
            complexity=max(1, min(5, complexity_scale)),
            entropy=float(min(1.0, entropy)),
        )
        self.logs.append(entry)
        with self.log_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry.__dict__) + "\n")
        return entry

    def recent(self, n: int = 7) -> List[ThoughtLogEntry]:
        return self.logs[-n:]


def estimate_complexity_and_entropy(message: str) -> tuple[int, float]:
    complexity = 1
    lowered = message.lower()
    if any(keyword in lowered for keyword in ["why", "paradox", "recursive"]):
        complexity += 2
    if len(message) > 300:
        complexity += 1
    complexity = max(1, min(5, complexity))
    unique_words = len(set(message.split()))
    entropy = min(1.0, unique_words / 40)
    return complexity, entropy


thought_logger = ThoughtComplexityLogger()


# --- 2-bit quantization ---
@torch.no_grad()
def quantize_2bit(model: nn.Module) -> None:
    """Quantize the model weights to 2-bit precision in-place."""
    for param in model.parameters():
        if param.dtype not in (torch.float32, torch.float64):
            continue
        max_val = param.abs().max()
        if max_val == 0:
            continue
        scale = max_val / 3
        q = (param / scale).round().clamp(-3, 3)
        signs = torch.sign(q)
        mags = torch.where(
            q.abs() > 2,
            torch.tensor(3.0, device=param.device),
            torch.tensor(1.0, device=param.device),
        )
        param.copy_(signs * mags * scale)


# --- Self-monitoring database ---
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


# --- Model definition ---
@dataclass
class AriannaCConfig:
    """Configuration for the Arianna-C transformer."""

    block_size: int = 1024
    vocab_size: int | None = None
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.vocab_size is None:
            self.vocab_size = tokenizer.vocab_size


class CausalSelfAttention(nn.Module):
    def __init__(self, config: AriannaCConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.key = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.query = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.value = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        k = self.key(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = torch.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, config: AriannaCConfig):
        super().__init__()
        self.fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.gelu(self.fc(x))
        x = self.proj(x)
        return self.dropout(x)


class Block(nn.Module):
    def __init__(self, config: AriannaCConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class AriannaC(nn.Module):
    """A minimal GPT-style model inspired by nanoGPT."""

    def __init__(self, config: AriannaCConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.block_size = config.block_size

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.size()
        if T > self.block_size:
            raise ValueError("Cannot forward, sequence too long")
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(pos)
        x = self.drop(tok_emb + pos_emb)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -self.block_size :])
            logits = logits[:, -1, :]
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# --- Reflection utility ---
def reflect(prompt: str, draft: str, max_new_tokens: int = 50, config: AriannaCConfig | None = None) -> str:
    """Critique a draft answer using the model."""
    critique_prompt = (
        "Provide feedback on the given answer. "
        f"Prompt: {prompt}\nAnswer: {draft}\nCritique:"
    )
    config = config or AriannaCConfig()
    model = AriannaC(config)
    quantize_2bit(model)
    model.eval()
    idx = tokenizer.encode(critique_prompt)
    out = model.generate(idx, max_new_tokens=max_new_tokens)
    critique = tokenizer.decode(out[0])
    return critique


# --- Text generation utilities ---
def generate_text(
    prompt: str | None = None,
    max_new_tokens: int = 50,
    config: AriannaCConfig | None = None,
    *,
    log_reasoning: bool = False,
    use_memory: bool = False,
    memory_limit: int = 3,
    self_reflect: bool = False,
) -> str | tuple[str, dict[str, float | int]]:
    """Generate a completion optionally enriched with past prompts."""
    prompt = prompt or CORE_PROMPT
    config = config or AriannaCConfig()
    monitor = SelfMonitor()
    if use_memory:
        examples = monitor.search(prompt, limit=memory_limit)
        if examples:
            combined = "\n".join(
                f"Prompt: {p}\nOutput: {o}" for p, o in examples
            )
            prompt = f"{combined}\n{prompt}"
    model = AriannaC(config)
    quantize_2bit(model)
    model.eval()
    idx = tokenizer.encode(prompt)
    out = model.generate(idx, max_new_tokens=max_new_tokens)
    text = tokenizer.decode(out[0])
    if self_reflect:
        critique = reflect(prompt, text, max_new_tokens=max_new_tokens, config=config)
        if "good" not in critique.lower():
            revision_prompt = (
                f"{prompt}\nDraft answer: {text}\nCritique: {critique}\nRevised answer:"
            )
            idx = tokenizer.encode(revision_prompt)
            out = model.generate(idx, max_new_tokens=max_new_tokens)
            text = tokenizer.decode(out[0])
    monitor.log(prompt, text)
    complexity, entropy = estimate_complexity_and_entropy(text)
    record = thought_logger.log_turn(text, complexity, entropy)
    if log_reasoning:
        return text, {
            "complexity": record.complexity,
            "entropy": record.entropy,
            "timestamp": record.timestamp,
        }
    return text


def reason_loop(
    prompt: str | None = None,
    *,
    max_steps: int = 5,
    stop_tokens: tuple[str, ...] = ("</think>", "</answer>"),
    max_new_tokens: int = 50,
    config: AriannaCConfig | None = None,
) -> str:
    """Iteratively alternate between ``<think>`` and ``<answer>`` phases."""
    prompt = prompt or CORE_PROMPT
    config = config or AriannaCConfig()
    monitor = SelfMonitor()
    model = AriannaC(config)
    quantize_2bit(model)
    model.eval()
    text = prompt
    final_answer = ""
    for _ in range(max_steps):
        think_prompt = f"{text}\n<think>"
        idx = tokenizer.encode(think_prompt)
        out = model.generate(idx, max_new_tokens=max_new_tokens)
        new_tokens = out[:, idx.shape[1] :]
        thought = tokenizer.decode(new_tokens)
        monitor.log("<think>", thought)
        text = tokenizer.decode(out[0])
        if any(tok in thought for tok in stop_tokens):
            break
        answer_prompt = f"{text}\n<answer>"
        idx = tokenizer.encode(answer_prompt)
        out = model.generate(idx, max_new_tokens=max_new_tokens)
        new_tokens = out[:, idx.shape[1] :]
        final_answer = tokenizer.decode(new_tokens)
        monitor.log("<answer>", final_answer)
        text = tokenizer.decode(out[0])
        if any(tok in final_answer for tok in stop_tokens):
            break
    return final_answer or text


def generate_with_think(
    prompt: str | None = None,
    max_new_tokens: int = 50,
    config: AriannaCConfig | None = None,
    **kwargs,
) -> str | tuple[str, dict[str, float | int]]:
    """Generate text while allowing a hook for reasoning steps."""
    return generate_text(
        prompt,
        max_new_tokens=max_new_tokens,
        config=config,
        log_reasoning=True,
        **kwargs,
    )


def generate_consistent_text(
    prompt: str | None = None,
    n: int = 5,
    **kwargs,
) -> str:
    """Generate multiple completions and return the most consistent answer."""
    prompt = prompt or CORE_PROMPT
    results: list[str] = []
    for _ in range(n):
        output = generate_with_think(prompt, **kwargs)
        final = output[-1] if isinstance(output, tuple) else output
        results.append(final)
    counts = Counter(results)
    most_common_answer, freq = counts.most_common(1)[0]
    tied = [ans for ans, c in counts.items() if c == freq]
    if len(tied) > 1:
        most_common_answer = min(tied, key=len)
    return most_common_answer


# --- CLI ---
def main() -> None:
    parser = argparse.ArgumentParser(description="Arianna-C text generation")
    parser.add_argument("prompt", nargs="?", help="prompt to complete")
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--verbose", action="store_true", help="show reasoning log")
    parser.add_argument(
        "--consistency",
        type=int,
        default=1,
        help="number of attempts to ensure answer consistency",
    )
    parser.add_argument(
        "--reflect",
        action="store_true",
        help="enable self-verification through reflection",
    )
    parser.add_argument(
        "--use-memory",
        action="store_true",
        help="prepend similar past prompts from memory",
    )
    parser.add_argument("--max-steps", type=int, default=0, help="max reasoning steps")
    parser.add_argument(
        "--stop-token",
        action="append",
        default=[],
        help="token that halts the reasoning loop; can be used multiple times",
    )
    args = parser.parse_args()

    config = AriannaCConfig(vocab_size=256)
    if args.max_steps or args.stop_token:
        loop_kwargs: dict[str, object] = {
            "max_new_tokens": args.max_new_tokens,
            "config": config,
        }
        if args.max_steps:
            loop_kwargs["max_steps"] = args.max_steps
        if args.stop_token:
            loop_kwargs["stop_tokens"] = tuple(args.stop_token)
        result = reason_loop(args.prompt, **loop_kwargs)
        print(result)
    elif args.consistency > 1:
        result = generate_consistent_text(
            args.prompt,
            n=args.consistency,
            max_new_tokens=args.max_new_tokens,
            config=config,
            self_reflect=args.reflect,
            use_memory=args.use_memory,
        )
        print(result)
    else:
        result = generate_text(
            args.prompt,
            max_new_tokens=args.max_new_tokens,
            config=config,
            log_reasoning=args.verbose,
            self_reflect=args.reflect,
            use_memory=args.use_memory,
        )
        if args.verbose:
            text, meta = result
            print(text)
            print(
                f"LOG@{meta['timestamp']} | Complexity: {meta['complexity']} | Entropy: {meta['entropy']:.2f}"
            )
        else:
            print(result)


if __name__ == "__main__":  # pragma: no cover
    main()


__all__ = [
    "AriannaC",
    "AriannaCConfig",
    "generate_text",
    "reason_loop",
    "reflect",
    "quantize_2bit",
    "SelfMonitor",
    "CORE_PROMPT",
    "ThoughtComplexityLogger",
    "estimate_complexity_and_entropy",
    "thought_logger",
    "generate_with_think",
    "generate_consistent_text",
    "tokenizer",
    "TokenizerWrapper",
]
