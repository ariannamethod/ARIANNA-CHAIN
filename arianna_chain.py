# flake8: noqa
# arianna_chain.py — "liquid weights" (CPU-only, W2A8 int-core)
# CHANGES (key):
# - Safer HTTP integration points remain outside; this file is self-contained.
# - W2A8 Linear: fixed byte-offset math + clarified per-group packing; tiny cache.
# - SelfMonitor: WAL + safe ALTER; optional embeddings off by ARIANNA_DISABLE_EMBED=1.
# - Tools: kept minimal/safe set; redaction hardened.
# - Reason loops: same API; better stagnation/consistency heuristics; more robust fallbacks.
# - ByteTokenizer: 2D batched tensors with correct token counting.
# - Minor cleanups, comments, and bounds.

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import math
import re
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

from arianna_core.config import settings

import numpy as np
try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover
    faiss = None
import torch
import torch.nn as nn

from arianna_core import (
    AriannaC,
    AriannaCConfig,
    CORE_PROMPT,
    PERSONA,
    SelfMonitor,
    ThoughtComplexityLogger,  # noqa: F401
    ByteTokenizer,  # noqa: F401
    call_liquid,
    call_liquid_stream,
    estimate_complexity_and_entropy,
    format_reward,
    quantize_2bit,
    reasoning_steps_reward,
    thought_logger,
    tokenizer,
    validate_reasoning_tags,  # noqa: F401
)

# Simple vector store
# ────────────────────────────────────────────────────────────────────────────────
class VectorStore:
    """Store documents as dense vectors and perform similarity search."""

    def __init__(self, documents: List[str] | None = None, dim: int = 128) -> None:
        self.dim = dim
        self.documents: List[str] = []
        if faiss:
            self.index = faiss.IndexFlatIP(dim)
        else:  # pragma: no cover
            self.index = None
            self.vectors: List[np.ndarray] = []
        if documents:
            self.add(documents)

    def _embed(self, text: str) -> np.ndarray:
        vec = np.frombuffer(text.encode("utf-8"), dtype="uint8").astype("float32")
        if vec.size < self.dim:
            vec = np.pad(vec, (0, self.dim - vec.size))
        else:
            vec = vec[: self.dim]
        norm = np.linalg.norm(vec) or 1.0
        return vec / norm

    def add(self, docs: List[str]) -> None:
        embeddings = (
            np.vstack([self._embed(d) for d in docs])
            if docs
            else np.empty((0, self.dim), dtype="float32")
        )
        if self.index is not None and embeddings.size:
            self.index.add(embeddings)
        else:  # pragma: no cover
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
        else:  # pragma: no cover
            sims = [float(np.dot(qvec.squeeze(), v)) for v in self.vectors]
            ids = np.argsort(sims)[::-1][:k]
        return [self.documents[i] for i in ids]

# ────────────────────────────────────────────────────────────────────────────────
# Tools (safe & redacted) + manifest
# ────────────────────────────────────────────────────────────────────────────────
_SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
    re.compile(r"(?i)(api[_-]?key|token)[\"'\s:]*[A-Za-z0-9\-_]{16,}"),
    re.compile(r"eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}"),  # JWT-ish
]

def _redact(text: str) -> str:
    red = text
    for pat in _SECRET_PATTERNS:
        red = pat.sub("[REDACTED]", red)
    red = re.sub(r"[A-Za-z0-9+/]{200,}={0,2}", "[BASE64_REDACTED]", red)
    return red

def _tool_memory_search(query: str, limit: int = 3) -> str:
    with SelfMonitor() as sm:
        pairs = sm.search_faiss(query, limit=limit)
        notes = sm._search_notes(query, limit=limit)
        if not pairs and not notes:
            return "(no hits)"
        out = []
        for p, o in pairs:
            p1 = _redact(p.strip().splitlines()[0][:160])
            o1 = _redact(o.strip().splitlines()[0][:200])
            out.append(f"- Q:{p1} | A:{o1}")
        for n in notes:
            out.append(f"- { _redact(n) }")
        return "\n".join(out[:limit])

def _tool_memory_note(text: str) -> str:
    with SelfMonitor() as sm:
        sm.note(_redact(text)[:1000])
    return "ok"

def _tool_memory_link(prompt_sha: str, note_sha: str, relation: str) -> str:
    with SelfMonitor() as sm:
        sm.link_prompt(prompt_sha, note_sha, relation)
    return "ok"

def _tool_math_eval(expr: str) -> str:
    import ast, operator as op
    allowed = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv,
               ast.Pow: op.pow, ast.USub: op.neg, ast.Mod: op.mod, ast.FloorDiv: op.floordiv}
    def guard(node, depth=0):
        if depth > 10:
            raise ValueError("expression too deep")
        if isinstance(node, ast.Num):
            if not isinstance(node.n, (int, float)):
                raise ValueError("number type")
            return node.n
        if isinstance(node, ast.UnaryOp) and type(node.op) in allowed:
            return allowed[type(node.op)](guard(node.operand, depth+1))
        if isinstance(node, ast.BinOp) and type(node.op) in allowed:
            return allowed[type(node.op)](guard(node.left, depth+1), guard(node.right, depth+1))
        raise ValueError("unsupported expression")
    if len(expr) > 128:
        raise ValueError("expr too long")
    node = ast.parse(expr, mode="eval").body
    result = guard(node)
    if isinstance(result, (int, float)) and (not math.isfinite(result) or abs(result) > 1e12):
        raise ValueError("result out of bounds")
    return str(result)

def _tool_time_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def _tool_text_regex_extract(pattern: str, text: str, limit: int = 10, flags: str = "") -> str:
    fl = 0
    if "i" in flags: fl |= re.IGNORECASE
    if "m" in flags: fl |= re.MULTILINE
    if "s" in flags: fl |= re.DOTALL
    try:
        rgx = re.compile(pattern, fl)
        matches = rgx.findall(text)
        if isinstance(matches, list):
            matches = matches[:max(1, min(limit, 50))]
            flat = []
            for m in matches:
                if isinstance(m, tuple):
                    flat.append("".join(map(str, m)))
                else:
                    flat.append(str(m))
            uniq, seen = [], set()
            for x in flat:
                if x not in seen:
                    seen.add(x)
                    uniq.append(x)
            return json.dumps(uniq, ensure_ascii=False)
        return "[]"
    except Exception as e:
        return json.dumps({"ok": False, "error": str(e)})

def _tool_date_parse(text: str) -> str:
    text = text.strip()
    fmts = ["%Y-%m-%d", "%d.%m.%Y", "%d/%m/%Y", "%d-%m-%Y", "%Y/%m/%d"]
    for f in fmts:
        try:
            dt = datetime.strptime(text, f)
            return dt.date().isoformat()
        except Exception:
            continue
    try:
        return datetime.fromisoformat(text).date().isoformat()
    except Exception:
        return json.dumps({"ok": False, "error": "unrecognized date"})

TOOLS: Dict[str, Callable[..., str]] = {
    "memory.search":      lambda **kw: _tool_memory_search(str(kw.get("query","")), int(kw.get("limit",3))),
    "memory.note":        lambda **kw: _tool_memory_note(str(kw.get("text",""))),
    "memory.link":        lambda **kw: _tool_memory_link(
        str(kw.get("prompt_sha","")),
        str(kw.get("note_sha","")),
        str(kw.get("relation","")),
    ),
    "math.eval":          lambda **kw: _tool_math_eval(str(kw.get("expr","0"))),
    "time.now":           lambda **kw: _tool_time_now(),
    "text.regex_extract": lambda **kw: _tool_text_regex_extract(str(kw.get("pattern","")), str(kw.get("text","")), int(kw.get("limit",10)), str(kw.get("flags",""))),
    "date.parse":         lambda **kw: _tool_date_parse(str(kw.get("text",""))),
}

def _tools_manifest() -> str:
    return json.dumps({
        "tools": {
            "memory.search":      {"args": {"query": "string", "limit": "int"}, "desc": "search previous prompts/answers/notes (redacted)."},
            "memory.note":        {"args": {"text": "string"}, "desc": "store a short note to memory."},
            "memory.link":        {"args": {"prompt_sha": "string", "note_sha": "string", "relation": "string"}, "desc": "link a prompt to a note."},
            "math.eval":          {"args": {"expr": "string"}, "desc": "evaluate a safe arithmetic expression."},
            "time.now":           {"args": {}, "desc": "UTC timestamp now."},
            "text.regex_extract": {"args": {"pattern": "string", "text": "string", "limit":"int", "flags":"string"}, "desc": "regex matches as JSON list."},
            "date.parse":         {"args": {"text": "string"}, "desc": "parse common date formats to ISO date."}
        }
    }, ensure_ascii=False)

# ────────────────────────────────────────────────────────────────────────────────
# Token budget + summarization
# ────────────────────────────────────────────────────────────────────────────────
def _approx_tokens(text: str) -> int:
    return max(1, len(text.encode("utf-8", errors="ignore")) // 4)

MAX_INPUT_TOKENS = settings.arianna_max_tokens
LAST_USAGE_SUMMARIZE_THRESHOLD = settings.arianna_last_usage_summary_tokens

def _summarize_trace(trace_text: str, *, trace_id: str) -> str:
    prompt = (
        "Summarize the following reasoning trace into <= 6 bullets capturing goals, constraints, used tools, "
        "observations, pending TODO, and assumptions/limits. Return JSON with 'mode':'reflect', 'answer': summary, 'stop': false.\n\nTRACE:\n" + trace_text
    )
    obj = call_liquid(prompt, trace_id=trace_id, temperature=0.2)
    return str(obj.get("answer", ""))[:2000]

# ────────────────────────────────────────────────────────────────────────────────
# Similarity helpers
# ────────────────────────────────────────────────────────────────────────────────
def _tf_cosine(a: str, b: str) -> float:
    def norm_tokens(s: str) -> List[str]:
        return re.findall(r"[A-Za-zА-Яа-яёЁ0-9]{2,}", s.lower())
    ca = Counter(norm_tokens(a))
    cb = Counter(norm_tokens(b))
    if not ca or not cb:
        return 0.0
    dot = sum(ca[t] * cb.get(t, 0) for t in ca)
    na = math.sqrt(sum(v*v for v in ca.values()))
    nb = math.sqrt(sum(v*v for v in cb.values()))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)

def _similarity(a: str, b: str) -> float:
    seq = difflib.SequenceMatcher(None, a, b).ratio()
    cos = _tf_cosine(a, b)
    return max(seq, cos)

# ────────────────────────────────────────────────────────────────────────────────
# ReAct reasoning + checkpoint reflect + critical verify
# ────────────────────────────────────────────────────────────────────────────────
_ALLOWED_TRANSITIONS = {
    "plan": {"act"},
    "act": {"reflect"},
    "reflect": {"plan", "final"},
    "final": set(),
}

def _normalize_step_text(step_obj: Dict[str, Any]) -> str:
    return (step_obj.get("think","") + " | " + step_obj.get("answer","")).strip()

def _force_action_from_text(answer: str) -> Dict[str, Any]:
    words = re.findall(r"[A-Za-zА-Яа-яёЁ]{3,}", answer)[:8]
    query = " ".join(words)
    return {"name": "memory.search", "args": {"query": query, "limit": 3}}

def _critical_check(trace_id: str, steps: List[Dict[str, Any]], user_prompt: str) -> str:
    ctx = "\n".join(json.dumps(s, ensure_ascii=False) for s in steps[-3:])
    prompt = (
        "Analyze for contradictions, leaps in logic, or missing facts in the following trace. "
        "Return JSON with mode='reflect', stop=false, answer with max 3 bullets: "
        "(1) potential flaw (2) what to verify (3) next concrete action.\n\nTRACE:\n" + ctx +
        "\n\nUSER PROMPT:\n" + user_prompt
    )
    obj = call_liquid(prompt, trace_id=trace_id, temperature=0.0)
    return str(obj.get("answer", ""))

def _verify_low_confidence(trace_id: str, user_prompt: str, draft: str) -> str:
    crit = (
        "Critique the following draft for factual errors, contradictions and missing steps. "
        "Then propose a corrected version. Return JSON with mode='reflect', stop=false, answer=ONLY corrected text.\n"
        f"PROMPT:\n{user_prompt}\n\nDRAFT:\n{draft}"
    )
    obj = call_liquid(crit, trace_id=trace_id, temperature=0.2)
    return str(obj.get("answer", draft))

def verify_step(trace_id: str, user_prompt: str, observation: str) -> str:
    prompt = (
        "Assess the following observation for factual correctness or potential issues. "
        "Return JSON with mode='verify', stop=false, answer=ONLY a brief comment.\n"
        f"PROMPT:\n{user_prompt}\n\nOBSERVATION:\n{observation}"
    )
    obj = call_liquid(prompt, trace_id=trace_id, temperature=0.0)
    return str(obj.get("answer", ""))

def reason_loop(
    prompt: Optional[str] = None,
    *,
    max_steps: int = 6,
    use_liquid: bool = True,
    progress_patience: int = 2,
    base_temperature: float = 0.3,
    checkpoint_every: int = 2,
    critical_every: int = 3,
    beams: int = 1,
    retrieve: bool = False,
) -> str:
    user_prompt = (prompt or CORE_PROMPT).strip()
    if retrieve:
        try:
            store = VectorStore()
            docs = store.search(user_prompt)
            if docs:
                user_prompt = "\n".join(docs) + "\n\n" + user_prompt
        except Exception:
            pass
    def _run(sm: SelfMonitor) -> str:
        trace_id = uuid.uuid4().hex
        steps: List[Dict[str, Any]] = []
        stagnation = 0
        final_answer = ""
        temperature = base_temperature
    
        def render_context(expected_next: str = "") -> str:
            lines = [
                "=== SYSTEM INSTRUCTION ===",
                CORE_PROMPT,
                "=== TOOLS (name -> args schema) ===",
                _tools_manifest(),
                "=== TRACE (latest first) ===",
            ]
            for s in reversed(steps[-6:]):
                lines.append(json.dumps(s, ensure_ascii=False))
            lines.append("=== USER PROMPT ===")
            lines.append(user_prompt)
            if expected_next:
                lines.append(f"=== MODE HINT ===\nExpected next step: {expected_next}")
            ctx = "\n".join(lines)
            ctx = re.sub(r"[A-Za-z0-9+/]{200,}={0,2}", "[BASE64_REDACTED]", ctx)
            return ctx
    
        def ensure_budget(ctx: str):
            if steps and isinstance(steps[-1].get("tokens_used"), dict):
                last_total = int(steps[-1]["tokens_used"].get("total", 0))
                if last_total > LAST_USAGE_SUMMARIZE_THRESHOLD:
                    full = "\n".join(json.dumps(s, ensure_ascii=False) for s in steps)
                    summary = _summarize_trace(full, trace_id=trace_id)
                    steps.clear()
                    steps.append({"trace_id": trace_id, "step": 0, "mode": "reflect", "think": "summary", "answer": summary, "stop": False, "meta": {"summarized": True}})
                    return
            if _approx_tokens(ctx) <= MAX_INPUT_TOKENS:
                return
            full = "\n".join(json.dumps(s, ensure_ascii=False) for s in steps)
            summary = _summarize_trace(full, trace_id=trace_id)
            steps.clear()
            steps.append({"trace_id": trace_id, "step": 0, "mode": "reflect", "think": "summary", "answer": summary, "stop": False, "meta": {"summarized": True}})
    
        last_mode = None
    
        for step_idx in range(1, max_steps + 1):
            # checkpoints
            if checkpoint_every and step_idx > 1 and (step_idx - 1) % checkpoint_every == 0 and steps:
                try:
                    chk = {
                        "trace_id": trace_id, "step": (steps[-1]["step"] + 1 if steps else 1),
                        "mode": "reflect", "think": "checkpoint",
                        "answer": _critical_check(trace_id, steps, user_prompt),
                        "stop": False, "confidence": 0.7
                    }
                    sm.log("<step>", json.dumps(chk, ensure_ascii=False))
                    steps.append(chk)
                except Exception:
                    pass
    
            temperature = max(0.3, min(0.9, base_temperature + 0.2 * stagnation))
            expected_next = "act" if (steps and steps[-1]["mode"] == "plan") else ""
    
            ctx = render_context(expected_next)
            ensure_budget(ctx)
    
            def _reason_reward(o: Dict[str, Any]) -> float:
                ans = str(o.get("answer", ""))
                feat = 0.0
                feat += 0.2 if any(ch.isdigit() for ch in ans) else 0.0
                feat += 0.2 if ("- " in ans or "1." in ans) else 0.0
                feat += -0.3 if _similarity(final_answer, ans) > 0.85 else 0.0
                feat += float(o.get("confidence", 0.7)) * 0.4
                feat += min(len(ans), 600) / 600.0 * 0.2
                return feat
    
            candidates: List[Dict[str, Any]] = []
            if use_liquid:
                for b in range(max(1, beams)):
                    t = min(0.9, temperature + 0.2 * b)
                    try:
                        candidates.append(call_liquid(ctx, trace_id=trace_id, temperature=t))
                    except Exception:
                        model = AriannaC(AriannaCConfig())
                        idx = tokenizer.encode(ctx)
                        out = model.generate(idx, max_new_tokens=128)
                        tok = out[0] if out.dim() > 1 else out
                        candidates.append({"mode": "final", "think": "", "answer": tokenizer.decode(tok), "stop": True, "step": step_idx, "trace_id": trace_id, "confidence": 0.6})
                        break
            else:
                model = AriannaC(AriannaCConfig())
                idx = tokenizer.encode(ctx)
                out = model.generate(idx, max_new_tokens=128)
                tok = out[0] if out.dim() > 1 else out
                candidates.append({"mode": "final", "think": "", "answer": tokenizer.decode(tok), "stop": True, "step": step_idx, "trace_id": trace_id, "confidence": 0.6})
    
            obj = max(candidates, key=_reason_reward)
    
            mode = str(obj.get("mode", "final"))
            think = str(obj.get("think", ""))
            answer = str(obj.get("answer", ""))
            stop = bool(obj.get("stop", mode == "final"))
            conf = float(obj.get("confidence", 0.7))
            act = obj.get("action") if isinstance(obj.get("action"), dict) or isinstance(obj.get("action"), list) else None
            observation: Optional[str] = None
    
            # enforce allowed transitions
            if last_mode is not None and mode not in _ALLOWED_TRANSITIONS.get(last_mode, {"final"}):
                nxt = next(iter(_ALLOWED_TRANSITIONS.get(last_mode, {"final"})))
                mode, stop = nxt, False
    
            if mode == "final" and conf < 0.6 and step_idx < max_steps:
                mode = "reflect"
                stop = False
    
            if mode in ("plan", "reflect") and act is None and stagnation >= 1:
                act = _force_action_from_text(answer)
                mode = "act"
                stop = False
    
            if mode == "act" and act:
                try:
                    if isinstance(act, list):
                        results = []
                        with ThreadPoolExecutor(max_workers=min(4, len(act))) as ex:
                            futs = []
                            for a in act:
                                if not (isinstance(a, dict) and isinstance(a.get("name"), str)):
                                    continue
                                name = a["name"]
                                args = a.get("args", {}) if isinstance(a.get("args"), dict) else {}
                                tool = TOOLS.get(name)
                                futs.append(ex.submit(lambda t=tool, kw=args, n=name: (n, t(**kw) if t else f"(unknown tool: {n})")))
                            for f in as_completed(futs):
                                name, res = f.result()
                                if isinstance(res, (dict, list)):
                                    res = json.dumps(res, ensure_ascii=False)
                                results.append({"tool": name, "result": str(res)})
                        observation = json.dumps(results, ensure_ascii=False)
                    else:
                        name = act["name"]
                        args = act.get("args", {}) if isinstance(act.get("args"), dict) else {}
                        tool = TOOLS.get(name)
                        obs_res = tool(**args) if tool else f"(unknown tool: {name})"
                        if isinstance(obs_res, (dict, list)):
                            observation = json.dumps(obs_res, ensure_ascii=False)
                        else:
                            observation = str(obs_res)
                except Exception as e:
                    observation = json.dumps({"ok": False, "error": str(e)})
    
            sm.log("<think>", think)
            sm.log("<answer>", answer)
            step_obj: Dict[str, Any] = {
                "trace_id": trace_id, "step": step_idx, "mode": mode,
                "think": think, "answer": answer, "stop": stop, "confidence": conf
            }
            if isinstance(obj.get("tokens_used"), dict):
                step_obj["tokens_used"] = obj["tokens_used"]
            if act: step_obj["action"] = act
            if observation: step_obj["observation"] = observation
    
            resp_text = f"<think>{think}</think>\n<answer>{answer}</answer>"
            fmt_score = format_reward(resp_text)
            steps_score = reasoning_steps_reward(resp_text)
            step_obj["rewards"] = {"format": fmt_score, "reasoning_steps": steps_score}
            sm.log("<reward>", json.dumps({"step": step_idx, "format": fmt_score, "reasoning_steps": steps_score}))
    
            sm.log("<step>", json.dumps(step_obj, ensure_ascii=False))
            steps.append(step_obj)
            if mode == "act" and observation:
                try:
                    comment = verify_step(trace_id, user_prompt, observation)
                    verify_obj = {
                        "trace_id": trace_id,
                        "step": step_idx + 0.1,
                        "mode": "verify",
                        "think": "",
                        "answer": comment,
                        "stop": False,
                        "confidence": 0.7,
                    }
                    sm.log("<step>", json.dumps(verify_obj, ensure_ascii=False))
                    steps.append(verify_obj)
                except Exception:
                    pass
            if answer:
                final_answer = answer
    
            if len(steps) >= 2:
                s_prev = _normalize_step_text(steps[-2])
                s_curr = _normalize_step_text(steps[-1])
                if _similarity(s_prev, s_curr) > 0.90:
                    stagnation += 1
                else:
                    stagnation = 0
    
            if critical_every and (step_idx % critical_every == 0) and step_idx < max_steps:
                try:
                    crit = _critical_check(trace_id, steps, user_prompt)
                    steps.append({"trace_id": trace_id, "step": step_idx + 0.5, "mode": "reflect", "think": "crit", "answer": crit, "stop": False, "confidence": 0.7})
                except Exception:
                    pass
    
            if stagnation >= 1 and step_idx < max_steps:
                temperature = min(0.9, temperature + 0.2)
                try:
                    alt_low = call_liquid(render_context(), trace_id=trace_id, temperature=0.2)
                    alt_hi  = call_liquid(render_context(), trace_id=trace_id, temperature=0.6)
                    cands = [obj, alt_low, alt_hi]
                    def score(o: Dict[str, Any]) -> float:
                        ans = str(o.get("answer",""))
                        feat = 0.0
                        feat += 0.2 if any(ch.isdigit() for ch in ans) else 0.0
                        feat += 0.2 if ("- " in ans or "1." in ans) else 0.0
                        feat += -0.3 if _similarity(final_answer, ans) > 0.85 else 0.0
                        feat += float(o.get("confidence", 0.7)) * 0.4
                        return feat + min(len(ans), 600)/600.0 * 0.2
                    best = max(cands, key=score)
                    if best is not obj:
                        obj = best
                        steps[-1]["think"] = str(obj.get("think",""))
                        steps[-1]["answer"] = str(obj.get("answer",""))
                        steps[-1]["confidence"] = float(obj.get("confidence", steps[-1]["confidence"]))
                        final_answer = steps[-1]["answer"] or final_answer
                except Exception:
                    pass
    
            if conf < 0.5 and step_idx < max_steps and not stop:
                try:
                    fixed = _verify_low_confidence(trace_id, user_prompt, final_answer or answer)
                    if fixed and fixed.strip() and _similarity(fixed, final_answer) < 0.92:
                        steps.append({"trace_id": trace_id, "step": step_idx + 0.6, "mode": "reflect", "think": "verify", "answer": fixed, "stop": False, "confidence": 0.7})
                        final_answer = fixed
                except Exception:
                    pass
    
            last_mode = mode
            if stop or mode == "final" or stagnation >= progress_patience:
                break
    
        text = final_answer or json.dumps(steps[-1], ensure_ascii=False)
        tokens, entropy, perplexity = estimate_complexity_and_entropy(text)
        final_conf = float(steps[-1].get("confidence", 0.0)) if steps else 0.0
        thought_logger.log_turn(text, tokens, entropy, perplexity, final_conf)
    
        plan = ""
        for s in steps:
            if s.get("mode") == "plan":
                plan = s.get("answer", "")
                break
        distill_path = Path("logs/distill.jsonl")
        distill_path.parent.mkdir(parents=True, exist_ok=True)
        with distill_path.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "prompt": user_prompt,
                        "plan": plan,
                        "answer": text,
                        "confidence": final_conf,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        return text
    with SelfMonitor() as sm:
        return _run(sm)

def tree_reason_loop(
    prompt: Optional[str] = None,
    *,
    beam_size: int = 2,
    depth: int = 2,
    score_fn: Callable[[str], float] | None = None,
    **reason_kwargs: Any,
) -> str:
    scorer = score_fn or (lambda ans: estimate_complexity_and_entropy(ans)[1])
    branches: List[Tuple[str, float]] = []
    for _ in range(max(1, beam_size)):
        ans = reason_loop(prompt, max_steps=depth, **reason_kwargs)
        branches.append((ans, scorer(ans)))
    best_answer, _ = max(branches, key=lambda x: x[1])
    return best_answer

def multi_reason(prompt: Optional[str] = None, paths: int = 5, **reason_kwargs) -> str:
    with SelfMonitor() as sm:
        temps = [0.2 + 0.6 * i / max(1, paths - 1) for i in range(max(1, paths))]
        results: List[Dict[str, Any]] = []
        for t in temps:
            ans = reason_loop(prompt, base_temperature=t, **reason_kwargs)
            conf = 1.0 - estimate_complexity_and_entropy(ans)[1]
            entry = {"temperature": t, "answer": ans, "confidence": conf}
            sm.log("<path>", json.dumps(entry, ensure_ascii=False))
            results.append(entry)

        counts = Counter(r["answer"] for r in results)
        unique_answers = list(counts.keys())

        def score(ans: str) -> float:
            freq = counts[ans]
            avg_conf = sum(r["confidence"] for r in results if r["answer"] == ans) / freq
            diversities = [1 - _similarity(ans, other) for other in unique_answers if other != ans]
            diversity = sum(diversities) / len(diversities) if diversities else 1.0
            return freq + avg_conf + diversity

        best = max(unique_answers, key=score)
        return best

# ────────────────────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────────
def reflect(prompt: str, draft: str, *, use_liquid: bool = True, max_new_tokens: int = 128, config: AriannaCConfig | None = None) -> str:
    critique_prompt = (
        "Critique the answer and propose fixes. Return JSON with keys trace_id, step, mode, think, answer, stop, confidence.\n"
        f"Prompt: {prompt}\nAnswer: {draft}"
    )
    if use_liquid:
        obj = call_liquid(critique_prompt, temperature=0.2)
        return str(obj.get("answer", "")) or json.dumps(obj, ensure_ascii=False)
    cfg = config or AriannaCConfig()
    model = AriannaC(cfg)
    idx = tokenizer.encode("Critique:\n" + critique_prompt)
    out = model.generate(idx, max_new_tokens=max_new_tokens, temperature=0.0)
    tok = out[0] if out.dim() > 1 else out
    return tokenizer.decode(tok)

def generate_text(
    prompt: Optional[str] = None,
    *,
    use_memory: bool = False,
    memory_limit: int = 3,
    self_reflect: bool = False,
    use_liquid: bool = True,
    max_new_tokens: int = 256,
    log_reasoning: bool = False,
    retrieve: bool = False,
    monitor: SelfMonitor | None = None,
) -> str | tuple[str, dict[str, float | int]]:
    prompt = (prompt or CORE_PROMPT).strip()
    if retrieve:
        try:
            store = VectorStore()
            docs = store.search(prompt)
            if docs:
                prompt = "\n".join(docs) + "\n\n" + prompt
        except Exception:
            pass
    def _run(sm: SelfMonitor) -> str | tuple[str, dict[str, float | int]]:
        prompt_local = prompt
        if use_memory:
            examples = sm.search_embedding(prompt_local, limit=memory_limit) or sm.search(prompt_local, limit=memory_limit)
            if examples:
                combined = "\n".join(f"PrevPrompt: {p}\nPrevOutput: {o}" for p, o in examples)
                prompt_local = f"{combined}\n\nCurrent:\n{prompt_local}"
        if use_liquid:
            try:
                plan_obj = call_liquid(f"Plan the steps to answer: {prompt_local}", temperature=0.3)
                plan = str(plan_obj.get("answer", ""))
                obj = call_liquid(prompt_local, temperature=0.3)
                think = str(obj.get("think", ""))
                answer = str(obj.get("answer", ""))
                try:
                    verify_obj = call_liquid(
                        f"Question: {prompt_local}\nAnswer: {answer}\nverify the previous answer",
                        temperature=0.0,
                    )
                    verified = str(verify_obj.get("answer", ""))
                    if verified:
                        answer = verified
                except Exception:
                    pass
                text = f"<plan>{plan}</plan>\n<think>{think}</think>\n<answer>{answer}</answer>"
                sm.log(prompt_local, text)
                tokens, entropy, perplexity = estimate_complexity_and_entropy(text)
                conf = float(obj.get("confidence", 1.0 - entropy))
                rec = thought_logger.log_turn(text, tokens, entropy, perplexity, conf)
                if self_reflect:
                    crit = reflect(prompt_local, text, use_liquid=True)
                    if "good" not in crit.lower():
                        repair = call_liquid(
                            f"Revise using this critique. Return JSON. Draft: {text}\nCritique: {crit}",
                            temperature=0.0,
                        )
                        text = str(repair.get("answer", text))
                        sm.log("revise", text)
                if log_reasoning:
                    return text, {"tokens": rec.tokens, "entropy": rec.entropy, "perplexity": rec.perplexity, "timestamp": rec.timestamp}
                return text
            except Exception:
                model = AriannaC(AriannaCConfig())
                idx = tokenizer.encode(prompt_local)
                out = model.generate(idx, max_new_tokens=max_new_tokens, temperature=0.0)
                tok = out[0] if out.dim() > 1 else out
                text = tokenizer.decode(tok)
                if self_reflect:
                    crit = reflect(prompt_local, text, use_liquid=False)
                    if "good" not in crit.lower():
                        idx = tokenizer.encode(f"Revise using this critique. Draft: {text}\nCritique: {crit}")
                        out = model.generate(idx, max_new_tokens=max_new_tokens, temperature=0.0)
                        tok = out[0] if out.dim() > 1 else out
                        text = tokenizer.decode(tok)
                        sm.log("revise", text)
                sm.log(prompt_local, text)
                tokens, entropy, perplexity = estimate_complexity_and_entropy(text, model)
                conf = 1.0 - entropy
                rec = thought_logger.log_turn(text, tokens, entropy, perplexity, conf)
                if log_reasoning:
                    return text, {"tokens": rec.tokens, "entropy": rec.entropy, "perplexity": rec.perplexity, "timestamp": rec.timestamp}
                return text
        model = AriannaC(AriannaCConfig())
        idx = tokenizer.encode(prompt_local)
        out = model.generate(idx, max_new_tokens=max_new_tokens, temperature=0.0)
        tok = out[0] if out.dim() > 1 else out
        text = tokenizer.decode(tok)
        if self_reflect:
            crit = reflect(prompt_local, text, use_liquid=False)
            if "good" not in crit.lower():
                idx = tokenizer.encode(f"Revise using this critique. Draft: {text}\nCritique: {crit}")
                out = model.generate(idx, max_new_tokens=max_new_tokens, temperature=0.0)
                tok = out[0] if out.dim() > 1 else out
                text = tokenizer.decode(tok)
                sm.log("revise", text)
        sm.log(prompt_local, text)
        tokens, entropy, perplexity = estimate_complexity_and_entropy(text, model)
        conf = 1.0 - entropy
        rec = thought_logger.log_turn(text, tokens, entropy, perplexity, conf)
        if log_reasoning:
            return text, {"tokens": rec.tokens, "entropy": rec.entropy, "perplexity": rec.perplexity, "timestamp": rec.timestamp}
        return text
    if monitor is None:
        with SelfMonitor() as sm:
            return _run(sm)
    return _run(monitor)

def generate_with_think(
    prompt: Optional[str] = None,
    *,
    max_new_tokens: int = 50,
    config: AriannaCConfig | None = None,
    retrieve: bool = False,
    **kwargs,
) -> str | tuple[str, dict[str, float | int]]:
    return generate_text(
        prompt,
        max_new_tokens=max_new_tokens,
        config=config,
        log_reasoning=True,
        retrieve=retrieve,
        **kwargs,
    )

def generate_consistent_text(prompt: Optional[str] = None, n: int = 3, **kwargs) -> str:
    prompt = (prompt or CORE_PROMPT).strip()
    results: List[str] = []
    for _ in range(n):
        out = generate_with_think(prompt, **kwargs)
        s = out[0] if isinstance(out, tuple) else out
        results.append(s)
    counts = Counter(results)
    ans, freq = counts.most_common(1)[0]
    tied = [a for a, c in counts.items() if c == freq]
    if len(tied) > 1:
        ans = min(tied, key=len)
    return ans

# ────────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Arianna-C (liquid-weights ReAct, W2A8 CPU core)")
    parser.add_argument("prompt", nargs="?", help="prompt to complete")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--verbose", action="store_true", help="show reasoning log")
    parser.add_argument("--consistency", type=int, default=1, help="n attempts for consistency vote")
    parser.add_argument("--reflect", action="store_true", help="self-reflection using liquid weights")
    parser.add_argument("--use-memory", action="store_true", help="prepend similar past prompts")
    parser.add_argument("--max-steps", type=int, default=0, help="ReAct steps (use reason_loop)")
    parser.add_argument("--no-liquid", action="store_true", help="disable liquid server (fallback to toy)")
    parser.add_argument("--stream", action="store_true", help="use SSE streaming endpoint")
    parser.add_argument("--checkpoint-every", type=int, default=2, help="insert checkpoint reflect every N steps")
    parser.add_argument("--progress-patience", type=int, default=2, help="allowed consecutive similar steps before halt")
    parser.add_argument("--critical-every", type=int, default=3, help="run critical check every N steps")
    parser.add_argument("--beams", type=int, default=1, help="number of candidate beams per step")
    parser.add_argument("--beam-size", type=int, default=1, help="number of reasoning branches for tree search")
    parser.add_argument("--retrieve", action="store_true", help="augment prompt with retrieved docs")
    args = parser.parse_args()

    use_liquid = not args.no_liquid

    if args.max_steps > 0:
        if args.beam_size > 1:
            result = tree_reason_loop(
                args.prompt,
                beam_size=args.beam_size,
                depth=args.max_steps,
                use_liquid=use_liquid,
                checkpoint_every=args.checkpoint_every,
                progress_patience=args.progress_patience,
                critical_every=args.critical_every,
                beams=args.beams,
                retrieve=args.retrieve,
            )
        else:
            result = reason_loop(
                args.prompt,
                max_steps=args.max_steps,
                use_liquid=use_liquid,
                checkpoint_every=args.checkpoint_every,
                progress_patience=args.progress_patience,
                critical_every=args.critical_every,
                beams=args.beams,
                retrieve=args.retrieve,
            )
        print(result)
    elif args.consistency > 1:
        result = generate_consistent_text(
            args.prompt,
            n=args.consistency,
            use_memory=args.use_memory,
            self_reflect=args.reflect,
            use_liquid=use_liquid,
            max_new_tokens=args.max_new_tokens,
            retrieve=args.retrieve,
        )
        print(result)
    else:
        if args.stream and use_liquid and args.prompt:
            buf = ""
            for etype, data in call_liquid_stream(args.prompt):
                if etype == "response.output_text.delta" and isinstance(data, dict):
                    buf += str(data.get("delta", ""))
                elif etype == "response.completed" and isinstance(data, dict):
                    buf = str(data.get("answer", buf))
            print(buf)
        else:
            result = generate_text(
                args.prompt,
                use_memory=args.use_memory,
                self_reflect=args.reflect,
                use_liquid=use_liquid,
                max_new_tokens=args.max_new_tokens,
                log_reasoning=args.verbose,
                retrieve=args.retrieve,
            )
            if args.verbose:
                text, meta = result  # type: ignore[assignment]
                print(text)
                print(f"LOG@{meta['timestamp']} | Tokens: {meta['tokens']} | Entropy: {meta['entropy']:.2f}")
            else:
                print(result)

if __name__ == "__main__":  # pragma: no cover
    main()

__all__ = [
    "AriannaC",
    "AriannaCConfig",
    "generate_text",
    "reason_loop",
    "tree_reason_loop",
    "multi_reason",
    "reflect",
    "quantize_2bit",
    "SelfMonitor",
    "CORE_PROMPT",
    "PERSONA",
    "ThoughtComplexityLogger",
    "estimate_complexity_and_entropy",
    "thought_logger",
    "generate_with_think",
    "generate_consistent_text",
    "tokenizer",
    "ByteTokenizer",
    "call_liquid",
    "call_liquid_stream",
    "validate_reasoning_tags",
    "reasoning_steps_reward",
]
