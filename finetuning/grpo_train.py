"""GRPO training loop for Arianna-C.

This script is inspired by `open_r1`'s implementation of Generalized
Rejection Policy Optimization but adapts the idea to the tiny Arianna-C
model. It combines simple reward functions—accuracy, reasoning tags and
length—to fine-tune the model.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import torch

from arianna_chain import AriannaC, AriannaCConfig, tokenizer


DEFAULT_LOGDIR = Path("logs/grpo")
DEFAULT_LOGDIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("grpo")
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
_file_handler = logging.FileHandler(DEFAULT_LOGDIR / "train.log")
_file_handler.setFormatter(formatter)
_stream_handler = logging.StreamHandler()
_stream_handler.setFormatter(formatter)
logger.addHandler(_file_handler)
logger.addHandler(_stream_handler)


def iter_dataset(path: str, min_confidence: float = 0.0) -> Iterator[Dict[str, str]]:
    """Yield dataset entries lazily from a JSONL file."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            conf = float(obj.get("confidence", 1.0))
            if conf < min_confidence:
                continue
            prompt = obj.get("prompt") or obj.get("question") or ""
            answer = obj.get("answer") or obj.get("solution") or ""
            plan = obj.get("plan", "")
            yield {"prompt": prompt, "answer": answer, "plan": plan, "confidence": conf}


def load_dataset(path: str, min_confidence: float = 0.0) -> List[Dict[str, str]]:
    """Eagerly load a JSONL dataset into memory."""
    return list(iter_dataset(path, min_confidence))


def accuracy_reward(pred: str, target: str) -> float:
    """Return 1.0 if ``pred`` exactly matches ``target`` (after stripping)."""
    return 1.0 if pred.strip() == target.strip() else 0.0


def reasoning_tag_reward(text: str) -> float:
    """Reward presence of both <analysis> and <final> tags."""
    has_analysis = "<analysis>" in text
    has_final = "<final>" in text
    return 1.0 if has_analysis and has_final else 0.0


def length_reward(text: str, max_tokens: int) -> float:
    """Penalty for outputs longer than ``max_tokens`` tokens."""
    toks = text.split()
    penalty = max(0, len(toks) - max_tokens)
    return -penalty / max_tokens


def load_model(model_path: str | None) -> AriannaC:
    """Load ``AriannaC`` from a checkpoint or initialize a new one."""
    config = AriannaCConfig(apply_quant=False)
    model = AriannaC(config)
    model.train()
    torch.set_grad_enabled(True)
    if model_path and Path(model_path).exists():
        state = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state)
        logger.info("Loaded model weights from %s", model_path)
    return model


def sample_with_grad(
    model: AriannaC, idx: torch.Tensor, max_new_tokens: int, temperature: float
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """Sample tokens while keeping gradients."""
    log_probs: List[torch.Tensor] = []
    for _ in range(max_new_tokens):
        logits, _ = model(idx[:, -model.block_size:])
        logits = logits[:, -1, :]
        probs = torch.softmax(logits / temperature, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        log_prob = torch.log(probs.gather(1, next_token))
        idx = torch.cat((idx, next_token), dim=1)
        log_probs.append(log_prob)
    return idx, log_probs


def train(args: argparse.Namespace) -> None:
    model = load_model(args.model_path)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    step = 0
    for epoch in range(args.epochs):
        for sample in iter_dataset(args.dataset, args.min_confidence):
            prompt = sample.get("prompt") or sample.get("question") or ""
            target = sample.get("solution") or sample.get("answer") or ""

            idx = tokenizer.encode(prompt).long()
            idx, log_probs = sample_with_grad(model, idx, args.max_new_tokens, args.temperature)
            generated = tokenizer.decode(idx[:, -args.max_new_tokens:])

            acc_r = accuracy_reward(generated, target)
            tag_r = reasoning_tag_reward(generated)
            len_r = length_reward(generated, args.max_new_tokens)
            reward = acc_r + tag_r + len_r

            loss = -reward * torch.stack(log_probs).sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            step += 1
            logger.info(
                "step=%d epoch=%d loss=%.4f reward=%.4f acc=%.2f tags=%.2f len=%.2f",
                step,
                epoch,
                loss.item(),
                reward,
                acc_r,
                tag_r,
                len_r,
            )

            if step % args.save_every == 0:
                ckpt = Path(args.logdir) / f"checkpoint_{step}.pt"
                torch.save(model.state_dict(), ckpt)
                logger.info("Saved checkpoint to %s", ckpt)

    final_ckpt = Path(args.logdir) / "final.pt"
    torch.save(model.state_dict(), final_ckpt)
    logger.info("Training complete. Final model saved to %s", final_ckpt)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GRPO training for Arianna-C")
    parser.add_argument("--dataset", required=True, help="Path to JSONL dataset")
    parser.add_argument("--model-path", help="Optional path to model checkpoint")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument(
        "--logdir",
        default=str(DEFAULT_LOGDIR),
        help="Directory for logs and checkpoints",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.0,
        help="Ignore dataset entries with confidence below this threshold",
    )
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
