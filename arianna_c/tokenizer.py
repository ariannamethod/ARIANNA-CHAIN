"""Tokenizer utilities for Arianna-C.

This module provides a simple byte-level Byte Pair Encoding (BPE) tokenizer
implemented with the `tokenizers` library. The tokenizer is trained at import
time on the project's core prompt so that encoding and decoding are consistent
throughout the package.

The global ``tokenizer`` instance exposes ``encode`` and ``decode`` methods used
by the generation utilities and tests. Its vocabulary size is also available via
the :attr:`vocab_size` attribute.
"""

from __future__ import annotations

from pathlib import Path

import torch
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

# Train a byte-level BPE tokenizer on the core prompt. This keeps the
# implementation lightweight while avoiding a dependency on any external
# dataset. The resulting vocabulary size is small but sufficient for tests and
# example generations.
_tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
_tokenizer.pre_tokenizer = ByteLevel()
trainer = BpeTrainer(special_tokens=["[UNK]"])
core_prompt = (
    Path(__file__).resolve().parent.parent / "core_prompt.txt"
).read_text(encoding="utf-8")
_tokenizer.train_from_iterator([core_prompt], trainer)


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


# Public tokenizer instance used throughout the package
tokenizer = TokenizerWrapper(_tokenizer)

__all__ = ["tokenizer", "TokenizerWrapper"]
