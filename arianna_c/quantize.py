"""2-bit weight quantization for Arianna-C.

The quantization squeezes model parameters to four discrete levels so
that the model fits on very small devices while keeping the reasoning
chain intact.
"""

from __future__ import annotations

import torch
from torch import nn


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
        mags = torch.where(q.abs() > 2, torch.tensor(3.0, device=param.device), torch.tensor(1.0, device=param.device))
        param.copy_(signs * mags * scale)


__all__ = ["quantize_2bit"]
