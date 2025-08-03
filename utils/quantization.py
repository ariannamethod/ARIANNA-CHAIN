import torch
import torch.nn as nn


def apply_dynamic_quant(model: nn.Module) -> nn.Module:
    """Apply dynamic quantization to reduce model size and improve inference."""
    return torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
