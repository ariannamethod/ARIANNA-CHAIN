from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


# ---- 2-bit quant utilities ---------------------------------------------------

def _calc_group_qparams(w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    max_abs = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    scale = (max_abs / 1.5).squeeze(-1)  # 1.5 ≈ max(|{-2,-1,0,1}|) for zp=2
    zp = torch.full_like(scale, 2, dtype=torch.int32)
    return scale.float(), zp.int()


def _quant_2bit_codes(w: torch.Tensor, scale: torch.Tensor, zp: torch.Tensor) -> torch.Tensor:
    q = torch.round(w / scale.unsqueeze(-1)) + zp.unsqueeze(-1)
    q.clamp_(0, 3)
    return q.to(torch.uint8)


def _pad_cols(t: torch.Tensor, multiple: int = 4) -> Tuple[torch.Tensor, int]:
    cols = t.size(-1)
    rem = cols % multiple
    if rem == 0:
        return t, 0
    pad = multiple - rem
    pad_t = torch.nn.functional.pad(t, (0, pad))
    return pad_t, pad


def _pack2(u2: torch.Tensor) -> torch.Tensor:
    assert u2.dtype == torch.uint8
    K = u2.size(-1)
    assert K % 4 == 0, "need len%4==0 to pack"
    u2 = u2.contiguous().view(*u2.shape[:-1], K // 4, 4)
    b = (
        u2[..., 0]
        | (u2[..., 1] << 2)
        | (u2[..., 2] << 4)
        | (u2[..., 3] << 6)
    ).contiguous()
    return b


def _unpack2(packed: torch.Tensor, K: int) -> torch.Tensor:
    assert packed.dtype == torch.uint8
    bytes_flat = packed.unsqueeze(-1)
    b0 = (bytes_flat & 0x03).squeeze(-1)
    b1 = ((bytes_flat >> 2) & 0x03).squeeze(-1)
    b2 = ((bytes_flat >> 4) & 0x03).squeeze(-1)
    b3 = ((bytes_flat >> 6) & 0x03).squeeze(-1)
    out = torch.stack([b0, b1, b2, b3], dim=-1).reshape(*packed.shape[:-1], -1)
    return out[..., :K].contiguous()


class LinearW2A8(nn.Module):
    """2-bit per weight (packed), per-group quant; activations are float32."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        group_size: int = 64,
        cache_groups: int = 0,
        device=None,
        dtype=torch.float32,
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.group_size = int(max(8, group_size))
        self.bias = (
            nn.Parameter(torch.zeros(out_features, dtype=dtype, device=device))
            if bias
            else None
        )
        self.register_buffer("w_packed", torch.empty(0, dtype=torch.uint8), persistent=True)
        self.register_buffer("scales", torch.empty(0, dtype=torch.float32), persistent=True)
        self.register_buffer("zps", torch.empty(0, dtype=torch.uint8), persistent=True)
        self.register_buffer("cols_padded", torch.tensor(0, dtype=torch.int32), persistent=True)
        self.cache_groups = int(cache_groups)
        self._unpacked_cache: Dict[int, torch.Tensor] = {}
        self._g_lens: List[int] = []

    @staticmethod
    def from_linear(
        lin: nn.Linear, group_size: int = 64, cache_groups: int = 0
    ) -> "LinearW2A8":
        with torch.no_grad():
            w = lin.weight.detach().to(torch.float32).cpu()
            b = lin.bias.detach().to(torch.float32).cpu() if lin.bias is not None else None
        m = LinearW2A8(
            w.size(1), w.size(0), bias=(b is not None), group_size=group_size, cache_groups=cache_groups
        )
        m.quantize_from_fp(w, b)
        return m

    @torch.no_grad()
    def quantize_from_fp(self, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> None:
        out, in_f = weight.shape
        assert in_f == self.in_features and out == self.out_features
        G = self.group_size
        n_groups = (in_f + G - 1) // G
        codes_packed: List[torch.Tensor] = []
        scales = []
        zps = []
        self._g_lens = []
        w_cpu = weight.detach().to(torch.float32).cpu()
        for g in range(n_groups):
            j0 = g * G
            j1 = min((g + 1) * G, in_f)
            self._g_lens.append(j1 - j0)
            wg = w_cpu[:, j0:j1]
            wg_pad, _ = _pad_cols(wg, multiple=4)
            sc, zp = _calc_group_qparams(wg_pad)
            q = _quant_2bit_codes(wg_pad, sc, zp)
            pk = _pack2(q)
            codes_packed.append(pk)
            scales.append(sc.unsqueeze(1))
            zps.append(zp.to(torch.uint8).unsqueeze(1))
        self.w_packed = (
            torch.cat(codes_packed, dim=1).contiguous() if codes_packed else torch.empty((out, 0), dtype=torch.uint8)
        )
        self.scales = (
            torch.cat(scales, dim=1).contiguous() if scales else torch.empty((out, 0), dtype=torch.float32)
        )
        self.zps = (
            torch.cat(zps, dim=1).contiguous() if zps else torch.empty((out, 0), dtype=torch.uint8)
        )
        self.cols_padded = torch.tensor(
            int(sum(((length + 3) // 4) for length in self._g_lens)), dtype=torch.int32
        )
        if self.bias is not None and bias is not None:
            with torch.no_grad():
                self.bias.copy_(bias.to(self.bias.dtype))
        self._unpacked_cache.clear()

    def _group_slice_packed(self, g: int) -> torch.Tensor:
        g_len_real = self._g_lens[g]
        bytes_g = ((g_len_real + 3) // 4)
        start = sum(((length + 3) // 4) for length in self._g_lens[:g])
        return self.w_packed[:, start : start + bytes_g]

    def _get_centered_scaled_group(self, g: int, device: torch.device) -> torch.Tensor:
        if self.cache_groups and (g in self._unpacked_cache):
            return self._unpacked_cache[g].to(device, non_blocking=True)
        pk = self._group_slice_packed(g)
        g_len_real = self._g_lens[g]
        q = _unpack2(pk, (g_len_real + 3) // 4 * 4)[:, :g_len_real]
        zp = self.zps[:, g].to(torch.int16)[:, None]
        sc = self.scales[:, g].to(torch.float32)[:, None]
        w_g = (q.to(torch.int16) - zp).to(torch.float32) * sc
        w_g = w_g.to(device)
        if self.cache_groups:
            self._unpacked_cache[g] = w_g.detach().cpu()
        return w_g

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 2 and x.size(1) == self.in_features
        B = x.size(0)
        device = x.device
        y = torch.zeros((B, self.out_features), dtype=torch.float32, device=device)
        n_groups = len(self._g_lens)
        G = self.group_size
        for g in range(n_groups):
            j0 = g * G
            j1 = j0 + self._g_lens[g]
            xg = x[:, j0:j1]
            wg = self._get_centered_scaled_group(g, device)
            y.add_(xg @ wg.t())
        if self.bias is not None:
            y.add_(self.bias)
        return y


# ---- Model --------------------------------------------------------------------


@dataclass
class AriannaCConfig:
    block_size: int = 1024
    vocab_size: int = 256
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 512
    dropout: float = 0.0
    apply_quant: bool = True
    w2_group_size: int = 64
    w2_cache_groups: int = 0


def _make_linear(in_f: int, out_f: int, bias: bool, cfg: AriannaCConfig) -> nn.Module:
    if cfg.apply_quant:
        lin = nn.Linear(in_f, out_f, bias=bias)
        with torch.no_grad():
            lin.weight.normal_(mean=0.0, std=0.02)
            if bias:
                lin.bias.zero_()
        return LinearW2A8.from_linear(
            lin, group_size=cfg.w2_group_size, cache_groups=cfg.w2_cache_groups
        )
    else:
        return nn.Linear(in_f, out_f, bias=bias)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: AriannaCConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.key = _make_linear(config.n_embd, config.n_embd, bias=False, cfg=config)
        self.query = _make_linear(config.n_embd, config.n_embd, bias=False, cfg=config)
        self.value = _make_linear(config.n_embd, config.n_embd, bias=False, cfg=config)
        self.proj = _make_linear(config.n_embd, config.n_embd, bias=True, cfg=config)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        x_flat = x.view(B * T, C)
        k = self.key(x_flat).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q = self.query(x_flat).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.value(x_flat).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = torch.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.proj(y.view(B * T, C)).view(B, T, C)
        return self.resid_dropout(y)


class MLP(nn.Module):
    def __init__(self, config: AriannaCConfig):
        super().__init__()
        self.fc = _make_linear(config.n_embd, 4 * config.n_embd, bias=True, cfg=config)
        self.proj = _make_linear(4 * config.n_embd, config.n_embd, bias=True, cfg=config)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        h = torch.nn.functional.gelu(self.fc(x.view(B * T, C)))
        h = self.proj(h).view(B, T, C)
        return self.dropout(h)


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
    def __init__(self, config: AriannaCConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = _make_linear(config.n_embd, config.vocab_size, bias=False, cfg=config)
        self.block_size = config.block_size
        self.eval()
        torch.set_grad_enabled(False)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None):
        if idx.dim() == 1:
            idx = idx.unsqueeze(0)
            if targets is not None and targets.dim() == 1:
                targets = targets.unsqueeze(0)
        if idx.dim() != 2:
            raise ValueError("idx must be [B,T]")
        B, T = idx.size()
        if T > self.block_size:
            raise ValueError("seq too long")
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(pos)
        x = self.drop(tok_emb + pos_emb)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x.view(B * T, self.config.n_embd)).view(B, T, self.config.vocab_size)
        loss = None
        if targets is not None:
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.0,
        top_k: int = 0,
        seed: int | None = None,
    ):
        if seed is not None:
            torch.manual_seed(seed)
        if idx.dim() == 1:
            idx = idx.unsqueeze(0)
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -self.block_size :])
            logits = logits[:, -1, :]
            if temperature <= 0:
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k and top_k < logits.size(-1):
                    v, ix = torch.topk(logits, top_k)
                    probs = torch.zeros_like(logits).scatter_(1, ix, torch.softmax(v, dim=-1))
                else:
                    probs = torch.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# ---- Legacy quantization API --------------------------------------------------


@torch.no_grad()
def quantize_2bit(model: nn.Module) -> None:
    for p in model.parameters():
        if not p.is_floating_point():
            continue
        max_val = p.detach().abs().max()
        if max_val == 0:
            continue
        scale = max_val / 3.0
        q = (p / scale).round().clamp(-3, 3)
        signs = torch.sign(q)
        mags = torch.where(q.abs() > 2, torch.tensor(3.0, device=p.device), torch.tensor(1.0, device=p.device))
        p.copy_(signs * mags * scale)


__all__ = [
    "AriannaC",
    "AriannaCConfig",
    "LinearW2A8",
    "quantize_2bit",
    "_pack2",
    "_unpack2",
]
