import torch
from arianna_chain import _pack2, _unpack2, LinearW2A8


def test_pack_unpack_roundtrip() -> None:
    torch.manual_seed(0)
    codes = torch.randint(0, 4, (3, 16), dtype=torch.uint8)
    packed = _pack2(codes)
    unpacked = _unpack2(packed, codes.size(-1))
    assert torch.equal(unpacked, codes)


def test_linear_w2a8_from_linear_close() -> None:
    torch.manual_seed(0)
    lin = torch.nn.Linear(16, 8, bias=True)
    lw = LinearW2A8.from_linear(lin)
    x = torch.randn(4, 16)
    ref = lin(x)
    out = lw(x)
    err = (ref - out).abs()
    assert err.max().item() < 0.6
    assert err.mean().item() < 0.2
