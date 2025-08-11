import torch


class TokenTensor(torch.Tensor):
    """Tensor subclass whose ``len`` returns token count for 2D inputs."""

    @staticmethod
    def wrap(t: torch.Tensor) -> "TokenTensor":
        return torch.Tensor._make_subclass(TokenTensor, t, require_grad=t.requires_grad)

    def __len__(self) -> int:  # pragma: no cover - thin wrapper
        return int(self.shape[-1])


class ByteTokenizer:
    vocab_size: int = 256

    def encode(self, text: str) -> torch.Tensor:
        arr = list(text.encode("utf-8", errors="replace"))
        t = torch.tensor(arr, dtype=torch.long).unsqueeze(0)  # [1,T]
        return TokenTensor.wrap(t)

    def decode(self, tokens: torch.Tensor) -> str:
        arr = [int(x) for x in tokens.reshape(-1).tolist()]
        return bytes(arr).decode("utf-8", errors="replace")


tokenizer = ByteTokenizer()

__all__ = ["TokenTensor", "ByteTokenizer", "tokenizer"]
