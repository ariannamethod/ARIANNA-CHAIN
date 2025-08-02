from __future__ import annotations

from .model import AriannaC, AriannaCConfig
from .quantize import quantize_2bit
from .tokenizer import tokenizer


def reflect(prompt: str, draft: str, max_new_tokens: int = 50, config: AriannaCConfig | None = None) -> str:
    """Critique a draft answer using the model.

    Args:
        prompt: The original prompt or question.
        draft: The draft answer to critique.
        max_new_tokens: Maximum tokens for the critique generation.
        config: Optional model configuration.

    Returns:
        A string containing the model's critique of the draft answer.
    """

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
