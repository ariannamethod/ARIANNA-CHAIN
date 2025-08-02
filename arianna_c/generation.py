from collections import Counter
from pathlib import Path

from .model import AriannaC, AriannaCConfig
from .monitor import SelfMonitor
from .quantize import quantize_2bit
from .logger import (
    estimate_complexity_and_entropy,
    thought_logger,
)
from .tokenizer import tokenizer
from .reflection import reflect

CORE_PROMPT = (
    Path(__file__).resolve().parent.parent / "core_prompt.txt"
).read_text(encoding="utf-8")
print("core_prompt.txt loaded [OK]")


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
    """Generate a completion optionally enriched with past prompts.

    Args:
        prompt: Initial text to complete. If ``None`` the core prompt is used.
        max_new_tokens: Maximum number of tokens to generate.
        config: Optional model configuration.
        log_reasoning: Whether to return reasoning metadata.
        use_memory: Fetch similar past prompts from :class:`SelfMonitor` and
            prepend them to the provided prompt.
        memory_limit: Maximum number of historical prompts to include.

    Returns:
        The generated text. If ``log_reasoning`` is ``True`` a tuple of the text
        and a dictionary with reasoning statistics is returned instead.
    """
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
    """Iteratively alternate between ``<think>`` and ``<answer>`` phases.

    At each step the model first produces a thought and then an answer. Each
    intermediate piece of text is logged via :class:`SelfMonitor` before
    optionally continuing to the next step.

    Args:
        prompt: Initial prompt to seed the loop. If ``None`` the core prompt is
            used.
        max_steps: Maximum number of ``<think>``/``<answer>`` pairs to run.
        stop_tokens: Collection of substrings that, if generated, terminate the
            loop early.
        max_new_tokens: Maximum tokens to generate per phase.
        config: Optional model configuration.

    Returns:
        The text produced in the final ``<answer>`` phase or the accumulated
        text if the loop exits early before producing an answer.
    """

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
    """Generate text while allowing a hook for reasoning steps.

    Currently this is a light wrapper around :func:`generate_text` so that it can
    be mocked in tests and extended in the future. The function requests
    reasoning metadata from :func:`generate_text` and therefore returns a tuple
    of the generated text and the associated statistics.
    """

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
    """Generate multiple completions and return the most consistent answer.

    Args:
        prompt: Optional prompt to complete. If ``None`` the core prompt is used.
        n: Number of attempts to generate a completion.
        **kwargs: Extra arguments passed to :func:`generate_with_think`.

    Returns:
        The most frequently produced final answer. In case of a tie, the
        shortest answer is returned.
    """

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
