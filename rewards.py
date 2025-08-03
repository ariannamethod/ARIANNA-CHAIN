import re


def format_reward(text: str) -> float:
    """Return 1.0 if text contains single <think>/<answer> blocks in order.

    The function checks that the response includes exactly one <think>...</think>
    block followed by one <answer>...</answer> block. If the format is not
    respected, ``0.0`` is returned.
    """
    think_blocks = re.findall(r"<think>(.*?)</think>", text, flags=re.DOTALL)
    answer_blocks = re.findall(r"<answer>(.*?)</answer>", text, flags=re.DOTALL)
    if len(think_blocks) != 1 or len(answer_blocks) != 1:
        return 0.0
    think_pos = text.find("<think>")
    answer_pos = text.find("<answer>")
    return 1.0 if 0 <= think_pos < answer_pos else 0.0


def reasoning_steps_reward(text: str) -> float:
    """Return 1.0 if at least three numbered/bulleted steps exist in <think>.

    The function inspects the content of the <think> block and counts lines
    starting with a number, ``-`` or ``*``. A reward of ``1.0`` is given when at
    least three such lines are found, otherwise ``0.0``.
    """
    match = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL)
    if not match:
        return 0.0
    think_content = match.group(1)
    lines = [line.strip() for line in think_content.splitlines()]
    count = 0
    for line in lines:
        if re.match(r"^(\d+\.\s+|-\s+|\*\s+)", line):
            count += 1
    return 1.0 if count >= 3 else 0.0
