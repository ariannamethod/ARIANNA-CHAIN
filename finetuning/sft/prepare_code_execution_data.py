import argparse
import json
import logging
import re
import subprocess
from pathlib import Path
from urllib.parse import urlparse

from rewards import format_reward, reasoning_steps_reward

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROMPT_RE = re.compile(r"#\s*prompt:(.*)", re.IGNORECASE)
SOLUTION_RE = re.compile(r"#\s*solution:(.*)", re.IGNORECASE)


def extract_tasks_from_file(path: Path):
    """Extract prompt/solution pairs from a file.

    The function searches for comment lines following the pattern::

        # prompt: <description>
        # solution: <expected answer>

    Args:
        path: Path to the source file.

    Returns:
        A list of dictionaries with ``prompt`` and ``solution`` keys.
    """

    tasks = []
    with path.open("r", encoding="utf-8") as file:
        lines = file.readlines()

    prompt = None
    for line in lines:
        prompt_match = PROMPT_RE.match(line)
        if prompt_match:
            prompt = prompt_match.group(1).strip()
            continue
        if prompt is not None:
            solution_match = SOLUTION_RE.match(line)
            if solution_match:
                solution = solution_match.group(1).strip()
                tasks.append({"prompt": prompt, "solution": solution})
                prompt = None

    return tasks


def clone_repo(url: str, workdir: Path) -> Path:
    """Clone a git repository if it does not exist."""
    repo_name = urlparse(url).path.split("/")[-1].replace(".git", "")
    dest = workdir / repo_name
    if dest.exists():
        logger.info("Repository %s already exists, skipping clone", url)
    else:
        subprocess.run(["git", "clone", url, str(dest)], check=True)
    return dest


def collect_from_repo(repo_path: Path):
    """Collect tasks from all Python files in a repository."""
    tasks = []
    for path in repo_path.rglob("*.py"):
        tasks.extend(extract_tasks_from_file(path))
    return tasks


def main():
    parser = argparse.ArgumentParser(
        description="Collect prompt/solution pairs from open repositories",
    )
    parser.add_argument(
        "--repos",
        nargs="+",
        required=True,
        help="Git repository URLs to process",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to output JSONL file",
    )
    parser.add_argument(
        "--workdir",
        default="external_repos",
        help="Directory where repositories will be cloned",
    )
    args = parser.parse_args()

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    all_tasks = []
    for repo_url in args.repos:
        repo_path = clone_repo(repo_url, workdir)
        all_tasks.extend(collect_from_repo(repo_path))

    with open(args.output, "w", encoding="utf-8") as outfile:
        for task in all_tasks:
            text = f"<think>{task['prompt']}</think>\n<answer>{task['solution']}</answer>"
            fmt_score = format_reward(text)
            steps_score = reasoning_steps_reward(text)
            task["format_reward"] = fmt_score
            task["reasoning_steps_reward"] = steps_score
            logger.info(
                "Rewards for %s: format=%.2f reasoning=%.2f",
                task["prompt"][:30],
                fmt_score,
                steps_score,
            )
            outfile.write(json.dumps(task, ensure_ascii=False) + "\n")

    logger.info("Wrote %d tasks to %s", len(all_tasks), args.output)


if __name__ == "__main__":
    main()
