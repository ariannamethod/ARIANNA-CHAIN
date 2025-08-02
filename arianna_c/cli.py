import argparse

from .generation import generate_consistent_text, generate_text, reason_loop
from .model import AriannaCConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Arianna-C text generation")
    parser.add_argument("prompt", nargs="?", help="prompt to complete")
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--verbose", action="store_true", help="show reasoning log")
    parser.add_argument(
        "--consistency",
        type=int,
        default=1,
        help="number of attempts to ensure answer consistency",
    )
    parser.add_argument(
        "--reflect",
        action="store_true",
        help="enable self-verification through reflection",
    )
    parser.add_argument(
        "--use-memory",
        action="store_true",
        help="prepend similar past prompts from memory",
    )
    parser.add_argument("--max-steps", type=int, default=0, help="max reasoning steps")
    parser.add_argument(
        "--stop-token",
        action="append",
        default=[],
        help="token that halts the reasoning loop; can be used multiple times",
    )
    args = parser.parse_args()

    config = AriannaCConfig(vocab_size=256)
    if args.max_steps or args.stop_token:
        loop_kwargs: dict[str, object] = {
            "max_new_tokens": args.max_new_tokens,
            "config": config,
        }
        if args.max_steps:
            loop_kwargs["max_steps"] = args.max_steps
        if args.stop_token:
            loop_kwargs["stop_tokens"] = tuple(args.stop_token)
        result = reason_loop(args.prompt, **loop_kwargs)
        print(result)
    elif args.consistency > 1:
        result = generate_consistent_text(
            args.prompt,
            n=args.consistency,
            max_new_tokens=args.max_new_tokens,
            config=config,
            self_reflect=args.reflect,
            use_memory=args.use_memory,
        )
        print(result)
    else:
        result = generate_text(
            args.prompt,
            max_new_tokens=args.max_new_tokens,
            config=config,
            log_reasoning=args.verbose,
            self_reflect=args.reflect,
            use_memory=args.use_memory,
        )
        if args.verbose:
            text, meta = result
            print(text)
            print(
                f"LOG@{meta['timestamp']} | Complexity: {meta['complexity']} | Entropy: {meta['entropy']:.2f}"
            )
        else:
            print(result)


if __name__ == "__main__":
    main()
