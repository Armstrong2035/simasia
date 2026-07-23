"""Command-line entry point: ``simasia train`` / ``simasia score``.

Loads keys from ``.env`` / ``.env.local`` (if present), reads ``simasia.toml``,
and runs the requested command.
"""

from __future__ import annotations

import argparse

from .config import build_guard, load_config, load_dotenv, run_training


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="simasia", description="Brand tone guardrail.")
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="Train a brand model from the config file.")
    train.add_argument("--config", default="simasia.toml")

    score = sub.add_parser("score", help="Score one response against a trained brand.")
    score.add_argument("--config", default="simasia.toml")
    score.add_argument("text", help="The response text to score.")

    explain = sub.add_parser(
        "explain", help="Score a response and show the closest on/off-brand samples."
    )
    explain.add_argument("--config", default="simasia.toml")
    explain.add_argument("text", help="The response text to explain.")

    args = parser.parse_args(argv)

    # Make keys in a local .env available before we build any backend.
    load_dotenv(".env")
    load_dotenv(".env.local")

    config = load_config(args.config)

    if args.command == "train":
        accuracy = run_training(config)
        print(f"Trained brand '{config['brand']['id']}'. Training accuracy: {accuracy:.3f}")
    elif args.command == "score":
        guard = build_guard(config)
        print(f"{guard.evaluate_response(args.text):.3f}")
    elif args.command == "explain":
        guard = build_guard(config)
        result = guard.explain(args.text)
        on = result["closest_on_brand"]
        off = result["closest_off_brand"]
        print(f"score:   {result['score']:.3f} ({result['verdict']})")
        print(f"on-brand  (sim {on['similarity']:.2f}): {on['text']}")
        print(f"off-brand (sim {off['similarity']:.2f}): {off['text']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
