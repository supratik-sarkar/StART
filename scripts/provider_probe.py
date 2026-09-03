#!/usr/bin/env python3
"""Non-interactive provider smoke utility.

Usage:
  python scripts/provider_probe.py --provider openai --model gpt-5-mini
  python scripts/provider_probe.py --provider anthropic
  python scripts/provider_probe.py --provider deepseek
  python scripts/provider_probe.py --provider gemini
  python scripts/provider_probe.py --provider grok
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src to sys.path if not installed
src_path = Path(__file__).resolve().parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from start.cli.provider_cli import run_provider_probe  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Finite, non-interactive live provider smoke probe.")
    parser.add_argument(
        "--provider",
        "-p",
        required=True,
        choices=["openai", "anthropic", "gemini", "deepseek", "grok"],
        help="Target provider name.",
    )
    parser.add_argument(
        "--model",
        "-m",
        default=None,
        help="Optional model ID override.",
    )
    parser.add_argument(
        "--prompt",
        default="Reply with exactly: START_PROVIDER_OK",
        help="Fixed probe prompt.",
    )
    parser.add_argument(
        "--system",
        default="You are a StART provider connectivity probe. Reply with exactly: START_PROVIDER_OK",
        help="Fixed probe system prompt.",
    )
    parser.add_argument(
        "--output-token-budget",
        "-b",
        type=int,
        default=512,
        help="Output token budget.",
    )

    args = parser.parse_args()
    code = run_provider_probe(
        provider=args.provider,
        model=args.model,
        prompt=args.prompt,
        system=args.system,
        output_token_budget=args.output_token_budget,
    )
    sys.exit(code)


if __name__ == "__main__":
    main()
