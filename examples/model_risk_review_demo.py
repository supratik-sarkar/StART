"""Full StART model-risk review — example script.

Runs the complete operating-system flow on the built-in demo dataset (or your
own CSV via --data), printing every stage as it executes. Deterministic by
default; no key required.

    python examples/model_risk_review_demo.py
    python examples/model_risk_review_demo.py --data mydata.csv --target churned --run-dl
    python examples/model_risk_review_demo.py --agent-mode llm --llm-provider openai
"""

from __future__ import annotations

import argparse

from start.interactive_review import ReviewConfig, run_interactive_review


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default=None, help="Dataset path (omit for the demo dataset).")
    p.add_argument("--target", default=None, help="Target column (omit to let discovery propose).")
    p.add_argument("--split-strategy", default="stratified")
    p.add_argument("--architecture", default="mlp")
    p.add_argument("--activation", default="relu")
    p.add_argument("--agent-mode", default="deterministic", choices=("deterministic", "llm"))
    p.add_argument("--llm-provider", default="none")
    p.add_argument("--run-dl", action="store_true", help="Train a tabular DL model.")
    p.add_argument("--output-root", default="start_output")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    cfg = ReviewConfig(
        data_path=args.data,
        target=args.target,
        split_strategy=args.split_strategy,
        architecture_family=args.architecture,
        activation=args.activation,
        agent_mode=args.agent_mode,
        llm_provider=args.llm_provider,
        run_dl=args.run_dl,
        output_root=args.output_root,
        seed=args.seed,
    )
    run_interactive_review(cfg)


if __name__ == "__main__":
    main()
