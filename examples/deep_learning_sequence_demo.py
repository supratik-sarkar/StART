"""Deep-learning model review — terminal demo (interactive or non-interactive).

Safe default run (no prompts, no key, deterministic governance):
    python examples/deep_learning_sequence_demo.py --non-interactive \
        --architecture mlp --agent-mode deterministic

Interactive run (prompts for architecture, epochs, explainability, agent mode,
and — only in LLM mode — a hidden API key):
    python examples/deep_learning_sequence_demo.py

The same workflow powers the Jupyter and Databricks notebooks. The LLM, when
enabled, reasons only over the evidence bundle and never sees raw data;
default mode is deterministic and requires no key.
"""

from __future__ import annotations

import argparse

from start.modeling.dl_training import DL_ARCHITECTURES, DLReviewOptions, run_dl_review


def _prompt(ask, label, choices, default):
    rendered = "/".join(c.upper() if c == default else c for c in choices)
    raw = ask(f"{label} [{rendered}] (default: {default}): ").strip().lower()
    return raw if raw in choices else default


def _interactive(opts: DLReviewOptions, ask=input) -> DLReviewOptions:
    print("\nStART deep-learning model review — interactive setup\n")
    opts.architecture = _prompt(ask, "Architecture", DL_ARCHITECTURES, opts.architecture)
    epochs_raw = ask(f"Epochs (1-10, default {opts.epochs}): ").strip()
    if epochs_raw.isdigit():
        opts.epochs = max(1, min(int(epochs_raw), 10))
    opts.explain_method = _prompt(
        ask, "Explainability", ("integrated_gradients", "gradient_shap"), opts.explain_method
    )
    opts.sensitivity_cohort = _prompt(
        ask, "Sensitivity cohort", ("test", "oos", "development"), opts.sensitivity_cohort
    )
    opts.agent_mode = _prompt(ask, "Agent mode", ("deterministic", "llm"), opts.agent_mode)
    if opts.agent_mode == "llm":
        opts.llm_provider = _prompt(
            ask,
            "LLM provider",
            ("none", "openai", "anthropic", "enterprise_llm_gateway"),
            opts.llm_provider or "none",
        )
    return opts


def _resolve_key_if_needed(opts: DLReviewOptions, prompt_for_key) -> None:
    """Secure session-only key handling before an LLM run; deterministic
    fallback (explicit) when no key is available."""
    if opts.agent_mode != "llm":
        return
    from start.providers.keys import ensure_provider_key, key_required

    if not key_required(opts.llm_provider):
        return
    status = ensure_provider_key(opts.llm_provider, prompt_for_key=prompt_for_key)
    if not status.ok:
        print(
            f"WARNING: {status.env_var} unavailable and prompting disabled; "
            "falling back to deterministic agent review explicitly."
        )
        opts.agent_mode = "deterministic"
        opts.llm_provider = ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--non-interactive", action="store_true", help="Run with safe defaults.")
    parser.add_argument("--architecture", default="mlp", choices=DL_ARCHITECTURES)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--agent-mode", default="deterministic", choices=("deterministic", "llm"))
    parser.add_argument(
        "--llm-provider",
        default="",
        choices=("", "none", "openai", "anthropic", "enterprise_llm_gateway"),
    )
    parser.add_argument(
        "--prompt-for-key",
        dest="prompt_for_key",
        action="store_true",
        default=None,
        help="Securely prompt for a missing API key.",
    )
    parser.add_argument("--no-prompt-for-key", dest="prompt_for_key", action="store_false")
    parser.add_argument(
        "--sensitivity-cohort", default="test", choices=("test", "oos", "development")
    )
    parser.add_argument(
        "--explain-method",
        default="integrated_gradients",
        choices=("integrated_gradients", "gradient_shap"),
    )
    parser.add_argument("--data-source", default="demo", choices=("demo", "files"))
    parser.add_argument("--train")
    parser.add_argument("--test")
    parser.add_argument("--oos")
    parser.add_argument("--target", default=None)
    parser.add_argument("--output-root", default="start_output")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    opts = DLReviewOptions(
        architecture=args.architecture,
        epochs=max(1, min(args.epochs, 10)),
        batch_size=min(args.batch_size, 128),
        learning_rate=args.learning_rate,
        agent_mode=args.agent_mode,
        llm_provider=args.llm_provider,
        sensitivity_cohort=args.sensitivity_cohort,
        explain_method=args.explain_method,
        data_source=args.data_source,
        train_path=args.train,
        test_path=args.test,
        oos_path=args.oos,
        output_root=args.output_root,
        seed=args.seed,
    )
    if args.target:
        opts.target_column = args.target
    if not args.non_interactive:
        opts = _interactive(opts)
    _resolve_key_if_needed(opts, args.prompt_for_key)
    run_dl_review(opts)


if __name__ == "__main__":
    main()
