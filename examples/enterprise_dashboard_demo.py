"""Enterprise dashboard example.

Runs the enterprise layered review on the built-in demo dataset and writes the
audit-ready dashboard (dashboard.html/.json/.md), the governance findings, the
AI-engineering control surface, and the review graph — all from one flow.

    python examples/enterprise_dashboard_demo.py
    python examples/enterprise_dashboard_demo.py --data mydata.csv --target churned --run-dl
"""

from __future__ import annotations

import argparse

from start.modeling.enterprise_orchestrator import EnterpriseReviewOrchestrator


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default=None, help="Dataset path (omit for the demo dataset).")
    p.add_argument("--target", default=None, help="Target column.")
    p.add_argument("--run-dl", action="store_true", help="Train a tabular DL model.")
    p.add_argument("--output-root", default="start_output")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if args.data:
        from start.data.loaders import load_any_tabular

        df = load_any_tabular(args.data)
        target = args.target
    else:
        from start.modeling.data import load_attrition_dataset

        df = load_attrition_dataset(seed=args.seed)
        target = args.target or "attrition"

    def show_layer(lr) -> None:
        if lr.status != "running":
            print(
                f"  {lr.name:16s} {lr.status:9s} {lr.runtime_seconds:.3f}s "
                f"findings={len(lr.findings)} artifacts={len(lr.artifacts)} "
                f"evidence={len(lr.evidence_ids)}"
            )

    print(f"Running enterprise review on {len(df)} rows (target: {target})\n")
    outcome = EnterpriseReviewOrchestrator(on_layer=show_layer).run(
        df,
        user_target=target,
        output_root=args.output_root,
        run_dl=args.run_dl,
        enterprise_mode=True,
        seed=args.seed,
    )

    s = outcome.findings_register.summary()
    print(
        f"\nReview {outcome.run_id} complete.\n"
        f"  findings: {s['total']} (Critical={s['Critical']} High={s['High']} "
        f"Medium={s['Medium']} Low={s['Low']})\n"
        f"  AI-engineering: {outcome.ai_engineering.available_count}"
        f"/{outcome.ai_engineering.total} adapters available\n"
        f"  evidence critique: {'PASSED' if outcome.critique_ok else 'FAILED'}\n"
        f"  dashboard (open in a browser): {outcome.dashboard_paths['html']}\n"
        f"  dashboard json: {outcome.dashboard_paths['json']}\n"
        f"  review graph: {outcome.graph_paths}"
    )


if __name__ == "__main__":
    main()
