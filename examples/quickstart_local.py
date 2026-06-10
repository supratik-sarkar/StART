"""Local CPU-only, no-LLM quickstart for StART.

Generates a toy propensity dataset, trains a small model, runs the full
agentic review pipeline with deterministic engines, and writes a
proof-carrying Markdown report plus a tamper-evident evidence ledger.

Run:  python examples/quickstart_local.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from start import build_context, load_config, run_review
from start.evidence.ledger import EvidenceLedger
from start.reporting import render_markdown


def make_toy_propensity(n: int = 4000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "tenure_months": rng.gamma(4, 12, n).round(1),
            "monthly_spend": rng.lognormal(4.0, 0.5, n).round(2),
            "n_products": rng.integers(1, 6, n),
            "support_calls": rng.poisson(1.5, n),
        }
    )
    logit = (
        -2.0
        + 0.015 * X["tenure_months"]
        + 0.004 * X["monthly_spend"]
        + 0.35 * X["n_products"]
        - 0.25 * X["support_calls"]
    )
    X["target"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    X.loc[rng.random(n) < 0.02, "monthly_spend"] = np.nan  # mild missingness
    return X


def main() -> None:
    config = load_config("configs/local_no_llm.yaml")
    df = make_toy_propensity(seed=config.seed)
    train_df, test_df = train_test_split(
        df, test_size=0.3, random_state=config.seed, stratify=df["target"]
    )

    features = ["tenure_months", "monthly_spend", "n_products", "support_calls"]
    model = LogisticRegression(max_iter=1000)
    model.fit(train_df[features].fillna(0), train_df["target"])
    train_df = train_df.assign(score=model.predict_proba(train_df[features].fillna(0))[:, 1])
    test_df = test_df.assign(score=model.predict_proba(test_df[features].fillna(0))[:, 1])

    ctx = build_context(config, train_df, test_df, model=model)
    result = run_review(config, ctx)

    out = Path(config.output.root) / config.output.reports_dir
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / f"{result.run_id}.md"
    report_path.write_text(render_markdown(result))

    print(f"Run: {result.run_id}")
    for rec in result.evidence:
        print(f"  [{rec.status.value:>7}] {rec.test_id:<38} {rec.evidence_id}")
    print(f"Narrative critique OK: {result.critique.ok if result.critique else 'n/a'}")
    ledger = EvidenceLedger(
        Path(config.output.root) / config.output.ledger_file,
        Path(config.output.root) / config.output.evidence_store,
    )
    print(f"Ledger integrity verified: {ledger.verify()}")
    print(f"Report written to: {report_path}")


if __name__ == "__main__":
    main()
