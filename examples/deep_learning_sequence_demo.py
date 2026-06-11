"""StART deep-learning model review demo.

This is a laptop-safe binary classification DL workflow. It intentionally uses
an MLP-family tabular model for the v0.5 demo; RNN/LSTM/GRU/TCN remain roadmap
for genuinely sequential data.

Run:
    python examples/deep_learning_sequence_demo.py
"""

from __future__ import annotations

from rich.console import Console
from rich.table import Table

from start.modeling.deep_learning import IMPLEMENTED_ARCHITECTURES, run_deep_learning_review
from start.modeling.dl_training import torch_available

console = Console()


def _print_metrics(result) -> None:
    table = Table(title="Deep Learning Cohort Metrics")
    for col in ["cohort", "auc_roc", "accuracy", "precision", "recall", "f1", "top_10_lift", "brier", "ece"]:
        table.add_column(col)
    cols = ["cohort", "auc_roc", "accuracy", "precision", "recall", "f1", "top_10_lift", "brier", "ece"]
    for row in result.metrics.to_dict(orient="records"):
        table.add_row(*[str(row[c]) if c == "cohort" else f"{row[c]:.4f}" for c in cols])
    console.print(table)


def main() -> None:
    console.print("StART DL model review demo — laptop-safe MLP workflow\n")
    console.print(f"torch available: {torch_available()}")
    console.print(f"implemented architectures: {', '.join(IMPLEMENTED_ARCHITECTURES)}")
    if not torch_available():
        console.print("Install torch first: python -m pip install torch")
        raise SystemExit(2)

    result = run_deep_learning_review(architecture="mlp", epochs=8, agent_mode="deterministic")
    _print_metrics(result)
    console.print("\nEvidence:")
    for ev in result.evidence:
        console.print(f"  [{ev.status:>4}] {ev.test_id:<38} {ev.evidence_id} — {ev.summary}")
    console.print("\nFigures:")
    for fig in result.figure_paths:
        console.print(f"  {fig}")
    console.print(f"\nReport: {result.report_path}")


if __name__ == "__main__":
    main()
