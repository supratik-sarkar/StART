"""Deep learning sequence demo — roadmap skeleton.

This demo is intentionally a scoped placeholder: RNN/LSTM/GRU/TCN models will
be demonstrated on a genuinely sequential (synthetic time-series
classification) dataset, NOT forced onto a non-sequential tabular dataset.
Explainability will use gradient methods (Integrated Gradients / DeepLIFT /
Gradient SHAP via Captum) plus occlusion analysis — tree SHAP does not apply.

Run: python examples/deep_learning_sequence_demo.py
"""

from __future__ import annotations

from start.modeling.deep_learning import (
    SUPPORTED_ARCHITECTURES,
    captum_available,
    torch_available,
)


def main() -> None:
    print("StART deep-learning sequence demo — roadmap skeleton\n")
    print(f"torch available:  {torch_available()}   (enable with: pip install -e \".[torch]\")")
    print(f"captum available: {captum_available()}")
    print(f"planned architectures: {', '.join(SUPPORTED_ARCHITECTURES)}")
    print(
        "\nPlanned flow: synthetic time-series classification dataset -> "
        "train/test/OOS sequence split -> architecture choice -> laptop-safe "
        "training defaults (large searches reserved for GPU clusters) -> cohort "
        "metrics -> gradient-based explainability -> shock sensitivity -> "
        "evidence pipeline.\n\nNothing is trained yet by design; see "
        "src/start/modeling/deep_learning.py for the scoped roadmap."
    )


if __name__ == "__main__":
    main()
