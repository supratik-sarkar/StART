"""Deep learning module — scoped skeleton (optional, gated behind [torch]).

Roadmap (deliberately NOT fully implemented in this release):

Architectures for tabular classification
    - MLP (ReLU / Leaky-ReLU variants, dropout, batch norm)
Architectures for sequence classification (genuinely sequential data only —
RNN-family models are not forced onto non-sequential tabular datasets):
    - RNN, LSTM, GRU, temporal convolution (TCN)

Explainability roadmap (model-appropriate, beyond tree SHAP):
    - Integrated Gradients, DeepLIFT, Gradient SHAP (via Captum)
    - permutation sensitivity, occlusion analysis

Tuning note: DL hyperparameter search is expensive; default configurations
must stay laptop-safe, with large searches reserved for dedicated clusters
(Databricks GPU / future Ray backend).

Everything here degrades gracefully when torch is absent.
"""

from __future__ import annotations

from typing import Any

SUPPORTED_ARCHITECTURES = ("mlp", "leaky_relu_mlp", "rnn", "lstm", "gru", "tcn")


def torch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except ImportError:
        return False


def captum_available() -> bool:
    try:
        import captum  # noqa: F401

        return True
    except ImportError:
        return False


def build_classifier(architecture: str, **kwargs: Any) -> Any:
    """Factory stub for DL classifiers. Raises with installation guidance
    until the torch-backed implementations land."""
    if architecture not in SUPPORTED_ARCHITECTURES:
        raise ValueError(f"Unknown architecture '{architecture}'. Roadmap: {SUPPORTED_ARCHITECTURES}")
    if not torch_available():
        raise ImportError(
            "Deep learning support requires the torch extra: pip install -e \".[torch]\""
        )
    raise NotImplementedError(
        f"'{architecture}' is on the StART deep-learning roadmap; implementations land "
        "behind the [torch] extra with sequence demos on genuinely sequential data."
    )
