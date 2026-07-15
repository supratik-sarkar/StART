"""Architecture registry: families and activations as orthogonal choices.

The historical naming baked activation into the architecture name
(``leaky_relu_mlp``). The registry separates them:

    architecture_family = mlp | residual_mlp | wide_deep | rnn | gru | lstm |
                          bi_lstm | tcn | transformer | tft | cnn
    activation          = relu | leaky_relu | gelu | tanh | selu | elu

Deprecated aliases (e.g. ``leaky_relu_mlp``) still resolve, mapping to a
(family, activation) pair and emitting a ``DeprecationWarning`` — old CLI and
notebook values keep working without abrupt removal.

Each family carries a modality (``tabular``, ``sequence``, ``vision``) and an
availability check, so callers can ask "is this implemented right now?"
without trial-and-error. Vision/sequence families that are not yet wired into
a training path report ``implemented=False`` honestly rather than pretending.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

ACTIVATIONS = ("relu", "leaky_relu", "gelu", "tanh", "selu", "elu", "swish", "mish", "sigmoid", "softplus")

# Families that have a real tabular training path today.
_TABULAR_IMPLEMENTED = {"mlp", "residual_mlp", "wide_deep", "rnn", "gru", "lstm", "bi_lstm", "cnn", "gnn", "dcn"}
# Sequence families implemented in the sequence track.
_SEQUENCE_IMPLEMENTED = {"rnn", "gru", "lstm", "bi_lstm"}
# Vision families implemented in the vision track (resnet18 gated separately).
_VISION_IMPLEMENTED = {"simple_cnn"}


@dataclass(frozen=True)
class ArchitectureSpec:
    family: str
    modality: str  # tabular | sequence | vision
    description: str
    default_activation: str = "relu"
    implemented: bool = True
    optional_backend: str | None = None  # e.g. "torchvision" for resnet18


# Canonical families.
_REGISTRY: dict[str, ArchitectureSpec] = {
    "mlp": ArchitectureSpec("mlp", "tabular", "Feed-forward MLP.", "relu", True),
    "residual_mlp": ArchitectureSpec(
        "residual_mlp", "tabular", "MLP with residual skip connections.", "relu", True
    ),
    "wide_deep": ArchitectureSpec(
        "wide_deep", "tabular", "Wide linear path + deep MLP.", "relu", True
    ),
    "rnn": ArchitectureSpec("rnn", "sequence", "Vanilla RNN classifier.", "tanh", True),
    "gru": ArchitectureSpec("gru", "sequence", "GRU classifier.", "tanh", True),
    "lstm": ArchitectureSpec("lstm", "sequence", "LSTM classifier.", "tanh", True),
    "bi_lstm": ArchitectureSpec(
        "bi_lstm", "sequence", "Bidirectional LSTM classifier.", "tanh", True
    ),

    "cnn": ArchitectureSpec(
        "cnn", "tabular", "1D Conv classifier.", "relu", True
    ),
    "gnn": ArchitectureSpec(
        "gnn", "tabular", "Graph Neural Network (GCN/GAT).", "relu", True
    ),
    "dcn": ArchitectureSpec(
        "dcn", "tabular", "Deep & Cross Network.", "relu", True
    ),
    "simple_cnn": ArchitectureSpec(
        "simple_cnn", "vision", "Compact pure-PyTorch CNN for image classification.", "relu", True
    ),
    "resnet18": ArchitectureSpec(
        "resnet18",
        "vision",
        "ResNet-18 preset (optional; requires torchvision).",
        "relu",
        True,
        optional_backend="torchvision",
    ),
    # Roadmap families: recognized, not yet wired to a training path.
    "tcn": ArchitectureSpec("tcn", "sequence", "Temporal conv net (roadmap).", "relu", False),
    "transformer": ArchitectureSpec(
        "transformer", "sequence", "Transformer encoder (roadmap).", "gelu", False
    ),
    "tft": ArchitectureSpec(
        "tft", "sequence", "Temporal Fusion Transformer (roadmap).", "gelu", False
    ),
}

# Deprecated architecture names -> (family, activation).
_DEPRECATED_ALIASES: dict[str, tuple[str, str]] = {
    "leaky_relu_mlp": ("mlp", "leaky_relu"),
}


@dataclass
class ResolvedArchitecture:
    family: str
    activation: str
    modality: str
    spec: ArchitectureSpec
    from_alias: str | None = None
    warnings_emitted: list[str] = field(default_factory=list)


def list_families(modality: str | None = None) -> list[str]:
    return [
        name
        for name, spec in _REGISTRY.items()
        if modality is None or spec.modality == modality
    ]


def torchvision_available() -> bool:
    try:
        import torchvision  # noqa: F401

        return True
    except ImportError:
        return False


def family_available(family: str) -> tuple[bool, str]:
    """Return (available, reason). Handles roadmap families and the
    torchvision-gated resnet18 preset with an explicit reason string."""
    spec = _REGISTRY.get(family)
    if spec is None:
        return False, f"Unknown family '{family}'."
    if not spec.implemented:
        return False, f"'{family}' is on the roadmap and not yet implemented."
    if spec.optional_backend == "torchvision" and not torchvision_available():
        return False, "resnet18 unavailable: torchvision not installed (pip install torchvision)."
    return True, ""


def resolve_architecture(
    architecture: str | None = None,
    *,
    activation: str | None = None,
    family: str | None = None,
) -> ResolvedArchitecture:
    """Resolve a user request into a (family, activation, modality) triple.

    Accepts either the new form (``family=...`` + ``activation=...``) or a
    legacy single ``architecture`` value. Deprecated aliases resolve with a
    ``DeprecationWarning``. Unknown activations are rejected.
    """
    emitted: list[str] = []
    from_alias: str | None = None

    # New-style: explicit family wins.
    if family is not None:
        resolved_family = family
    else:
        token = architecture or "mlp"
        if token in _DEPRECATED_ALIASES:
            resolved_family, alias_activation = _DEPRECATED_ALIASES[token]
            from_alias = token
            msg = (
                f"Architecture name '{token}' is deprecated; use "
                f"family='{resolved_family}', activation='{alias_activation}'. "
                "The alias still works for now."
            )
            warnings.warn(msg, DeprecationWarning, stacklevel=2)
            emitted.append(msg)
            if activation is None:
                activation = alias_activation
        else:
            resolved_family = token

    spec = _REGISTRY.get(resolved_family)
    if spec is None:
        raise ValueError(
            f"Unknown architecture family '{resolved_family}'. "
            f"Known: {tuple(_REGISTRY)} (or deprecated aliases {tuple(_DEPRECATED_ALIASES)})."
        )

    resolved_activation = activation or spec.default_activation
    if resolved_activation not in ACTIVATIONS:
        raise ValueError(
            f"Unknown activation '{resolved_activation}'. Known: {ACTIVATIONS}."
        )

    return ResolvedArchitecture(
        family=resolved_family,
        activation=resolved_activation,
        modality=spec.modality,
        spec=spec,
        from_alias=from_alias,
        warnings_emitted=emitted,
    )
