"""PyTorch network modules for StART's implemented DL architectures.

All builders return an ``nn.Module`` mapping standardized features to a single
logit (binary classification). Architectures are intentionally compact and
laptop-safe; sequence models (RNN/LSTM/GRU/TCN/Transformer/TFT) are on the
roadmap and are never forced onto tabular data.

This module imports torch lazily through the builder functions so that
``import start.modeling.dl_models`` never fails when torch is absent.
"""

from __future__ import annotations

from typing import Any

ARCHITECTURE_DESCRIPTIONS: dict[str, str] = {
    "mlp": "Feed-forward MLP with ReLU activations and dropout.",
    "leaky_relu_mlp": "Feed-forward MLP with Leaky-ReLU activations and dropout.",
    "residual_mlp": "MLP with residual (skip) connections between equal-width blocks.",
    "wide_deep": "Wide linear path plus a deep MLP path, summed at the logit.",
}


def _activation(name: str) -> Any:
    import torch.nn as nn

    return nn.LeakyReLU() if name == "leaky_relu" else nn.ReLU()


def build_mlp(n_features: int, hidden_dims: tuple[int, ...], dropout: float, activation: str):
    """Plain feed-forward MLP."""
    import torch.nn as nn

    layers: list[Any] = []
    prev = n_features
    for width in hidden_dims:
        layers += [nn.Linear(prev, width), _activation(activation), nn.Dropout(dropout)]
        prev = width
    layers.append(nn.Linear(prev, 1))
    return nn.Sequential(*layers)


def build_residual_mlp(
    n_features: int, hidden_dims: tuple[int, ...], dropout: float, activation: str
):
    """MLP with residual connections. Blocks of equal width add a skip path;
    a projection adapts dimensions when consecutive widths differ."""
    import torch
    import torch.nn as nn

    class ResidualBlock(nn.Module):
        def __init__(self, in_dim: int, out_dim: int) -> None:
            super().__init__()
            self.linear = nn.Linear(in_dim, out_dim)
            self.act = _activation(activation)
            self.drop = nn.Dropout(dropout)
            self.proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.drop(self.act(self.linear(x)) + self.proj(x))

    class ResidualMLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            blocks: list[Any] = []
            prev = n_features
            for width in hidden_dims:
                blocks.append(ResidualBlock(prev, width))
                prev = width
            self.blocks = nn.Sequential(*blocks)
            self.head = nn.Linear(prev, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.head(self.blocks(x))

    return ResidualMLP()


def build_wide_deep(
    n_features: int, hidden_dims: tuple[int, ...], dropout: float, activation: str
):
    """Wide & Deep: a wide linear path (memorization) summed with a deep MLP
    path (generalization) at the logit."""
    import torch
    import torch.nn as nn

    class WideDeep(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.wide = nn.Linear(n_features, 1)
            deep_layers: list[Any] = []
            prev = n_features
            for width in hidden_dims:
                deep_layers += [nn.Linear(prev, width), _activation(activation), nn.Dropout(dropout)]
                prev = width
            deep_layers.append(nn.Linear(prev, 1))
            self.deep = nn.Sequential(*deep_layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.wide(x) + self.deep(x)

    return WideDeep()


_BUILDERS = {
    "mlp": build_mlp,
    "leaky_relu_mlp": build_mlp,
    "residual_mlp": build_residual_mlp,
    "wide_deep": build_wide_deep,
}


def build_network(
    architecture: str,
    n_features: int,
    hidden_dims: tuple[int, ...],
    dropout: float,
    activation: str,
):
    """Dispatch to the network builder for an implemented architecture."""
    if architecture not in _BUILDERS:
        raise ValueError(f"No network builder for architecture '{architecture}'.")
    return _BUILDERS[architecture](n_features, hidden_dims, dropout, activation)
