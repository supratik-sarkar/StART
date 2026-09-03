"""PyTorch network modules for StART's implemented DL architectures.

All builders return an ``nn.Module`` mapping standardized features to a single
logit (binary classification) or multiple outputs. Architectures are intentionally
compact and laptop-safe.

This module imports torch lazily through the builder functions so that
``import start.modeling.dl_models`` never fails when torch is absent.
"""

from __future__ import annotations

from typing import Any

ARCHITECTURE_DESCRIPTIONS: dict[str, str] = {
    "mlp": "Feed-forward MLP with activations and dropout.",
    "leaky_relu_mlp": "Feed-forward MLP with Leaky-ReLU activations and dropout.",
    "residual_mlp": "MLP with residual (skip) connections between equal-width blocks.",
    "wide_deep": "Wide linear path plus a deep MLP path, summed at the logit.",
    "rnn": "Vanilla RNN tabular classifier.",
    "lstm": "LSTM tabular classifier.",
    "gru": "GRU tabular classifier.",
    "bi_lstm": "Bidirectional LSTM tabular classifier.",
    "cnn": "1D Convolutional Neural Network tabular classifier.",
    "gnn": "Graph Neural Network batch-relational classifier.",
    "dcn": "Deep & Cross Network feature interaction model.",
}


ACTIVATIONS = ("relu", "leaky_relu", "gelu", "tanh", "selu", "elu", "swish", "mish", "sigmoid", "softplus")


def _activation(name: str) -> Any:
    import torch.nn as nn

    table = {
        "relu": nn.ReLU,
        "leaky_relu": nn.LeakyReLU,
        "gelu": nn.GELU,
        "tanh": nn.Tanh,
        "selu": nn.SELU,
        "elu": nn.ELU,
        "swish": nn.SiLU,
        "mish": nn.Mish,
        "sigmoid": nn.Sigmoid,
        "softplus": nn.Softplus,
    }
    if name not in table:
        raise ValueError(f"Unknown activation '{name}'. Known: {tuple(table)}")
    return table[name]()


def build_mlp(
    n_features: int, hidden_dims: tuple[int, ...], dropout: float, activation: str, n_outputs: int = 1
):
    """Plain feed-forward MLP."""
    import torch.nn as nn

    layers: list[Any] = []
    prev = n_features
    for width in hidden_dims:
        layers += [nn.Linear(prev, width), _activation(activation), nn.Dropout(dropout)]
        prev = width
    layers.append(nn.Linear(prev, n_outputs))
    return nn.Sequential(*layers)


def build_residual_mlp(
    n_features: int, hidden_dims: tuple[int, ...], dropout: float, activation: str, n_outputs: int = 1
):
    """MLP with residual connections."""
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
            self.head = nn.Linear(prev, n_outputs)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.head(self.blocks(x))

    return ResidualMLP()


def build_wide_deep(
    n_features: int, hidden_dims: tuple[int, ...], dropout: float, activation: str, n_outputs: int = 1
):
    """Wide & Deep architecture."""
    import torch
    import torch.nn as nn

    class WideDeep(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.wide = nn.Linear(n_features, n_outputs)
            deep_layers: list[Any] = []
            prev = n_features
            for width in hidden_dims:
                deep_layers += [nn.Linear(prev, width), _activation(activation), nn.Dropout(dropout)]
                prev = width
            deep_layers.append(nn.Linear(prev, n_outputs))
            self.deep = nn.Sequential(*deep_layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.wide(x) + self.deep(x)

    return WideDeep()


def build_rnn(
    n_features: int, hidden_dims: tuple[int, ...], dropout: float, activation: str, n_outputs: int = 1
):
    """Tabular RNN architecture."""
    import torch
    import torch.nn as nn

    class RecurrentTabular(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            hidden_size = hidden_dims[0]
            self.rnn = nn.RNN(
                input_size=n_features,
                hidden_size=hidden_size,
                num_layers=1,
                batch_first=True,
                nonlinearity="relu" if activation == "relu" else "tanh",
            )
            layers: list[Any] = []
            prev = hidden_size
            for width in hidden_dims[1:]:
                layers += [nn.Linear(prev, width), _activation(activation), nn.Dropout(dropout)]
                prev = width
            layers.append(nn.Linear(prev, n_outputs))
            self.mlp = nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x_seq = x.unsqueeze(1)
            out, _ = self.rnn(x_seq)
            last = out[:, -1, :]
            return self.mlp(last)

    return RecurrentTabular()


def build_lstm(
    n_features: int, hidden_dims: tuple[int, ...], dropout: float, activation: str, n_outputs: int = 1
):
    """Tabular LSTM architecture."""
    import torch
    import torch.nn as nn

    class LSTMTabular(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            hidden_size = hidden_dims[0]
            self.lstm = nn.LSTM(
                input_size=n_features, hidden_size=hidden_size, num_layers=1, batch_first=True
            )
            layers: list[Any] = []
            prev = hidden_size
            for width in hidden_dims[1:]:
                layers += [nn.Linear(prev, width), _activation(activation), nn.Dropout(dropout)]
                prev = width
            layers.append(nn.Linear(prev, n_outputs))
            self.mlp = nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x_seq = x.unsqueeze(1)
            out, _ = self.lstm(x_seq)
            last = out[:, -1, :]
            return self.mlp(last)

    return LSTMTabular()


def build_gru(
    n_features: int, hidden_dims: tuple[int, ...], dropout: float, activation: str, n_outputs: int = 1
):
    """Tabular GRU architecture."""
    import torch
    import torch.nn as nn

    class GRUTabular(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            hidden_size = hidden_dims[0]
            self.gru = nn.GRU(input_size=n_features, hidden_size=hidden_size, num_layers=1, batch_first=True)
            layers: list[Any] = []
            prev = hidden_size
            for width in hidden_dims[1:]:
                layers += [nn.Linear(prev, width), _activation(activation), nn.Dropout(dropout)]
                prev = width
            layers.append(nn.Linear(prev, n_outputs))
            self.mlp = nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x_seq = x.unsqueeze(1)
            out, _ = self.gru(x_seq)
            last = out[:, -1, :]
            return self.mlp(last)

    return GRUTabular()


def build_bi_lstm(
    n_features: int, hidden_dims: tuple[int, ...], dropout: float, activation: str, n_outputs: int = 1
):
    """Tabular Bidirectional LSTM architecture."""
    import torch
    import torch.nn as nn

    class BiLSTMTabular(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            hidden_size = hidden_dims[0]
            self.lstm = nn.LSTM(
                input_size=n_features,
                hidden_size=hidden_size,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
            )
            layers: list[Any] = []
            prev = hidden_size * 2
            for width in hidden_dims[1:]:
                layers += [nn.Linear(prev, width), _activation(activation), nn.Dropout(dropout)]
                prev = width
            layers.append(nn.Linear(prev, n_outputs))
            self.mlp = nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x_seq = x.unsqueeze(1)
            out, _ = self.lstm(x_seq)
            last = out[:, -1, :]
            return self.mlp(last)

    return BiLSTMTabular()


def build_cnn(
    n_features: int, hidden_dims: tuple[int, ...], dropout: float, activation: str, n_outputs: int = 1
):
    """Tabular 1D Convolutional Neural Network."""
    import torch
    import torch.nn as nn

    class CNNTabular(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=3, padding=1)
            self.act = _activation(activation)
            self.pool = nn.AdaptiveAvgPool1d(1)

            layers: list[Any] = []
            prev = 8
            for width in hidden_dims:
                layers += [nn.Linear(prev, width), _activation(activation), nn.Dropout(dropout)]
                prev = width
            layers.append(nn.Linear(prev, n_outputs))
            self.mlp = nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x_conv = x.unsqueeze(1)
            out = self.pool(self.act(self.conv(x_conv)))
            last = out.squeeze(2)
            return self.mlp(last)

    return CNNTabular()


def build_gnn(
    n_features: int, hidden_dims: tuple[int, ...], dropout: float, activation: str, n_outputs: int = 1
):
    """Tabular dynamic relational Graph Neural Network (GCN/GAT similarity aggregation)."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class GNNTabular(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            hidden_size = hidden_dims[0]
            self.proj = nn.Linear(n_features, hidden_size)
            self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            self.act = _activation(activation)
            self.drop = nn.Dropout(dropout)

            layers: list[Any] = []
            prev = hidden_size
            for width in hidden_dims[1:]:
                layers += [nn.Linear(prev, width), _activation(activation), nn.Dropout(dropout)]
                prev = width
            layers.append(nn.Linear(prev, n_outputs))
            self.mlp = nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = self.act(self.drop(self.proj(x)))
            # Compute dynamic batch-relational graph structure
            q = self.q_proj(h)
            k = self.k_proj(h)
            scores = torch.matmul(q, k.transpose(0, 1)) / (h.shape[-1] ** 0.5)
            adj = F.softmax(scores, dim=-1)

            # GCN-style aggregation
            h_graph = torch.matmul(adj, h)
            h = self.drop(self.act(h_graph) + h)
            return self.mlp(h)

    return GNNTabular()


def build_dcn(
    n_features: int, hidden_dims: tuple[int, ...], dropout: float, activation: str, n_outputs: int = 1
):
    """Deep & Cross Network for tabular data."""
    import torch
    import torch.nn as nn

    class CrossNetwork(nn.Module):
        def __init__(self, input_dim: int, num_layers: int = 2) -> None:
            super().__init__()
            self.num_layers = num_layers
            self.weights = nn.ParameterList(
                [nn.Parameter(torch.randn(input_dim, 1) * 0.01) for _ in range(num_layers)]
            )
            self.biases = nn.ParameterList(
                [nn.Parameter(torch.zeros(input_dim, 1)) for _ in range(num_layers)]
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x0 = x.unsqueeze(2)
            xl = x0
            for i in range(self.num_layers):
                dot = torch.matmul(xl.transpose(1, 2), self.weights[i])
                xl = torch.matmul(x0, dot).squeeze(2) + self.biases[i].squeeze(1) + xl.squeeze(2)
                xl = xl.unsqueeze(2)
            return xl.squeeze(2)

    class DCNTabular(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.cross = CrossNetwork(n_features, num_layers=2)
            deep_layers: list[Any] = []
            prev = n_features
            for width in hidden_dims:
                deep_layers += [nn.Linear(prev, width), _activation(activation), nn.Dropout(dropout)]
                prev = width
            self.deep = nn.Sequential(*deep_layers)
            self.head = nn.Linear(n_features + prev, n_outputs)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x_cross = self.cross(x)
            x_deep = self.deep(x)
            combined = torch.cat([x_cross, x_deep], dim=1)
            return self.head(combined)

    return DCNTabular()


_BUILDERS = {
    "mlp": build_mlp,
    "leaky_relu_mlp": build_mlp,
    "residual_mlp": build_residual_mlp,
    "wide_deep": build_wide_deep,
    "rnn": build_rnn,
    "lstm": build_lstm,
    "gru": build_gru,
    "bi_lstm": build_bi_lstm,
    "cnn": build_cnn,
    "gnn": build_gnn,
    "dcn": build_dcn,
}


def build_network(
    architecture: str,
    n_features: int,
    hidden_dims: tuple[int, ...],
    dropout: float,
    activation: str,
    n_outputs: int = 1,
):
    """Dispatch to the network builder for an implemented architecture."""
    if architecture not in _BUILDERS:
        raise ValueError(f"No network builder for architecture '{architecture}'.")
    return _BUILDERS[architecture](n_features, hidden_dims, dropout, activation, n_outputs)
