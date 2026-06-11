"""Laptop-safe PyTorch training utilities for StART DL model review."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


def detect_torch_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass(frozen=True)
class DLTrainingConfig:
    architecture: Literal["mlp", "residual_mlp", "wide_deep"] = "mlp"
    hidden_dim: int = 64
    epochs: int = 8
    batch_size: int = 128
    learning_rate: float = 1e-3
    dropout: float = 0.10
    seed: int = 42
    patience: int = 3


@dataclass
class DLTrainingResult:
    model: object
    history: dict[str, list[float]]
    device: str
    architecture: str


def _build_model(input_dim: int, config: DLTrainingConfig):
    import torch
    from torch import nn

    class MLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, config.hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim, config.hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(config.hidden_dim // 2, 1),
            )

        def forward(self, x):
            return self.net(x).squeeze(-1)

    class ResidualMLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.inp = nn.Linear(input_dim, config.hidden_dim)
            self.block = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.ReLU(),
                nn.Linear(config.hidden_dim, config.hidden_dim),
            )
            self.out = nn.Linear(config.hidden_dim, 1)

        def forward(self, x):
            h = self.inp(x)
            h = h + self.block(h)
            return self.out(torch.relu(h)).squeeze(-1)

    class WideDeep(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.wide = nn.Linear(input_dim, 1)
            self.deep = nn.Sequential(
                nn.Linear(input_dim, config.hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim, 1),
            )

        def forward(self, x):
            return (self.wide(x) + self.deep(x)).squeeze(-1)

    if config.architecture == "residual_mlp":
        return ResidualMLP()
    if config.architecture == "wide_deep":
        return WideDeep()
    return MLP()


def train_binary_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    config: DLTrainingConfig | None = None,
) -> DLTrainingResult:
    if not torch_available():
        raise ImportError("Deep learning demo requires torch. Install with: python -m pip install torch")

    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    config = config or DLTrainingConfig()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = detect_torch_device()

    model = _build_model(X_train.shape[1], config).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.BCEWithLogitsLoss()

    ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    loader = DataLoader(ds, batch_size=config.batch_size, shuffle=True)
    Xv = torch.tensor(X_val, device=device)
    yv = torch.tensor(y_val, device=device)
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    bad_epochs = 0

    for _epoch in range(config.epochs):
        model.train()
        losses = []
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(Xv), yv).detach().cpu())
        history["train_loss"].append(float(np.mean(losses)))
        history["val_loss"].append(val_loss)
        if val_loss + 1e-6 < best_val:
            best_val = val_loss
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= config.patience:
                break

    return DLTrainingResult(model=model, history=history, device=device, architecture=config.architecture)


def predict_proba(model: object, X: np.ndarray) -> np.ndarray:
    import torch

    device = next(model.parameters()).device  # type: ignore[attr-defined]
    model.eval()  # type: ignore[attr-defined]
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32, device=device))  # type: ignore[operator]
        return torch.sigmoid(logits).detach().cpu().numpy().astype(float)
