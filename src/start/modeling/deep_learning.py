"""Deep learning for StART — real, laptop-safe MLP plus a scoped roadmap.

Implemented now (optional, behind the [torch] extra):
    - TorchMLPClassifier: a sklearn-compatible PyTorch MLP for binary
      classification. CPU and Apple-MPS compatible, no GPU required;
      defaults are laptop-safe (epochs <= 10, batch size <= 128) and a full
      train/evaluate cycle completes in well under a minute on the demo data.
    - Integrated Gradients global attribution via Captum (optional; honest
      permutation fallback when Captum is absent — never claimed otherwise,
      and SHAP is never claimed for DL models).

Roadmap (deliberately NOT implemented yet): RNN / LSTM / GRU / TCN on
genuinely sequential data — sequence models are not forced onto tabular
datasets — with DeepLIFT, Gradient SHAP, and occlusion analysis. DL
hyperparameter searches stay laptop-safe by default; large searches belong
on dedicated clusters.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

SUPPORTED_ARCHITECTURES = (
    "mlp",
    "leaky_relu_mlp",
    "residual_mlp",
    "wide_deep",
    "rnn",
    "lstm",
    "gru",
    "tcn",
    "transformer",
    "tft",
)
_IMPLEMENTED = {"mlp", "leaky_relu_mlp", "residual_mlp", "wide_deep"}


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


def resolve_torch_device() -> str:
    """CUDA -> Apple MPS -> CPU, mirroring start.providers.compute."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class TorchMLPClassifier:
    """sklearn-compatible PyTorch MLP for binary classification.

    Laptop-safe by construction: defaults of 10 epochs and batch size 128,
    two hidden layers, internal feature standardization, deterministic
    seeding, and automatic CUDA/MPS/CPU device selection. Implements
    get_params/set_params/fit/predict_proba/predict so the rest of StART
    (metrics, evidence engines, sensitivity shocks, sklearn tooling) treats
    it exactly like a classical model.
    """

    _start_model_family = "deep_learning"

    def __init__(
        self,
        architecture: str = "mlp",
        hidden_dims: tuple[int, ...] = (64, 32),
        epochs: int = 10,
        batch_size: int = 128,
        learning_rate: float = 1e-3,
        dropout: float = 0.1,
        activation: str = "relu",  # relu | leaky_relu
        validation_fraction: float = 0.2,
        early_stopping_patience: int = 3,
        device: str | None = None,
        random_state: int = 42,
        verbose: bool = False,
    ) -> None:
        if epochs > 10:
            raise ValueError("Laptop-safe constraint: default-configurable epochs must be <= 10.")
        if batch_size > 128:
            raise ValueError("Laptop-safe constraint: batch_size must be <= 128.")
        self.architecture = architecture
        if architecture == "leaky_relu_mlp":
            activation = "leaky_relu"
        self.hidden_dims = tuple(hidden_dims)
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.activation = activation
        self.validation_fraction = validation_fraction
        self.early_stopping_patience = early_stopping_patience
        self.device = device
        self.random_state = random_state
        self.verbose = verbose
        self._net = None
        self.classes_ = np.array([0, 1])  # sklearn classifier contract
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None
        self._device_used: str = "cpu"
        # populated by fit() for training diagnostics
        self.history_: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
        self.best_epoch_: int = 0
        self.stopped_early_: bool = False

    # -- sklearn protocol -------------------------------------------------- #
    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {
            "architecture": self.architecture,
            "hidden_dims": self.hidden_dims,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "dropout": self.dropout,
            "activation": self.activation,
            "validation_fraction": self.validation_fraction,
            "early_stopping_patience": self.early_stopping_patience,
            "device": self.device,
            "random_state": self.random_state,
            "verbose": self.verbose,
        }

    def set_params(self, **params: Any) -> TorchMLPClassifier:
        for key, value in params.items():
            if not hasattr(self, key):
                raise ValueError(f"Unknown parameter '{key}' for TorchMLPClassifier.")
            setattr(self, key, value)
        return self

    # -- internals --------------------------------------------------------- #
    def _build_net(self, n_features: int):
        from start.modeling.dl_models import build_network

        return build_network(
            self.architecture, n_features, self.hidden_dims, self.dropout, self.activation
        )

    def _standardize(self, X: np.ndarray, fit: bool) -> np.ndarray:
        if fit:
            self._mean = X.mean(axis=0)
            self._std = X.std(axis=0)
            self._std[self._std == 0] = 1.0
        assert self._mean is not None and self._std is not None
        return (X - self._mean) / self._std

    @staticmethod
    def _to_numpy(X: Any) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            return X.to_numpy(dtype=np.float64)
        return np.asarray(X, dtype=np.float64)

    # -- training / inference ---------------------------------------------- #
    def fit(self, X: Any, y: Any) -> TorchMLPClassifier:
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        X_arr = self._standardize(self._to_numpy(X), fit=True)
        y_arr = np.asarray(y, dtype=np.float32).reshape(-1, 1)
        self._device_used = self.device or resolve_torch_device()
        device = torch.device(self._device_used)

        # internal train/validation split for learning curves + early stopping
        n = len(X_arr)
        rng = np.random.default_rng(self.random_state)
        perm = rng.permutation(n)
        n_val = int(n * self.validation_fraction) if self.validation_fraction else 0
        val_idx, tr_idx = perm[:n_val], perm[n_val:]
        has_val = n_val > 0

        self._net = self._build_net(X_arr.shape[1]).to(device)
        optimizer = torch.optim.Adam(self._net.parameters(), lr=self.learning_rate)
        loss_fn = torch.nn.BCEWithLogitsLoss()

        train_ds = TensorDataset(
            torch.tensor(X_arr[tr_idx], dtype=torch.float32),
            torch.tensor(y_arr[tr_idx]),
        )
        generator = torch.Generator().manual_seed(self.random_state)
        loader = DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=True, generator=generator
        )
        if has_val:
            xv = torch.tensor(X_arr[val_idx], dtype=torch.float32).to(device)
            yv = torch.tensor(y_arr[val_idx]).to(device)

        self.history_ = {"train_loss": [], "val_loss": []}
        best_val = float("inf")
        best_state: dict[str, Any] | None = None
        epochs_no_improve = 0
        self.stopped_early_ = False

        for epoch in range(self.epochs):
            self._net.train()
            total = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = loss_fn(self._net(xb), yb)
                loss.backward()
                optimizer.step()
                total += float(loss.item()) * len(xb)
            train_loss = total / max(len(tr_idx), 1)
            self.history_["train_loss"].append(round(train_loss, 6))

            if has_val:
                self._net.eval()
                with torch.no_grad():
                    val_loss = float(loss_fn(self._net(xv), yv).item())
                self.history_["val_loss"].append(round(val_loss, 6))
                if val_loss < best_val - 1e-5:
                    best_val = val_loss
                    best_state = {k: v.detach().clone() for k, v in self._net.state_dict().items()}
                    self.best_epoch_ = epoch + 1
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                if self.verbose:
                    print(f"epoch {epoch + 1}/{self.epochs} train={train_loss:.4f} val={val_loss:.4f}")
                if epochs_no_improve >= self.early_stopping_patience:
                    self.stopped_early_ = True
                    break
            elif self.verbose:
                print(f"epoch {epoch + 1}/{self.epochs} train={train_loss:.4f}")

        if best_state is not None:
            self._net.load_state_dict(best_state)
        else:
            self.best_epoch_ = len(self.history_["train_loss"])
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        import torch

        if self._net is None:
            raise RuntimeError("TorchMLPClassifier is not fitted; call fit() first.")
        X_arr = self._standardize(self._to_numpy(X), fit=False)
        device = torch.device(self._device_used)
        self._net.eval()
        with torch.no_grad():
            logits = self._net(torch.tensor(X_arr, dtype=torch.float32).to(device))
            p1 = torch.sigmoid(logits).cpu().numpy().reshape(-1)
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X: Any) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def score(self, X: Any, y: Any) -> float:
        """Accuracy, matching the sklearn classifier contract (needed by
        sklearn tooling such as permutation_importance)."""
        return float(np.mean(self.predict(X) == np.asarray(y)))

    @property
    def device_used(self) -> str:
        return self._device_used


def build_classifier(architecture: str, **kwargs: Any) -> Any:
    """Factory for DL classifiers. MLP variants are real; sequence models
    raise with roadmap guidance (never forced onto tabular data)."""
    if architecture not in SUPPORTED_ARCHITECTURES:
        raise ValueError(
            f"Unknown architecture '{architecture}'. Roadmap: {SUPPORTED_ARCHITECTURES}"
        )
    if not torch_available():
        raise ImportError(
            "Deep learning support requires the torch extra: pip install -e \".[torch]\""
        )
    if architecture in _IMPLEMENTED:
        return TorchMLPClassifier(architecture=architecture, **kwargs)
    raise NotImplementedError(
        f"'{architecture}' is on the StART deep-learning roadmap: sequence models "
        "will be demonstrated on genuinely sequential data, not tabular datasets."
    )


def integrated_gradients_importance(
    model: TorchMLPClassifier,
    X: pd.DataFrame,
    *,
    n_samples: int = 200,
    seed: int = 42,
) -> tuple[str, list[tuple[str, float]], str]:
    """Global attribution for the MLP via Captum Integrated Gradients.

    Returns (method, ranked importance, note). Degrades honestly: if Captum
    or torch is unavailable, or the model is not a fitted TorchMLPClassifier,
    returns method='unavailable' with the reason — never a fabricated result
    and never a SHAP claim."""
    if not torch_available() or not captum_available():
        return (
            "unavailable",
            [],
            "Captum/torch not installed (pip install -e \".[torch]\"); "
            "use permutation importance instead.",
        )
    if not isinstance(model, TorchMLPClassifier) or model._net is None:
        return ("unavailable", [], "Integrated Gradients requires a fitted TorchMLPClassifier.")

    import torch
    from captum.attr import IntegratedGradients

    sample = X.sample(n=min(n_samples, len(X)), random_state=seed)
    X_arr = model._standardize(model._to_numpy(sample), fit=False)
    device = torch.device(model.device_used)
    inputs = torch.tensor(X_arr, dtype=torch.float32, device=device, requires_grad=True)
    baseline = torch.zeros_like(inputs)
    model._net.eval()
    ig = IntegratedGradients(lambda t: model._net(t))
    attributions = ig.attribute(inputs, baselines=baseline, n_steps=32)
    mean_abs = attributions.abs().mean(dim=0).detach().cpu().numpy()
    order = np.argsort(mean_abs)[::-1]
    ranked = [(str(sample.columns[i]), round(float(mean_abs[i]), 6)) for i in order]
    return ("integrated_gradients", ranked, "")
