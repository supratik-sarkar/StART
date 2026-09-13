"""Task-aware tabular deep-learning classifier/regressor.

A single sklearn-compatible estimator that branches by task type:

    binary_classification       -> 1 logit, BCEWithLogits, sigmoid
    multiclass_classification   -> C logits, CrossEntropy, softmax
    multilabel_classification   -> K logits, BCEWithLogits per label, sigmoid
    regression / forecasting    -> 1 output, MSELoss, linear

It reuses the implemented architecture families (mlp / residual_mlp /
wide_deep) and the full activation set via the architecture registry.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from start.modeling.architecture_registry import resolve_architecture
from start.modeling.deep_learning import resolve_torch_device

TABULAR_TASKS = (
    "binary_classification",
    "multiclass_classification",
    "multilabel_classification",
    "regression",
    "forecasting",
)


class TabularDLClassifier:
    """Branches by task type; sklearn-style fit/predict/predict_proba."""

    _start_model_family = "deep_learning"

    def __init__(
        self,
        task: str = "binary_classification",
        family: str = "mlp",
        activation: str | None = None,
        hidden_dims: tuple[int, ...] = (64, 32),
        epochs: int = 10,
        batch_size: int = 128,
        learning_rate: float = 1e-3,
        dropout: float = 0.1,
        validation_fraction: float = 0.2,
        early_stopping_patience: int = 3,
        device: str | None = None,
        random_state: int = 42,
        verbose: bool = False,
        class_weight: str | None = None,
        winsorize: bool = False,
        cost_specification: dict[str, Any] | None = None,
    ) -> None:
        if task not in TABULAR_TASKS:
            raise ValueError(f"Unknown tabular task '{task}'. Known: {TABULAR_TASKS}")
        if epochs > 10:
            raise ValueError("Laptop-safe constraint: epochs must be <= 10.")
        if batch_size > 128:
            raise ValueError("Laptop-safe constraint: batch_size must be <= 128.")
        resolved = resolve_architecture(family=family, activation=activation)
        self.task = task
        self.family = resolved.family
        self.activation = resolved.activation
        self.hidden_dims = tuple(hidden_dims)
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.validation_fraction = validation_fraction
        self.early_stopping_patience = early_stopping_patience
        self.device = device
        self.random_state = random_state
        self.verbose = verbose
        self.class_weight = class_weight
        self.winsorize = winsorize
        self.cost_specification = cost_specification or {"type": "balanced"}

        self._net = None
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None
        self._medians: np.ndarray | None = None
        self._lower_bounds: np.ndarray | None = None
        self._upper_bounds: np.ndarray | None = None
        self._device_used = "cpu"
        self.classes_: np.ndarray | None = None
        self.label_columns_: list[str] | None = None  # multilabel
        self.n_outputs_ = 1
        self.history_: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
        self.best_epoch_ = 0
        self.stopped_early_ = False

    # -- sklearn protocol -------------------------------------------------- #
    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {
            "task": self.task,
            "family": self.family,
            "activation": self.activation,
            "hidden_dims": self.hidden_dims,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "dropout": self.dropout,
            "validation_fraction": self.validation_fraction,
            "early_stopping_patience": self.early_stopping_patience,
            "device": self.device,
            "random_state": self.random_state,
            "verbose": self.verbose,
            "class_weight": self.class_weight,
            "winsorize": self.winsorize,
            "cost_specification": self.cost_specification,
        }

    def set_params(self, **params: Any) -> TabularDLClassifier:
        for k, v in params.items():
            if not hasattr(self, k):
                raise ValueError(f"Unknown parameter '{k}'.")
            setattr(self, k, v)
        return self

    # -- helpers ----------------------------------------------------------- #
    @staticmethod
    def _to_numpy(X: Any) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            return X.to_numpy(dtype=np.float64)
        return np.asarray(X, dtype=np.float64)

    def _standardize(self, X: np.ndarray, fit: bool) -> np.ndarray:
        if fit:
            self._medians = np.nanmedian(X, axis=0)
            self._medians[np.isnan(self._medians)] = 0.0

        # Median Imputation
        nan_mask = np.isnan(X)
        if np.any(nan_mask):
            X = X.copy()
            for col_idx in range(X.shape[1]):
                col_nan = nan_mask[:, col_idx]
                if np.any(col_nan):
                    X[col_nan, col_idx] = self._medians[col_idx]

        # Winsorization clipping
        if self.winsorize:
            if fit:
                self._lower_bounds = np.percentile(X, 1, axis=0)
                self._upper_bounds = np.percentile(X, 99, axis=0)
            X = np.clip(X, self._lower_bounds, self._upper_bounds)

        if fit:
            self._mean = X.mean(axis=0)
            self._std = X.std(axis=0)
            self._std[self._std == 0] = 1.0
        return (X - self._mean) / self._std

    def _prepare_targets(self, y: Any):
        """Return (y_tensor_ready ndarray, n_outputs) and set class metadata."""
        if self.task in ("regression", "forecasting"):
            if isinstance(y, pd.DataFrame):
                arr = y.to_numpy(dtype=np.float32)
            else:
                arr = np.asarray(y, dtype=np.float32)
                if arr.ndim == 1:
                    arr = arr.reshape(-1, 1)
            self.n_outputs_ = arr.shape[1]
            self.classes_ = None
            self._target_mean = arr.mean(axis=0)
            self._target_std = arr.std(axis=0)
            self._target_std[self._target_std == 0] = 1.0
            arr = (arr - self._target_mean) / self._target_std
            return arr, self.n_outputs_

        if self.task == "multilabel_classification":
            if isinstance(y, pd.DataFrame):
                self.label_columns_ = list(y.columns)
                arr = y.to_numpy(dtype=np.float32)
            else:
                arr = np.asarray(y, dtype=np.float32)
                if arr.ndim == 1:
                    arr = arr.reshape(-1, 1)
                self.label_columns_ = [f"label_{i}" for i in range(arr.shape[1])]
            self.n_outputs_ = arr.shape[1]
            self.classes_ = np.array([0, 1])
            return arr, self.n_outputs_

        # single-column classification
        y_arr = np.asarray(y).reshape(-1)
        classes = np.unique(y_arr)
        self.classes_ = classes
        if self.task == "binary_classification":
            self.n_outputs_ = 1
            mapping = {c: i for i, c in enumerate(classes)}
            mapped = np.array([mapping[v] for v in y_arr], dtype=np.float32).reshape(-1, 1)
            return mapped, 1

        # multiclass
        self.n_outputs_ = len(classes)
        mapping = {c: i for i, c in enumerate(classes)}
        mapped = np.array([mapping[v] for v in y_arr], dtype=np.int64)
        return mapped, self.n_outputs_

    def _loss_fn(self, y_arr, device):
        import torch

        if self.task in ("regression", "forecasting"):
            return torch.nn.MSELoss()

        weights = None
        if self.task == "multiclass_classification":
            if self.class_weight == "balanced":
                from sklearn.utils.class_weight import compute_class_weight
                classes = np.arange(self.n_outputs_)
                weights = compute_class_weight("balanced", classes=classes, y=y_arr.ravel())
                weights = np.asarray(weights, dtype=np.float32)

            cost_spec = getattr(self, "cost_specification", None) or {"type": "balanced"}
            if cost_spec.get("type") == "critical_class":
                crit_class = cost_spec.get("critical_class")
                rel_cost = cost_spec.get("relative_cost", 5.0)
                if self.classes_ is not None:
                    if weights is None:
                        weights = np.ones(self.n_outputs_, dtype=np.float32)
                    for i, c in enumerate(self.classes_):
                        if str(c) == str(crit_class):
                            weights[i] *= rel_cost

            if weights is not None:
                return torch.nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
            return torch.nn.CrossEntropyLoss()

        if self.class_weight == "balanced" and self.task == "binary_classification":
            n_pos = float(np.sum(y_arr == 1))
            n_neg = float(np.sum(y_arr == 0))
            if n_pos > 0:
                pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32, device=device)
                return torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        return torch.nn.BCEWithLogitsLoss()

    # -- training ---------------------------------------------------------- #
    def fit(self, X: Any, y: Any) -> TabularDLClassifier:
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        X_arr = self._standardize(self._to_numpy(X), fit=True)
        y_arr, n_outputs = self._prepare_targets(y)
        from start.modeling.dl_models import build_network

        self._device_used = self.device or resolve_torch_device()
        device = torch.device(self._device_used)
        self._net = build_network(
            self.family, X_arr.shape[1], self.hidden_dims, self.dropout, self.activation, n_outputs
        ).to(device)

        # validation split
        n = len(X_arr)
        rng = np.random.default_rng(self.random_state)
        perm = rng.permutation(n)
        n_val = int(n * self.validation_fraction)
        val_idx, tr_idx = perm[:n_val], perm[n_val:]
        has_val = n_val > 0

        if self.task == "multiclass_classification":
            y_tensor = torch.tensor(y_arr, dtype=torch.long)
        else:
            y_tensor = torch.tensor(y_arr, dtype=torch.float32)
        X_tensor = torch.tensor(X_arr, dtype=torch.float32)

        train_ds = TensorDataset(X_tensor[tr_idx], y_tensor[tr_idx])
        gen = torch.Generator().manual_seed(self.random_state)
        loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, generator=gen)
        if has_val:
            xv, yv = X_tensor[val_idx].to(device), y_tensor[val_idx].to(device)

        optimizer = torch.optim.Adam(self._net.parameters(), lr=self.learning_rate)
        loss_fn = self._loss_fn(y_arr, device)
        self.history_ = {"train_loss": [], "val_loss": []}
        best_val, best_state, no_improve = float("inf"), None, 0
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
            self.history_["train_loss"].append(round(total / max(len(tr_idx), 1), 6))
            if has_val:
                self._net.eval()
                with torch.no_grad():
                    vloss = float(loss_fn(self._net(xv), yv).item())
                self.history_["val_loss"].append(round(vloss, 6))
                if vloss < best_val - 1e-5:
                    best_val, no_improve = vloss, 0
                    best_state = {k: v.detach().clone() for k, v in self._net.state_dict().items()}
                    self.best_epoch_ = epoch + 1
                else:
                    no_improve += 1
                if no_improve >= self.early_stopping_patience:
                    self.stopped_early_ = True
                    break
        if best_state is not None:
            self._net.load_state_dict(best_state)
        else:
            self.best_epoch_ = len(self.history_["train_loss"])
        return self

    # -- inference --------------------------------------------------------- #
    def _logits(self, X: Any) -> np.ndarray:
        import torch

        if self._net is None:
            raise RuntimeError("Not fitted; call fit() first.")
        X_arr = self._standardize(self._to_numpy(X), fit=False)
        device = torch.device(self._device_used)
        self._net.eval()
        with torch.no_grad():
            return self._net(torch.tensor(X_arr, dtype=torch.float32).to(device)).cpu().numpy()

    def predict_proba(self, X: Any) -> np.ndarray:
        if self.task in ("regression", "forecasting"):
            raise ValueError("predict_proba is not supported for regression/forecasting tasks.")
        logits = self._logits(X)
        if self.task == "binary_classification":
            p1 = 1.0 / (1.0 + np.exp(-logits.reshape(-1)))
            return np.column_stack([1.0 - p1, p1])
        if self.task == "multiclass_classification":
            e = np.exp(logits - logits.max(axis=1, keepdims=True))
            return e / e.sum(axis=1, keepdims=True)
        # multilabel: independent sigmoids, shape (n, K)
        return 1.0 / (1.0 + np.exp(-logits))

    def predict(self, X: Any) -> np.ndarray:
        if self.task in ("regression", "forecasting"):
            preds = self._logits(X)
            if hasattr(self, "_target_mean"):
                preds = preds * self._target_std + self._target_mean
            if self.n_outputs_ == 1:
                return preds.reshape(-1)
            return preds
        if self.task == "multilabel_classification":
            return (self.predict_proba(X) >= 0.5).astype(int)
        
        proba = self.predict_proba(X)
        idx = proba.argmax(axis=1)
        argmax_preds = np.asarray(self.classes_)[idx]
        self.last_predictions_argmax_ = argmax_preds

        # Check if cost specification is set and not balanced
        cost_spec = getattr(self, "cost_specification", None) or {"type": "balanced"}
        cost_matrix = None
        if cost_spec.get("type", "balanced") != "balanced":
            from start.modeling.cost_sensitive import cost_spec_to_matrix, validate_cost_matrix
            classes_list = [str(c) for c in self.classes_]
            cost_matrix = cost_spec_to_matrix(cost_spec, classes_list)
            if cost_matrix is not None:
                # Assertions to validate cost matrix before applying expected-cost minimization
                assert cost_matrix.shape == (len(self.classes_), len(self.classes_)), f"Cost matrix shape must be {len(self.classes_)}x{len(self.classes_)}"
                assert np.all(np.isfinite(cost_matrix)), "Cost matrix must contain only finite values"
                assert np.all(cost_matrix >= 0), "Cost matrix must contain non-negative values"
                
                # Check for critical errors from validate_cost_matrix if matrix format was directly supplied
                if cost_spec.get("type") == "matrix":
                    errors = validate_cost_matrix(cost_spec["matrix"], classes_list)
                    critical_errors = [e for e in errors if "Warning" not in e]
                    assert len(critical_errors) == 0, f"Cost matrix validation failed: {critical_errors}"

        if cost_matrix is not None:
            from start.modeling.cost_sensitive import cost_sensitive_predictions
            cost_sensitive_preds = cost_sensitive_predictions(proba, cost_matrix, self.classes_)
            self.last_predictions_cost_sensitive_ = cost_sensitive_preds
            return cost_sensitive_preds
        else:
            self.last_predictions_cost_sensitive_ = argmax_preds
            return argmax_preds

    def score(self, X: Any, y: Any) -> float:
        if self.task in ("regression", "forecasting"):
            from sklearn.metrics import r2_score
            preds = self.predict(X)
            return float(r2_score(np.asarray(y).reshape(-1), preds))
        if self.task == "multilabel_classification":
            preds = self.predict(X)
            y_arr = y.to_numpy() if isinstance(y, pd.DataFrame) else np.asarray(y)
            return float((preds == y_arr).mean())
        return float(np.mean(self.predict(X) == np.asarray(y).reshape(-1)))

    @property
    def device_used(self) -> str:
        return self._device_used
