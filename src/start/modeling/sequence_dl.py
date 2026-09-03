"""Sequence DL classifier, metrics, explainability, and robustness.

A trainable recurrent classifier over (n, timesteps, features) tensors with
the laptop-safe constraints, internal validation split, early stopping, and
device routing. Sequence-specific diagnostics:

    metrics       - AUC/accuracy/precision/recall/F1 on held-out sequences
    explainability- per-timestep and per-feature gradient saliency (which
                    parts of the sequence drove the prediction)
    robustness    - Gaussian input noise, time-warp (temporal jitter), and
                    feature-dropout drift tables
"""

from __future__ import annotations

from typing import Any

import numpy as np

from start.modeling.deep_learning import resolve_torch_device
from start.modeling.sequence_models import SEQUENCE_FAMILIES, build_sequence_network


class SequenceClassifier:
    """sklearn-style recurrent classifier for sequence binary classification."""

    _start_model_family = "deep_learning_sequence"

    def __init__(
        self,
        family: str = "lstm",
        hidden_size: int = 32,
        num_layers: int = 1,
        epochs: int = 10,
        batch_size: int = 128,
        learning_rate: float = 3e-3,
        dropout: float = 0.1,
        validation_fraction: float = 0.2,
        early_stopping_patience: int = 3,
        device: str | None = None,
        random_state: int = 42,
        class_weight: str | None = None,
        task: str = "binary_classification",
        cost_specification: dict[str, Any] | None = None,
    ) -> None:
        if family not in SEQUENCE_FAMILIES:
            raise ValueError(f"Unknown sequence family '{family}'. Known: {SEQUENCE_FAMILIES}")
        if epochs > 10:
            raise ValueError("Laptop-safe constraint: epochs must be <= 10.")
        if batch_size > 128:
            raise ValueError("Laptop-safe constraint: batch_size must be <= 128.")
        self.family = family
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.validation_fraction = validation_fraction
        self.early_stopping_patience = early_stopping_patience
        self.device = device
        self.random_state = random_state
        self.class_weight = class_weight
        self.task = task
        self.cost_specification = cost_specification or {"type": "balanced"}
        self._net = None
        self._device_used = "cpu"
        self.classes_ = np.array([0, 1])
        self.history_: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
        self.best_epoch_ = 0
        self.stopped_early_ = False

    # -- sklearn protocol -------------------------------------------------- #
    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {
            "family": self.family,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "dropout": self.dropout,
            "validation_fraction": self.validation_fraction,
            "early_stopping_patience": self.early_stopping_patience,
            "device": self.device,
            "random_state": self.random_state,
            "class_weight": self.class_weight,
            "task": self.task,
            "cost_specification": self.cost_specification,
        }

    def set_params(self, **params: Any) -> SequenceClassifier:
        for k, v in params.items():
            if not hasattr(self, k):
                raise ValueError(f"Unknown parameter '{k}'.")
            setattr(self, k, v)
        return self

    def fit(self, X: np.ndarray, y: np.ndarray) -> SequenceClassifier:
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 2:
            X = X[:, np.newaxis, :]

        classes = np.unique(y)
        self.classes_ = classes

        if self.task == "multiclass_classification":
            self.n_outputs_ = len(classes)
            mapping = {c: i for i, c in enumerate(classes)}
            y_arr = np.array([mapping[v] for v in y], dtype=np.int64)
        else:
            self.n_outputs_ = 1
            mapping = {c: i for i, c in enumerate(classes)}
            y_arr = np.array([mapping[v] for v in y], dtype=np.float32).reshape(-1, 1)

        self._device_used = self.device or resolve_torch_device()
        device = torch.device(self._device_used)
        self._net = build_sequence_network(
            self.family, X.shape[2], self.hidden_size, self.num_layers, self.dropout, self.n_outputs_
        ).to(device)

        n = len(X)
        rng = np.random.default_rng(self.random_state)
        perm = rng.permutation(n)
        n_val = int(n * self.validation_fraction)
        val_idx, tr_idx = perm[:n_val], perm[n_val:]
        has_val = n_val > 0

        Xt = torch.tensor(X)
        if self.task == "multiclass_classification":
            yt = torch.tensor(y_arr, dtype=torch.long)
        else:
            yt = torch.tensor(y_arr, dtype=torch.float32)

        loader = DataLoader(
            TensorDataset(Xt[tr_idx], yt[tr_idx]),
            batch_size=self.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(self.random_state),
        )
        if has_val:
            xv, yv = Xt[val_idx].to(device), yt[val_idx].to(device)
        opt = torch.optim.Adam(self._net.parameters(), lr=self.learning_rate)

        weights = None
        if self.task == "multiclass_classification":
            if self.class_weight == "balanced":
                from sklearn.utils.class_weight import compute_class_weight

                weights = compute_class_weight("balanced", classes=np.arange(self.n_outputs_), y=y_arr)
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
                loss_fn = torch.nn.CrossEntropyLoss(
                    weight=torch.tensor(weights, dtype=torch.float32, device=device)
                )
            else:
                loss_fn = torch.nn.CrossEntropyLoss()
        else:
            if self.class_weight == "balanced":
                n_pos = float(np.sum(y_arr == 1))
                n_neg = float(np.sum(y_arr == 0))
                if n_pos > 0:
                    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32, device=device)
                    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                else:
                    loss_fn = torch.nn.BCEWithLogitsLoss()
            else:
                loss_fn = torch.nn.BCEWithLogitsLoss()

        self.history_ = {"train_loss": [], "val_loss": []}
        best_val, best_state, no_improve = float("inf"), None, 0
        self.stopped_early_ = False

        for epoch in range(self.epochs):
            self._net.train()
            total = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                loss = loss_fn(self._net(xb), yb)
                loss.backward()
                opt.step()
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

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        import torch

        if self._net is None:
            raise RuntimeError("Not fitted; call fit() first.")
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 2:
            X = X[:, np.newaxis, :]
        device = torch.device(self._device_used)
        self._net.eval()
        with torch.no_grad():
            logits = self._net(torch.tensor(X).to(device)).cpu().numpy()

        if self.task == "multiclass_classification":
            e = np.exp(logits - logits.max(axis=1, keepdims=True))
            return e / e.sum(axis=1, keepdims=True)
        else:
            p1 = 1.0 / (1.0 + np.exp(-logits.reshape(-1)))
            return np.column_stack([1.0 - p1, p1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.task == "multiclass_classification":
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
                    # Assertions to validate cost matrix before prediction
                    assert cost_matrix.shape == (len(self.classes_), len(self.classes_)), (
                        f"Cost matrix shape must be {len(self.classes_)}x{len(self.classes_)}"
                    )
                    assert np.all(np.isfinite(cost_matrix)), "Cost matrix must contain only finite values"
                    assert np.all(cost_matrix >= 0), "Cost matrix must contain non-negative values"

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
        else:
            return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == np.asarray(y).reshape(-1)))

    @property
    def device_used(self) -> str:
        return self._device_used


def sequence_metrics(
    y_true: np.ndarray, proba: np.ndarray, task: str = "binary_classification", classes: Any = None
) -> dict[str, float]:
    from start.modeling.tabular_dl_metrics import dl_task_metrics

    return dl_task_metrics(task, y_true, proba, classes)


def sequence_saliency(
    model: SequenceClassifier, X: np.ndarray, *, n_samples: int = 64, seed: int = 42
) -> dict[str, Any]:
    """Gradient saliency: mean absolute input gradient per timestep and per
    feature, identifying which parts of the sequence drove predictions."""
    import torch

    if model._net is None:
        return {"method": "unavailable", "note": "model not fitted"}
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(n_samples, len(X)), replace=False)
    device = torch.device(model.device_used)
    x = torch.tensor(np.asarray(X[idx], dtype=np.float32), device=device, requires_grad=True)
    model._net.eval()
    out = model._net(x).sum()
    out.backward()
    grads = x.grad.abs().detach().cpu().numpy()  # (n, timesteps, features)
    return {
        "method": "gradient_saliency",
        "per_timestep": grads.mean(axis=(0, 2)).round(6).tolist(),
        "per_feature": grads.mean(axis=(0, 1)).round(6).tolist(),
        "most_salient_timestep": int(grads.mean(axis=(0, 2)).argmax()),
        "most_salient_feature": int(grads.mean(axis=(0, 1)).argmax()),
    }


def sequence_robustness(
    model: SequenceClassifier,
    X: np.ndarray,
    y: np.ndarray,
    *,
    noise_levels: tuple[float, ...] = (0.0, 0.01, 0.05, 0.1),
    seed: int = 42,
) -> dict[str, Any]:
    """Robustness drift tables: input noise and temporal jitter (time-warp)."""
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(seed)
    baseline = float(roc_auc_score(y, model.predict_proba(X)[:, 1]))
    noise_rows = []
    std = float(np.std(X))
    for level in noise_levels:
        if level == 0.0:
            auc = baseline
        else:
            Xn = X + rng.normal(0, level * std, size=X.shape).astype(np.float32)
            auc = float(roc_auc_score(y, model.predict_proba(Xn)[:, 1]))
        noise_rows.append({"noise": level, "auc": round(auc, 6), "drift": round(auc - baseline, 6)})

    # temporal jitter: roll each sequence by a small random shift
    jitter_rows = []
    for shift in (0, 1, 2, 3):
        if shift == 0:
            auc = baseline
        else:
            Xj = np.stack([np.roll(seq, rng.integers(-shift, shift + 1), axis=0) for seq in X])
            auc = float(roc_auc_score(y, model.predict_proba(Xj.astype(np.float32))[:, 1]))
        jitter_rows.append({"max_shift": shift, "auc": round(auc, 6), "drift": round(auc - baseline, 6)})

    return {
        "baseline_auc": round(baseline, 6),
        "noise": noise_rows,
        "time_jitter": jitter_rows,
        "max_abs_noise_drift": round(max(abs(r["drift"]) for r in noise_rows), 6),
        "max_abs_jitter_drift": round(max(abs(r["drift"]) for r in jitter_rows), 6),
    }
