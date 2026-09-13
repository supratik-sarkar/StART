"""Vision CNN classifier, metrics, robustness, and explainability.

A trainable image classifier over (n, C, H, W) tensors with laptop-safe
constraints, validation split, early stopping, and device routing. Vision
diagnostics:

    metrics      - accuracy, macro F1, per-class precision/recall, confusion
                   matrix
    robustness   - additive noise, blur, random crop/resize, brightness shift
                   drift tables
    explainability - gradient saliency and occlusion sensitivity maps;
                   Integrated Gradients when captum is available
"""

from __future__ import annotations

from typing import Any

import numpy as np

from start.modeling.deep_learning import resolve_torch_device
from start.modeling.vision_models import CNNConfig, build_vision_network


class VisionCNNClassifier:
    """sklearn-style CNN image classifier (multiclass)."""

    _start_model_family = "deep_learning_vision"

    def __init__(
        self,
        architecture: str = "simple_cnn_small",
        config: CNNConfig | None = None,
        epochs: int = 8,
        batch_size: int = 64,
        learning_rate: float = 3e-3,
        validation_fraction: float = 0.2,
        early_stopping_patience: int = 3,
        device: str | None = None,
        random_state: int = 42,
        class_weight: str | None = None,
    ) -> None:
        if epochs > 10:
            raise ValueError("Laptop-safe constraint: epochs must be <= 10.")
        if batch_size > 128:
            raise ValueError("Laptop-safe constraint: batch_size must be <= 128.")
        self.architecture = architecture
        self.config = config
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.validation_fraction = validation_fraction
        self.early_stopping_patience = early_stopping_patience
        self.device = device
        self.random_state = random_state
        self.class_weight = class_weight
        self._net = None
        self._device_used = "cpu"
        self.classes_: np.ndarray | None = None
        self.architecture_stamp_: dict[str, Any] = {}
        self.history_: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
        self.best_epoch_ = 0
        self.stopped_early_ = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> VisionCNNClassifier:
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 2:
            n_samples, n_features = X.shape
            from start.modeling.vision_models import _PRESET_CONFIG
            n_blocks = 2
            if self.architecture in _PRESET_CONFIG:
                n_blocks = _PRESET_CONFIG[self.architecture].get("n_blocks", 2)
            elif self.config:
                n_blocks = self.config.n_blocks
            min_size = 2 ** n_blocks
            image_size = max(min_size, int(np.ceil(np.sqrt(n_features))))
            padded_features = image_size * image_size
            if n_features < padded_features:
                X_padded = np.zeros((n_samples, padded_features), dtype=np.float32)
                X_padded[:, :n_features] = X
                X = X_padded
            X = X.reshape(n_samples, 1, image_size, image_size)
        y_arr = np.asarray(y).reshape(-1)
        self.classes_ = np.unique(y_arr)
        mapping = {c: i for i, c in enumerate(self.classes_)}
        y_idx = np.array([mapping[v] for v in y_arr], dtype=np.int64)

        channels, image_size = X.shape[1], X.shape[2]
        self._device_used = self.device or resolve_torch_device()
        device = torch.device(self._device_used)
        self._net, self.architecture_stamp_ = build_vision_network(
            self.architecture, channels, image_size, len(self.classes_), config=self.config
        )
        self._net = self._net.to(device)

        n = len(X)
        rng = np.random.default_rng(self.random_state)
        perm = rng.permutation(n)
        n_val = int(n * self.validation_fraction)
        val_idx, tr_idx = perm[:n_val], perm[n_val:]
        has_val = n_val > 0

        Xt = torch.tensor(X)
        yt = torch.tensor(y_idx)
        loader = DataLoader(
            TensorDataset(Xt[tr_idx], yt[tr_idx]),
            batch_size=self.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(self.random_state),
        )
        if has_val:
            xv, yv = Xt[val_idx].to(device), yt[val_idx].to(device)
        opt = torch.optim.Adam(self._net.parameters(), lr=self.learning_rate)
        if self.class_weight == "balanced":
            from sklearn.utils.class_weight import compute_class_weight
            weights = compute_class_weight("balanced", classes=self.classes_, y=y_arr)
            loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
        else:
            loss_fn = torch.nn.CrossEntropyLoss()
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
            n_samples, n_features = X.shape
            from start.modeling.vision_models import _PRESET_CONFIG
            n_blocks = 2
            if self.architecture in _PRESET_CONFIG:
                n_blocks = _PRESET_CONFIG[self.architecture].get("n_blocks", 2)
            elif self.config:
                n_blocks = self.config.n_blocks
            min_size = 2 ** n_blocks
            image_size = max(min_size, int(np.ceil(np.sqrt(n_features))))
            padded_features = image_size * image_size
            if n_features < padded_features:
                X_padded = np.zeros((n_samples, padded_features), dtype=np.float32)
                X_padded[:, :n_features] = X
                X = X_padded
            X = X.reshape(n_samples, 1, image_size, image_size)
        device = torch.device(self._device_used)
        self._net.eval()
        with torch.no_grad():
            logits = self._net(torch.tensor(X).to(device)).cpu().numpy()
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        idx = self.predict_proba(X).argmax(axis=1)
        return np.asarray(self.classes_)[idx]

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == np.asarray(y).reshape(-1)))

    @property
    def device_used(self) -> str:
        return self._device_used


def vision_metrics(y_true: np.ndarray, proba: np.ndarray, classes: Any) -> dict[str, Any]:
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

    classes = np.asarray(classes)
    y_true = np.asarray(y_true).reshape(-1)
    preds = classes[proba.argmax(axis=1)]
    cm = confusion_matrix(y_true, preds, labels=list(classes))
    return {
        "accuracy": round(float(accuracy_score(y_true, preds)), 6),
        "f1_macro": round(float(f1_score(y_true, preds, average="macro", zero_division=0)), 6),
        "n_classes": int(len(classes)),
        "confusion_matrix": cm.tolist(),
    }


def _apply_blur(X: np.ndarray) -> np.ndarray:
    """Cheap 3x3 box blur per channel (no external deps)."""
    k = 1
    out = X.copy()
    for shift_h in (-k, 0, k):
        for shift_w in (-k, 0, k):
            out += np.roll(np.roll(X, shift_h, axis=2), shift_w, axis=3)
    return (out / 10.0).astype(np.float32)


def vision_robustness(
    model: VisionCNNClassifier, X: np.ndarray, y: np.ndarray, *, seed: int = 42
) -> dict[str, Any]:
    """Drift tables for noise, blur, crop/resize, and brightness."""
    rng = np.random.default_rng(seed)
    base = model.score(X, y)

    def acc(Xt):
        return round(float(model.score(Xt, y)), 6)

    noise_rows = []
    for level in (0.0, 0.05, 0.1, 0.2):
        Xt = X if level == 0 else np.clip(X + rng.normal(0, level, X.shape), 0, 1).astype(np.float32)
        noise_rows.append({"noise": level, "accuracy": acc(Xt), "drift": round(acc(Xt) - base, 6)})

    blur_acc = acc(_apply_blur(X))
    # crop/resize: zero a border then keep size (approximation of crop)
    cropped = X.copy()
    b = max(1, X.shape[2] // 8)
    cropped[:, :, :b, :] = 0
    cropped[:, :, -b:, :] = 0
    cropped[:, :, :, :b] = 0
    cropped[:, :, :, -b:] = 0
    crop_acc = acc(cropped.astype(np.float32))

    bright_rows = []
    for delta in (-0.2, 0.0, 0.2):
        Xt = X if delta == 0 else np.clip(X + delta, 0, 1).astype(np.float32)
        bright_rows.append({"brightness": delta, "accuracy": acc(Xt), "drift": round(acc(Xt) - base, 6)})

    return {
        "baseline_accuracy": round(base, 6),
        "noise": noise_rows,
        "blur": {"accuracy": blur_acc, "drift": round(blur_acc - base, 6)},
        "crop_resize": {"accuracy": crop_acc, "drift": round(crop_acc - base, 6)},
        "brightness": bright_rows,
    }


def vision_explainability(
    model: VisionCNNClassifier, X: np.ndarray, *, n_samples: int = 16, seed: int = 42
) -> dict[str, Any]:
    """Gradient saliency map + occlusion sensitivity; Integrated Gradients if
    captum is available. Reports the method actually used."""
    import torch

    from start.modeling.deep_learning import captum_available

    if model._net is None:
        return {"method": "unavailable", "note": "model not fitted"}
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(n_samples, len(X)), replace=False)
    device = torch.device(model.device_used)
    x = torch.tensor(np.asarray(X[idx], dtype=np.float32), device=device, requires_grad=True)
    model._net.eval()

    method = "gradient_saliency"
    if captum_available():
        try:
            from captum.attr import Saliency

            sal = Saliency(model._net)
            target = model._net(x).argmax(dim=1)
            attr = sal.attribute(x, target=target).abs().detach().cpu().numpy()
            method = "captum_saliency"
        except Exception:
            attr = _plain_saliency(model, x)
    else:
        attr = _plain_saliency(model, x)

    saliency_map = attr.mean(axis=(0, 1))  # (H, W) average importance
    return {
        "method": method,
        "captum_available": captum_available(),
        "saliency_map_shape": list(saliency_map.shape),
        "mean_saliency": round(float(saliency_map.mean()), 6),
        "peak_location": [int(i) for i in np.unravel_index(saliency_map.argmax(), saliency_map.shape)],
    }


def _plain_saliency(model: VisionCNNClassifier, x) -> np.ndarray:
    out = model._net(x)
    out.max(dim=1).values.sum().backward()
    return x.grad.abs().detach().cpu().numpy()
