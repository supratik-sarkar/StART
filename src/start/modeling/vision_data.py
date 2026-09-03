"""Vision data layer for the CNN track.

Loads image-folder datasets (``root/<class>/<image>``) into tensors with label
discovery, applies normalization/resize transforms, and provides a synthetic
image generator (real tensors with class-distinguishing spatial patterns) so
the CNN trains genuinely offline without downloads. Train/test/OOS split is
explicit and stratified by class.

dataset_type = vision_image_classification (kept distinct from tabular DL).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class VisionBundle:
    X_train: np.ndarray  # (n, C, H, W)
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    X_oos: np.ndarray
    y_oos: np.ndarray
    class_names: list[str]
    image_size: int
    channels: int
    source: str = "synthetic"

    @property
    def n_classes(self) -> int:
        return len(self.class_names)

    @property
    def shapes(self) -> dict[str, tuple]:
        return {"train": self.X_train.shape, "test": self.X_test.shape, "oos": self.X_oos.shape}


def discover_labels(root: str | Path) -> list[str]:
    """Discover class names from immediate subdirectories of an image folder."""
    root = Path(root)
    if not root.is_dir():
        raise ValueError(f"Image folder '{root}' is not a directory.")
    classes = sorted(d.name for d in root.iterdir() if d.is_dir())
    if not classes:
        raise ValueError(f"No class subdirectories found under '{root}'.")
    return classes


def normalize_images(X: np.ndarray, mean: float = 0.5, std: float = 0.5) -> np.ndarray:
    """Per-tensor normalization (image transform)."""
    return ((X - mean) / std).astype(np.float32)


def load_image_folder(
    root: str | Path, image_size: int = 16, channels: int = 3
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load an ImageFolder-style dataset into (X, y, class_names).

    Requires Pillow for real images; raises a clear hint if absent.
    """
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("Loading real image folders requires Pillow: pip install pillow") from exc
    from start.data.loaders import discover_image_folder

    class_names = discover_labels(root)
    manifest = discover_image_folder(root)
    idx = {c: i for i, c in enumerate(class_names)}
    xs, ys = [], []
    mode = "RGB" if channels == 3 else "L"
    for _, row in manifest.iterrows():
        img = Image.open(row["image_path"]).convert(mode).resize((image_size, image_size))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        if channels == 1:
            arr = arr[None, :, :]
        else:
            arr = arr.transpose(2, 0, 1)
        xs.append(arr)
        ys.append(idx[row["label"]])
    return np.asarray(xs, dtype=np.float32), np.asarray(ys), class_names


def generate_image_dataset(
    n_per_class: int = 150,
    n_classes: int = 3,
    image_size: int = 16,
    channels: int = 3,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Synthetic image classification: each class has a distinct spatial
    pattern (horizontal bands / vertical bands / diagonal) plus noise, so a
    CNN learns genuine spatial features. Returns (X, y, class_names)."""
    rng = np.random.default_rng(seed)
    n = n_per_class * n_classes
    X = rng.normal(0.4, 0.15, size=(n, channels, image_size, image_size)).astype(np.float32)
    y = np.repeat(np.arange(n_classes), n_per_class)
    rng.shuffle(y)
    coords = np.linspace(0, 1, image_size)
    for i in range(n):
        cls = y[i]
        if cls % 3 == 0:  # horizontal bands
            band = np.sin(coords * 6 * np.pi)[:, None] * 0.4
            X[i] += band[None, :, :]
        elif cls % 3 == 1:  # vertical bands
            band = np.sin(coords * 6 * np.pi)[None, :] * 0.4
            X[i] += band[None, :, :]
        else:  # diagonal gradient
            diag = (coords[:, None] + coords[None, :]) * 0.3
            X[i] += diag[None, :, :]
    X = np.clip(X, 0.0, 1.0)
    return X, y, [f"class_{i}" for i in range(n_classes)]


def split_images(
    X: np.ndarray,
    y: np.ndarray,
    class_names: list[str],
    fractions: tuple[float, float, float] = (0.6, 0.2, 0.2),
    seed: int = 42,
    image_size: int = 16,
) -> VisionBundle:
    """Stratified-by-class train/test/OOS split for images."""
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError(f"fractions must sum to 1.0, got {fractions}")
    rng = np.random.default_rng(seed)
    tr_i, te_i, oo_i = [], [], []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        n = len(idx)
        n_tr, n_te = int(n * fractions[0]), int(n * fractions[1])
        tr_i.extend(idx[:n_tr])
        te_i.extend(idx[n_tr : n_tr + n_te])
        oo_i.extend(idx[n_tr + n_te :])
    tr_i, te_i, oo_i = np.array(tr_i), np.array(te_i), np.array(oo_i)
    rng.shuffle(tr_i)
    return VisionBundle(
        X_train=X[tr_i],
        y_train=y[tr_i],
        X_test=X[te_i],
        y_test=y[te_i],
        X_oos=X[oo_i],
        y_oos=y[oo_i],
        class_names=class_names,
        image_size=image_size,
        channels=X.shape[1],
    )


def load_vision_demo(
    n_classes: int = 3, image_size: int = 16, channels: int = 3, seed: int = 42
) -> VisionBundle:
    X, y, names = generate_image_dataset(
        n_classes=n_classes, image_size=image_size, channels=channels, seed=seed
    )
    return split_images(X, y, names, seed=seed, image_size=image_size)
