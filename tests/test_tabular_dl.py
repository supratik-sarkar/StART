from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch", reason="Layer 5 requires the [torch] extra")

from sklearn.datasets import make_classification  # noqa: E402

from start.modeling.tabular_dl import TABULAR_TASKS, TabularDLClassifier  # noqa: E402
from start.modeling.tabular_dl_metrics import dl_task_metrics  # noqa: E402


@pytest.fixture(scope="module")
def binary_data():
    X, y = make_classification(n_samples=400, n_features=10, n_informative=6, random_state=0)
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(10)]), y


@pytest.fixture(scope="module")
def multiclass_data():
    X, y = make_classification(
        n_samples=600, n_features=12, n_informative=8, n_classes=4,
        n_clusters_per_class=1, random_state=0,
    )
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(12)]), y


@pytest.fixture(scope="module")
def multilabel_data():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(400, 8)), columns=[f"f{i}" for i in range(8)])
    y = pd.DataFrame(
        {"a": (X.f0 > 0).astype(int), "b": (X.f1 > 0).astype(int), "c": (X.f2 > 0).astype(int)}
    )
    return X, y


def test_task_set():
    assert set(TABULAR_TASKS) == {
        "binary_classification",
        "multiclass_classification",
        "multilabel_classification",
    }


def test_unknown_task_raises():
    with pytest.raises(ValueError, match="Unknown tabular task"):
        TabularDLClassifier(task="ranking")


def test_binary_classification(binary_data):
    from sklearn.metrics import roc_auc_score

    X, y = binary_data
    clf = TabularDLClassifier(task="binary_classification", family="mlp", epochs=8, random_state=0)
    clf.fit(X, y)
    proba = clf.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert abs(proba.sum(axis=1) - 1.0).max() < 1e-6
    assert roc_auc_score(y, proba[:, 1]) > 0.7
    assert set(clf.predict(X)) <= {0, 1}
    metrics = dl_task_metrics("binary_classification", y, proba, clf.classes_)
    assert "auc_roc" in metrics and "brier_score" in metrics


@pytest.mark.parametrize("family", ["mlp", "residual_mlp", "wide_deep"])
def test_multiclass_classification(family, multiclass_data):
    X, y = multiclass_data
    lr = 3e-3 if family == "wide_deep" else 1e-3
    clf = TabularDLClassifier(
        task="multiclass_classification", family=family, epochs=10,
        learning_rate=lr, random_state=0,
    )
    clf.fit(X, y)
    proba = clf.predict_proba(X)
    assert proba.shape == (len(X), 4)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)
    metrics = dl_task_metrics("multiclass_classification", y, proba, clf.classes_)
    assert metrics["n_classes"] == 4
    assert metrics["accuracy"] > 0.6
    assert set(clf.predict(X)) <= set(clf.classes_)


def test_multiclass_with_string_labels():
    X, yi = make_classification(
        n_samples=300, n_features=8, n_informative=6, n_classes=3,
        n_clusters_per_class=1, random_state=1,
    )
    X = pd.DataFrame(X, columns=[f"f{i}" for i in range(8)])
    y = np.array(["low", "mid", "high"])[yi]
    clf = TabularDLClassifier(
        task="multiclass_classification", epochs=10, learning_rate=3e-3, random_state=0
    ).fit(X, y)
    assert set(clf.classes_) == {"low", "mid", "high"}
    assert set(clf.predict(X)) <= {"low", "mid", "high"}


def test_multilabel_classification(multilabel_data):
    X, y = multilabel_data
    clf = TabularDLClassifier(
        task="multilabel_classification", family="mlp", epochs=10,
        learning_rate=3e-3, random_state=0,
    )
    clf.fit(X, y)
    proba = clf.predict_proba(X)
    assert proba.shape == (len(X), 3)
    preds = clf.predict(X)
    assert preds.shape == (len(X), 3)
    assert set(np.unique(preds)) <= {0, 1}
    metrics = dl_task_metrics("multilabel_classification", y.to_numpy(), proba)
    assert metrics["n_labels"] == 3
    assert metrics["subset_accuracy"] >= 0.0 and metrics["f1_micro"] > 0.5


def test_metrics_branch_by_task(binary_data):
    X, y = binary_data
    clf = TabularDLClassifier(task="binary_classification", epochs=5, random_state=0).fit(X, y)
    proba = clf.predict_proba(X)
    bm = dl_task_metrics("binary_classification", y, proba, clf.classes_)
    assert "top_decile_lift" in bm  # binary-only metric
    with pytest.raises(ValueError, match="No metric branch"):
        dl_task_metrics("forecasting", y, proba)


def test_sklearn_protocol_and_history(binary_data):
    X, y = binary_data
    clf = TabularDLClassifier(task="binary_classification", epochs=6, random_state=0).fit(X, y)
    params = clf.get_params()
    assert params["task"] == "binary_classification" and "learning_rate" in params
    assert clf.history_["train_loss"] and len(clf.history_["val_loss"]) >= 1
    assert 1 <= clf.best_epoch_ <= 6
    assert clf.score(X, y) > 0.6
    assert clf.device_used in {"cpu", "mps", "cuda"}


def test_laptop_safe_constraints():
    with pytest.raises(ValueError, match="epochs"):
        TabularDLClassifier(epochs=50)
    with pytest.raises(ValueError, match="batch_size"):
        TabularDLClassifier(batch_size=9999)


def test_activation_selection(binary_data):
    X, y = binary_data
    for act in ("relu", "leaky_relu", "gelu", "tanh", "selu", "elu"):
        clf = TabularDLClassifier(
            task="binary_classification", activation=act, epochs=3, random_state=0
        ).fit(X, y)
        assert clf.activation == act
