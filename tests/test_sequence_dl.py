from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch", reason="Layer 6 requires the [torch] extra")

from sklearn.metrics import roc_auc_score  # noqa: E402

from start.modeling.sequence_data import (  # noqa: E402
    generate_sequence_dataset,
    load_sequence_demo,
    sliding_windows,
    split_sequences,
)
from start.modeling.sequence_dl import (  # noqa: E402
    SequenceClassifier,
    sequence_metrics,
    sequence_robustness,
    sequence_saliency,
)
from start.modeling.sequence_models import SEQUENCE_FAMILIES, build_sequence_network


@pytest.fixture(scope="module")
def bundle():
    return load_sequence_demo(timesteps=24, n_features=3, seed=0)


def test_generate_has_temporal_structure():
    X, y = generate_sequence_dataset(n_series=200, timesteps=20, n_features=3, seed=0)
    assert X.shape == (200, 20, 3)
    assert set(np.unique(y)) <= {0, 1}
    # positive class should have a higher end-of-sequence mean in feature 0 (trend)
    pos_end = X[y == 1, -1, 0].mean()
    neg_end = X[y == 0, -1, 0].mean()
    assert pos_end > neg_end


def test_split_preserves_order_and_sizes():
    X, y = generate_sequence_dataset(n_series=100, timesteps=10, n_features=2, seed=1)
    b = split_sequences(X, y, fractions=(0.6, 0.2, 0.2))
    assert len(b.X_train) == 60 and len(b.X_test) == 20 and len(b.X_oos) == 20
    assert b.n_features == 2 and b.timesteps == 10
    # order preserved: train is the earliest block
    assert np.array_equal(b.X_train[0], X[0])
    with pytest.raises(ValueError, match="sum to 1.0"):
        split_sequences(X, y, fractions=(0.5, 0.2, 0.2))


def test_sliding_windows():
    series = np.arange(40).reshape(20, 2).astype(float)
    labels = np.arange(20)
    xs, ys = sliding_windows(series, labels, window=5, stride=5)
    assert xs.shape == (4, 5, 2)
    assert ys[0] == labels[4]  # window label = last step
    with pytest.raises(ValueError, match="larger than series"):
        sliding_windows(series, labels, window=50)


@pytest.mark.parametrize("family", ["rnn", "gru", "lstm", "bi_lstm"])
def test_each_sequence_architecture_learns(family, bundle):
    clf = SequenceClassifier(family=family, epochs=8, learning_rate=3e-3, random_state=0)
    clf.fit(bundle.X_train, bundle.y_train)
    proba = clf.predict_proba(bundle.X_test)
    assert proba.shape == (len(bundle.X_test), 2)
    assert abs(proba.sum(axis=1) - 1.0).max() < 1e-6
    auc = roc_auc_score(bundle.y_test, proba[:, 1])
    assert auc > 0.75, f"{family} should learn temporal signal; AUC {auc:.3f}"
    assert clf.family == family
    assert set(clf.predict(bundle.X_test)) <= {0, 1}


def test_sequence_network_shapes():
    import torch

    for family in SEQUENCE_FAMILIES:
        net = build_sequence_network(family, n_features=3, hidden_size=16, n_outputs=1)
        out = net(torch.randn(5, 12, 3))
        assert out.shape == (5, 1)
    with pytest.raises(ValueError, match="Unknown sequence family"):
        build_sequence_network("transformer", 3)


def test_laptop_safe_constraints():
    with pytest.raises(ValueError, match="epochs"):
        SequenceClassifier(epochs=99)
    with pytest.raises(ValueError, match="batch_size"):
        SequenceClassifier(batch_size=9999)
    with pytest.raises(ValueError, match="Unknown sequence family"):
        SequenceClassifier(family="tcn")


def test_sequence_metrics(bundle):
    clf = SequenceClassifier(family="lstm", epochs=8, learning_rate=3e-3, random_state=0)
    clf.fit(bundle.X_train, bundle.y_train)
    metrics = sequence_metrics(bundle.y_oos, clf.predict_proba(bundle.X_oos))
    assert "auc_roc" in metrics and "f1" in metrics
    assert metrics["auc_roc"] > 0.7


def test_sequence_saliency_identifies_structure(bundle):
    clf = SequenceClassifier(family="gru", epochs=10, learning_rate=3e-3, random_state=0)
    clf.fit(bundle.X_train, bundle.y_train)
    sal = sequence_saliency(clf, bundle.X_test, seed=0)
    assert sal["method"] == "gradient_saliency"
    assert len(sal["per_timestep"]) == bundle.timesteps
    assert len(sal["per_feature"]) == bundle.n_features
    # feature 0 (with injected trend+seasonality) should be among the salient
    assert sal["most_salient_feature"] in (0, 1)


def test_sequence_robustness_baseline(bundle):
    clf = SequenceClassifier(family="lstm", epochs=8, learning_rate=3e-3, random_state=0)
    clf.fit(bundle.X_train, bundle.y_train)
    rob = sequence_robustness(clf, bundle.X_test, bundle.y_test, seed=0)
    zero = next(r for r in rob["noise"] if r["noise"] == 0.0)
    assert zero["drift"] == 0.0  # baseline by construction
    assert "max_abs_noise_drift" in rob and "max_abs_jitter_drift" in rob
    assert len(rob["time_jitter"]) == 4
