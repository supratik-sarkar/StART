"""Scientific temporal contract tests.

Validates that sequence architectures (LSTM, GRU, RNN, Bi-LSTM) enforce genuine
rank-3 temporal structure (samples, timesteps, features) with timesteps > 1, and
fail closed when passed ordinary rank-2 tabular matrices.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch", reason="Sequence DL requires torch")

import start.modeling.sequence_dl as seq_mod
from start.modeling.models import resolve_model
from start.modeling.sequence_data import generate_sequence_dataset
from start.modeling.sequence_dl import SequenceClassifier

# Use SequenceInputContractError if defined, else ValueError for initial RED state
SequenceInputContractError = getattr(seq_mod, "SequenceInputContractError", ValueError)


def test_lstm_rejects_rank2_tabular_input():
    """RED TEST: LSTM must NOT silently accept rank-2 (N, F) tabular data."""
    X_tabular = np.random.randn(100, 10).astype(np.float32)
    y = np.random.randint(0, 2, size=100)
    clf = SequenceClassifier(family="lstm", epochs=2, random_state=42)
    with pytest.raises(SequenceInputContractError, match="requires rank-3 input"):
        clf.fit(X_tabular, y)


def test_gru_rejects_rank2_tabular_input():
    """RED TEST: GRU must NOT silently accept rank-2 (N, F) tabular data."""
    X_tabular = np.random.randn(100, 10).astype(np.float32)
    y = np.random.randint(0, 2, size=100)
    clf = SequenceClassifier(family="gru", epochs=2, random_state=42)
    with pytest.raises(SequenceInputContractError, match="requires rank-3 input"):
        clf.fit(X_tabular, y)


def test_sequence_rejects_timesteps_le_1():
    """RED TEST: Sequence models must reject degenerate timesteps <= 1 (fake sequences)."""
    X_fake_seq = np.random.randn(100, 1, 10).astype(np.float32)
    y = np.random.randint(0, 2, size=100)
    clf = SequenceClassifier(family="lstm", epochs=2, random_state=42)
    with pytest.raises(SequenceInputContractError, match="timesteps > 1"):
        clf.fit(X_fake_seq, y)


def test_sequence_accepts_genuine_rank3_with_t_gt_1():
    """Genuine rank-3 input with T=24 and F=3 is accepted and fitted."""
    X, y = generate_sequence_dataset(n_series=60, timesteps=24, n_features=3, seed=42)
    clf = SequenceClassifier(family="lstm", epochs=2, random_state=42)
    clf.fit(X, y)
    probs = clf.predict_proba(X)
    assert probs.shape == (60, 2)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_sequence_rejects_wrong_ranks():
    """1D or 4D tensors must be rejected."""
    clf = SequenceClassifier(family="lstm", epochs=2, random_state=42)
    y = np.array([0, 1])

    # 1D
    with pytest.raises(SequenceInputContractError, match="requires rank-3 input"):
        clf.fit(np.array([1.0, 2.0]), y)

    # 4D
    with pytest.raises(SequenceInputContractError, match="requires rank-3 input"):
        clf.fit(np.zeros((10, 5, 4, 3)), np.zeros(10))


def test_sequence_rejects_non_finite_values():
    """NaN or Inf in sequence input must fail closed."""
    X, y = generate_sequence_dataset(n_series=20, timesteps=10, n_features=2, seed=42)
    X[0, 2, 0] = np.nan
    clf = SequenceClassifier(family="lstm", epochs=2, random_state=42)
    with pytest.raises(ValueError, match="non-finite"):
        clf.fit(X, y)


def test_tabular_mlp_preserves_rank2_tabular():
    """MLP and tabular models continue to accept ordinary rank-2 tabular data."""
    clf, name, _ = resolve_model("mlp", seed=42)
    X_tabular = np.random.randn(50, 8).astype(np.float32)
    y = np.random.randint(0, 2, size=50)
    clf.fit(X_tabular, y)
    probs = clf.predict_proba(X_tabular)
    assert probs.shape == (50, 2)


def test_tabular_dl_rejects_sequence_dl_in_orchestrator():
    """ReviewOrchestrator._run_tabular_dl fails closed when given a sequence architecture."""
    import pandas as pd

    from start.modeling.review_orchestrator import ReviewOrchestrator
    from start.modeling.split_planner import SplitPlan

    df = pd.DataFrame({"feat1": [1.0, 2.0, 3.0], "feat2": [4.0, 5.0, 6.0], "target": [0, 1, 0]})
    plan = SplitPlan(strategy="holdout", fractions=(0.6, 0.2, 0.2), train=df, test=df.iloc[:0], oos=df.iloc[:0])
    orch = ReviewOrchestrator()
    with pytest.raises(SequenceInputContractError, match="cannot be trained on rank-2 tabular"):
        orch._run_tabular_dl(plan, "target", [], 42, architecture="lstm")


def test_model_execution_rejects_sequence_dl_without_bundle():
    """run_model_execution fails closed when given a sequence architecture without sequence_bundle."""
    import pandas as pd

    from start.modeling.model_execution import run_model_execution

    df = pd.DataFrame({
        "feat1": np.random.randn(50),
        "feat2": np.random.randn(50),
        "target": np.random.randint(0, 2, size=50),
    })
    with pytest.raises(SequenceInputContractError, match="requires a rank-3 sequence bundle"):
        run_model_execution(df, "target", architecture="lstm", sequence_bundle=None)


def test_interactive_review_fail_closed_contract():
    """Interactive review validates sequence vs tabular contract at entry."""
    from start.data.selection import select_temporal_sequence
    from start.interactive_review import ReviewConfig, run_interactive_review

    # Case 1: Recurrent architecture without sequence_bundle -> SequenceInputContractError
    cfg_bad_arch = ReviewConfig(
        data_path=None,
        target="target",
        architecture_family="lstm",
        non_interactive=True,
        sequence_bundle=None,
    )
    with pytest.raises(SequenceInputContractError, match="requires a genuine rank-3 temporal sequence"):
        run_interactive_review(cfg_bad_arch)

    # Case 2: Sequence bundle with tabular model -> ValueError
    sel = select_temporal_sequence(n_series=60, timesteps=10, n_features=2, seed=42)
    cfg_bad_model = ReviewConfig(
        data_path=None,
        target="target",
        architecture_family="mlp",
        non_interactive=True,
        dataset_selection=sel,
        sequence_bundle=sel.sequence_bundle,
    )
    with pytest.raises(ValueError, match="requiring recurrent architectures"):
        run_interactive_review(cfg_bad_model)


def test_review_orchestrator_sequence_review_execution(tmp_path):
    """ReviewOrchestrator runs full sequence pipeline and produces diagnostics evidence."""
    from start.data.selection import select_temporal_sequence
    from start.modeling.review_orchestrator import ReviewOrchestrator

    sel = select_temporal_sequence(n_series=100, timesteps=12, n_features=3, seed=42)
    orch = ReviewOrchestrator()
    outcome = orch.run(
        sel.frame,
        user_target=sel.target_column,
        run_dl=True,
        architecture="lstm",
        output_root=str(tmp_path),
        sequence_bundle=sel.sequence_bundle,
    )

    assert outcome.modality == "temporal_sequence"
    assert outcome.task_type == "binary_classification"
    assert outcome.recommended_family == "lstm"
    assert "train" in outcome.cohort_metrics
    assert "test" in outcome.cohort_metrics
    assert "oos" in outcome.cohort_metrics

    # Check evidence records
    test_ids = [r.test_id for r in outcome.evidence]
    assert "discovery.dataset_profile" in test_ids
    assert "deep_learning.performance_diagnostics" in test_ids
    assert "deep_learning.explainability_diagnostics" in test_ids
    assert "deep_learning.robustness_diagnostics" in test_ids

    # Discovery evidence checks
    disc_rec = next(r for r in outcome.evidence if r.test_id == "discovery.dataset_profile")
    assert disc_rec.metrics.get("modality") == "temporal_sequence"
    assert disc_rec.metrics.get("tensor_shape") == "[100, 12, 3]"
    assert disc_rec.metrics.get("timesteps") == 12


def test_enterprise_orchestrator_sequence_review_execution(tmp_path):
    """EnterpriseReviewOrchestrator executes sequence pipeline and produces registered artifacts."""
    from start.data.selection import select_temporal_sequence
    from start.modeling.enterprise_orchestrator import EnterpriseReviewOrchestrator

    sel = select_temporal_sequence(n_series=100, timesteps=12, n_features=3, seed=42)
    orch = EnterpriseReviewOrchestrator()
    outcome = orch.run(
        sel.frame,
        user_target=sel.target_column,
        run_dl=True,
        architecture="lstm",
        output_root=str(tmp_path),
        sequence_bundle=sel.sequence_bundle,
    )

    assert outcome.modality == "temporal_sequence"
    assert outcome.task_type == "binary_classification"
    assert outcome.model_execution is not None
    assert len(outcome.model_execution.split_table) == 3
    assert "train" in outcome.model_execution.metrics_by_split
    assert outcome.model_execution.explainability_method == "Temporal Input-Gradient Saliency"
    assert len(outcome.model_execution.artifacts) > 0
