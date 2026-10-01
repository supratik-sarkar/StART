"""Deterministic tests for sequence tensor preservation and split semantics.

Verifies that:
1. Genuine rank-3 sequence tensors (samples, timesteps, features) are preserved completely
   during model execution, without tabular coercion, flattening, or feature truncation.
2. The order-preserving 480/160/160 holdout is correctly reported and not falsely described
   as forecasting future chronological events over independent sequence samples.
"""

from __future__ import annotations

from start.data.selection import select_temporal_sequence
from start.modeling.model_execution import run_model_execution


def test_sequence_tensor_preservation_during_execution(tmp_path) -> None:
    """Prove that sequence tensor shape (480, 24, 3) is strictly preserved without tabular distortion."""
    selection = select_temporal_sequence(seed=42)
    bundle = selection.sequence_bundle
    assert bundle is not None

    # Initial shape assertions
    assert bundle.X_train.shape == (480, 24, 3)
    assert bundle.X_test.shape == (160, 24, 3)
    assert bundle.X_oos.shape == (160, 24, 3)
    assert bundle.timesteps == 24
    assert bundle.n_features == 3

    # Store exact snapshot of train tensor bytes before execution
    train_bytes_before = bundle.X_train.tobytes()

    # Run model execution with sequence bundle
    exec_result = run_model_execution(
        df=selection.frame,
        target="target",
        split_props=(0.6, 0.2, 0.2),
        metric_name="auc_roc",
        seed=42,
        output_root=str(tmp_path),
        run_id="TEST-SEQ-PRESERVE",
        architecture="lstm",
        sequence_bundle=bundle,
    )

    assert exec_result is not None
    # Verify train tensor was NOT mutated
    assert bundle.X_train.tobytes() == train_bytes_before
    assert bundle.X_train.shape == (480, 24, 3)
    assert bundle.X_test.shape == (160, 24, 3)
    assert bundle.X_oos.shape == (160, 24, 3)

    # Verify features in result match sequence channels
    assert len(exec_result.feature_columns) == 3


def test_split_semantics_no_forecasting_claims() -> None:
    """Verify split documentation describes order-preserving sequence holdout, not forecasting."""
    selection = select_temporal_sequence(seed=42)
    notes = " ".join(selection.notes).lower()

    # Must confirm order-preserving / sequence classification
    assert "order-preserving" in notes or "contiguous" in notes
    assert "sequence binary classification" in notes or "trajectory pattern detection" in notes

    # Must NOT claim forecasting
    assert "forecasting" not in notes
    assert "future observation" not in notes
