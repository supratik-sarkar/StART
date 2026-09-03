import io
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from start.interactive_checkpoints import read_multiline_paste_normalized, resolve_checkpoint
from start.interactive_review import ReviewConfig, run_interactive_review
from start.modeling.kfold_tuning import _make_estimator, _metric_fn
from start.progress import progress_bar, spinner


# 1. Multiline Paste Isolation Tests
def test_multiline_paste_isolation():
    # Simulate a multiline paste into stdin
    paste_content = "Q\nHow does this checkpoint work?\nAnd some other lines\n"

    with patch("sys.stdin", io.StringIO(paste_content)):
        # Stdin is mocked as StringIO which behaves like a TTY here if we mock isatty
        with patch("sys.stdin.isatty", return_value=True):
            sel_effect = [([sys.stdin], [], []), ([sys.stdin], [], []), ([], [], [])]
            with patch("select.select", side_effect=sel_effect):
                result = read_multiline_paste_normalized("Prompt: ")
                # Verify that it joins all lines with spaces
                assert result == "Q How does this checkpoint work? And some other lines"
                # Verify that stdin was fully consumed
                assert sys.stdin.read() == ""


# 2. Repeated Q&A at Specific Checkpoints
def test_repeated_qa_checkpoints():
    # Setup mock answers for a checkpoint loop
    # 1. Q: What is architecture?
    # 2. Q: Tell me more.
    # 3. A (Accept recommendation)
    answers = ["Q What is architecture?", "Q Tell me more.", "A"]
    ask_mock = MagicMock(side_effect=answers)

    on_ask_call_count = 0

    def on_ask_mock(question):
        nonlocal on_ask_call_count
        on_ask_call_count += 1
        return f"Answer to: {question}"

    say_calls = []

    def say_mock(msg):
        say_calls.append(msg)

    decision = resolve_checkpoint(
        name="architecture",
        user_value="mlp",
        recommended_value="wide_deep",
        reason="Better representation",
        ask=ask_mock,
        on_ask=on_ask_mock,
        emit=say_mock,
        interactive=True,
    )

    # Assert that the decision was resolved after the repeated Q&A
    assert decision.choice == "accept"
    assert decision.effective_value == "wide_deep"
    assert on_ask_call_count == 2
    assert any("Answer to: What is architecture?" in call for call in say_calls)
    assert any("Answer to: Tell me more." in call for call in say_calls)


# 3. Binary/Multiclass Mismatch Verification
def test_binary_multiclass_mismatch_reconciliation():
    # Verify non-interactive raises ValueError
    cfg = ReviewConfig(
        data_path="dummy.csv",
        target="label",
        task_override="binary_classification",
        non_interactive=True,
    )
    df = pd.DataFrame({"feat": [1, 2, 3], "label": [0, 1, 2]})  # 3 classes

    with patch("start.data.loaders.load_any_tabular", return_value=df):
        with pytest.raises(ValueError) as excinfo:
            run_interactive_review(cfg)
        assert "requires multiclass_classification" in str(excinfo.value)

    # Verify interactive prompting switches to multiclass when confirmed
    cfg_interactive = ReviewConfig(
        data_path="dummy.csv",
        target="label",
        task_override="binary_classification",
        non_interactive=False,
    )

    # Mock Confirm.ask to return True
    with patch("start.data.loaders.load_any_tabular", return_value=df):
        with patch("rich.prompt.Confirm.ask", return_value=True):
            with patch("start.interactive_review._run_enterprise"):
                run_interactive_review(cfg_interactive)
                # Check that task_override was switched to multiclass_classification
                assert cfg_interactive.task_override == "multiclass_classification"


# 4. Softmax vs Sigmoid & BCE loss routing
def test_loss_routing_dl_classifier():
    import torch

    from start.modeling.tabular_dl import TabularDLClassifier

    # Binary classifier routing: sigmoid, 1 output, BCEWithLogitsLoss
    bin_clf = TabularDLClassifier(task="binary_classification")
    assert bin_clf.n_outputs_ == 1

    X = np.random.randn(10, 5)
    y_bin = np.random.choice([0, 1], size=10)
    bin_clf.fit(X, y_bin)

    loss_fn_bin = bin_clf._loss_fn(bin_clf._prepare_targets(y_bin)[0], "cpu")
    assert isinstance(loss_fn_bin, torch.nn.BCEWithLogitsLoss)

    # Multiclass classifier routing: softmax, K outputs, CrossEntropyLoss
    multi_clf = TabularDLClassifier(task="multiclass_classification")
    y_multi = np.random.choice([0, 1, 2], size=10)
    multi_clf.fit(X, y_multi)
    assert multi_clf.n_outputs_ == 3

    loss_fn_multi = multi_clf._loss_fn(multi_clf._prepare_targets(y_multi)[0], "cpu")
    assert isinstance(loss_fn_multi, torch.nn.CrossEntropyLoss)


# 5. Class-aligned Multiclass metrics
def test_class_aligned_multiclass_metrics():
    # Mock estimator and data
    classes_order = np.array([3, 5, 9])
    y_true = np.array([3, 5, 9, 3, 5])
    # Shape: (5, 3)
    p_matrix = np.array(
        [
            [0.8, 0.1, 0.1],  # predicts 3
            [0.2, 0.7, 0.1],  # predicts 5
            [0.1, 0.1, 0.8],  # predicts 9
            [0.6, 0.3, 0.1],  # predicts 3
            [0.1, 0.8, 0.1],  # predicts 5
        ]
    )

    # F1 score metric function
    f1_fn = _metric_fn("f1", is_multiclass=True, classes_order=classes_order)
    score_f1 = f1_fn(y_true, p_matrix)
    # y_true is exactly matching predicted labels, so F1 should be 1.0
    assert score_f1 == 1.0

    # Test specificity metric function (should calculate correctly from confusion matrix)
    spec_fn = _metric_fn("specificity", is_multiclass=True, classes_order=classes_order)
    score_spec = spec_fn(y_true, p_matrix)
    assert score_spec == 1.0


# 6. Fold-local Preprocessing Pipeline Checks
def test_fold_local_pipeline_preprocessing():
    # Setup data with infinite values and NaNs
    df = pd.DataFrame(
        {
            "feat1": [1.0, np.inf, 3.0, np.nan, 5.0, -np.inf],
            "feat2": [2.0, 4.0, np.nan, 8.0, 10.0, 12.0],
            "target": [0, 1, 0, 1, 0, 1],
        }
    )

    # Verify _make_estimator pipeline processes these non-finite inputs cleanly
    est = _make_estimator(C=1.0, class_weight=None, seed=42)

    X = df[["feat1", "feat2"]].to_numpy()
    y = df["target"].to_numpy()

    # Fit the pipeline
    # This must fit without raising any ValueError or overflow warnings
    est.fit(X, y)

    # Check that predictions can be made cleanly
    preds = est.predict_proba(X)
    assert preds.shape == (6, 2)
    assert np.isfinite(preds).all()


# 7. Exception Propagation in Context Managers
def test_progress_exception_propagation():
    # Spinner exception propagation
    with pytest.raises(ZeroDivisionError):
        with spinner("Testing spinner", enabled=True):
            raise ZeroDivisionError("Inside spinner")

    # Progress Bar exception propagation
    with pytest.raises(ValueError):
        with progress_bar(total=10, description="Testing progress", enabled=True):
            raise ValueError("Inside progress bar")


# 8. Sequence Classifier Multiclass & Tuning Run Multiclass Tests
def test_sequence_classifier_multiclass():
    from start.modeling.sequence_dl import SequenceClassifier

    # 3 classes target
    clf = SequenceClassifier(task="multiclass_classification")
    X = np.random.randn(15, 4, 6)  # (n_samples, timesteps, features)
    y = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2])

    # Fit the classifier
    clf.fit(X, y)
    assert clf.n_outputs_ == 3
    assert len(clf.classes_) == 3

    # Predict and predict_proba shape
    probs = clf.predict_proba(X)
    assert probs.shape == (15, 3)

    preds = clf.predict(X)
    assert preds.shape == (15,)

    # Score
    sc = clf.score(X, y)
    assert 0.0 <= sc <= 1.0


def test_tuning_run_multiclass():
    from start.modeling.tuning_run import run_tuning

    # Create simple dataframe with 3 classes for multiclass classification tuning
    df = pd.DataFrame(
        {
            "feat1": np.random.randn(45),
            "feat2": np.random.randn(45),
            "target": np.random.choice([0, 1, 2], size=45),
        }
    )

    space = {"hidden_size": [16, 32], "learning_rate": [1e-3, 5e-3], "dropout": [0.0, 0.2]}

    # Run tuning with K-fold validation scheme
    run = run_tuning(
        df=df,
        target="target",
        features=["feat1", "feat2"],
        architecture="lstm",
        task_type="multiclass_classification",
        primary_metric="f1",
        validation="k_fold",
        k_folds=2,
        n_trials=2,
        seed=42,
        custom_space=space,
    )

    assert run is not None
    assert run.best_params != {}
    assert len(run.trials) == 2
