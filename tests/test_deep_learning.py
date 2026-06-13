from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from start.modeling.data import TARGET_COLUMN, feature_columns, load_attrition_dataset, three_way_split
from start.modeling.deep_learning import (
    SUPPORTED_ARCHITECTURES,
    build_classifier,
    captum_available,
    integrated_gradients_importance,
)

torch = pytest.importorskip("torch", reason="DL tests require the [torch] extra")

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def splits():
    df = load_attrition_dataset(seed=0)
    return three_way_split(df, seed=0)


@pytest.fixture(scope="module")
def fitted_mlp(splits):
    train, _, _ = splits
    features = feature_columns(train)
    model = build_classifier("mlp", epochs=8, random_state=0)
    model.fit(train[features], train[TARGET_COLUMN])
    return model, features


def test_laptop_safe_constraints():
    model = build_classifier("mlp")
    assert model.epochs <= 10 and model.batch_size <= 128
    with pytest.raises(ValueError, match="epochs"):
        build_classifier("mlp", epochs=50)
    with pytest.raises(ValueError, match="batch_size"):
        build_classifier("mlp", batch_size=4096)
    with pytest.raises(NotImplementedError, match="roadmap"):
        build_classifier("lstm")
    with pytest.raises(ValueError, match="Unknown architecture"):
        build_classifier("quantum_net")
    assert set(SUPPORTED_ARCHITECTURES) >= {"mlp", "rnn", "lstm", "gru", "tcn"}


def test_mlp_fit_learns_signal(fitted_mlp, splits):
    from sklearn.metrics import roc_auc_score

    model, features = fitted_mlp
    _, test, _ = splits
    proba = model.predict_proba(test[features])
    assert proba.shape == (len(test), 2)
    assert ((proba >= 0) & (proba <= 1)).all()
    assert abs(proba.sum(axis=1) - 1.0).max() < 1e-6
    auc = roc_auc_score(test[TARGET_COLUMN], proba[:, 1])
    assert auc > 0.9, f"MLP should learn the signal; got AUC {auc:.3f}"
    assert model.device_used in {"cpu", "mps", "cuda"}
    preds = model.predict(test[features])
    assert set(preds) <= {0, 1}


def test_mlp_sklearn_protocol(fitted_mlp):
    model, _ = fitted_mlp
    params = model.get_params()
    assert params["epochs"] == 8 and "learning_rate" in params
    clone = build_classifier("mlp").set_params(**{**params, "epochs": 5})
    assert clone.epochs == 5
    with pytest.raises(ValueError, match="Unknown parameter"):
        clone.set_params(banana=1)
    leaky = build_classifier("leaky_relu_mlp")
    assert leaky.activation == "leaky_relu"


def test_dl_metrics_match_classical_contract(fitted_mlp, splits):
    from start.modeling.metrics import METRIC_NAMES, compute_cohort_metrics

    model, features = fitted_mlp
    _, test, _ = splits
    metrics = compute_cohort_metrics(
        test[TARGET_COLUMN].to_numpy(), model.predict_proba(test[features])[:, 1]
    )
    assert set(metrics) == set(METRIC_NAMES)
    assert metrics["top_decile_lift"] > 1.5


def test_integrated_gradients_or_honest_unavailable(fitted_mlp, splits):
    model, features = fitted_mlp
    _, test, _ = splits
    method, ranked, note = integrated_gradients_importance(model, test[features], seed=0)
    if captum_available():
        assert method == "integrated_gradients" and len(ranked) == len(features)
        assert not note
    else:
        assert method == "unavailable" and "Captum" in note


def test_ig_degrades_when_captum_missing(monkeypatch, fitted_mlp, splits):
    import start.modeling.deep_learning as dl

    model, features = fitted_mlp
    _, test, _ = splits
    monkeypatch.setattr(dl, "captum_available", lambda: False)
    method, ranked, note = dl.integrated_gradients_importance(model, test[features])
    assert method == "unavailable" and ranked == []
    assert "permutation" in note  # honest redirection, no SHAP claim
    assert "shap" not in note.lower()


def test_dl_evidence_records_like_classical(tmp_path, monkeypatch, fitted_mlp, splits):
    """The MLP must flow through the evidence pipeline exactly like a
    classical model: cohort metrics, lift, sensitivity, importance — plus the
    DL-specific Integrated Gradients record."""
    monkeypatch.chdir(tmp_path)
    from start.modeling.data import SCORE_COLUMN
    from start.orchestration import review_dataframes

    model, features = fitted_mlp
    train, test, oos = (f.copy() for f in splits)
    for frame in (train, test, oos):
        frame[SCORE_COLUMN] = model.predict_proba(frame[features])[:, 1]
    result = review_dataframes(train, test, oos, target_column=TARGET_COLUMN, model=model, seed=0)
    by_id = {rec.test_id: rec for rec in result.evidence}
    assert by_id["supervised.cohort_metrics_comparison"].status.value in {"pass", "warn", "fail"}
    assert "oos_auc_roc" in by_id["supervised.cohort_metrics_comparison"].metrics
    assert by_id["xai.feature_sensitivity"].metrics["drift_+0pct"] == 0.0
    importance = by_id["xai.global_importance"]
    assert importance.metrics["method"] == "permutation"  # never SHAP for DL
    ig = by_id["xai.integrated_gradients"]
    if captum_available():
        assert ig.status.value == "pass"
        assert ig.metrics["method"] == "integrated_gradients"
    else:
        assert ig.status.value == "skipped"
    assert result.critique is not None and result.critique.ok
    assert (Path("start_output") / "ledger.jsonl").exists()


def test_ig_engine_skips_for_classical_models(tmp_path, monkeypatch, splits):
    monkeypatch.chdir(tmp_path)
    from sklearn.ensemble import RandomForestClassifier

    from start.modeling.data import SCORE_COLUMN
    from start.orchestration import review_dataframes

    train, test, oos = (f.copy() for f in splits)
    features = feature_columns(train)
    rf = RandomForestClassifier(n_estimators=50, random_state=0, n_jobs=-1)
    rf.fit(train[features], train[TARGET_COLUMN])
    for frame in (train, test, oos):
        frame[SCORE_COLUMN] = rf.predict_proba(frame[features])[:, 1]
    result = review_dataframes(train, test, oos, target_column=TARGET_COLUMN, model=rf, seed=0)
    ig = next(r for r in result.evidence if r.test_id == "xai.integrated_gradients")
    assert ig.status.value == "skipped"
    assert "not applicable" in ig.interpretation


def test_explainability_router_recognizes_mlp(fitted_mlp):
    from start.modeling.explain import detect_model_family, route_explainability

    model, _ = fitted_mlp
    assert detect_model_family(model) == "deep_learning"
    plan = route_explainability(model)
    if captum_available():
        assert "integrated_gradients" in plan.implemented()
    assert "permutation_sensitivity" in plan.implemented()


def test_mlp_resolves_via_model_factory():
    from start.modeling.models import HYPERPARAM_SPACES, resolve_model

    model, name, note = resolve_model("mlp", seed=0)
    assert name == "mlp" and note == ""
    assert set(HYPERPARAM_SPACES["mlp"]) == {
        "epochs",
        "batch_size",
        "learning_rate",
        "dropout",
        "activation",
    }


def test_notebook_03_smoke_execution(tmp_path):
    """The DL notebook must run end-to-end as a plain script (the local
    VS Code / Jupyter contract) using fast-mode training."""
    env = {**os.environ, "START_NB_FAST": "1"}
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "notebooks" / "03_deep_learning_model_review.py")],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=420,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "Sign-off recommendation" in proc.stdout
    assert "Report:" in proc.stdout
    assert (tmp_path / "start_output" / "ledger.jsonl").exists()
