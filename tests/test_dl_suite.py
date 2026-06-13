from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="DL suite requires the [torch] extra")

from start.modeling.deep_learning import build_classifier, captum_available  # noqa: E402
from start.modeling.dl_data import (  # noqa: E402
    TARGET_COLUMN,
    feature_columns,
    load_dl_bundle,
    load_dl_demo_dataset,
)
from start.modeling.dl_metrics import (  # noqa: E402
    DL_METRIC_NAMES,
    compute_dl_cohort_metrics,
    expected_calibration_error,
    training_diagnostics,
)
from start.modeling.dl_sensitivity import (  # noqa: E402
    feature_masking_robustness,
    feature_shock_sensitivity,
    input_noise_robustness,
)
from start.modeling.dl_training import DL_ARCHITECTURES, DLReviewOptions, run_dl_review  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
FAKE_KEY = "sk-dl-suite-FAKE-do-not-leak-9999"


@pytest.fixture(scope="module")
def dl_splits():
    df = load_dl_demo_dataset(seed=0, n=900)
    from start.modeling.dl_data import three_way_split

    return three_way_split(df, seed=0)


# --------------------------------------------------------------------------- #
# Architectures
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("architecture", ["mlp", "leaky_relu_mlp", "residual_mlp", "wide_deep"])
def test_each_architecture_builds_and_trains(architecture, dl_splits):
    from sklearn.metrics import roc_auc_score

    train, test, _ = dl_splits
    features = feature_columns(train)
    # Wide & Deep's wide linear path needs a slightly higher LR to converge in
    # the few-epoch laptop-safe regime; this is a known training dynamic.
    lr = 3e-3 if architecture == "wide_deep" else 1e-3
    model = build_classifier(architecture, epochs=8, learning_rate=lr, random_state=0)
    model.fit(train[features], train[TARGET_COLUMN])
    proba = model.predict_proba(test[features])
    assert proba.shape == (len(test), 2)
    assert abs(proba.sum(axis=1) - 1.0).max() < 1e-6
    auc = roc_auc_score(test[TARGET_COLUMN], proba[:, 1])
    assert auc > 0.65, f"{architecture} should learn signal; got AUC {auc:.3f}"
    assert model.architecture == architecture
    assert len(model.history_["train_loss"]) >= 1


def test_roadmap_architectures_raise():
    for arch in ("lstm", "gru", "tcn", "transformer", "tft"):
        with pytest.raises(NotImplementedError, match="roadmap"):
            build_classifier(arch)
    assert set(DL_ARCHITECTURES) == {"mlp", "leaky_relu_mlp", "residual_mlp", "wide_deep"}


def test_training_history_and_early_stopping(dl_splits):
    train, _, _ = dl_splits
    features = feature_columns(train)
    model = build_classifier("mlp", epochs=10, early_stopping_patience=1, random_state=0)
    model.fit(train[features], train[TARGET_COLUMN])
    assert model.history_["val_loss"], "validation loss should be tracked"
    assert 1 <= model.best_epoch_ <= 10
    diag = training_diagnostics(model.history_, model.best_epoch_, model.stopped_early_)
    assert "generalization_gap" in diag and diag["epochs_run"] >= 1


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def test_dl_metrics_include_calibration(dl_splits):
    train, test, _ = dl_splits
    features = feature_columns(train)
    model = build_classifier("mlp", epochs=8, random_state=0)
    model.fit(train[features], train[TARGET_COLUMN])
    scores = model.predict_proba(test[features])[:, 1]
    metrics = compute_dl_cohort_metrics(test[TARGET_COLUMN].to_numpy(), scores)
    assert set(metrics) == set(DL_METRIC_NAMES)
    assert 0.0 <= metrics["brier_score"] <= 1.0
    assert 0.0 <= metrics["ece"] <= 1.0


def test_ece_perfect_calibration_is_low():
    # scores equal to the true rate in each region -> small ECE
    y = np.array([0, 0, 1, 1] * 50)
    perfect = y.astype(float) * 0.99 + 0.005
    assert expected_calibration_error(y, perfect) < 0.05


# --------------------------------------------------------------------------- #
# Sensitivity / robustness
# --------------------------------------------------------------------------- #
def test_sensitivity_and_robustness_baselines(dl_splits):
    train, test, _ = dl_splits
    features = feature_columns(train)
    model = build_classifier("mlp", epochs=8, random_state=0)
    model.fit(train[features], train[TARGET_COLUMN])
    y = test[TARGET_COLUMN].to_numpy()

    shocks = feature_shock_sensitivity(model, test[features], y, features[:5])
    zero = next(r for r in shocks if r["shock"] == 0.0)
    assert zero["auc_drift"] == 0.0  # baseline by construction
    assert len(shocks) == 7

    noise = input_noise_robustness(model, test[features], y, features[:5], seed=0)
    zero_noise = next(r for r in noise if r["noise"] == 0.0)
    assert zero_noise["auc_drift"] == 0.0
    assert len(noise) == 5

    masks = feature_masking_robustness(model, test[features], y, features[:5])
    assert [r["masked_top_k"] for r in masks] == [1, 3, 5]
    assert all("masked_features" in r for r in masks)


# --------------------------------------------------------------------------- #
# Explainability routing
# --------------------------------------------------------------------------- #
def test_dl_explain_routes_and_falls_back(dl_splits):
    from start.modeling.dl_explain import dl_global_importance

    train, test, _ = dl_splits
    features = feature_columns(train)
    model = build_classifier("mlp", epochs=6, random_state=0)
    model.fit(train[features], train[TARGET_COLUMN])

    result = dl_global_importance(
        model, test[features], test[TARGET_COLUMN].to_numpy(), prefer="integrated_gradients", seed=0
    )
    if captum_available():
        assert result.method == "integrated_gradients" and not result.note
        assert len(result.top_features(5)) == 5
    else:
        assert result.method == "permutation" and "Captum" in result.note


def test_dl_explain_permutation_fallback_when_captum_absent(monkeypatch, dl_splits):
    import start.modeling.deep_learning as dl

    monkeypatch.setattr(dl, "captum_available", lambda: False)
    from start.modeling.dl_explain import dl_global_importance

    train, test, _ = dl_splits
    features = feature_columns(train)
    model = build_classifier("mlp", epochs=5, random_state=0)
    model.fit(train[features], train[TARGET_COLUMN])
    result = dl_global_importance(
        model, test[features], test[TARGET_COLUMN].to_numpy(), prefer="integrated_gradients", seed=0
    )
    assert result.method == "permutation"
    assert "Captum" in result.note and "shap" not in result.note.lower()


# --------------------------------------------------------------------------- #
# Connector-backed data (bring your own data)
# --------------------------------------------------------------------------- #
def test_load_dl_bundle_pandas_and_files(tmp_path):
    df = load_dl_demo_dataset(seed=1, n=300)
    bundle = load_dl_bundle("pandas", train_df=df, target_column=TARGET_COLUMN, seed=0)
    assert bundle.test is not None and bundle.oos is not None
    total = len(bundle.train) + len(bundle.test) + len(bundle.oos)
    assert total == len(df)

    path = tmp_path / "dl.csv"
    df.to_csv(path, index=False)
    file_bundle = load_dl_bundle("files", train_path=str(path), target_column=TARGET_COLUMN)
    assert len(file_bundle.train) > 0


# --------------------------------------------------------------------------- #
# Full review workflow: evidence, figures, agent review, report
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def review_result(tmp_path_factory):
    out = tmp_path_factory.mktemp("dlrun")
    opts = DLReviewOptions(architecture="mlp", epochs=6, seed=0, output_root=str(out))
    result = run_dl_review(opts)
    return result, out


def test_review_produces_all_evidence_labels(review_result):
    result, _ = review_result
    labels = {rec.artifacts.get("dl_evidence_label") for rec in result.evidence}
    assert labels == {f"EV-DL-{i:04d}" for i in range(1, 8)}
    assert all(rec.policy_hash for rec in result.evidence)


def test_review_cohort_metrics_and_device(review_result):
    result, _ = review_result
    assert set(result.cohort_metrics) == {"train", "test", "oos"}
    for cohort in result.cohort_metrics.values():
        assert set(cohort) == set(DL_METRIC_NAMES)
    assert result.device in {"cpu", "mps", "cuda"}


def test_review_generates_figures(review_result):
    result, out = review_result
    from start.modeling.dl_figures import matplotlib_available

    if matplotlib_available():
        assert set(result.figures) >= {
            "learning_curve",
            "calibration_curve",
            "attribution_top_features",
            "top_feature_shock_sensitivity",
        }
        fig_dir = out / "figures" / "deep_learning" / result.run_id
        for path in result.figures.values():
            assert Path(path).exists() and Path(path).stat().st_size > 0
        assert fig_dir.exists()


def test_review_agent_section_and_ledger(review_result):
    result, out = review_result
    assert result.agent_review is not None
    assert result.agent_review.mode == "deterministic"
    assert result.agent_review.signoff
    assert result.narrative_ok  # citation gate passes
    assert (out / "ledger.jsonl").exists()
    from start.evidence.ledger import EvidenceLedger

    ledger = EvidenceLedger(out / "ledger.jsonl", out / "evidence_store")
    assert ledger.verify()  # tamper-evident chain intact


def test_review_report_has_required_sections(review_result):
    result, _ = review_result
    report = Path(result.report_path).read_text()
    for section in (
        "## Run metadata",
        "## Cohort metrics",
        "## Evidence table",
        "## Explainability",
        "## Sensitivity",
        "## Robustness",
        "## Generated figures",
        "## Agentic review",
        "### Sign-off recommendation",
        "## Reproducibility metadata",
    ):
        assert section in report, f"missing {section}"
    assert "EV-DL-0001" in report and "EV-DL-0007" in report
    assert f"Architecture: `{result.architecture}`" in report


# --------------------------------------------------------------------------- #
# Agent modes
# --------------------------------------------------------------------------- #
def test_llm_mode_without_key_falls_back_explicitly(tmp_path, monkeypatch):
    for var in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(var, raising=False)

    opts = DLReviewOptions(
        architecture="mlp", epochs=4, seed=0, output_root=str(tmp_path),
        agent_mode="llm", llm_provider="none",
    )
    result = run_dl_review(opts)
    # provider 'none' is unavailable -> explicit deterministic fallback
    assert result.agent_review.mode == "deterministic"
    assert any("fell back" in n.lower() or "deterministic" in n.lower()
               for n in result.agent_review.notes)


def test_llm_mode_with_fake_provider_passes_gate(tmp_path):
    from tests.test_agent_review import FakeLLM

    from start.modeling import dl_training

    train_opts = DLReviewOptions(
        architecture="mlp", epochs=4, seed=0, output_root=str(tmp_path),
        agent_mode="llm", llm_provider="openai",
    )

    class CitingFake(FakeLLM):
        def complete(self, system: str, user: str, *, max_tokens: int = 1024) -> str:
            import re

            self.calls.append((system, user))
            ids = re.findall(r"\[EV-[A-Za-z0-9]+\]|EV-DL-\d{4}", user)
            ev = next((i for i in ids if i.startswith("[")), "")
            return f"- Evidence reviewed across all diagnostics. {ev}".strip()

    # inject the fake provider in place of the real resolver
    fake = CitingFake([])
    dl_training._resolve_llm = lambda opts: fake  # type: ignore[attr-defined]
    try:
        result = run_dl_review(train_opts)
    finally:
        import importlib

        importlib.reload(dl_training)
    assert result.agent_review.mode == "llm"
    assert result.agent_review.llm_provider == "fake"


# --------------------------------------------------------------------------- #
# No secret leakage
# --------------------------------------------------------------------------- #
def test_no_key_leakage_in_dl_outputs(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", FAKE_KEY)
    opts = DLReviewOptions(architecture="mlp", epochs=4, seed=0, output_root=str(tmp_path))
    result = run_dl_review(opts)
    report = Path(result.report_path).read_text()
    assert FAKE_KEY not in report
    for artifact in Path(tmp_path).rglob("*"):
        if artifact.is_file() and artifact.suffix in {".md", ".jsonl", ".json", ".txt"}:
            assert FAKE_KEY not in artifact.read_text(errors="ignore"), artifact


# --------------------------------------------------------------------------- #
# Notebook + config sanity
# --------------------------------------------------------------------------- #
def test_dl_notebook_compiles():
    import py_compile

    py_compile.compile(
        str(REPO_ROOT / "notebooks" / "03_deep_learning_model_review.py"), doraise=True
    )


def test_dl_ipynb_is_valid_notebook():
    import json

    nb = json.loads((REPO_ROOT / "notebooks" / "03_deep_learning_model_review.ipynb").read_text())
    assert nb["nbformat"] == 4
    assert len(nb["cells"]) > 5
    code = "\n".join(
        "".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"
    )
    assert "run_dl_review" in code
    assert "ensure_provider_key" in code  # secure key handling present
    # no hardcoded secrets / endpoints in the notebook
    assert "sk-" not in code and "api.openai.com" not in code


def test_dl_config_loads():
    import yaml

    cfg = yaml.safe_load((REPO_ROOT / "configs" / "dl_default.yaml").read_text())
    assert cfg["agent"]["mode"] == "deterministic"
    assert cfg["deep_learning"]["epochs"] <= 10
    assert cfg["deep_learning"]["batch_size"] <= 128
