"""v4.0.1 regression tests.

Every test here names the incident it prevents. A contract test with no
remembered failure behind it gets deleted the first time it is inconvenient.
"""

from __future__ import annotations

import pytest

from start.data.selection import (
    WIZARD_OPTIONS,
    DatasetKind,
    DatasetSelection,
    resolve_wizard_choice,
)
from start.modeling.config_propagation import (
    RULES,
    audit_propagation,
    resolve_path,
)


class _Frame:
    """Minimal stand-in so contract tests do not require pandas."""

    shape = (1000, 26)
    columns = ["is_fraud", "txn_amount_zscore", "merchant_risk_band"]


# --------------------------------------------------------------------------- #
# Dataset selection contract
#
# Incident: the wizard returned a display label, and the caller recovered a file
# path from it with `dataset_vector.split(": ", 1)[1]`. Every dataset option
# aborted with `ValueError: Unsupported tabular format ''` because none of the
# labels contained the substring the caller was sniffing for.
# --------------------------------------------------------------------------- #
def test_selection_carries_a_frame_not_a_label() -> None:
    """The resolved data travels as a value. Nothing parses a display string."""
    sel = DatasetSelection(
        kind=DatasetKind.SYNTHETIC,
        display_name="Synthetic — AML / transaction monitoring",
        frame=_Frame(),
        source_reference="generated locally; no external source",
        target_column="is_fraud",
    )
    assert sel.frame is not None
    assert sel.n_rows == 1000
    # A display name may say anything; it must never be load-bearing.
    sel.display_name = "anything at all"
    assert sel.frame is not None and sel.n_rows == 1000


def test_synthetic_data_may_not_cite_an_external_url() -> None:
    """Incident: a synthetic-labelled preset cited a UCI breast-cancer URL."""
    sel = DatasetSelection(
        kind=DatasetKind.SYNTHETIC,
        display_name="Synthetic Anomaly Detection & Transaction Monitoring",
        frame=_Frame(),
        source_reference="https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic",
        target_column="is_fraud",
    )
    errors = sel.consistency_errors()
    assert errors, "a synthetic dataset citing a URL must be rejected"
    assert any("no external source" in e for e in errors)


def test_medical_dataset_may_not_be_presented_as_something_else() -> None:
    """Incident: breast-cancer data shipped relabelled as fraud, then as attrition."""
    sel = DatasetSelection(
        kind=DatasetKind.PUBLIC_BENCHMARK,
        display_name="Synthetic AML / Transaction Monitoring",
        frame=_Frame(),
        source_reference="https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic",
        target_column="is_fraud",
    )
    assert any("presented as what it is" in e for e in sel.consistency_errors())


def test_public_benchmark_must_cite_its_source() -> None:
    sel = DatasetSelection(
        kind=DatasetKind.PUBLIC_BENCHMARK,
        display_name="UCI Statlog German Credit",
        frame=_Frame(),
        source_reference="a benchmark",
        target_column="is_fraud",
    )
    assert any("must cite its source URL" in e for e in sel.consistency_errors())


def test_user_supplied_data_records_its_path() -> None:
    sel = DatasetSelection(
        kind=DatasetKind.USER_SUPPLIED,
        display_name="Local file — x.csv",
        frame=_Frame(),
        source_reference="operator-supplied local file",
        target_column="is_fraud",
    )
    assert any("path it was read from" in e for e in sel.consistency_errors())


def test_declared_target_must_exist_in_the_frame() -> None:
    sel = DatasetSelection(
        kind=DatasetKind.SYNTHETIC,
        display_name="Synthetic",
        frame=_Frame(),
        source_reference="generated locally",
        target_column="not_a_column",
    )
    assert any("is not a column" in e for e in sel.consistency_errors())


def test_honest_selection_has_no_errors() -> None:
    sel = DatasetSelection(
        kind=DatasetKind.SYNTHETIC,
        display_name="Synthetic — AML / transaction monitoring",
        frame=_Frame(),
        source_reference="generated locally; no external source",
        target_column="is_fraud",
    )
    assert sel.consistency_errors() == []


def test_provenance_block_is_complete() -> None:
    """What goes in the evidence chain must answer 'where did this come from?'."""
    sel = DatasetSelection(
        kind=DatasetKind.SYNTHETIC,
        display_name="Synthetic",
        frame=_Frame(),
        source_reference="generated locally",
        target_column="is_fraud",
    )
    block = sel.provenance_dict()
    for key in ("kind", "source_reference", "target_derivation", "n_rows", "parameters"):
        assert key in block


def test_wizard_options_are_stable_keys() -> None:
    """Branching happens on the key, never on the menu text."""
    keys = [k for k, _ in WIZARD_OPTIONS]
    assert keys == ["1", "2", "3", "4"]


def test_unknown_choice_falls_back_to_synthetic_not_to_a_preset() -> None:
    """Incident: an unrecognised key fell through to the breast-cancer default."""
    pytest.importorskip("start.data.synthetic")
    messages: list[str] = []
    sel = resolve_wizard_choice("99", prompt=lambda *_: "", echo=messages.append)
    assert sel.kind is DatasetKind.SYNTHETIC
    assert sel.consistency_errors() == []


def test_failed_load_announces_the_fallback() -> None:
    """A silent substitution is how a demo ends up on data nobody chose."""
    pytest.importorskip("start.data.synthetic")
    messages: list[str] = []
    sel = resolve_wizard_choice("4", prompt=lambda *_: "/does/not/exist.csv", echo=messages.append)
    assert sel.kind is DatasetKind.SYNTHETIC
    assert any("Falling back" in m for m in messages)


# --------------------------------------------------------------------------- #
# Configuration propagation
#
# Incident: the reviewer enabled class-weight balancing, the agent recommended
# it, the reviewer accepted it, and the estimator trained without it. The model
# predicted zero positives out of sample and the review sealed anyway.
# --------------------------------------------------------------------------- #
class _Config:
    class_weight = "balanced"
    stratify = True
    train_prop = 0.7
    tuning_strategy = "bounded_random"
    tuning_trials = 5
    validation_scheme = "holdout"
    explain_method = "integrated_gradients"
    costlier_errors = "recall"
    architecture_family = "mlp"
    activation = "tanh"
    seed = 42


class _Estimator:
    class_weight = "balanced"
    family = "mlp"
    activation = "tanh"
    random_state = 42


def _propagated_context() -> dict:
    return {
        "estimator": _Estimator(),
        "fit_kwargs": {},
        "model_params": {
            "family": "mlp",
            "activation": "tanh",
            "random_state": 42,
            "class_weight": "balanced",
        },
        "split_plan": {"strategy": "stratified", "train_pct": 0.7},
        "tuning": {
            "strategy": "bounded_random",
            "n_trials": 5,
            "validation": "holdout",
            "primary_metric": "recall",
        },
        "explainability": {"method": "integrated_gradients"},
    }


def test_fully_propagated_run_passes_cleanly() -> None:
    """A correct run must produce no findings, or the audit is noise."""
    report = audit_propagation(_Config(), _propagated_context())
    assert report.ok, [f.setting for f in report.failures]


def test_dropped_class_weight_is_caught() -> None:
    """The actual D3 defect, as a test."""
    context = _propagated_context()
    context["estimator"].class_weight = None
    context["model_params"]["class_weight"] = None

    report = audit_propagation(_Config(), context)
    assert not report.ok
    failed = {f.setting for f in report.failures}
    assert "class_weight" in failed

    finding = next(f for f in report.failures if f.setting == "class_weight")
    assert "zero positives" in finding.consequence


def test_sample_weight_counts_as_class_weight_propagation() -> None:
    """sklearn paths apply balancing via sample_weight, not the constructor."""
    context = _propagated_context()
    context["estimator"].class_weight = None
    context["model_params"]["class_weight"] = None
    context["fit_kwargs"] = {"sample_weight": [0.5, 0.5, 9.0, 9.0]}

    report = audit_propagation(_Config(), context)
    assert "class_weight" not in {f.setting for f in report.failures}


def test_degenerate_sample_weight_is_not_evidence() -> None:
    """An all-ones weight vector applies no balancing at all."""
    context = _propagated_context()
    context["estimator"].class_weight = None
    context["model_params"]["class_weight"] = None
    context["fit_kwargs"] = {"sample_weight": [1.0, 1.0, 1.0, 1.0]}

    report = audit_propagation(_Config(), context)
    assert "class_weight" in {f.setting for f in report.failures}


def test_ignored_metric_override_is_caught() -> None:
    """Incident: the reviewer overrode to recall; tuning still reported auc_roc."""
    context = _propagated_context()
    context["tuning"]["primary_metric"] = "auc_roc"

    report = audit_propagation(_Config(), context)
    assert "metric_priority" in {f.setting for f in report.failures}


def test_recall_override_accepts_pr_auc_as_equivalent() -> None:
    """Routing 'recall' to PR-AUC is correct, not a mismatch."""
    context = _propagated_context()
    context["tuning"]["primary_metric"] = "pr_auc"
    report = audit_propagation(_Config(), context)
    assert "metric_priority" not in {f.setting for f in report.failures}


def test_stratify_boolean_matches_strategy_string() -> None:
    """stratify=True legitimately arrives as strategy='stratified'."""
    report = audit_propagation(_Config(), _propagated_context())
    assert "stratify" not in {f.setting for f in report.failures}


def test_unset_settings_are_skipped_not_failed() -> None:
    class Sparse:
        class_weight = None
        stratify = False
        train_prop = 0.0
        tuning_strategy = ""
        tuning_trials = 0
        validation_scheme = ""
        explain_method = ""
        costlier_errors = ""
        architecture_family = ""
        activation = ""
        seed = 0

    report = audit_propagation(Sparse(), {})
    assert report.ok
    assert all(f.status == "skipped" for f in report.findings)


def test_every_rule_explains_its_consequence() -> None:
    """A finding that does not say why it matters gets ignored."""
    for rule in RULES:
        assert len(rule.consequence) > 30, rule.setting
        assert rule.observation_points


def test_path_resolver_handles_nesting_and_absence() -> None:
    assert resolve_path({"a": [{"b": 7}]}, "a[0].b") == (True, 7)
    assert resolve_path({"a": 1}, "a.b.c") == (False, None)
    assert resolve_path({"a": {"b": None}}, "a.b") == (True, None)


def test_report_serialises_for_the_evidence_chain() -> None:
    report = audit_propagation(_Config(), _propagated_context())
    block = report.as_dict()
    assert block["ok"] is True
    assert block["checked"] == len(RULES)
    assert isinstance(block["findings"], list)
