"""v4.0.2 regression tests.

Each test names the incident it prevents. A contract test with no remembered failure
behind it gets deleted the first time it is inconvenient.
"""

from __future__ import annotations

import numpy as np
import pytest

from start.governance.challenge_disposition import (
    ChallengeDisposition,
    OverrideClass,
    challenge_factor,
    classify_challenge,
    classify_override,
    override_factor,
)
from start.reporting.figure_viewer import (
    FigurePresentation,
    FigureSpec,
    should_open_figures,
)


# --------------------------------------------------------------------------- #
# Challenge disposition
#
# Incident: the reviewer challenged ValidationAgent on age as an ECOA-protected
# characteristic. The agent conceded, citing evidence, that sign-off could not
# proceed without a fair-lending disparity analysis. Governance reported
# "no outstanding reviewer challenges" and READY WITH CONDITIONS, 0 blockers.
# --------------------------------------------------------------------------- #
def test_conceded_challenge_blocks_signoff() -> None:
    """The exact defect: a concession must reach the disposition."""
    conceded = classify_challenge(
        {
            "challenge_id": "c1",
            "agent": "ValidationAgent",
            "status": "closed",
            "conceded": True,
            "text": "age is protected under ECOA",
            "response": "This model cannot be signed off without a fair-lending analysis.",
            "evidence_used": ["sensitivity_analysis.rows"],
        }
    )
    assert conceded.disposition is ChallengeDisposition.CONCEDED
    assert conceded.blocks

    status, detail, _ = challenge_factor([conceded])
    assert status == "blocker"
    assert "conceded" in detail
    assert "ValidationAgent" in detail


def test_the_misleading_phrase_cannot_appear_when_a_challenge_was_raised() -> None:
    """'no outstanding reviewer challenges' is what hid the concession."""
    resolved = classify_challenge(
        {"challenge_id": "c1", "agent": "A", "status": "closed", "response": "answered"}
    )
    _, detail, _ = challenge_factor([resolved])
    assert "no outstanding reviewer challenges" not in detail
    assert "raised and resolved" in detail


def test_no_challenges_at_all_reads_correctly() -> None:
    status, detail, _ = challenge_factor([])
    assert status == "ok"
    assert "no reviewer challenges raised" in detail


def test_unanswered_challenge_blocks() -> None:
    outstanding = classify_challenge({"challenge_id": "c1", "agent": "A", "status": "open", "response": ""})
    assert outstanding.disposition is ChallengeDisposition.OUTSTANDING
    assert challenge_factor([outstanding])[0] == "blocker"


def test_concession_is_never_inferred_from_prose() -> None:
    """Keyword-sniffing an LLM response would fire on hedged wording and miss real
    concessions. The reviewer confirms; the machine records."""
    looks_conceding = classify_challenge(
        {
            "challenge_id": "c1",
            "agent": "A",
            "status": "closed",
            "response": "This cannot be signed off without evidence, which is present.",
            "conceded": False,
        }
    )
    assert looks_conceding.disposition is ChallengeDisposition.RESOLVED


# --------------------------------------------------------------------------- #
# Override classification
#
# Incident: both overrides in the live run were counted as governance concerns —
# including one made after the agent conceded, and one citing the dataset's own
# published 5:1 cost matrix.
# --------------------------------------------------------------------------- #
def test_override_after_agent_concession_is_not_a_concern() -> None:
    verdict = classify_override(
        {
            "key": "architecture",
            "recommended": "lstm",
            "effective": "mlp",
            "rationale": "Agent conceded recurrence is unjustified for static tabular data.",
            "agent_rationale": "User choice is appropriate; no change recommended.",
        },
        conceded_keys={"architecture"},
    )
    assert verdict.classification is OverrideClass.AGENT_ENDORSED
    assert verdict.severity == "informational"


def test_reasoned_override_is_informational_not_a_concern() -> None:
    verdict = classify_override(
        {
            "key": "metric_priority",
            "recommended": "balanced",
            "effective": "recall",
            "rationale": "Dataset publishes a 5:1 cost matrix; recall-weighted is defensible.",
        }
    )
    assert verdict.classification is OverrideClass.REASONED
    assert verdict.severity == "informational"


def test_unexplained_override_is_a_concern() -> None:
    verdict = classify_override(
        {"key": "architecture", "recommended": "lstm", "effective": "mlp", "rationale": ""}
    )
    assert verdict.classification is OverrideClass.UNEXPLAINED
    assert verdict.severity == "concern"
    assert override_factor([verdict])[0] == "concern"


def test_agent_boilerplate_does_not_count_as_reviewer_rationale() -> None:
    """A rationale copied from the agent is the agent's reasoning, not the reviewer's."""
    boilerplate = "User choice is appropriate for the data; no change recommended."
    verdict = classify_override(
        {
            "key": "architecture",
            "recommended": "lstm",
            "effective": "mlp",
            "rationale": boilerplate,
            "agent_rationale": boilerplate,
        }
    )
    assert verdict.classification is OverrideClass.UNEXPLAINED


def test_no_op_override_is_not_an_override() -> None:
    """Incident: metric_priority recorded status 'overridden' with recommended ==
    chosen, and governance counted it as a concern."""
    verdict = classify_override(
        {"key": "metric_priority", "recommended": "balanced", "effective": "balanced"}
    )
    assert verdict.classification is OverrideClass.NO_OP
    assert verdict.severity == "ok"
    assert override_factor([verdict])[0] == "ok"


def test_accepted_choice_is_no_op_even_if_recommended_and_effective_strings_differ() -> None:
    """The session's own record of what the reviewer did outranks a string diff."""
    v1 = classify_override(
        {"key": "fe:encoding", "recommended": "apply", "effective": "onehot", "choice": "accepted"}
    )
    assert v1.classification is OverrideClass.NO_OP
    assert v1.severity == "ok"

    v2 = classify_override(
        {"key": "fe:scaling", "recommended": "apply", "effective": "standardize", "status": "accepted"}
    )
    assert v2.classification is OverrideClass.NO_OP
    assert v2.severity == "ok"

    v3 = classify_override(
        {"key": "fe:outliers", "recommended": "apply", "effective": "iqr_1_5", "choice": "accept"}
    )
    assert v3.classification is OverrideClass.NO_OP
    assert v3.severity == "ok"


def test_architecture_override_impact_wording() -> None:
    from start.review_tables import _impact_for

    impact_overridden = _impact_for("architecture", "override", "mlp", recommended="lstm")
    assert impact_overridden == "trained user-selected mlp (recommended: lstm)"

    impact_accepted = _impact_for("architecture", "accept", "mlp", recommended="mlp")
    assert impact_accepted == "trained recommended mlp"


def test_override_factor_names_the_checkpoints() -> None:
    verdicts = [
        classify_override(
            {"key": "architecture", "recommended": "lstm", "effective": "mlp", "rationale": "r"},
            conceded_keys={"architecture"},
        ),
        classify_override(
            {"key": "metric_priority", "recommended": "balanced", "effective": "recall", "rationale": "r"}
        ),
    ]
    status, detail, _ = override_factor(verdicts)
    assert status == "informational"
    assert "architecture" in detail and "metric_priority" in detail
    assert "concession" in detail


# --------------------------------------------------------------------------- #
# Figure presentation
# --------------------------------------------------------------------------- #
def test_figures_never_open_under_pytest() -> None:
    """A test suite that spawns image viewers is one people stop running."""
    allowed, reason = should_open_figures(explicit=True)
    assert not allowed
    assert "pytest" in reason


def test_explicit_disable_wins() -> None:
    assert should_open_figures(explicit=False)[0] is False


def test_ci_suppresses_opening() -> None:
    allowed, reason = should_open_figures(explicit=True, env={"CI": "true"})
    assert not allowed


def test_figures_are_announced_even_when_not_opened() -> None:
    """A reviewer over SSH still needs to know what was produced and where."""
    messages: list[str] = []
    presentation = FigurePresentation(enabled=False, suppressed_reason="not a terminal", echo=messages.append)
    presentation.present(
        [FigureSpec(key="roc_curve", path="/tmp/roc.png", headline="AUC 0.8913", cohort="OOS")]
    )
    joined = "\n".join(messages)
    assert "[FIGURE 1/1]" in joined
    assert "/tmp/roc.png" in joined
    assert "AUC 0.8913" in joined
    assert "not a terminal" in joined


def test_a_viewer_failure_never_raises() -> None:
    messages: list[str] = []
    presentation = FigurePresentation(
        enabled=True,
        echo=messages.append,
        sleep=lambda _: None,
        opener=lambda _: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    with pytest.raises(RuntimeError):
        presentation.opener("/tmp/x.png")  # the injected opener does raise

    presentation.opener = lambda _: (False, "no viewer")
    result = presentation.present([FigureSpec(key="roc_curve", path="/tmp/roc.png")])
    assert result[0]["opened"] is False
    assert "could not open" in "\n".join(messages)


def test_figure_spec_carries_an_interpretation() -> None:
    spec = FigureSpec(key="calibration_curve", path="/tmp/c.png")
    assert "probabilities" in spec.reading().lower()
    assert spec.title().startswith("Calibration")


def test_presentation_summary_is_evidence_shaped() -> None:
    presentation = FigurePresentation(enabled=False, echo=lambda _: None)
    presentation.present([FigureSpec(key="roc_curve", path="/tmp/a.png")])
    block = presentation.as_evidence()
    assert block["figures_produced"] == 1
    assert block["figures_opened"] == 0
    assert block["opening_enabled"] is False


# --------------------------------------------------------------------------- #
# Reviewer control surface
# --------------------------------------------------------------------------- #
@pytest.fixture()
def frame():
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(42)
    n = 500
    data = pd.DataFrame(
        {
            "amount": np.abs(rng.normal(3000, 3500, n)),
            "duration": rng.integers(4, 72, n).astype(float),
            "age": rng.integers(19, 75, n).astype(float),
            "purpose": rng.choice(list("ABCDEFGHIJ"), n),
            "housing": rng.choice(["own", "rent", "free"], n),
        }
    )
    data.loc[rng.choice(n, 25, replace=False), "age"] = np.nan
    return data


def test_outlier_options_compute_real_counts(frame) -> None:
    """Every count must come from running the rule on the data, not from a table."""
    from start.modeling.method_options import outlier_options

    menu = outlier_options(frame, ["amount", "duration", "age"])
    keys = [o.key for o in menu.options]
    assert {"iqr", "iqr_wide", "percentile", "zscore", "none", "custom"} <= set(keys)

    iqr = menu.option("iqr")
    wide = menu.option("iqr_wide")
    assert iqr.affected > 0
    # A wider multiplier must flag no more points than a narrower one.
    assert wide.affected <= iqr.affected
    assert menu.option("none").affected == 0


def test_iqr_multiplier_is_adjustable(frame) -> None:
    """The original defect: 1.5 was hard-coded and invisible."""
    from start.modeling.method_options import outlier_options

    tight = outlier_options(frame, ["amount"], iqr_multiplier=1.0)
    loose = outlier_options(frame, ["amount"], iqr_multiplier=3.0)
    assert tight.option("iqr").affected >= loose.option("iqr").affected
    assert tight.option("iqr").parameters["multiplier"] == 1.0


def test_imputation_reports_rows_and_cells_distinctly(frame) -> None:
    from start.modeling.method_options import imputation_options

    menu = imputation_options(frame)
    assert menu.option("median_mode").affected == 25
    assert menu.option("drop_rows").affected == 25
    assert "indicator" in {o.key for o in menu.options}


def test_encoding_shows_cardinality_not_a_meaningless_ratio(frame) -> None:
    """A 'retained 0.0%' column on an encoding choice reads as data loss."""
    from start.modeling.method_options import encoding_options

    menu = encoding_options(frame, ["purpose", "housing"])
    onehot = menu.option("onehot")
    assert onehot.affected is None
    assert "13 columns" in onehot.detail or "columns" in onehot.detail
    assert "LEAKAGE" in menu.option("target").detail


def test_imbalance_warns_about_double_correction() -> None:
    """Incident: class weights enabled at configuration, then resampling proposed
    on top, with nothing telling the reviewer they compound."""
    from start.modeling.method_options import imbalance_options

    y = np.zeros(1000, dtype=int)
    y[:55] = 1
    menu = imbalance_options(y, already_weighted=True)
    assert "ALREADY ENABLED" in menu.option("class_weight").detail
    assert "already enabled" in menu.agent_reason
    assert "CALIBRATION" in menu.option("oversample").detail


def test_threshold_options_expose_the_operating_point() -> None:
    """Every metric in this product was computed at 0.5, which at low prevalence
    produced a model predicting zero positives."""
    from start.modeling.method_options import threshold_options

    rng = np.random.default_rng(7)
    y = rng.binomial(1, 0.30, 800)
    scores = np.clip(y * 0.35 + rng.normal(0.32, 0.16, 800), 0, 1)

    menu = threshold_options(y, scores, cost_false_negative=5.0, cost_false_positive=1.0)
    keys = {o.key for o in menu.options}
    assert {"fixed_050", "f1_optimal", "f2_optimal", "cost_optimal", "alert_budget"} <= keys

    for option in menu.options:
        if option.key != "custom":
            assert "TP=" in option.detail and "FN=" in option.detail

    # A 5:1 cost matrix must not select a threshold stricter than F1's.
    assert menu.option("cost_optimal").parameters["threshold"] <= (
        menu.option("f1_optimal").parameters["threshold"] + 1e-9
    )
    assert menu.recommended_option().key == "f2_optimal"


def test_alert_budget_respects_capacity() -> None:
    from start.modeling.method_options import threshold_options

    rng = np.random.default_rng(3)
    y = rng.binomial(1, 0.10, 1000)
    scores = rng.uniform(0, 1, 1000)
    menu = threshold_options(y, scores, alert_budget=0.05)
    alerts = menu.option("alert_budget").affected
    assert 30 <= alerts <= 70  # ~5% of 1000, allowing for ties


def test_every_option_carries_a_rationale(frame) -> None:
    """A menu without reasons is a quiz."""
    from start.modeling.method_options import (
        encoding_options,
        imputation_options,
        outlier_options,
        scaling_options,
    )

    menus = [
        outlier_options(frame, ["amount", "age"]),
        imputation_options(frame),
        encoding_options(frame, ["purpose"]),
        scaling_options(frame, ["amount", "age"]),
    ]
    for menu in menus:
        assert menu.options
        for option in menu.options:
            assert len(option.rationale) > 15, f"{menu.decision}/{option.key}"


def test_exactly_one_recommendation_per_menu(frame) -> None:
    from start.modeling.method_options import outlier_options

    menu = outlier_options(frame, ["amount"])
    assert sum(1 for o in menu.options if o.recommended) == 1


# --------------------------------------------------------------------------- #
# Benchmarking
# --------------------------------------------------------------------------- #
def test_a_model_that_barely_beats_a_stump_is_flagged() -> None:
    """Incident: MLP at OOS AUC 0.6717 vs a stump at 0.66, signed off
    READY WITH CONDITIONS with nobody having computed the stump."""
    pytest.importorskip("sklearn")
    pd = pytest.importorskip("pandas")
    from start.modeling.benchmark import benchmark_against_baselines

    rng = np.random.default_rng(11)
    n = 600
    signal = rng.normal(0, 1, n)
    y = (signal + rng.normal(0, 1.4, n) > 1.0).astype(int)
    X = pd.DataFrame({"signal": signal, "noise": rng.normal(0, 1, n)})

    split = 400
    weak_scores = X["signal"].to_numpy()[split:] * 0.5 + rng.normal(0, 2.0, n - split)

    report = benchmark_against_baselines(X.iloc[:split], y[:split], X.iloc[split:], y[split:], weak_scores)
    assert report.stump() is not None
    assert report.stump().auc is not None
    assert report.status in {"ok", "concern", "blocker"}
    assert report.verdict


def test_benchmark_includes_all_three_baselines() -> None:
    pytest.importorskip("sklearn")
    pd = pytest.importorskip("pandas")
    from start.modeling.benchmark import benchmark_against_baselines

    rng = np.random.default_rng(5)
    n = 400
    X = pd.DataFrame({"a": rng.normal(0, 1, n), "b": rng.normal(0, 1, n)})
    y = rng.binomial(1, 0.3, n)
    report = benchmark_against_baselines(X.iloc[:300], y[:300], X.iloc[300:], y[300:], rng.uniform(0, 1, 100))
    names = {b.name for b in report.baselines}
    assert names == {"majority_class", "base_rate", "decision_stump"}


def test_single_class_cohort_is_reported_not_crashed() -> None:
    pytest.importorskip("sklearn")
    pd = pytest.importorskip("pandas")
    from start.modeling.benchmark import benchmark_against_baselines

    X = pd.DataFrame({"a": np.arange(50, dtype=float)})
    y = np.zeros(50, dtype=int)
    report = benchmark_against_baselines(X, y, X, y, np.zeros(50))
    assert report.status == "unknown"
    assert "single class" in report.verdict
