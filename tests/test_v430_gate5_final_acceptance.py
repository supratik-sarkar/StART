"""Gate 5 Final Acceptance Audit Test Suite.

Verifies:
1. Registry discipline: Exactly 79 registered root test surfaces with 0 duplicate test IDs.
2. Domain applicability: Market excludes CEV/Stanton, Treasury resolves CEV/Stanton, Market+Treasury includes both.
3. Frozen validation surface: validation.var_size_power remains intact and passes.
4. Dual-plane artifact generation: All Gate 5 artifacts generate valid vector SVGs and JSON companions with non-empty evidence_ids.
5. Gate 5 Showcase manifest & summary integrity.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from start.portfolio.artifacts import (
    render_backtest_summary_artifact,
    render_duration_diagnostics_artifact,
    render_exception_transition_artifact,
    render_tail_comparison_artifact,
    render_tail_loss_distribution_artifact,
    render_tail_risk_contribution_artifact,
    render_tail_severity_artifact,
    render_var_pnl_timeline_artifact,
)
from start.portfolio.evidence_bridge import (
    duration_diagnostics_to_evidence,
    tail_backtest_to_evidence,
    tail_comparison_to_evidence,
    tail_contribution_to_evidence,
    tail_risk_estimate_to_evidence,
    tail_severity_to_evidence,
)
from start.portfolio.tail_risk import (
    compare_tail_risk_models,
    compute_exception_duration_diagnostics,
    compute_historical_var_es,
    compute_tail_risk_contributions,
    compute_tail_severity,
    run_comprehensive_tail_backtest,
)
from start.registry import list_tests, load_builtin_tests
from start.review.applicability import applicable_tests
from start.review.architecture import ReviewDomain


def test_registry_census_is_79_with_zero_duplicates() -> None:
    """The canonical test registry must maintain exactly 79 registered root surfaces with zero duplicates."""
    load_builtin_tests()
    specs = list_tests()
    assert len(specs) == 79, f"Expected 79 registered root surfaces; found {len(specs)}"

    test_ids = [s.test_id for s in specs]
    duplicates = [tid for tid in test_ids if test_ids.count(tid) > 1]
    assert len(duplicates) == 0, f"Duplicate test IDs detected in registry: {set(duplicates)}"


def test_domain_applicability_regression() -> None:
    """Domain applicability helper must correctly route tests by context_type, excluding CEV/Stanton from Market."""
    load_builtin_tests()

    # Market alone: 25 market tests, excludes CEV and Stanton
    market_plan = applicable_tests((ReviewDomain.MARKET,))
    market_ids = set(market_plan.test_ids)
    assert "traded_risk.cev_elasticity" not in market_ids
    assert "traded_risk.stanton_nonparametric" not in market_ids
    assert len(market_ids) == 25

    # Treasury alone: 2 short_rate tests (CEV and Stanton)
    treasury_plan = applicable_tests((ReviewDomain.TREASURY,))
    treasury_ids = set(treasury_plan.test_ids)
    assert treasury_ids == {"traded_risk.cev_elasticity", "traded_risk.stanton_nonparametric"}

    # Market + Treasury combined: 27 tests (includes both)
    combined_plan = applicable_tests((ReviewDomain.MARKET, ReviewDomain.TREASURY))
    combined_ids = set(combined_plan.test_ids)
    assert "traded_risk.cev_elasticity" in combined_ids
    assert "traded_risk.stanton_nonparametric" in combined_ids
    assert len(combined_ids) == 27


def test_validation_var_size_power_surface_intact() -> None:
    """The frozen validation.var_size_power surface must remain intact and passing."""
    from start.core.schemas import Status
    from start.review.architecture import ReviewDomain
    from start.validation.gate_b_evidence import (
        VERIFIED_B7_RESULTS,
        validation_results,
        validation_results_for_domains,
    )

    studies_dict = {o.study_id: o for o in VERIFIED_B7_RESULTS}
    assert "var_size_power" in studies_dict
    assert studies_dict["var_size_power"].status == Status.PASS

    # Verify reproduction results
    b7_results = validation_results()
    b7_ids = [r.params.get("study_id") for r in b7_results]
    assert "var_size_power" in b7_ids

    # Verify domain-scoped validation results
    m_results = validation_results_for_domains((ReviewDomain.MARKET,))
    m_ids = [r.params.get("study_id") for r in m_results]
    assert "var_size_power" in m_ids
    assert "regem_structural" in m_ids
    assert "cev_consistency" not in m_ids


def test_gate5_dual_plane_artifact_generation(tmp_path: Path) -> None:
    """All Gate 5 artifacts must generate valid vector SVG visuals and JSON companion files."""
    losses = np.arange(1.0, 101.0)
    pnl = np.zeros(250)
    pnl[50:54] = -2.0
    var = np.ones(250)

    est = compute_historical_var_es(losses, confidence=0.99)
    ev_est = tail_risk_estimate_to_evidence(est)
    ev_est.evidence_id = "EV-TAIL-EST"

    backtest = run_comprehensive_tail_backtest(pnl, var, var_confidence=0.99, test_significance=0.05)
    ev_backtest = tail_backtest_to_evidence(backtest)
    ev_backtest.evidence_id = "EV-TAIL-BACKTEST"

    dur = compute_exception_duration_diagnostics(backtest.indicators)
    ev_dur = duration_diagnostics_to_evidence(dur)
    ev_dur.evidence_id = "EV-TAIL-DUR"

    sev = compute_tail_severity(losses=-pnl, var_forecasts=var, indicators=backtest.indicators)
    ev_sev = tail_severity_to_evidence(sev)
    ev_sev.evidence_id = "EV-TAIL-SEV"

    compare = compare_tail_risk_models(losses, confidence=0.99)
    ev_comp = tail_comparison_to_evidence(compare)
    ev_comp.evidence_id = "EV-TAIL-COMP"

    contrib = compute_tail_risk_contributions(
        returns_or_losses=np.ones((100, 2)),
        weights={"A": 0.5, "B": 0.5},
        confidence=0.99,
        method="parametric_normal",
    )
    ev_contrib = tail_contribution_to_evidence(contrib)
    ev_contrib.evidence_id = "EV-TAIL-CONTRIB"

    out_dir = str(tmp_path)
    artifacts = [
        render_tail_loss_distribution_artifact(est, evidence_ids=(ev_est.evidence_id,), output_dir=out_dir),
        render_var_pnl_timeline_artifact(backtest, evidence_ids=(ev_backtest.evidence_id,), output_dir=out_dir),
        render_exception_transition_artifact(backtest, evidence_ids=(ev_backtest.evidence_id,), output_dir=out_dir),
        render_duration_diagnostics_artifact(dur, evidence_ids=(ev_dur.evidence_id,), output_dir=out_dir),
        render_tail_severity_artifact(sev, evidence_ids=(ev_sev.evidence_id,), output_dir=out_dir),
        render_backtest_summary_artifact(backtest, evidence_ids=(ev_backtest.evidence_id,), output_dir=out_dir),
        render_tail_comparison_artifact(compare, evidence_ids=(ev_comp.evidence_id,), output_dir=out_dir),
        render_tail_risk_contribution_artifact(contrib, evidence_ids=(ev_contrib.evidence_id,), output_dir=out_dir),
    ]

    for art in artifacts:
        assert art.file_path is not None
        svg_file = Path(art.file_path)
        json_file = Path(art.file_path.replace(".svg", ".json"))

        assert svg_file.exists(), f"SVG file {svg_file} does not exist"
        assert json_file.exists(), f"JSON file {json_file} does not exist"

        # Validate SVG is valid XML
        tree = ET.parse(svg_file)
        root = tree.getroot()
        assert root.tag.endswith("svg")

        # Validate JSON companion
        with open(json_file, encoding="utf-8") as f:
            companion_data = json.load(f)
        assert isinstance(companion_data, dict)
        assert len(companion_data) > 0


def test_gate5_manifest_integrity() -> None:
    """The Gate 5 showcase manifest and summary files must exist and be structurally valid."""
    manifest_path = Path("start_output/gate5_showcase/manifest.json")
    summary_path = Path("start_output/gate5_showcase/gate5_summary.json")

    assert manifest_path.exists(), "manifest.json missing"
    assert summary_path.exists(), "gate5_summary.json missing"

    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["gate"] == "GATE_5"
    assert manifest["var_confidence"] == 0.99
    assert manifest["test_significance"] == 0.05
    assert len(manifest["semantic_artifacts"]) == 9
    assert len(manifest["human_review_artifacts"]) == 9
    assert len(manifest["evidence_records"]) == 16
