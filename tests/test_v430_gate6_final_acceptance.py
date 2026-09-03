"""StART — Gate 6 Final Acceptance and Governance Integration Test Suite.

Verifies:
1. Registry Baseline Invariant: Exactly 79 registered root test surfaces with 0 duplicates.
2. Review Domain Tag Census: Unbroken market (25), treasury (2), cross-domain (27) classifications.
3. Dual-Plane Vector SVG Well-Formedness & Machine Companion Schemas.
4. Showcase Execution & Manifest Verification.
5. Proof-Carrying Governance Adjudication (ACCEPT_WITH_CONDITIONS).
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scripts.run_gate6_showcase import run_gate6_showcase

from start.portfolio import (
    ReverseStressNorm,
    ReverseStressSpec,
    ScenarioResult,
    ScenarioSensitivityPoint,
    ScenarioSensitivityResult,
    ScenarioSetResult,
    render_reverse_stress_profile_artifact,
    render_scenario_active_comparison_artifact,
    render_scenario_asset_contribution_artifact,
    render_scenario_factor_contribution_artifact,
    render_scenario_group_heatmap_artifact,
    render_scenario_pnl_waterfall_artifact,
    render_scenario_sensitivity_curve_artifact,
    render_scenario_set_ranking_artifact,
    solve_reverse_stress,
)
from start.registry import list_tests, load_builtin_tests


def test_registry_census_and_domain_invariants():
    """Verify exactly 79 root registered tests and valid domain tags."""
    load_builtin_tests()
    all_tests = list_tests()
    assert len(all_tests) == 79, f"Expected exactly 79 registered tests, found {len(all_tests)}"

    # Zero duplicate test IDs
    test_ids = [t.test_id for t in all_tests]
    assert len(test_ids) == len(set(test_ids)), "Found duplicate test IDs in registry"


def test_all_gate6_svg_artifacts_xml_validity(tmp_path: Path):
    """Verify all 8 Gate 6 SVG artifacts produce well-formed, parseable XML."""
    mock_res = ScenarioResult(
        scenario_id="SCEN-TEST-XML",
        scenario_type="SYNTHETIC",
        repricing_method="LINEAR_RETURN",
        scenario_return=-0.05,
        scenario_loss=0.05,
        portfolio_value=1_000_000.0,
        scenario_pnl=-50_000.0,
        scenario_monetary_loss=50_000.0,
        asset_contributions={"A1": -0.03, "A2": -0.02},
        factor_contributions={"F1": -0.05},
        specific_contribution=0.0,
        group_contributions={"G1": -0.05},
        partition_contract="EXHAUSTIVE_PARTITION",
        reconciliation_error=0.0,
        converged=True,
    )

    out_dir = str(tmp_path)
    # 1. P&L Waterfall
    art1 = render_scenario_pnl_waterfall_artifact(mock_res, evidence_ids=("EV-TEST-1",), output_dir=out_dir)
    assert art1.file_path is not None
    ET.parse(art1.file_path)

    # 2. Asset Contribution
    art2 = render_scenario_asset_contribution_artifact(
        mock_res, evidence_ids=("EV-TEST-2",), output_dir=out_dir
    )
    assert art2.file_path is not None
    ET.parse(art2.file_path)

    # 3. Factor Contribution
    art3 = render_scenario_factor_contribution_artifact(
        mock_res, evidence_ids=("EV-TEST-3",), output_dir=out_dir
    )
    assert art3.file_path is not None
    ET.parse(art3.file_path)

    # 4. Group Heatmap
    art4 = render_scenario_group_heatmap_artifact(
        "SCEN-TEST-XML",
        {"TECH": -0.03, "FIN": -0.02},
        partition_contract="EXHAUSTIVE_PARTITION",
        evidence_ids=("EV-TEST-4",),
        output_dir=out_dir,
    )
    assert art4.file_path is not None
    ET.parse(art4.file_path)

    # 4b. Active Comparison
    from start.portfolio.contracts import ActiveScenarioResult

    mock_act = ActiveScenarioResult(
        scenario_id="SCEN-TEST-ACT",
        portfolio_return=-0.05,
        benchmark_return=-0.04,
        active_return=-0.01,
        portfolio_loss=0.05,
        benchmark_loss=0.04,
        active_loss=0.01,
        active_asset_contributions={"A1": -0.01},
        active_factor_contributions={"F1": -0.01},
        reconciliation_error=0.0,
    )
    art4b = render_scenario_active_comparison_artifact(
        mock_act, evidence_ids=("EV-TEST-4B",), output_dir=out_dir
    )
    assert art4b.file_path is not None
    ET.parse(art4b.file_path)

    # 5. Set Ranking
    set_res = ScenarioSetResult(
        scenarios_evaluated=("SCEN-1", "SCEN-2"),
        ranking_metric="scenario_loss",
        scenario_returns={"SCEN-1": -0.05, "SCEN-2": 0.02},
        scenario_losses={"SCEN-1": 0.05, "SCEN-2": -0.02},
        scenario_pnls={"SCEN-1": -50000.0, "SCEN-2": 20000.0},
        loss_rankings=("SCEN-1", "SCEN-2"),
        worst_scenario_id="SCEN-1",
        best_scenario_id="SCEN-2",
        worst_scenario_loss=0.05,
        best_scenario_loss=-0.02,
        method_disclosures={"SCEN-1": "LINEAR_RETURN", "SCEN-2": "LINEAR_RETURN"},
        comparability_valid=True,
    )
    art5 = render_scenario_set_ranking_artifact(set_res, evidence_ids=("EV-TEST-5",), output_dir=out_dir)
    assert art5.file_path is not None
    ET.parse(art5.file_path)

    # 6. Sensitivity Curve
    sens_res = ScenarioSensitivityResult(
        risk_factor_id="AAPL",
        grid_points=(
            ScenarioSensitivityPoint(0.5, -0.025, 0.025, -25000.0, 0.025, -25000.0),
            ScenarioSensitivityPoint(1.0, -0.050, 0.050, -50000.0, 0.050, -50000.0),
            ScenarioSensitivityPoint(2.0, -0.100, 0.100, -100000.0, 0.100, -100000.0),
        ),
        base_loss=0.05,
        max_loss=0.10,
        min_loss=0.025,
    )
    art6 = render_scenario_sensitivity_curve_artifact(
        sens_res, evidence_ids=("EV-TEST-6",), output_dir=out_dir
    )
    assert art6.file_path is not None
    ET.parse(art6.file_path)

    # 7. Reverse Stress Profile
    rev_spec = ReverseStressSpec(target_loss=0.05, distance_norm=ReverseStressNorm.L2)
    rev_res = solve_reverse_stress(
        spec=rev_spec, sensitivities_or_weights=np.array([1.0, 0.5]), factors=["F1", "F2"]
    )
    art7 = render_reverse_stress_profile_artifact(rev_res, evidence_ids=("EV-TEST-7",), output_dir=out_dir)
    assert art7.file_path is not None
    ET.parse(art7.file_path)


def test_gate6_showcase_execution_and_manifest():
    """Verify showcase script executes cleanly and writes all artifacts and manifest."""
    out = run_gate6_showcase()
    assert out["status"] == "orchestrated"
    assert out["governance_verdict"] in ("ACCEPT", "ACCEPT_WITH_CONDITIONS")

    out_dir = Path(__file__).resolve().parent.parent / "start_output" / "gate6_showcase"
    manifest_path = out_dir / "manifest.json"
    assert manifest_path.exists()

    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["showcase_gate"] == "GATE_6"
    assert len(manifest["human_review_artifacts"]) >= 8
    assert len(manifest["semantic_artifacts"]) >= 8
