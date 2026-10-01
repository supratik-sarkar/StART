"""Combined Gate 7-9 Final Acceptance and Integration Suite for StART.

Invariants:
- Zero confidential enterprise data: all datasets and fixtures are synthetic and public-safe.
- Public-release hazard scan checks repository tree without scanning outside the project root.
- Built-in root registry census remains strictly 79 total, 79 unique, 0 duplicates.
- Deterministic cross-analytical committee produces evidence graph, typed claims, and conditional sign-off.
- Artifact semantic payload hashing uses full 64-hex lowercase SHA-256.
"""

from __future__ import annotations

import os
from pathlib import Path

from start.agents.committee import CrossAnalyticalCommittee
from start.core.schemas import EvidenceRecord, Status
from start.data.adapters import DataFrameAdapter, execute_chunked
from start.evidence.claims import ClaimType
from start.portfolio.artifacts import (
    render_scenario_asset_contribution_artifact,
    render_scenario_pnl_waterfall_artifact,
)
from start.portfolio.contracts import (
    PartitionContract,
    RepricingMethod,
    ScenarioResult,
)
from start.registry import list_tests, load_builtin_tests


def _make_synth_evidence(
    test_id: str,
    name: str,
    metrics: dict,
    params: dict | None = None,
    status: Status = Status.PASS,
) -> EvidenceRecord:
    return EvidenceRecord(
        test_id=test_id,
        test_name=name,
        model_id="MOD-PUBLIC-ACCEPTANCE",
        dataset_id="DS-SYNTH-ACCEPTANCE",
        run_id="RUN-GATE79",
        metrics=metrics,
        params=params or {},
        status=status,
        interpretation="Public acceptance synthetic evidence.",
        limitations=["Deterministic test fixture."],
        input_artifact_hash="HASH-ACCEPTANCE-001",
    )


def test_registry_census_strictly_79_unique():
    """Verify built-in registry census is invariant at 79 total, 79 unique, 0 duplicates."""
    load_builtin_tests()
    all_tests = list_tests()
    test_ids = [t.test_id for t in all_tests]

    assert len(all_tests) == 79, f"Expected exactly 79 tests, got {len(all_tests)}"
    assert len(set(test_ids)) == 79, f"Expected 79 unique test IDs, got {len(set(test_ids))}"
    assert len(test_ids) - len(set(test_ids)) == 0, "Found duplicate test IDs in registry"


def test_public_repo_release_hazard_scan():
    """Scan local repository tree for accidental public-release hazards (credentials, .env, private keys)."""
    repo_root = Path(__file__).resolve().parent.parent
    assert repo_root.exists(), "Repository root must exist"

    # Scan for forbidden hazard file patterns
    hazard_filenames = {".env", ".env.local", ".env.production", "id_rsa", "id_ed25519"}
    found_hazards: list[str] = []

    # Check source, test, script, doc, config, and example subtrees
    scan_subdirs = ["src", "tests", "scripts", "docs", "configs", "examples"]
    for subdir in scan_subdirs:
        subpath = repo_root / subdir
        if not subpath.exists():
            continue
        for root, dirs, files in os.walk(subpath):
            dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__"]
            for f in files:
                if f in hazard_filenames:
                    found_hazards.append(str(Path(root) / f))

    # Also verify .gitignore excludes .env
    gitignore_path = repo_root / ".gitignore"
    assert gitignore_path.exists(), ".gitignore must exist"
    gitignore_content = gitignore_path.read_text(encoding="utf-8")
    assert ".env" in gitignore_content, ".gitignore must ignore .env files"

    assert len(found_hazards) == 0, f"Found public release hazard files: {found_hazards}"


def test_end_to_end_cross_analytical_committee_integration():
    """Verify end-to-end multi-lens review produces evidence graph, typed claims, distinct diagnostic evidence, and conditional sign-off."""
    # Construct complete synthetic evidence bundle spanning all 5 quantitative lenses
    ev_opt = _make_synth_evidence(
        "portfolio.mean_variance", "Mean Variance", {"converged": True, "sharpe": 1.45}
    )
    ev_cov_s = _make_synth_evidence(
        "covariance.empirical", "Sample Cov", {"is_psd": True}, params={"weights": {"A": 0.6, "B": 0.4}}
    )
    ev_cov_lw = _make_synth_evidence(
        "covariance.ledoit_wolf", "LW Cov", {"is_psd": True}, params={"weights": {"A": 0.45, "B": 0.55}}
    )
    ev_factor = _make_synth_evidence(
        "factor_risk.decomposition", "Factor Risk", {"beta_MKT": 1.10, "beta_SMB": 0.40}
    )
    ev_attrib = _make_synth_evidence(
        "attribution.brinson", "Brinson Attribution", {"reconciliation_error": 0.0}
    )
    ev_kupiec = _make_synth_evidence(
        "traded_risk.kupiec_pof", "Kupiec POF", {"reject_unconditional_coverage": False, "gamma_test": 0.05}
    )
    ev_christ = _make_synth_evidence(
        "traded_risk.christoffersen_independence",
        "Christoffersen Independence",
        {"reject_independence": True, "gamma_test": 0.05},
    )
    ev_scen = _make_synth_evidence(
        "scenario.linear_return",
        "Macro Shock",
        {"scenario_loss": 0.15, "contrib_MKT": -0.12, "contrib_SMB": -0.03},
    )
    ev_rev = _make_synth_evidence(
        "scenario.reverse_stress", "Reverse Stress", {"target_loss": 0.30, "minimum_distance": 0.22}
    )

    full_evidence_set = [
        ev_opt,
        ev_cov_s,
        ev_cov_lw,
        ev_factor,
        ev_attrib,
        ev_kupiec,
        ev_christ,
        ev_scen,
        ev_rev,
    ]

    committee = CrossAnalyticalCommittee()
    res = committee.conduct_committee_review(full_evidence_set)

    # 1. Evidence Graph Structure
    assert res.graph.node_count >= len(full_evidence_set)
    assert res.graph.edge_count > 0

    # 2. Typed Claims
    claim_types = {c.claim_type for c in res.claims}
    assert ClaimType.UNRESOLVED_RISK in claim_types
    assert ClaimType.SENSITIVITY in claim_types
    assert ClaimType.OBSERVATION in claim_types
    assert ClaimType.DEPENDENCY in claim_types

    # 3. Diagnostic Evidence Uniqueness
    assert len(res.diagnostic_evidence) > 0
    all_source_ids = {r.evidence_id for r in full_evidence_set}
    for diag in res.diagnostic_evidence:
        assert diag.evidence_id not in all_source_ids

    # 4. Separation of Powers: Critic is READY, Governance is ACCEPT_WITH_CONDITIONS
    assert res.critic_disposition == "READY_FOR_GOVERNANCE"
    assert res.governance_decision == "ACCEPT_WITH_CONDITIONS"
    assert len(res.governance_conditions) > 0


def test_system_wide_artifact_sha256_integrity():
    """Verify artifact renderers produce collision-resistant 64-hex SHA-256 semantic payload hashes."""
    scen_res = ScenarioResult(
        scenario_id="SCEN-ACCEPTANCE",
        scenario_type="SYNTHETIC",
        repricing_method=RepricingMethod.LINEAR_RETURN.value,
        scenario_return=-0.10,
        scenario_loss=0.10,
        portfolio_value=2_000_000.0,
        scenario_pnl=-200_000.0,
        scenario_monetary_loss=200_000.0,
        asset_contributions={"AAPL": -0.06, "MSFT": -0.04},
        factor_contributions={},
        specific_contribution=None,
        group_contributions={},
        partition_contract=PartitionContract.EXHAUSTIVE_PARTITION.value,
        reconciliation_error=0.0,
        converged=True,
        limitations=("Acceptance scenario.",),
        data_fingerprint="FP-ACCEPT-001",
    )

    art_wf = render_scenario_pnl_waterfall_artifact(scen_res, evidence_ids=("EV-001",))
    art_contrib = render_scenario_asset_contribution_artifact(scen_res, evidence_ids=("EV-001",))

    assert len(art_wf.semantic_payload_hash) == 64
    assert len(art_contrib.semantic_payload_hash) == 64
    assert art_wf.semantic_hash_algorithm == "sha256"
    assert art_contrib.semantic_hash_algorithm == "sha256"


def test_data_adapter_and_chunked_execution_invariance():
    """Verify DataFrameAdapter and chunked execution produce deterministically invariant results."""
    import pandas as pd

    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "y": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]})
    adapter = DataFrameAdapter(df)

    # Chunked sum reduction with chunk_size=2
    res_chunked = execute_chunked(
        adapter=adapter,
        chunk_size=2,
        map_fn=lambda c: (float(c["x"].sum()), float(c["y"].sum())),
        reduce_fn=lambda parts: (sum(p[0] for p in parts), sum(p[1] for p in parts)),
    )

    assert res_chunked == (21.0, 210.0)
    assert res_chunked[0] == float(df["x"].sum())
    assert res_chunked[1] == float(df["y"].sum())
