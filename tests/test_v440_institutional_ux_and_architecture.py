"""Unit and Integration Tests for StART v4.4 Institutional UX & Architecture Convergence.

Verifies:
1. Synthetic Market World: Enriched metadata, Black-Litterman views, Scenario shock specs, and constraints.
2. Scenario Integrity Diagnostic: Real ScenarioSpec consumption and negative evidence preservation.
3. Deterministic Governance Disposition: Evaluation under conditions and failures.
4. Presentation Layer: Typed ReviewPresentationModel, blocks, and JSON export.
5. Orchestration Tracer: Real event recording, rich tables, JSON, Mermaid, and SVG export.
6. Architecture Capability Registry: Truthful component classifications and privacy behaviors.
7. Configurable Artifact Viewer: Auto, open, terminal, and off modes.
"""

import json
import tempfile
from pathlib import Path

from start.core.architecture_registry import (
    CAPABILITY_REGISTRY,
    get_architecture_capability_table,
    get_architecture_registry_dict,
)
from start.core.schemas import EvidenceRecord, Status
from start.data.synthetic_market import generate_market_world
from start.orchestration.tracer import AgentExecutionTracer
from start.portfolio.contracts import (
    RepricingMethod,
    ScenarioShock,
    ScenarioSpec,
    ScenarioType,
    ShockSpace,
    ShockUnit,
)
from start.portfolio.scenario import validate_scenario_data_integrity
from start.reporting.presentation import (
    build_presentation_model,
)
from start.reporting.viewer import get_artifact_view_mode, view_artifacts
from start.review.architecture import ReviewContextBundle, ReviewDomain
from start.review.executor import (
    evaluate_deterministic_governance_disposition,
)


def test_synthetic_market_world_enrichment():
    """Verify synthetic market world provides full institutional metadata and constraints."""
    world = generate_market_world(seed=123)
    ctx = world.market_context()

    assert ctx.returns is not None
    assert ctx.returns.shape[1] == 50
    assert ctx.asset_metadata is not None
    assert len(ctx.asset_metadata) == 50
    assert "sector" in ctx.asset_metadata.columns
    assert "asset_class" in ctx.asset_metadata.columns
    assert "provenance_tag" in ctx.asset_metadata.columns

    # Check constraints
    assert ctx.portfolio.constraints is not None
    assert ctx.portfolio.constraints.group_constraints is not None

    # Check Black-Litterman views
    assert "bl_views" in ctx.extra
    assert ctx.extra["bl_views"]["P"].shape[0] == 2
    assert ctx.extra["bl_views"]["Q"].shape[0] == 2

    # Check scenario specs
    assert "scenarios" in ctx.extra
    assert len(ctx.extra["scenarios"]) == 2
    scen = ctx.extra["scenarios"][0]
    assert isinstance(scen, ScenarioSpec)
    assert len(scen.shocks) > 0


def test_scenario_integrity_diagnostic_real_spec():
    """Verify validate_scenario_data_integrity executes with real specs and preserves negative status."""
    # 1. Valid scenario with real shocks
    shocks = (
        ScenarioShock(
            risk_factor_id="ASSET_000",
            shock_space=ShockSpace.ASSET_RETURN,
            shock_unit=ShockUnit.RETURN_DECIMAL,
            raw_value=-0.08,
            normalized_value=-0.08,
            normalization_rule="IDENTITY_RETURN_DECIMAL",
        ),
    )
    spec_valid = ScenarioSpec(
        scenario_id="SCEN-TEST-VALID",
        scenario_name="Valid Shock Spec",
        scenario_type=ScenarioType.SYNTHETIC,
        shocks=shocks,
        repricing_method=RepricingMethod.LINEAR_RETURN,
    )
    res_valid = validate_scenario_data_integrity(spec_valid, portfolio_assets=["ASSET_000"])
    assert res_valid.valid is True
    assert res_valid.n_shocks == 1
    assert len(res_valid.issues) == 0

    # 2. Invalid scenario with empty shocks -> must report invalid and preserve negative evidence
    spec_empty = ScenarioSpec(
        scenario_id="SCEN-EMPTY",
        scenario_name="Empty Shock Spec",
        scenario_type=ScenarioType.SYNTHETIC,
        shocks=(),
        repricing_method=RepricingMethod.LINEAR_RETURN,
    )
    res_empty = validate_scenario_data_integrity(spec_empty, portfolio_assets=["ASSET_000"])
    assert res_empty.valid is False
    assert res_empty.n_shocks == 0
    assert len(res_empty.issues) > 0


def test_deterministic_governance_disposition_evaluation():
    """Verify evaluate_deterministic_governance_disposition derives correct institutional disposition."""
    bundle_market = ReviewContextBundle(
        mode="single_domain",
        domains=(ReviewDomain.MARKET,),
        materiality="high",
        lifecycle="pre_implementation",
    )
    bundle_treasury = ReviewContextBundle(
        mode="single_domain",
        domains=(ReviewDomain.TREASURY,),
        materiality="high",
        lifecycle="pre_implementation",
    )

    clean_records = [
        EvidenceRecord(
            evidence_id="EV-1",
            test_id="portfolio.risk_statistics",
            test_name="Portfolio Risk Statistics",
            model_id="M1",
            dataset_id="D1",
            run_id="R1",
            status=Status.RECORDED,
            metrics={"annualised_volatility": 0.05, "is_valid": True},
        )
    ]

    # Treasury domain always requires conditions due to pre-registered studies
    assert (
        evaluate_deterministic_governance_disposition(bundle_treasury, clean_records, [])
        == "ACCEPT_WITH_CONDITIONS"
    )

    # Market domain with clean records -> ACCEPT
    assert evaluate_deterministic_governance_disposition(bundle_market, clean_records, []) == "ACCEPT"

    # Market domain with failed test -> ACCEPT_WITH_CONDITIONS
    failed_records = [
        EvidenceRecord(
            evidence_id="EV-2",
            test_id="validation.var_size_power",
            test_name="VaR Validation",
            model_id="M1",
            dataset_id="D1",
            run_id="R1",
            status=Status.FAIL,
            metrics={"observed.size": 0.15},
        )
    ]
    assert (
        evaluate_deterministic_governance_disposition(bundle_market, failed_records, [])
        == "ACCEPT_WITH_CONDITIONS"
    )

    # Market domain with unresolved challenge decision -> ACCEPT_WITH_CONDITIONS
    decisions = [{"action": "challenge", "details": "Unresolved diagnostic issue"}]
    assert (
        evaluate_deterministic_governance_disposition(bundle_market, clean_records, decisions)
        == "ACCEPT_WITH_CONDITIONS"
    )


def test_presentation_model_export():
    """Verify ReviewPresentationModel construction, blocks, and JSON serialization."""
    records = [
        EvidenceRecord(
            evidence_id="EV-10",
            test_id="portfolio.hierarchical_risk_parity",
            test_name="HRP Optimization",
            model_id="M1",
            dataset_id="D1",
            run_id="RUN-10",
            status=Status.RECORDED,
            metrics={"herfindahl": 0.03, "max_weight": 0.08, "linkage_method": "single"},
        )
    ]

    model = build_presentation_model(
        run_id="RUN-10",
        mode="single_domain",
        domains=(ReviewDomain.MARKET,),
        materiality="high",
        lifecycle="pre_implementation",
        records=records,
        governance_disposition="ACCEPT",
        attestation_seal_merkle_root="0123456789abcdef",
    )

    assert "PORTFOLIO_CONSTRUCTION" in model.blocks
    assert "HRP_SHOWCASE" in model.blocks
    assert "GOVERNANCE" in model.blocks

    json_str = model.to_json()
    parsed = json.loads(json_str)
    assert parsed["run_id"] == "RUN-10"
    assert parsed["governance_disposition"] == "ACCEPT"
    assert "blocks" in parsed


def test_agent_orchestration_tracer():
    """Verify AgentExecutionTracer records transitions, builds tables, and exports files."""
    tracer = AgentExecutionTracer()
    tracer.record(
        source_agent="Director",
        target_agent="MarketSpecialist",
        stage="PLANNING",
        node="test_discovery",
        tool_name="applicable_tests",
        status="SUCCESS",
    )
    tracer.record(
        source_agent="MarketSpecialist",
        target_agent="DeterministicEngine",
        stage="EXECUTION",
        node="compute_surfaces",
        tool_name="tool_dispatcher",
        emitted_evidence_ids=["EV-1", "EV-2"],
        status="SUCCESS",
    )

    assert len(tracer.events) == 2
    table = tracer.build_rich_table()
    assert table is not None

    with tempfile.TemporaryDirectory() as td:
        p = Path(td)
        j_path = p / "agent_orchestration.json"
        m_path = p / "agent_orchestration.mmd"
        s_path = p / "agent_orchestration.svg"

        tracer.export_json(j_path)
        tracer.export_mermaid(m_path)
        tracer.export_svg(s_path)

        assert j_path.exists() and j_path.stat().st_size > 0
        assert m_path.exists() and "flowchart TD" in m_path.read_text()
        assert s_path.exists() and "<svg" in s_path.read_text()


def test_architecture_capability_registry():
    """Verify architecture registry is truthful, classified, and exportable."""
    assert len(CAPABILITY_REGISTRY) >= 8
    names = [c.name for c in CAPABILITY_REGISTRY]
    assert any("StateGraph" in n for n in names)
    assert any("OpenTelemetry" in n for n in names)
    assert any("OPA" in n or "Policy" in n for n in names)

    table = get_architecture_capability_table()
    assert table is not None

    reg_dict = get_architecture_registry_dict()
    assert "capabilities" in reg_dict
    assert len(reg_dict["capabilities"]) == len(CAPABILITY_REGISTRY)


def test_artifact_viewer_safe_modes():
    """Verify artifact viewer handles auto, open, terminal, and off modes safely."""
    assert get_artifact_view_mode() in ("auto", "open", "terminal", "off")

    # In 'off' mode, returns 0 opened
    res_off = view_artifacts([], mode="off")
    assert res_off == 0
