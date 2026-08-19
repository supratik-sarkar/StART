"""Tests for terminal presentation layer (theme, badges, glyphs, and panels)."""

from __future__ import annotations

from start.attestation.invariance import InvarianceAttestation
from start.attestation.seal import build_seal
from start.cli.panels import (
    render_adapters_table,
    render_agent_stream,
    render_attestation_panel,
    render_governance_panel,
    render_ledger_view,
    render_metrics_panel,
    render_run_header,
    render_seal_panel,
)
from start.cli.theme import create_console, format_agent_badge, format_status_glyph


def test_theme_and_badges_render() -> None:
    badge = format_agent_badge("DatasetDiscoveryAgent")
    assert "[DSC]" in badge.plain
    glyph = format_status_glyph("pass")
    assert "✓" in glyph.plain


def test_all_eight_panels_render_cleanly() -> None:
    console = create_console(width=100)

    # 1. Run Header
    p1 = render_run_header(
        review_id="R-2026-TEST",
        target="is_default",
        task_type="binary_classification",
        dataset_shape=(1000, 15),
    )
    with console.capture() as cap:
        console.print(p1)
    out1 = cap.get()
    assert "R-2026-TEST" in out1

    # 2. Agent Stream
    events = [
        {"agent": "DatasetDiscoveryAgent", "status": "complete", "stage": "dataset_discovery", "detail": "15 features", "runtime": 0.12},
        {"agent": "FeatureEngineeringAgent", "status": "warn", "stage": "leakage_check", "detail": "1 potential leak", "runtime": 0.35},
    ]
    p2 = render_agent_stream(events)
    with console.capture() as cap:
        console.print(p2)
    out2 = cap.get()
    assert "[DSC]" in out2
    assert "[FTR]" in out2

    # 3. Metrics Panel
    metrics = {
        "train": {"auc": 0.85, "brier": 0.10},
        "test": {"auc": 0.78, "brier": 0.12},
        "oos": {"auc": 0.77, "brier": 0.13},
    }
    p3 = render_metrics_panel(metrics)
    with console.capture() as cap:
        console.print(p3)
    out3 = cap.get()
    assert "AUC" in out3

    # 4. Ledger View
    records = [
        {"evidence_id": "EV-001", "status": "pass", "test_name": "Cohort Metric", "metrics": {"gap": 0.04}},
        {"evidence_id": "EV-002", "status": "warn", "test_name": "Calibration", "metrics": {"ece": 0.12}},
    ]
    p4 = render_ledger_view(records)
    with console.capture() as cap:
        console.print(p4)
    out4 = cap.get()
    assert "EV-001" in out4

    # 5. Governance Panel
    controls = [
        {"name": "Conceptual Soundness", "reference": "SR 11-7", "status": "pass", "notes": "Approved"}
    ]
    p5 = render_governance_panel(tier="Tier-1", controls=controls)
    with console.capture() as cap:
        console.print(p5)
    out5 = cap.get()
    assert "SR 11-7" in out5

    # 6. Attestation Panel
    att = InvarianceAttestation(
        section="test",
        invariant=True,
        divergences=(),
        deterministic_binding={"bound_claims": 5, "total_claims": 5},
        model_binding={"bound_claims": 5, "total_claims": 5, "unbound_claims": 0},
        tolerance=5e-4,
        narration_path="model_narrated",
        provider_name="openai / gpt-4o-mini",
    )
    p6 = render_attestation_panel(att)
    with console.capture() as cap:
        console.print(p6)
    out6 = cap.get()
    assert "INVARIANT" in out6

    # 7. Seal Panel
    seal = build_seal(
        review_id="R-TEST",
        plan={"scope": "demo"},
        policy={"policy": "demo"},
        evidence_head="aa" * 32,
    )
    p7 = render_seal_panel(seal)
    with console.capture() as cap:
        console.print(p7)
    out7 = cap.get()
    assert "Merkle Root" in out7
    assert "start-seal/2:R-TEST" in out7

    # 8. Adapters Table
    adapters = [
        {"name": "LangSmith", "category": "observability", "status": "active", "role": "Lineage"},
        {"name": "Phoenix", "category": "observability", "status": "blocked_by_profile", "role": "Evals"},
    ]
    p8 = render_adapters_table(adapters)
    with console.capture() as cap:
        console.print(p8)
    out8 = cap.get()
    assert "LangSmith" in out8
    assert "Phoenix" in out8
