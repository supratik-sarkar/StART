"""Focused contract checks for the final presentation-only integration pass."""

from __future__ import annotations

import base64
import io
import json
import os
from pathlib import Path

import pytest
from demo_artifact_board import (
    build_artifact_board_html,
    landscape_geometry,
    load_current_run_manifest,
)
from demo_controller import ArtifactBoardPresenter, PTYController, TypingPolicy
from rich.console import Console

from start.portfolio.artifacts import (
    render_reverse_stress_profile_artifact,
    render_scenario_pnl_waterfall_artifact,
)
from start.portfolio.contracts import (
    RepricingMethod,
    ReverseStressResult,
    ScenarioResult,
    ScenarioShock,
    ScenarioSpec,
    ScenarioType,
    ShockSpace,
    ShockUnit,
)
from start.review.terminal_observability import PresentationPacing, TerminalReviewObserver


def _entry(root: Path, run_id: str, name: str, title: str, test_id: str) -> dict[str, object]:
    artifact = root / name
    artifact.write_bytes(b"fixture-image")
    return {
        "artifact_id": f"ART-{name}",
        "artifact_type": name.removesuffix(".png"),
        "title": title,
        "test_id": test_id,
        "evidence_ids": [f"EV-{name}"],
        "owner_run_id": run_id,
        "parent_review_run_id": run_id,
        "file_path": str(artifact),
        "semantic_companion": None,
        "checkpoint": test_id,
    }


def _manifest(tmp_path: Path) -> tuple[Path, str]:
    run_id = "RUN-REVIEW-PRESENTATION"
    root = tmp_path / run_id
    root.mkdir()
    groups = {
        "flight_a": [
            _entry(
                root,
                run_id,
                "temporal-saliency.png",
                "Temporal Input-Gradient Saliency",
                "deep_learning.explainability_diagnostics",
            )
        ],
        "flight_b": [
            _entry(
                root,
                run_id,
                "portfolio-weights.png",
                "Hierarchical Risk Parity (HRP) Allocation",
                "portfolio.hierarchical_risk_parity",
            )
        ],
        "flight_c": [
            _entry(
                root,
                run_id,
                "var-backtest.png",
                "VaR Backtest Exception Timeline",
                "traded_risk.var_kupiec_pof",
            )
        ],
    }
    payload = {
        "schema": "start.current-run-artifacts/2",
        "review_run_id": run_id,
        "execution_run_ids": [],
        "report_run_id": run_id,
        "presentation_root": str(root.resolve()),
        "scientific_artifact_roots": [str(root.resolve())],
        "groups": groups,
    }
    manifest = root / "artifact_manifest.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    return manifest, run_id


def _signal(action: str, **payload: object) -> bytes:
    body = json.dumps({"action": action, **payload}, separators=(",", ":")).encode()
    encoded = base64.urlsafe_b64encode(body).rstrip(b"=")
    return b"\x1b]777;StART=" + encoded + b"\x07"


def test_runtime_cards_do_not_create_or_reorder_semantic_events(tmp_path: Path) -> None:
    stream = io.StringIO()
    observer = TerminalReviewObserver(
        "RUN-PRESENTATION",
        console=Console(file=stream, width=132, color_system=None),
        session_kind="A",
        pacing=PresentationPacing(enabled=True),
    )
    before = list(observer.events)
    observer.flight_context("FLIGHT A", (("Dataset", "temporal"), ("Shape", "64×24×3")))
    observer.review_story(
        "ARCHITECTURE DECISION",
        question="Which implemented family matches rank-3 input?",
        alternatives=("lstm", "gru"),
        deterministic_result="lstm",
        evidence=("EV-ARCH",),
        outcome="accepted",
    )
    observer.artifact_board_ready(tmp_path / "artifact_manifest.json", run_id="RUN-PRESENTATION")

    assert observer.events == before
    assert "RUNTIME CONTEXT" in stream.getvalue()
    assert "ARCHITECTURE DECISION" in stream.getvalue()
    assert "\x1b]777;StART=" in stream.getvalue()


def test_viewer_dwell_buffers_only_display_not_transcript() -> None:
    controller = PTYController(presentation_mode=True)
    viewer = io.BytesIO()
    controller._viewer_stream = viewer

    signal = _signal("dwell", seconds=5.0, reason="fixture")
    controller.consume_data(b"visible-before" + signal[:7])
    controller.consume_data(signal[7:] + b"visible-after")

    assert controller.transcript == "visible-beforevisible-after"
    assert viewer.getvalue() == b"visible-before"
    controller.release_viewer_hold()
    assert viewer.getvalue() == b"visible-beforevisible-after"


def test_presentation_typing_is_exact_and_policy_is_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    policy = TypingPolicy()
    assert policy.action_key_pre_delay == 0.4
    assert policy.ordinary_chars_per_second == 50.0
    assert policy.long_chars_per_second == 75.0
    assert policy.long_text_threshold == 120
    assert policy.post_message_dwell == 0.65

    read_fd, write_fd = os.pipe()
    controller = PTYController(presentation_mode=True, typing_policy=policy)
    controller.fd = write_fd
    monkeypatch.setattr("demo_controller.time.sleep", lambda _seconds: None)
    controller.send_line_presented("A")
    os.close(write_fd)
    controller.fd = None
    try:
        assert os.read(read_fd, 16) == b"A\n"
    finally:
        os.close(read_fd)


def test_v2_manifest_builds_exact_run_landscape_board(tmp_path: Path) -> None:
    manifest_path, run_id = _manifest(tmp_path)
    manifest = load_current_run_manifest(manifest_path.parent, run_id)
    board = build_artifact_board_html(manifest)
    geometry = landscape_geometry()

    assert geometry["terminal"].width == 1190
    assert geometry["artifact_board"].x == 1190
    assert geometry["artifact_board"].width == 730
    assert "Temporal Input-Gradient Saliency" in board
    assert "Hierarchical Risk Parity (HRP) Allocation" in board
    assert "VaR Backtest Exception Timeline" in board
    assert run_id in board
    assert "no historical fallback" in board
    assert 'data-group="flight_a"' in board
    assert 'data-group="flight_b"' in board
    assert 'data-group="flight_c"' in board

    manifest["groups"]["flight_a"][0]["evidence_ids"] = []
    manifest["groups"]["flight_a"][0]["semantic_payload_hash"] = "a" * 64
    semantic_board = build_artifact_board_html(manifest)
    assert "Semantic sha256:aaaaaaaaaaaaaaaa…" in semantic_board


def test_manifest_rejects_alias_mismatch_and_root_escape(tmp_path: Path) -> None:
    manifest_path, run_id = _manifest(tmp_path)
    with pytest.raises(ValueError, match="exact non-alias"):
        load_current_run_manifest(manifest_path.parent, "latest")
    with pytest.raises(ValueError, match="run ID mismatch"):
        load_current_run_manifest(manifest_path.parent, "RUN-WRONG")

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    escaped = tmp_path / "escaped.png"
    escaped.write_bytes(b"fixture")
    payload["groups"]["flight_a"][0]["file_path"] = str(escaped)
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="escapes declared current-run roots"):
        load_current_run_manifest(manifest_path.parent, run_id)


def test_presenter_launches_only_rendered_exact_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path, run_id = _manifest(tmp_path)
    captured: dict[str, object] = {}

    class FakeProcess:
        def poll(self) -> None:
            return None

        def terminate(self) -> None:
            captured["terminated"] = True

        def wait(self, timeout: float) -> int:
            captured["wait_timeout"] = timeout
            return 0

        def kill(self) -> None:
            captured["killed"] = True

    def fake_popen(command: list[str], **kwargs: object) -> FakeProcess:
        captured["command"] = command
        captured["kwargs"] = kwargs
        return FakeProcess()

    monkeypatch.setattr(
        ArtifactBoardPresenter,
        "_browser_executable",
        lambda _self: "/browser",
    )
    monkeypatch.setattr("demo_controller.subprocess.Popen", fake_popen)
    presenter = ArtifactBoardPresenter()
    board = presenter.show(manifest_path=manifest_path, run_id=run_id)
    command = captured["command"]
    assert isinstance(command, list)
    assert f"--app={board.as_uri()}" in command
    assert "--window-position=1190,0" in command
    assert "--window-size=730,1080" in command
    assert board.parent == manifest_path.parent
    presenter.close()
    assert captured["terminated"] is True


def test_source_wiring_is_runtime_owned_and_observer_has_no_sleep() -> None:
    root = Path(__file__).parents[1]
    observer_source = (root / "src/start/review/terminal_observability.py").read_text()
    temporal_source = (root / "src/start/interactive_review.py").read_text()
    market_source = (root / "src/start/review/executor.py").read_text()
    controller_source = (root / "demo_controller.py").read_text()

    assert "time.sleep(" not in observer_source
    assert "observer.flight_context(" in temporal_source
    assert "observer.review_story(" in temporal_source
    assert "observer.artifact_board_ready(" in temporal_source
    assert '"start.scientific-artifacts/1"' in temporal_source
    assert '"execution-produced-scientific-inventory"' in temporal_source
    assert "observer.flight_context(" in market_source
    assert "observer.review_story(" in market_source
    assert "observer.artifact_board_ready(" in market_source
    assert "ctrl.consume_data(d)" in controller_source
    assert "ctrl.send_line_presented(send_val)" in controller_source


def test_scenario_artifact_preserves_explicit_shock_semantics(tmp_path: Path) -> None:
    scenario_spec = ScenarioSpec(
        scenario_id="SCEN-RATES",
        scenario_name="Rates up 100bp",
        scenario_type=ScenarioType.USER_DEFINED,
        shocks=(
            ScenarioShock(
                risk_factor_id="USD_10Y",
                shock_space=ShockSpace.YIELD,
                shock_unit=ShockUnit.BASIS_POINTS,
                raw_value=100.0,
                normalized_value=0.01,
                normalization_rule="basis_points / 10000",
                source_reference="fixture://rates",
            ),
        ),
        repricing_method=RepricingMethod.FACTOR_LINEAR,
        frequency="DAILY",
        currency="USD",
        specific_shock_policy="NONE",
        assumptions=("Parallel move only",),
    )
    scenario_result = ScenarioResult(
        scenario_id="SCEN-RATES",
        scenario_type=ScenarioType.USER_DEFINED.value,
        repricing_method=RepricingMethod.FACTOR_LINEAR.value,
        scenario_return=-0.04,
        scenario_loss=0.04,
        portfolio_value=1_000_000.0,
        scenario_pnl=-40_000.0,
        scenario_monetary_loss=40_000.0,
        asset_contributions={},
        factor_contributions={"USD_10Y": -0.04},
        specific_contribution=None,
        group_contributions={},
        partition_contract="EXHAUSTIVE_PARTITION",
        reconciliation_error=0.0,
        converged=True,
        limitations=("Linear factor approximation",),
        horizon="ONE_DAY",
        currency="USD",
    )

    artifact = render_scenario_pnl_waterfall_artifact(
        scenario_result,
        ("EV-SCENARIO",),
        output_dir=tmp_path,
        scenario_spec=scenario_spec,
    )
    payload = artifact.semantic_payload
    shock = payload["scenario_spec"]["shocks"][0]
    svg = Path(artifact.file_path or "").read_text(encoding="utf-8")

    assert shock["shock_unit"] == "BASIS_POINTS"
    assert shock["raw_value"] == 100.0
    assert shock["normalized_value"] == 0.01
    assert payload["horizon"] == "ONE_DAY"
    assert payload["currency"] == "USD"
    assert "1/1 non-zero explicit shocks" in svg
    assert "missing/specific policy NONE" in svg


def test_reverse_stress_artifact_never_labels_failed_solution_solved(tmp_path: Path) -> None:
    result = ReverseStressResult(
        target_loss=0.10,
        achieved_loss=0.06,
        achieved_return=-0.06,
        loss_gap=0.04,
        shock_vector={"MKT": -0.15},
        normalized_shocks={"MKT": -1.5},
        distance=1.5,
        distance_norm="L2",
        bounds_satisfied=False,
        solver_status="INFEASIBLE",
        converged=False,
        is_closed_form=False,
        limitations=("Target cannot be reached inside supplied bounds",),
    )

    artifact = render_reverse_stress_profile_artifact(
        result,
        ("EV-REVERSE",),
        output_dir=tmp_path,
    )
    payload = artifact.semantic_payload
    svg = Path(artifact.file_path or "").read_text(encoding="utf-8")

    assert payload["bounds_satisfied"] is False
    assert payload["solution_kind"] == "CONSTRAINED_OPTIMIZATION"
    assert payload["normalized_shocks"] == {"MKT": -1.5}
    assert "OUT_OF_BOUNDS" in svg
    assert "SOLVED" not in svg
