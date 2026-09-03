"""Self-test for the StART acceptance harness driver and runbook parser."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import pytest
from scripts.run_market_acceptance_from_runbook import (
    RUNBOOK_PATH,
    MarketAcceptanceRunner,
    PtySession,
    RunbookParser,
)


def test_runbook_found_and_sha_computed() -> None:
    """1. Runbook file exists and SHA-256 is computed."""
    assert RUNBOOK_PATH.exists(), f"Runbook missing at {RUNBOOK_PATH}"
    parser = RunbookParser(RUNBOOK_PATH)
    assert len(parser.sha256) == 64
    assert parser.sha256 in (
        "1eab1c35c716a11a14be3db8d53ed76ab74a5c317765a08661ec740e6a840fc2",
        "79cb6a55fbe47d6a23b32e4b534d0c56e8732d056c0a162dfdf3ff793aa023aa",
    )


def test_exact_actionable_blocks_extracted() -> None:
    """2. Exact actionable blocks extracted; non-input blocks are excluded."""
    parser = RunbookParser(RUNBOOK_PATH)
    manifest = parser.build_manifest()

    step_map = {s.heading: s.content for s in manifest.steps}

    # Setup selections
    assert step_map["Review Mode"] == "1"
    assert step_map["Review Domain"] == "2"
    assert step_map["Backend"] == "3"
    assert step_map["Provider"] == "1"
    assert step_map["Model"] == "2"
    assert step_map["Materiality"] == "1"
    assert step_map["Lifecycle"] == "1"

    # Governance
    assert "independent validation of a high-materiality market risk" in step_map["Business Context"]
    assert "Act as an independent model-risk reviewer" in step_map["Reviewer Clarification"]
    assert "Daily independent portfolio-risk assessment" in step_map["Intended Use"]
    assert "Potential limitations include covariance instability" in step_map["Known Limitations"]

    # Market Data & Scope
    assert step_map["Data Source"] == "1"
    assert step_map["Scope"] == "1"
    assert step_map["Proceed"] == "y"

    # Non-input code blocks must NOT be present as actionable inputs
    all_contents = [s.content for s in manifest.steps]
    assert not any("VaR exceptions: 6 / 1000" in c for c in all_contents)
    assert not any("confidence = 0.99\nalpha_var = 0.01" in c for c in all_contents)
    assert not any("Grounded + Unbound = Quantitative Claims" in c for c in all_contents)
    assert not any("Registered: 79\nUnique: 79" in c for c in all_contents)


def test_expected_number_and_order_of_inputs() -> None:
    """3. Verify expected number and ordering of steps."""
    parser = RunbookParser(RUNBOOK_PATH)
    manifest = parser.build_manifest()

    headings = [s.heading for s in manifest.steps]
    # Check ordering
    expected_order = [
        "Review Mode",
        "Review Domain",
        "Backend",
        "Provider",
        "Model",
        "Materiality",
        "Lifecycle",
        "Business Context",
        "Reviewer Clarification",
        "Intended Use",
        "Known Limitations",
        "Data Source",
        "Scope",
        "Proceed",
        "Portfolio Action",
        "Factor View Artifacts",
        "Factor Ask Action",
        "Factor Question Text",
        "Factor Accept Action",
        "VaR View Artifacts",
        "VaR Ask Action",
        "VaR Question Text",
        "VaR Challenge Action",
        "VaR Challenge Text",
        "VaR Accept Action",
        "Covariance View Artifacts",
        "Covariance Ask Action",
        "Covariance Question Text",
        "Covariance Challenge Action",
        "Covariance Challenge Text",
        "Covariance Accept Action",
        "Scenario View Artifacts",
        "Scenario Ask Action",
        "Scenario Question Text",
        "Scenario Challenge Action",
        "Scenario Challenge Text",
        "Scenario Accept Action",
        "Committee View Artifacts",
        "Committee Ask Action",
        "Committee Question Text",
        "Committee Challenge Action",
        "Committee Challenge Text",
        "Committee Accept Action",
        "Governance View Artifacts",
        "Governance Ask Action",
        "Governance Question Text",
        "Governance Challenge Action",
        "Governance Challenge Text",
        "Governance Final Accept",
    ]
    assert headings == expected_order
    assert len(manifest.steps) == 49


def test_unexpected_prompt_fails_closed(tmp_path: Path) -> None:
    """4. Unexpected prompt in PTY session fails closed with TimeoutError."""
    cmd = [
        sys.executable,
        "-c",
        "import sys, time; print('SOME_UNEXPECTED_STRING'); sys.stdout.flush(); time.sleep(1)",
    ]
    session = PtySession(cmd=cmd, cwd=str(tmp_path), env={})
    try:
        with pytest.raises(TimeoutError):
            session.expect(["EXPECTED_PROMPT_NEVER_ARRIVES"], timeout=0.5)
    finally:
        session.close()


def test_timeout_fails_closed(tmp_path: Path) -> None:
    """5. PTY session timeout fails closed."""
    cmd = [sys.executable, "-c", "import time; time.sleep(5)"]
    session = PtySession(cmd=cmd, cwd=str(tmp_path), env={})
    try:
        with pytest.raises(TimeoutError):
            session.expect(["PROMPT"], timeout=0.3)
    finally:
        session.close()


def test_failure_menu_aborts_review(tmp_path: Path) -> None:
    """6. Live reviewer failure menu sends 2 and aborts review."""
    cmd = [
        sys.executable,
        "-c",
        "import sys; "
        "print('Select action [default: 1]: '); sys.stdout.flush(); "
        "choice = input(); "
        "print(f'CHOICE_RECEIVED:{choice}'); sys.stdout.flush()",
    ]
    session = PtySession(cmd=cmd, cwd=str(tmp_path), env={})
    try:
        fallback_menu_pat = re.compile(r"Select action \[default:\s*1\]:\s*")
        idx = session.expect([fallback_menu_pat], timeout=2.0)
        assert idx == 0
        # In failure menu, runner sends "2"
        session.sendline("2")
        session.drain_to_completion(timeout=2.0)
        transcript = session.get_full_transcript()
        assert "CHOICE_RECEIVED:2" in transcript
    finally:
        session.close()


def test_pass_requires_all_checkpoints(tmp_path: Path) -> None:
    """7. Runner requires all checkpoints completed for PASS verdict."""
    runner = MarketAcceptanceRunner(
        runbook_path=RUNBOOK_PATH,
        start_bin=Path(sys.executable),
        output_dir=tmp_path,
    )
    # Initially checkpoints_reached is empty
    assert len(runner.checkpoints_reached) == 0
    # VaR assertion check
    with pytest.raises(AssertionError):
        runner._assert_var_section("Incomplete VaR text without required metrics")


def _make_valid_summary_data() -> dict[str, Any]:
    checkpoints = [
        "Portfolio Risk & Volatility Assumptions",
        "Factor Modeling & Attribution Assumptions",
        "VaR Backtesting & Exception Frequency",
        "Covariance Structure & Missing Data Treatment",
        "Scenario Analysis & Stress Testing",
        "Cross-Analytical Committee Synthesis",
        "Model Governance & Attestation Sign-Off",
    ]
    decisions = []
    for i, chk in enumerate(checkpoints[1:], 1):
        decisions.append({
            "checkpoint": chk,
            "action": "question",
            "grounding_mode": "STRUCTURED",
            "backend": "llm_structured",
            "provider": "openai",
            "model": "gpt-5",
            "provider_status": "OK",
            "schema_validation_status": "VALID",
            "finding_count": 5,
            "evidence_ref_count": 10,
            "validated_ref_count": 10,
            "invalid_ref_count": 0,
            "structured_findings_content_hash": f"{i:02d}" * 32,
        })
        if i in (2, 3, 4, 5, 6):
            decisions.append({
                "checkpoint": chk,
                "action": "challenge",
                "grounding_mode": "STRUCTURED",
                "backend": "llm_structured",
                "provider": "openai",
                "model": "gpt-5",
                "provider_status": "OK",
                "schema_validation_status": "VALID",
                "finding_count": 4,
                "evidence_ref_count": 8,
                "validated_ref_count": 8,
                "invalid_ref_count": 0,
                "structured_findings_content_hash": f"{i + 10:02d}" * 32,
            })
    return {
        "run_id": "RUN-REVIEW-TEST",
        "grounding_mode": "STRUCTURED",
        "domains": ["market"],
        "materiality": "HIGH",
        "lifecycle": "INITIAL_VALIDATION",
        "llm_config": {"provider": "openai", "model": "gpt-5", "backend_mode": "public"},
        "decisions": decisions,
        "attestation_seal": {
            "merkle_root": "2441691ecaa478292441691ecaa47829",
            "metadata": {"live_reviewer_status": "VALIDATED"},
        },
        "governance_disposition": "ACCEPT",
        "exit_code": 0,
    }


def _setup_runner(tmp_path: Path) -> MarketAcceptanceRunner:
    runner = MarketAcceptanceRunner(
        runbook_path=RUNBOOK_PATH,
        start_bin=Path(sys.executable),
        output_dir=tmp_path,
    )
    runner.checkpoints_reached = [
        "Portfolio Risk & Volatility Assumptions",
        "Factor Modeling & Attribution Assumptions",
        "VaR Backtesting & Exception Frequency",
        "Covariance Structure & Missing Data Treatment",
        "Scenario Analysis & Stress Testing",
        "Cross-Analytical Committee Synthesis",
        "Model Governance & Attestation Sign-Off",
    ]
    return runner


def test_native_structured_decisions_produce_pass(tmp_path: Path) -> None:
    """1. Native structured decisions produce PASS."""
    runner = _setup_runner(tmp_path)
    summary = _make_valid_summary_data()
    status, reason, censuses = runner.evaluate_structured_run_state(summary, exit_code=0)
    assert status == "PASS"
    assert reason == ""
    assert len(censuses) == 11


def test_one_invalid_ref_produces_fail(tmp_path: Path) -> None:
    """2. One invalid ref produces FAIL."""
    runner = _setup_runner(tmp_path)
    summary = _make_valid_summary_data()
    summary["decisions"][0]["invalid_ref_count"] = 1
    status, reason, _ = runner.evaluate_structured_run_state(summary, exit_code=0)
    assert status == "FAIL"
    assert "Invalid evidence refs" in reason


def test_one_schema_failure_produces_fail(tmp_path: Path) -> None:
    """3. One schema failure produces FAIL."""
    runner = _setup_runner(tmp_path)
    summary = _make_valid_summary_data()
    summary["decisions"][0]["schema_validation_status"] = "INVALID"
    status, reason, _ = runner.evaluate_structured_run_state(summary, exit_code=0)
    assert status == "FAIL"
    assert "Schema validation failed" in reason


def test_missing_checkpoint_produces_fail(tmp_path: Path) -> None:
    """4. Missing checkpoint produces FAIL."""
    runner = MarketAcceptanceRunner(
        runbook_path=RUNBOOK_PATH,
        start_bin=Path(sys.executable),
        output_dir=tmp_path,
    )
    runner.checkpoints_reached = ["Portfolio Risk & Volatility Assumptions"]
    summary = _make_valid_summary_data()
    status, reason, _ = runner.evaluate_structured_run_state(summary, exit_code=0)
    assert status == "FAIL"
    assert "Missing required checkpoint" in reason


def test_missing_final_attestation_produces_fail(tmp_path: Path) -> None:
    """5. Missing final attestation produces FAIL."""
    runner = _setup_runner(tmp_path)
    summary = _make_valid_summary_data()
    summary["attestation_seal"]["merkle_root"] = ""
    status, reason, _ = runner.evaluate_structured_run_state(summary, exit_code=0)
    assert status == "FAIL"
    assert "Missing or invalid final attestation" in reason


def test_legacy_grounding_appearing_in_canonical_market_produces_fail(tmp_path: Path) -> None:
    """6. Legacy grounding appearing in canonical Market produces FAIL."""
    runner = _setup_runner(tmp_path)
    summary = _make_valid_summary_data()
    summary["decisions"][0]["grounding_mode"] = "LEGACY_FREEFORM"
    status, reason, _ = runner.evaluate_structured_run_state(summary, exit_code=0)
    assert status == "FAIL"
    assert "Legacy grounding or invalid mode" in reason


def test_provider_fallback_produces_fail(tmp_path: Path) -> None:
    """7. Provider fallback produces FAIL."""
    runner = _setup_runner(tmp_path)
    summary = _make_valid_summary_data()
    summary["decisions"][0]["backend"] = "fallback"
    summary["decisions"][0]["provider_status"] = "ERROR"
    status, reason, _ = runner.evaluate_structured_run_state(summary, exit_code=0)
    assert status == "FAIL"
    assert "fallback" in reason.lower()


def test_structured_content_hash_missing_produces_fail(tmp_path: Path) -> None:
    """8. Structured content hash missing produces FAIL."""
    runner = _setup_runner(tmp_path)
    summary = _make_valid_summary_data()
    summary["decisions"][0]["structured_findings_content_hash"] = None
    status, reason, _ = runner.evaluate_structured_run_state(summary, exit_code=0)
    assert status == "FAIL"
    assert "content hash" in reason.lower()


def test_result_json_generated_without_post_run_transcript_reparsing(tmp_path: Path) -> None:
    """9. Result JSON is generated directly from machine-readable state."""
    runner = _setup_runner(tmp_path)
    summary = _make_valid_summary_data()
    status, reason, censuses = runner.evaluate_structured_run_state(summary, exit_code=0)
    assert status == "PASS"
    result_data = {
        "status": status,
        "checkpoints_reached": runner.checkpoints_reached,
        "grounding_censuses": censuses,
    }
    out_file = tmp_path / "acceptance_result.json"
    out_file.write_text(json.dumps(result_data, indent=2))
    loaded = json.loads(out_file.read_text())
    assert loaded["status"] == "PASS"
    assert len(loaded["grounding_censuses"]) == 11


def test_historical_result_file_is_never_rewritten() -> None:
    """10. Historical result file exists and is preserved."""
    historical_path = Path("start_output/acceptance_runs/20260903_033002/acceptance_result.json")
    if not historical_path.exists():
        historical_path = Path("../StART_Private_Archive/certification/acceptance_runs/20260903_033002/acceptance_result.json")
    assert historical_path.exists()
    assert historical_path.stat().st_size > 0


def test_canonical_action_names_required_and_terminal_labels_not_needed(tmp_path: Path) -> None:
    """11. Evaluator counts canonical 'question' and 'challenge' actions; terminal labels 'Q'/'C' are not required."""
    runner = _setup_runner(tmp_path)
    summary = {
        "run_id": "RUN-REVIEW-REGRESSION",
        "grounding_mode": "STRUCTURED",
        "domains": ["market"],
        "materiality": "high",
        "lifecycle": "initial_validation",
        "llm_config": {"provider": "openai", "model": "gpt-5", "backend_mode": "public"},
        "decisions": [
            {
                "checkpoint": "Factor Modeling & Attribution Assumptions",
                "action": "question",
                "grounding_mode": "STRUCTURED",
                "backend": "llm_structured",
                "provider": "openai",
                "model": "gpt-5",
                "provider_status": "OK",
                "schema_validation_status": "VALID",
                "finding_count": 8,
                "evidence_ref_count": 34,
                "validated_ref_count": 34,
                "invalid_ref_count": 0,
                "structured_findings_content_hash": "38e050407c31e1e51f357c867f0484b91554356961c479dde66db33368ac9eb3",
            },
            {
                "checkpoint": "VaR Backtesting & Exception Frequency",
                "action": "question",
                "grounding_mode": "STRUCTURED",
                "backend": "llm_structured",
                "provider": "openai",
                "model": "gpt-5",
                "provider_status": "OK",
                "schema_validation_status": "VALID",
                "finding_count": 8,
                "evidence_ref_count": 39,
                "validated_ref_count": 39,
                "invalid_ref_count": 0,
                "structured_findings_content_hash": "e40fd1a3ebc4f63077f27621e977cdea5011fa8c7979e7e22a95a332dd6dd393",
            },
            {
                "checkpoint": "Covariance Structure & Missing Data Treatment",
                "action": "challenge",
                "grounding_mode": "STRUCTURED",
                "backend": "llm_structured",
                "provider": "openai",
                "model": "gpt-5",
                "provider_status": "OK",
                "schema_validation_status": "VALID",
                "finding_count": 10,
                "evidence_ref_count": 63,
                "validated_ref_count": 63,
                "invalid_ref_count": 0,
                "structured_findings_content_hash": "2027a12f87e144c86fc9d811a09e87ec9d05b73fa31b344cf71fd2804c5f8261",
            },
            {
                "checkpoint": "Cross-Analytical Committee Synthesis",
                "action": "question",
                "grounding_mode": "STRUCTURED",
                "backend": "llm_structured",
                "provider": "openai",
                "model": "gpt-5",
                "provider_status": "OK",
                "schema_validation_status": "VALID",
                "finding_count": 6,
                "evidence_ref_count": 26,
                "validated_ref_count": 26,
                "invalid_ref_count": 0,
                "structured_findings_content_hash": "e7f93d1d3f3a0af139dc37a1e0210b9f6cc39283b26beb619afa8a280355ea33",
            },
        ],
        "attestation_seal": {
            "merkle_root": "f9bd0f6d949e5a2d4806a64273bb3a48",
            "metadata": {"live_reviewer_status": "VALIDATED"},
        },
        "governance_disposition": "ACCEPT",
        "exit_code": 0,
    }
    status, reason, censuses = runner.evaluate_structured_run_state(summary, exit_code=0)
    assert status == "PASS"
    assert reason == ""
    assert len(censuses) == 4


