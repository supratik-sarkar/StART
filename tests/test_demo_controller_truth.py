from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from demo_controller import SessionStatus, run_session_result
from demo_result_collector import collect_results


@pytest.mark.parametrize(
    ("mode", "expected", "exit_code", "final_hold"),
    [
        ("success", SessionStatus.COMPLETE, 0, True),
        ("cancelled", SessionStatus.FAIL_CLOSED, 7, False),
        ("crashed", SessionStatus.CRASHED, 9, False),
        ("wrong_prompt", SessionStatus.CONTROLLER_ERROR, 0, False),
    ],
)
def test_controller_child_exit_truth(
    tmp_path: Path,
    mode: str,
    expected: SessionStatus,
    exit_code: int,
    final_hold: bool,
) -> None:
    fixture = Path(__file__).parent / "fixtures" / "controller_process_fixture.py"
    manifest = tmp_path / "artifact_manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    result = run_session_result(
        "SESSION_C_POLICY",
        f"{sys.executable} {fixture} {mode} {manifest}",
        timeout=5.0,
        result_root=tmp_path / mode,
    )
    assert result.controller_status is expected
    assert result.child_exit_code == exit_code
    assert result.final_hold_reached is final_hold
    payload = json.loads(Path(result.result_path or "").read_text())
    assert payload["controller_status"] == expected.value
    assert payload["child_exit_code"] == exit_code
    assert payload["final_hold_reached"] is final_hold
    if mode == "success":
        assert payload["review_run_id"] == "RUN-REVIEW-FIXTURE"
        assert payload["execution_run_ids"] == ["RUN-ENT-FIXTURE"]
        assert payload["evidence_record_count"] == 3
        assert payload["artifact_manifest_path"] == str(manifest.resolve())
        assert payload["opa_decision"] == "OPA_LOCAL:ALLOW"
        assert payload["attestation_merkle_state"] == "fixture-root"


def test_result_collector_copies_exact_session_results(tmp_path: Path) -> None:
    fixture = Path(__file__).parent / "fixtures" / "controller_process_fixture.py"
    manifest = tmp_path / "artifact_manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    result = run_session_result(
        "SESSION_C_POLICY",
        f"{sys.executable} {fixture} success {manifest}",
        timeout=5.0,
        result_root=tmp_path / "run",
    )
    summary_path = collect_results(
        [Path(result.result_path or "")], tmp_path / "diagnostic_summary.json"
    )
    summary = json.loads(summary_path.read_text())
    assert summary["sessions"] == [result.to_dict()]
    assert summary["all_complete"] is True
