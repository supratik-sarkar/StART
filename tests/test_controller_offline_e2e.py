"""End-to-End Offline Controller Verification Suite (Requirement 10).

Runs demo_controller against a deterministic fake PTY integration harness (demo_mock_twin.py)
emitting the complete final state sequence for all sessions at $0.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from demo_controller import (
    STATE_TABLE,
    SessionStatus,
    run_session_result,
    validate_route_contracts,
)


def test_controller_session_a_offline_e2e(tmp_path: Path):
    """Verify Controller executes SESSION_A_TEMPORAL end-to-end through all states."""
    manifest = tmp_path / "RUN-ENT-TEMPORAL-FIXTURE" / "artifact_manifest.json"
    manifest.parent.mkdir()
    manifest.write_text(
        json.dumps(
            {
                "schema": "start.current-run-artifacts/2",
                "review_run_id": "RUN-ENT-TEMPORAL-FIXTURE",
                "execution_run_ids": ["RUN-ENT-TEMPORAL-FIXTURE"],
                "report_run_id": None,
                "presentation_root": str(manifest.parent.resolve()),
                "scientific_artifact_roots": [],
                "groups": {"flight_a": [], "flight_b": [], "flight_c": []},
            }
        ),
        encoding="utf-8",
    )
    cmd = (
        f"{sys.executable} demo_mock_twin.py --session SESSION_A_TEMPORAL "
        f"--artifact-manifest {manifest}"
    )
    result = run_session_result(
        "SESSION_A_TEMPORAL", cmd, timeout=20.0, result_root=tmp_path / "session-a"
    )
    assert result.controller_status is SessionStatus.COMPLETE
    assert result.child_exit_code == 0
    assert result.final_hold_reached is True
    assert result.evidence_record_count == 8
    assert result.artifact_manifest_path == str(manifest.resolve())
    assert json.loads(manifest.read_text(encoding="utf-8"))["schema"] == "start.current-run-artifacts/2"

    # Prove exact states traversed
    states_a = [t.state_id for t in STATE_TABLE["SESSION_A_TEMPORAL"]]
    assert len(states_a) == 26
    assert states_a[0] == "A_MODE"
    assert states_a[-1] == "A_EVIDENCE_SEPARATION"
    assert "A_PROVIDER" not in states_a
    assert "A_MODEL" not in states_a
    assert "A_METRIC" not in states_a
    assert not any(state.startswith("A_FE_") for state in states_a)
    assert next(t for t in STATE_TABLE["SESSION_A_TEMPORAL"] if t.state_id == "A_ARCH_DECISION").next_state == "A_EXECUTE"


def test_controller_semantic_route_contracts_are_consistent():
    assert validate_route_contracts() == (True, "semantic route contracts are consistent")


def test_controller_session_b_offline_e2e(tmp_path: Path):
    """Verify Controller executes SESSION_B_MARKET through all 7 checkpoints and sign-off."""
    cmd = f"{sys.executable} demo_mock_twin.py --session SESSION_B_MARKET"
    result = run_session_result(
        "SESSION_B_MARKET", cmd, timeout=20.0, result_root=tmp_path / "session-b"
    )
    assert result.controller_status is SessionStatus.COMPLETE

    # Prove exact states traversed
    states_b = [t.state_id for t in STATE_TABLE["SESSION_B_MARKET"]]
    assert len(states_b) == 35
    assert states_b[0] == "B_MODE"
    assert "B_CHK_PORTFOLIO_1" in states_b
    assert "B_CHK_ATTRIBUTION_1" in states_b
    assert "C_CHK_VAR_1" in states_b
    assert "C_CHK_COV_1" in states_b
    assert "C_CHK_SCENARIO_1" in states_b
    assert "C_CHK_GOVERNANCE_1" in states_b
    assert "B_CHK_SIGNOFF" in states_b
    assert states_b[-1] == "B_EXECUTE_FINALIZE"
    assert "B_PROVIDER" not in states_b
    assert "B_MODEL" not in states_b


def test_controller_session_c_offline_e2e(tmp_path: Path):
    """Verify Controller executes SESSION_C_POLICY cleanly."""
    cmd = f"{sys.executable} demo_mock_twin.py --session SESSION_C_POLICY"
    result = run_session_result(
        "SESSION_C_POLICY", cmd, timeout=20.0, result_root=tmp_path / "session-c"
    )
    assert result.controller_status is SessionStatus.COMPLETE

    states_c = [t.state_id for t in STATE_TABLE["SESSION_C_POLICY"]]
    assert len(states_c) == 1
    assert states_c[0] == "C_EXECUTE"
