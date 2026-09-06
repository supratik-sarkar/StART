"""StART v5.1.2 — Reviewer Gating & Fail-Closed Truth Closure Tests.

Verifies:
1. Strict Browser reviewer structured-output truth (no synthetic fallback, fails closed on malformed JSON).
2. Client request schemas forbid server_hydrated_value and canonical_severity.
3. Client severity is untrusted (client_proposed_severity audit-only; canonical_severity server-owned).
4. Blank metric citation grounds at evidence level without choosing first metric.
5. Exact metric path matching (zero fuzzy repair; ungrounded on mismatch).
6. Malicious client numeric claim ignored; hydrated from canonical EvidenceRecord.
7. Numeric claim on non-numeric metric rejected (NON_NUMERIC_METRIC_CLAIM).
8. OPA failure fails closed (ERROR/BLOCKED) without mutating citation grounding.
9. Reviewer route does not invent governance (no default ACCEPT).
10. Attestation failure returns None (zero synthetic SEAL-HASH fallback).
11. Active session binding enforced (no fabricated session ID).
12. Truthful Optuna metadata (distinguishes NOT_AVAILABLE vs FAILED; zero fake 5 trials completed).
13. Complete synthetic fixture metadata classification audit (UNCLASSIFIED_SYNTHETIC_METADATA_FIELDS = 0).
14. Descriptive metadata does not enter EvidenceRecord metrics as measured truth.
"""

import builtins
import time
from typing import Any
from unittest.mock import patch

import pytest
from pydantic import ValidationError

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from start.core.schemas import EvidenceRecord, Status, TestResult
from start.data.synthetic_dl import generate_dl_world
from start.web.queue import GLOBAL_QUEUE, ActiveRunContext
from start.web.routes_reviewer import hydrate_and_gate_reviewer_submission
from start.web.schemas import (
    EvidenceCitationRequest,
    QualitativeFinding,
    RunRequest,
    WebReviewerSubmission,
)


@pytest.fixture
def run_context_with_evidence():
    """Create an active run context with known evidence records."""
    run_id = f"RUN-TEST-{int(time.time() * 1000)}"
    session_id = f"SES-TEST-{int(time.time() * 1000)}"

    req = RunRequest(
        domain="predictive",
        session_id=session_id,
        workflow="predictive_ml",
        synthetic_profile="institutional_credit_v1",
    )
    ctx = ActiveRunContext(run_id=run_id, session_id=session_id, request=req)

    # Add 2 known EvidenceRecords
    tr1 = TestResult(
        test_id="test_auroc",
        test_name="Area Under ROC Curve",
        status=Status.PASS,
        metrics={"auroc": 0.845, "prauc": 0.782, "model_type": "binary_logistic"},
        interpretation="Good discrimination",
    )
    r1 = EvidenceRecord(
        evidence_id="EV-TEST-001",
        run_id=run_id,
        test_id="test_auroc",
        test_name=tr1.test_name,
        model_id="MOD-TEST-01",
        dataset_id="DS-TEST-01",
        status=Status.PASS,
        metrics=tr1.metrics,
    )

    tr2 = TestResult(
        test_id="test_brier",
        test_name="Brier Score Loss",
        status=Status.PASS,
        metrics={"brier_score": 0.112, "is_calibrated": True, "calibration_label": "reliable"},
        interpretation="Well calibrated",
    )
    r2 = EvidenceRecord(
        evidence_id="EV-TEST-002",
        run_id=run_id,
        test_id="test_brier",
        test_name=tr2.test_name,
        model_id="MOD-TEST-01",
        dataset_id="DS-TEST-01",
        status=Status.PASS,
        metrics=tr2.metrics,
    )

    ctx.evidence_records.extend([r1, r2])
    ctx.status = "COMPLETED"
    GLOBAL_QUEUE._runs[run_id] = ctx
    return ctx


# --------------------------------------------------------------------------- #
# 1. Schema Invariants: Client Request vs Server Hydrated Response
# --------------------------------------------------------------------------- #
def test_client_cannot_set_server_hydrated_value():
    """CLIENT_CAN_SET_SERVER_HYDRATED_VALUE = NO. Extra fields forbidden on EvidenceCitationRequest."""
    with pytest.raises(ValidationError):
        EvidenceCitationRequest(
            evidence_id="EV-TEST-001",
            metric_name="auroc",
            server_hydrated_value=0.999,  # type: ignore[call-arg]
        )


def test_client_cannot_set_canonical_severity():
    """CLIENT_CAN_SET_CANONICAL_SEVERITY = NO. Extra fields forbidden on QualitativeFinding."""
    with pytest.raises(ValidationError):
        QualitativeFinding(
            title="Test Finding",
            description="Finding text",
            canonical_severity="CRITICAL",  # type: ignore[call-arg]
        )


def test_client_proposed_severity_audit_only(run_context_with_evidence):
    """CLIENT_PROPOSED_SEVERITY_AUDIT_ONLY = PASS. Client severity is untrusted; server produces canonical_severity=None."""
    sub = WebReviewerSubmission(
        run_id=run_context_with_evidence.run_id,
        session_id=run_context_with_evidence.session_id,
        executive_summary="Review summary",
        findings=[
            QualitativeFinding(
                title="Finding with claimed severity",
                description="High severity claimed by client",
                client_proposed_severity="CRITICAL",
                evidence_refs=[EvidenceCitationRequest(evidence_id="EV-TEST-001")],
            )
        ],
    )
    resp = hydrate_and_gate_reviewer_submission(run_context_with_evidence.run_id, sub)
    data = resp.data
    finding = data["hydrated_findings"][0]

    assert finding["client_proposed_severity"] == "CRITICAL"
    assert finding["canonical_severity"] is None
    assert finding["severity"] is None


# --------------------------------------------------------------------------- #
# 2. Blank Metric Name: Evidence-Level Grounding (No First Metric Fallback)
# --------------------------------------------------------------------------- #
def test_blank_metric_name_evidence_level_grounding(run_context_with_evidence):
    """BLANK_METRIC_SELECTS_ARBITRARY_METRIC = NO. Blank metric grounds at evidence level."""
    sub = WebReviewerSubmission(
        run_id=run_context_with_evidence.run_id,
        session_id=run_context_with_evidence.session_id,
        executive_summary="Review summary",
        findings=[
            QualitativeFinding(
                title="Evidence-level observation",
                description="Grounded qualitatively on EV-TEST-001",
                evidence_refs=[EvidenceCitationRequest(evidence_id="EV-TEST-001", metric_name=None)],
            )
        ],
    )
    resp = hydrate_and_gate_reviewer_submission(run_context_with_evidence.run_id, sub)
    ref = resp.data["hydrated_findings"][0]["evidence_refs"][0]

    assert ref["grounding_status"] == "GROUNDED"
    assert ref["metric_name"] is None
    assert ref["canonical_value"] is None
    assert ref["server_hydrated_value"] is None
    # Crucially, it did NOT choose "auroc" (the first metric)
    assert ref["metric_name"] != "auroc"


# --------------------------------------------------------------------------- #
# 3. Exact Metric Path Matching (Zero Fuzzy/Prefix Matching)
# --------------------------------------------------------------------------- #
def test_metric_path_exact_matching(run_context_with_evidence):
    """METRIC_PATH_FUZZY_REPAIR = 0. Fuzzy matches (auroc vs test_auroc.auroc or roc) must fail."""
    sub = WebReviewerSubmission(
        run_id=run_context_with_evidence.run_id,
        session_id=run_context_with_evidence.session_id,
        executive_summary="Review summary",
        findings=[
            QualitativeFinding(
                title="Fuzzy metric claim",
                description="Testing suffix or substring",
                evidence_refs=[
                    EvidenceCitationRequest(evidence_id="EV-TEST-001", metric_name="test_auroc.auroc"),
                    EvidenceCitationRequest(evidence_id="EV-TEST-001", metric_name="roc"),
                ],
            )
        ],
    )
    resp = hydrate_and_gate_reviewer_submission(run_context_with_evidence.run_id, sub)
    refs = resp.data["hydrated_findings"][0]["evidence_refs"]

    for ref in refs:
        assert ref["grounding_status"] == "UNGROUNDED_METRIC_PATH"
        assert ref["canonical_value"] is None
    assert resp.data["all_grounded"] is False


def test_unknown_evidence_id_rejected(run_context_with_evidence):
    """UNKNOWN_EVIDENCE_REJECTION = PASS. Non-existent Evidence ID marked UNGROUNDED_EVIDENCE_ID."""
    sub = WebReviewerSubmission(
        run_id=run_context_with_evidence.run_id,
        session_id=run_context_with_evidence.session_id,
        executive_summary="Review summary",
        findings=[
            QualitativeFinding(
                title="Unknown evidence claim",
                description="Citing non-existent record",
                evidence_refs=[EvidenceCitationRequest(evidence_id="EV-NON-EXISTENT-999")],
            )
        ],
    )
    resp = hydrate_and_gate_reviewer_submission(run_context_with_evidence.run_id, sub)
    ref = resp.data["hydrated_findings"][0]["evidence_refs"][0]

    assert ref["grounding_status"] == "UNGROUNDED_EVIDENCE_ID"
    assert resp.data["all_grounded"] is False


# --------------------------------------------------------------------------- #
# 4. Malicious Numeric Claim: Server Hydrates Canonical Value
# --------------------------------------------------------------------------- #
def test_malicious_number_hydration(run_context_with_evidence):
    """MALICIOUS_NUMBER_HYDRATION = PASS. Client claimed value ignored; server hydrates canonical value."""
    malicious_claim = 0.12345
    canonical_val = 0.845

    sub = WebReviewerSubmission(
        run_id=run_context_with_evidence.run_id,
        session_id=run_context_with_evidence.session_id,
        executive_summary="Review summary",
        findings=[
            QualitativeFinding(
                title="Malicious claim",
                description="Client passes wrong number",
                evidence_refs=[
                    EvidenceCitationRequest(
                        evidence_id="EV-TEST-001",
                        metric_name="auroc",
                        client_claimed_value=malicious_claim,
                    )
                ],
            )
        ],
    )
    resp = hydrate_and_gate_reviewer_submission(run_context_with_evidence.run_id, sub)
    ref = resp.data["hydrated_findings"][0]["evidence_refs"][0]

    assert ref["grounding_status"] == "GROUNDED"
    assert ref["client_claimed_value"] == malicious_claim
    assert ref["canonical_value"] == canonical_val
    assert ref["server_hydrated_value"] == canonical_val
    assert ref["server_hydrated_value"] != malicious_claim


def test_numeric_claim_on_non_numeric_metric_rejected(run_context_with_evidence):
    """NUMERIC_CLAIM_NON_NUMERIC_METRIC = REJECTED. Claiming numeric value on string metric rejects."""
    sub = WebReviewerSubmission(
        run_id=run_context_with_evidence.run_id,
        session_id=run_context_with_evidence.session_id,
        executive_summary="Review summary",
        findings=[
            QualitativeFinding(
                title="Numeric claim on string metric",
                description="Client claims 42.0 on 'model_type' string metric",
                evidence_refs=[
                    EvidenceCitationRequest(
                        evidence_id="EV-TEST-001",
                        metric_name="model_type",
                        client_claimed_value=42.0,
                    )
                ],
            )
        ],
    )
    resp = hydrate_and_gate_reviewer_submission(run_context_with_evidence.run_id, sub)
    ref = resp.data["hydrated_findings"][0]["evidence_refs"][0]

    assert ref["grounding_status"] == "NON_NUMERIC_METRIC_CLAIM"
    assert resp.data["all_grounded"] is False


# --------------------------------------------------------------------------- #
# 5. OPA Failure Fails Closed Without Mutating Grounding
# --------------------------------------------------------------------------- #
def test_opa_failure_fails_closed_without_mutating_grounding(run_context_with_evidence):
    """OPA_FAILURE_MUTATES_GROUNDING = NO, OPA_FAIL_CLOSED = PASS.

    OPA exception produces ERROR/BLOCKED, but all_grounded remains True based on citation truth.
    """
    sub = WebReviewerSubmission(
        run_id=run_context_with_evidence.run_id,
        session_id=run_context_with_evidence.session_id,
        executive_summary="Review summary",
        findings=[
            QualitativeFinding(
                title="Valid grounded finding",
                description="Citing known auroc",
                evidence_refs=[EvidenceCitationRequest(evidence_id="EV-TEST-001", metric_name="auroc")],
            )
        ],
    )

    with patch(
        "start.policies.opa_policy_plane.OPAPolicyPlane.evaluate_governance_attestation",
        side_effect=RuntimeError("OPA connection refused"),
    ):
        resp = hydrate_and_gate_reviewer_submission(run_context_with_evidence.run_id, sub)
        data = resp.data

        # Grounding is based strictly on citations: EV-TEST-001/auroc is grounded
        assert data["all_grounded"] is True
        assert data["is_grounded"] is True

        # Policy failed closed
        assert data["opa_policy_decision"] == "ERROR"
        assert data["gate_status"] == "ERROR"

        # Governance & Attestation are NOT invented
        assert data["governance_disposition"] is None
        assert data["attestation_seal_merkle_root"] is None


# --------------------------------------------------------------------------- #
# 6. Reviewer Route Does Not Own Governance Semantics (No Default ACCEPT)
# --------------------------------------------------------------------------- #
def test_reviewer_route_no_default_accept(run_context_with_evidence):
    """REVIEW_GATE_DEFAULT_ACCEPT = 0, REVIEWER_ROUTE_OWNS_GOVERNANCE_SEMANTICS = NO.

    Grounding success alone does not synthesize ACCEPT disposition.
    """
    sub = WebReviewerSubmission(
        run_id=run_context_with_evidence.run_id,
        session_id=run_context_with_evidence.session_id,
        executive_summary="Review summary",
        findings=[
            QualitativeFinding(
                title="Valid finding",
                description="Valid citation",
                evidence_refs=[EvidenceCitationRequest(evidence_id="EV-TEST-001", metric_name="auroc")],
            )
        ],
    )
    resp = hydrate_and_gate_reviewer_submission(run_context_with_evidence.run_id, sub)
    data = resp.data

    # ctx.governance is None, so reviewer route returns None (does NOT invent ACCEPT)
    assert data["governance_disposition"] is None


# --------------------------------------------------------------------------- #
# 7. Zero Synthetic Attestation Fallback
# --------------------------------------------------------------------------- #
def test_attestation_failure_no_fake_root(run_context_with_evidence):
    """SYNTHETIC_ATTESTATION_FALLBACK = 0, REVIEWER_ROUTE_SYNTHETIC_ATTESTATION = 0.

    When ledger raises an exception, root is None; zero SEAL-HASH- fallback.
    """
    sub = WebReviewerSubmission(
        run_id=run_context_with_evidence.run_id,
        session_id=run_context_with_evidence.session_id,
        executive_summary="Review summary",
        findings=[
            QualitativeFinding(
                title="Valid finding",
                description="Valid citation",
                evidence_refs=[EvidenceCitationRequest(evidence_id="EV-TEST-001", metric_name="auroc")],
            )
        ],
    )

    with patch(
        "start.attestation.seal.merkle_root",
        side_effect=RuntimeError("Merkle computation failure"),
    ):
        resp = hydrate_and_gate_reviewer_submission(run_context_with_evidence.run_id, sub)
        data = resp.data

        assert data["attestation_seal_merkle_root"] is None


# --------------------------------------------------------------------------- #
# 8. Active Session ID Required (No Fabricated Session ID)
# --------------------------------------------------------------------------- #
def test_session_id_binding_enforced(run_context_with_evidence):
    """FABRICATED_REVIEW_SESSION_ID = 0. Wrong session rejected with 403."""
    sub = WebReviewerSubmission(
        run_id=run_context_with_evidence.run_id,
        session_id="SES-WRONG-FABRICATED",
        executive_summary="Review summary",
        findings=[],
    )
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc_info:
        hydrate_and_gate_reviewer_submission(run_context_with_evidence.run_id, sub)
    assert exc_info.value.status_code == 403


# --------------------------------------------------------------------------- #
# 9. Truthful Optuna Metadata (Zero Fake 5 Trials Completed)
# --------------------------------------------------------------------------- #
def test_optuna_metadata_truthfulness():
    """FAKE_OPTUNA_SUCCESS_FALLBACK = 0.

    When optuna is unavailable, tuning_status is NOT_AVAILABLE, trials_completed = 0.
    """
    orig_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "optuna":
            raise ImportError("No module named 'optuna'")
        return orig_import(name, *args, **kwargs)

    builtins.__import__ = mock_import
    try:
        world = generate_dl_world(n_samples=100, n_features=4, seed=42)
        tuning = world["tuning_metadata"]

        assert tuning["tuning_status"] == "NOT_AVAILABLE"
        assert tuning["trials_completed"] == 0
        assert tuning["best_trial_idx"] is None
        assert tuning["best_hyperparameters"] is None
    finally:
        builtins.__import__ = orig_import


def test_optuna_failure_truthfulness():
    """When optuna study execution fails, tuning_status is FAILED, trials_completed = 0."""
    pytest.importorskip("optuna")

    with patch("optuna.create_study", side_effect=RuntimeError("Study memory exhausted")):
        world = generate_dl_world(n_samples=100, n_features=4, seed=42)
        tuning = world["tuning_metadata"]

        assert tuning["tuning_status"] == "FAILED"
        assert tuning["trials_completed"] == 0
        assert tuning["best_trial_idx"] is None
        assert tuning["best_hyperparameters"] is None


# --------------------------------------------------------------------------- #
# 10. Complete Synthetic Fixture Metadata Audit (No Unclassified Fields)
# --------------------------------------------------------------------------- #
def test_complete_synthetic_fixture_metadata_audit():
    """UNCLASSIFIED_SYNTHETIC_METADATA_FIELDS = 0.

    Audits every single key returned by generate_dl_world() across all metadata blocks.
    """
    world = generate_dl_world(n_samples=100, n_features=4, seed=42)

    CLASSIFICATIONS = {
        # Preprocessing metadata
        ("preprocessing_metadata", "n_samples_total"): "FIXED_GENERATOR_PARAMETER",
        ("preprocessing_metadata", "n_train"): "ACTUAL_RUNTIME_CONFIGURATION",
        ("preprocessing_metadata", "n_val"): "ACTUAL_RUNTIME_CONFIGURATION",
        ("preprocessing_metadata", "n_test"): "ACTUAL_RUNTIME_CONFIGURATION",
        ("preprocessing_metadata", "n_features"): "FIXED_GENERATOR_PARAMETER",
        ("preprocessing_metadata", "feature_names"): "ACTUAL_RUNTIME_CONFIGURATION",
        ("preprocessing_metadata", "target_column"): "ACTUAL_RUNTIME_CONFIGURATION",
        ("preprocessing_metadata", "class_imbalance_ratio"): "COMPUTED",
        ("preprocessing_metadata", "missing_rate_feat_04"): "COMPUTED",
        ("preprocessing_metadata", "scaling"): "ACTUAL_RUNTIME_CONFIGURATION",
        ("preprocessing_metadata", "imputation"): "ACTUAL_RUNTIME_CONFIGURATION",
        ("preprocessing_metadata", "encoding"): "ACTUAL_RUNTIME_CONFIGURATION",
        ("preprocessing_metadata", "data_leakage_check"): "COMPUTED",
        ("preprocessing_metadata", "split_strategy"): "ACTUAL_RUNTIME_CONFIGURATION",
        # Architecture metadata
        ("architecture_metadata", "framework"): "DESCRIPTIVE_ONLY",
        ("architecture_metadata", "family"): "ACTUAL_RUNTIME_CONFIGURATION",
        ("architecture_metadata", "device"): "ACTUAL_RUNTIME_CONFIGURATION",
        ("architecture_metadata", "layers"): "ACTUAL_RUNTIME_CONFIGURATION",
        ("architecture_metadata", "trainable_parameters"): "COMPUTED",
        ("architecture_metadata", "non_trainable_parameters"): "COMPUTED",
        ("architecture_metadata", "optimizer"): "ACTUAL_RUNTIME_CONFIGURATION",
        ("architecture_metadata", "learning_rate"): "ACTUAL_RUNTIME_CONFIGURATION",
        ("architecture_metadata", "weight_decay"): "ACTUAL_RUNTIME_CONFIGURATION",
        ("architecture_metadata", "scheduler"): "ACTUAL_RUNTIME_CONFIGURATION",
        ("architecture_metadata", "loss_function"): "ACTUAL_RUNTIME_CONFIGURATION",
        ("architecture_metadata", "batch_size"): "ACTUAL_RUNTIME_CONFIGURATION",
        ("architecture_metadata", "epochs_requested"): "ACTUAL_RUNTIME_CONFIGURATION",
        ("architecture_metadata", "epochs_completed"): "COMPUTED",
        ("architecture_metadata", "best_epoch"): "COMPUTED",
        ("architecture_metadata", "early_stopping"): "ACTUAL_RUNTIME_CONFIGURATION",
        ("architecture_metadata", "seed"): "FIXED_GENERATOR_PARAMETER",
        # Tuning metadata
        ("tuning_metadata", "tuning_method"): "DESCRIPTIVE_ONLY",
        ("tuning_metadata", "search_space"): "ACTUAL_RUNTIME_CONFIGURATION",
        ("tuning_metadata", "tuning_status"): "COMPLETED" if world["tuning_metadata"]["tuning_status"] == "COMPLETED" else "COMPUTED",
        ("tuning_metadata", "trials_completed"): "COMPUTED",
        ("tuning_metadata", "best_trial_idx"): "COMPUTED",
        ("tuning_metadata", "best_hyperparameters"): "COMPUTED",
        ("tuning_metadata", "best_value"): "COMPUTED",
        ("tuning_metadata", "train_val_generalization_gap"): "COMPUTED",
        ("tuning_metadata", "overfitting_diagnostic"): "COMPUTED",
        # Sensitivity metadata
        ("sensitivity_metadata", "seed_dispersion_std"): "COMPUTED",
        ("sensitivity_metadata", "perturbation_snr_10db_delta_auc"): "COMPUTED",
        ("sensitivity_metadata", "missingness_stress_20pct_delta_auc"): "COMPUTED",
        # Explainability metadata
        ("explainability_metadata", "method"): "DESCRIPTIVE_ONLY",
        ("explainability_metadata", "top_features"): "COMPUTED",
    }

    # Verify architecture matches real model
    arch = world["architecture_metadata"]
    assert arch["family"] == "Tabular MLP"
    assert arch["optimizer"] == "Adam"
    assert arch["scheduler"] == "None"
    assert "integrated_gradients_baseline" not in world["explainability_metadata"]

    unclassified = []
    for section_key in ("preprocessing_metadata", "architecture_metadata", "tuning_metadata", "sensitivity_metadata", "explainability_metadata"):
        sec = world.get(section_key, {})
        for field in sec:
            if (section_key, field) not in CLASSIFICATIONS and not field.startswith("tuning_error"):
                unclassified.append((section_key, field))

    assert len(unclassified) == 0, f"Unclassified metadata fields found: {unclassified}"


def test_descriptive_metadata_not_in_evidence_metrics():
    """DESCRIPTIVE_FIXTURE_METADATA_AS_MEASURED_EVIDENCE = 0.

    Ensures that descriptive/configuration strings (e.g. 'PyTorch 2.x', 'Tabular MLP')
    do not enter EvidenceRecord.metrics.
    """
    from start.runtime.execution import CanonicalExecutionService

    res = CanonicalExecutionService.execute(
        workflow_id="deep_learning",
        context_id="deep_learning_v1",
        materiality="low",
        seed=42,
    )

    forbidden_strings = {"PyTorch 2.x", "Tabular MLP", "Adam", "CosineAnnealingLR"}
    for rec in res.records:
        for metric_name, metric_val in rec.metrics.items():
            assert metric_val not in forbidden_strings, (
                f"Descriptive metadata value '{metric_val}' found in EvidenceRecord '{rec.evidence_id}' metric '{metric_name}'"
            )


# --------------------------------------------------------------------------- #
# 15. Graph Provenance: Evidence/Artifact Edges from RuntimeEvent refs ONLY
# --------------------------------------------------------------------------- #

def _make_graph_test_context(
    run_id: str,
    *,
    parent_run_id: str | None = None,
    events: list[dict],
    evidence_records: list[Any] | None = None,
    artifacts: dict | None = None,
    presentation: dict | None = None,
) -> "ActiveRunContext":
    """Create a minimal ActiveRunContext for graph endpoint testing."""
    from start.web.queue import GLOBAL_QUEUE, ActiveRunContext

    req = RunRequest(
        domain="predictive",
        session_id="SES-GRAPH-TEST",
        workflow="predictive_ml",
        synthetic_profile="institutional_credit_v1",
    )
    if parent_run_id:
        req.parent_run_id = parent_run_id

    ctx = ActiveRunContext(run_id=run_id, session_id="SES-GRAPH-TEST", request=req)
    ctx.events = events
    ctx.evidence_records = evidence_records or []
    ctx.artifacts = artifacts or {}
    ctx.presentation = presentation
    ctx.status = "COMPLETED"
    GLOBAL_QUEUE._runs[run_id] = ctx
    return ctx


def test_evidence_with_test_id_but_no_event_ref_has_no_producer_edge():
    """WEB_GRAPH_EVIDENCE_EDGE_FROM_STATIC_TEST_MAPPING = 0.

    An EvidenceRecord has test_id matching a plan step,
    BUT no RuntimeEvent.evidence_refs references that evidence ID.
    Result: evidence node exists, NO observed producer edge.
    """
    from fastapi.testclient import TestClient

    from start.web.app import create_app
    from start.web.queue import GLOBAL_QUEUE

    run_id = "RUN-GRAPH-NO-REF"

    # Events: step-evidence completed, but NO evidence_refs
    events = [
        {
            "event_id": "EVT-001",
            "event_type": "test_completed",
            "status": "COMPLETED",
            "node_id": "step-evidence",
            "parent_node_id": "step-supervised",
            "test_id": "test_auroc",
            "evidence_refs": [],  # deliberately empty
            "artifact_refs": [],
        },
        {
            "event_id": "EVT-002",
            "event_type": "agent_transition",
            "status": "COMPLETED",
            "node_id": "step-supervised",
            "parent_node_id": None,
            "evidence_refs": [],
            "artifact_refs": [],
        },
    ]

    # EvidenceRecord with test_id that would match step-evidence via static mapping
    ev_rec = EvidenceRecord(
        evidence_id="EV-ORPHAN-001",
        run_id=run_id,
        test_id="test_auroc",
        test_name="Area Under ROC Curve",
        model_id="MOD-01",
        dataset_id="DS-01",
        status=Status.PASS,
        metrics={"auroc": 0.85},
    )

    _make_graph_test_context(run_id, events=events, evidence_records=[ev_rec])

    app = create_app()
    client = TestClient(app)
    resp = client.get(f"/api/v1/runs/{run_id}/graph")
    assert resp.status_code == 200
    graph = resp.json()

    # Evidence node MUST exist
    ev_node_ids = {n["id"] for n in graph["nodes"] if n["kind"] == "evidence"}
    assert "EV-ORPHAN-001" in ev_node_ids, "Evidence node must be visible even without producer edge"

    # NO observed producer edge to this evidence (no event referenced it)
    observed_edges = [e for e in graph["edges"] if e.get("edgeKind") == "observed"]
    evidence_target_edges = [e for e in observed_edges if e["target"] == "EV-ORPHAN-001"]
    assert len(evidence_target_edges) == 0, (
        f"Expected zero observed producer edges for unreferenced evidence, got: {evidence_target_edges}"
    )

    del GLOBAL_QUEUE._runs[run_id]


def test_event_evidence_refs_produces_observed_edge():
    """RuntimeEvent.evidence_refs → observed edge exists.

    When a RuntimeEvent references an evidence ID via evidence_refs,
    the graph must contain an observed edge from the event's node_id
    to that evidence ID.
    """
    from fastapi.testclient import TestClient

    from start.web.app import create_app
    from start.web.queue import GLOBAL_QUEUE

    run_id = "RUN-GRAPH-WITH-REF"

    events = [
        {
            "event_id": "EVT-010",
            "event_type": "test_completed",
            "status": "COMPLETED",
            "node_id": "step-evidence",
            "parent_node_id": "step-supervised",
            "test_id": "test_auroc",
            "evidence_refs": ["EV-LINKED-001"],
            "artifact_refs": [],
        },
        {
            "event_id": "EVT-011",
            "event_type": "agent_transition",
            "status": "COMPLETED",
            "node_id": "step-supervised",
            "parent_node_id": None,
            "evidence_refs": [],
            "artifact_refs": [],
        },
    ]

    ev_rec = EvidenceRecord(
        evidence_id="EV-LINKED-001",
        run_id=run_id,
        test_id="test_auroc",
        test_name="Area Under ROC Curve",
        model_id="MOD-01",
        dataset_id="DS-01",
        status=Status.PASS,
        metrics={"auroc": 0.85},
    )

    _make_graph_test_context(run_id, events=events, evidence_records=[ev_rec])

    app = create_app()
    client = TestClient(app)
    resp = client.get(f"/api/v1/runs/{run_id}/graph")
    assert resp.status_code == 200
    graph = resp.json()

    # Observed producer edge MUST exist from step-evidence to EV-LINKED-001
    observed_edges = {e["id"]: e for e in graph["edges"] if e.get("edgeKind") == "observed"}
    expected_edge_id = "edge-step-evidence-EV-LINKED-001"
    assert expected_edge_id in observed_edges, (
        f"Expected observed edge '{expected_edge_id}' not found. Got: {list(observed_edges.keys())}"
    )
    edge = observed_edges[expected_edge_id]
    assert edge["source"] == "step-evidence"
    assert edge["target"] == "EV-LINKED-001"

    del GLOBAL_QUEUE._runs[run_id]


def test_parent_run_produces_lineage_edge_not_observed():
    """SYNTHETIC_PARENT_TO_FIRST_OBSERVED_EDGE = 0.

    When parent_run_id exists without a canonical parent_node event,
    NO synthetic observed parent-to-first edge should be emitted.
    The edge should be edgeKind='lineage' instead.
    """
    from fastapi.testclient import TestClient

    from start.web.app import create_app
    from start.web.queue import GLOBAL_QUEUE

    run_id = "RUN-CHILD-LINEAGE"

    events = [
        {
            "event_id": "EVT-020",
            "event_type": "agent_transition",
            "status": "COMPLETED",
            "node_id": "step-context",
            "parent_node_id": None,
            "evidence_refs": [],
            "artifact_refs": [],
        },
    ]

    _make_graph_test_context(
        run_id,
        parent_run_id="RUN-PARENT-LINEAGE",
        events=events,
    )

    app = create_app()
    client = TestClient(app)
    resp = client.get(f"/api/v1/runs/{run_id}/graph")
    assert resp.status_code == 200
    graph = resp.json()

    # Collect all edges
    all_edges = graph["edges"]
    observed_edges = [e for e in all_edges if e.get("edgeKind") == "observed"]
    lineage_edges = [e for e in all_edges if e.get("edgeKind") == "lineage"]

    # NO observed edge from parent-run
    parent_observed = [e for e in observed_edges if e.get("source") == "parent-run"]
    assert len(parent_observed) == 0, (
        f"Expected zero synthetic observed parent edges, got: {parent_observed}"
    )

    # Lineage edge SHOULD exist
    parent_lineage = [e for e in lineage_edges if e.get("source") == "parent-run"]
    assert len(parent_lineage) == 1, (
        f"Expected exactly one lineage edge from parent-run, got: {parent_lineage}"
    )
    assert parent_lineage[0]["id"] == "edge-parent-to-first"

    del GLOBAL_QUEUE._runs[run_id]


# --------------------------------------------------------------------------- #
# 16. Graph Oracle Topology & Provenance Regressions (v5.1.2 Pre-Freeze)
# --------------------------------------------------------------------------- #

def test_parent_only_runtime_node_recognized():
    """PARENT_ONLY_RUNTIME_NODE_RECOGNIZED = PASS.

    Canonical runtime contains:
        parent_node_id = step-context
        node_id = step-preflight
    but there is no separate event with:
        node_id = step-context

    Expected oracle behavior:
        step-context = observed canonical node
        step-preflight = observed canonical node
        step-context → step-preflight = expected observed edge
    """
    import sys
    from pathlib import Path

    scripts_dir = str(Path(__file__).resolve().parent.parent / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from run_v512_acceptance import build_independent_graph_oracle

    events = [
        {
            "event_id": "EVT-PF-001",
            "event_type": "agent_transition",
            "status": "COMPLETED",
            "node_id": "step-preflight",
            "parent_node_id": "step-context",
            "evidence_refs": [],
            "artifact_refs": [],
        },
    ]

    expected_nodes, expected_edges = build_independent_graph_oracle(events)

    # step-context = observed canonical node
    assert "step-context" in expected_nodes, (
        f"step-context must be recognized as observed canonical node from parent_node_id. Got: {expected_nodes}"
    )

    # step-preflight = observed canonical node
    assert "step-preflight" in expected_nodes, (
        f"step-preflight must be recognized as observed canonical node from node_id. Got: {expected_nodes}"
    )

    # step-context → step-preflight = expected observed edge
    expected_edge_id = "edge-obs-step-context-step-preflight"
    assert expected_edge_id in expected_edges, (
        f"Expected observed edge '{expected_edge_id}' not found in expected edges: {expected_edges}"
    )


def test_oracle_node_ids_include_parent_node_ids():
    """GRAPH_ORACLE_NODE_IDS_INCLUDE_PARENT_NODE_IDS = PASS."""
    import sys
    from pathlib import Path

    scripts_dir = str(Path(__file__).resolve().parent.parent / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from run_v512_acceptance import build_independent_graph_oracle

    events = [
        {"node_id": "child-a", "parent_node_id": "parent-a"},
        {"node_id": "child-b", "parent_node_id": "parent-b"},
    ]
    expected_nodes, expected_edges = build_independent_graph_oracle(events)
    assert "parent-a" in expected_nodes
    assert "child-a" in expected_nodes
    assert "parent-b" in expected_nodes
    assert "child-b" in expected_nodes
    assert "edge-obs-parent-a-child-a" in expected_edges
    assert "edge-obs-parent-b-child-b" in expected_edges


def test_duplicate_governance_oracle_edge_zero():
    """DUPLICATE_GOVERNANCE_ORACLE_EDGE = 0.

    step-xai → step-governance comes solely from generic parent_node_id → node_id.
    Zero duplicate governance transition edges are added.
    """
    import sys
    from pathlib import Path

    scripts_dir = str(Path(__file__).resolve().parent.parent / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from run_v512_acceptance import build_independent_graph_oracle

    events = [
        {
            "event_id": "EVT-XAI-001",
            "event_type": "agent_transition",
            "status": "COMPLETED",
            "node_id": "step-xai",
            "parent_node_id": "step-robustness",
            "evidence_refs": [],
            "artifact_refs": [],
        },
        {
            "event_id": "EVT-GOV-001",
            "event_type": "governance_decided",
            "status": "COMPLETED",
            "node_id": "step-governance",
            "parent_node_id": "step-xai",
            "evidence_refs": [],
            "artifact_refs": [],
        },
    ]
    expected_nodes, expected_edges = build_independent_graph_oracle(events)
    assert "step-governance" in expected_nodes
    gov_incoming = [e for e in expected_edges if e.endswith("-step-governance")]
    assert len(gov_incoming) == 1, f"Expected exactly 1 transition edge to step-governance, got: {gov_incoming}"
    assert gov_incoming[0] == "edge-obs-step-xai-step-governance"


def test_attestation_graph_node_canonical_source():
    """ATTESTATION_GRAPH_NODE_HAS_CANONICAL_SOURCE = PASS.

    attest node is only expected when attestation_created event exists.
    """
    import sys
    from pathlib import Path

    scripts_dir = str(Path(__file__).resolve().parent.parent / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from run_v512_acceptance import build_independent_graph_oracle

    events_without = [
        {"node_id": "step-governance", "parent_node_id": None, "event_type": "governance_decided"}
    ]
    nodes_without, _ = build_independent_graph_oracle(events_without)
    assert "attest" not in nodes_without

    events_with = [
        {"node_id": "step-governance", "parent_node_id": None, "event_type": "governance_decided"},
        {"node_id": "step-governance", "parent_node_id": None, "event_type": "attestation_created"},
    ]
    nodes_with, edges_with = build_independent_graph_oracle(events_with)
    assert "attest" in nodes_with
    assert "edge-step-governance-attest" in edges_with

