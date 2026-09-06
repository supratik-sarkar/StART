"""Untrusted Browser WebLLM Reviewer Ingestion, Hydration & Governance Gating Routes for StART v4.5."""

from __future__ import annotations

import logging
import time

from fastapi import APIRouter, HTTPException

from start.web.queue import GLOBAL_QUEUE
from start.web.schemas import (
    START_SCHEMA_VERSION,
    APIResponseEnvelope,
    HydratedEvidenceCitation,
    HydratedFindingView,
    ReviewerHydrationResponse,
    WebReviewerSubmission,
)

logger = logging.getLogger("start.web.routes_reviewer")
router = APIRouter(prefix="/api/v1/runs", tags=["reviewer"])


@router.post("/{run_id}/reviewer/hydrate-and-gate", response_model=APIResponseEnvelope)
def hydrate_and_gate_reviewer_submission(
    run_id: str,
    submission: WebReviewerSubmission,
) -> APIResponseEnvelope:
    """Validate untrusted browser WebLLM reviewer submission, hydrate metrics, and apply OPA gating.

    Strict Architectural Invariants:
    1. Zero arithmetic by the client browser or LLM.
    2. All numeric values are deterministically resolved from immutable EvidenceRecord objects.
    3. Rejects unknown Evidence IDs or references outside the active run universe.
    4. OPA governance policy decision and Merkle attestation are determined exclusively server-side.
    """
    ctx = GLOBAL_QUEUE.get_run(run_id, submission.session_id)
    if not ctx:
        if any(r.run_id == run_id for r in GLOBAL_QUEUE.list_runs()):
            raise HTTPException(status_code=403, detail=f"Session access denied for run '{run_id}'")
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")

    records = ctx.evidence_records
    evidence_universe = {r.evidence_id: r for r in records}

    hydrated_findings: list[HydratedFindingView] = []
    all_citations_grounded = True

    # 1. Process and hydrate each qualitative finding
    for finding in submission.findings:
        finding_grounded = True
        hydrated_refs: list[HydratedEvidenceCitation] = []

        for ref in finding.evidence_refs:
            ev_id = ref.evidence_id.strip("[]")
            metric_name = ref.metric_name if ref.metric_name else None
            client_val = ref.client_claimed_value

            if ev_id not in evidence_universe:
                logger.warning("Unknown Evidence ID '%s' rejected during server hydration", ev_id)
                finding_grounded = False
                all_citations_grounded = False
                hydrated_refs.append(
                    HydratedEvidenceCitation(
                        evidence_id=ev_id,
                        metric_name=metric_name,
                        client_claimed_value=client_val,
                        canonical_value=None,
                        server_hydrated_value=None,
                        grounding_status="UNGROUNDED_EVIDENCE_ID",
                        grounding_reason=f"Evidence ID '{ev_id}' not in active run evidence universe",
                    )
                )
                continue

            rec = evidence_universe[ev_id]

            # Case A: Evidence-level citation (no metric claimed)
            if not metric_name:
                hydrated_refs.append(
                    HydratedEvidenceCitation(
                        evidence_id=ev_id,
                        metric_name=None,
                        client_claimed_value=client_val,
                        canonical_value=None,
                        server_hydrated_value=None,
                        grounding_status="GROUNDED",
                        grounding_reason="Evidence-level citation grounded against immutable EvidenceRecord",
                        test_id=rec.test_id,
                        record_status=str(rec.status),
                    )
                )
                continue

            # Case B: Metric-level citation (requires exact canonical match; zero fuzzy repair)
            if metric_name not in rec.metrics:
                logger.warning("Metric '%s' not found in EvidenceRecord '%s'", metric_name, ev_id)
                finding_grounded = False
                all_citations_grounded = False
                hydrated_refs.append(
                    HydratedEvidenceCitation(
                        evidence_id=ev_id,
                        metric_name=metric_name,
                        client_claimed_value=client_val,
                        canonical_value=None,
                        server_hydrated_value=None,
                        grounding_status="UNGROUNDED_METRIC_PATH",
                        grounding_reason=(
                            f"Metric '{metric_name}' not found in EvidenceRecord '{ev_id}' metrics"
                        ),
                        test_id=rec.test_id,
                        record_status=str(rec.status),
                    )
                )
                continue

            # Case C: Exact metric found
            canonical_val = rec.metrics[metric_name]

            # If client claimed a numeric value, verify the canonical metric is numeric
            if (
                client_val is not None
                and isinstance(client_val, (int, float))
                and not isinstance(client_val, bool)
            ):
                is_canonical_numeric = (
                    isinstance(canonical_val, (int, float)) and not isinstance(canonical_val, bool)
                )
                if not is_canonical_numeric:
                    logger.warning(
                        "Client claimed numeric value %s on non-numeric metric '%s' (%s)",
                        client_val,
                        metric_name,
                        type(canonical_val).__name__,
                    )
                    finding_grounded = False
                    all_citations_grounded = False
                    hydrated_refs.append(
                        HydratedEvidenceCitation(
                            evidence_id=ev_id,
                            metric_name=metric_name,
                            client_claimed_value=client_val,
                            canonical_value=canonical_val,
                            server_hydrated_value=canonical_val,
                            grounding_status="NON_NUMERIC_METRIC_CLAIM",
                            grounding_reason=(
                                f"Client claimed numeric value {client_val}, "
                                f"but canonical metric '{metric_name}' is non-numeric "
                                f"({type(canonical_val).__name__})"
                            ),
                            test_id=rec.test_id,
                            record_status=str(rec.status),
                        )
                    )
                    continue

            # Hydrate exact canonical value (client numeric claim is untrusted)
            hydrated_refs.append(
                HydratedEvidenceCitation(
                    evidence_id=ev_id,
                    metric_name=metric_name,
                    client_claimed_value=client_val,
                    canonical_value=canonical_val,
                    server_hydrated_value=canonical_val,
                    grounding_status="GROUNDED",
                    test_id=rec.test_id,
                    record_status=str(rec.status),
                )
            )

        hydrated_findings.append(
            HydratedFindingView(
                finding_id=finding.finding_id,
                canonical_severity=None,  # server-owned authoritative severity only
                client_proposed_severity=finding.client_proposed_severity,
                severity=None,
                title=finding.title,
                description=finding.description,
                grounded=finding_grounded,
                evidence_refs=hydrated_refs,
                recommendation=finding.recommendation,
            )
        )

    # Grounding Stage: derived solely from citation grounding
    all_grounded = all_citations_grounded and all(f.grounded for f in hydrated_findings)
    is_grounded = all_grounded

    # 2. Server-side OPA policy evaluation stage (OPA failure does NOT mutate grounding)
    opa_decision: str | None = None
    opa_reasons: list[str] = []
    try:
        from start.policies.opa_policy_plane import OPAPolicyPlane

        policy_plane = OPAPolicyPlane()
        n_ungrounded = sum(1 for f in hydrated_findings if not f.grounded)
        n_failures = sum(1 for r in records if r.status.value == "fail")
        disp_candidate = "ACCEPT" if n_failures == 0 else "CONDITIONAL_ACCEPT"

        pol_dec = policy_plane.evaluate_governance_attestation(
            n_ungrounded_claims=n_ungrounded,
            n_validation_failures=n_failures,
            committee_disposition=disp_candidate,
            run_id=run_id,
        )
        opa_decision = pol_dec.decision
        opa_reasons.append(pol_dec.reason)
    except Exception as exc:
        logger.warning("OPA policy plane evaluation failed (failing closed): %s", exc)
        opa_decision = "ERROR"
        opa_reasons.append(f"OPA policy evaluation failed: {exc}")

    # 3. Canonical Governance Stage (Reviewer route does not own governance semantics)
    governance_disp = None
    if getattr(ctx, "governance", None) and getattr(ctx.governance, "disposition", None):
        governance_disp = ctx.governance.disposition

    # 4. Canonical Attestation Stage (Zero synthetic fallback)
    merkle_root: str | None = None
    if all_grounded and opa_decision == "ALLOW" and records:
        try:
            import hashlib

            from start.attestation.seal import merkle_root as compute_merkle_root

            leaf_hashes = [
                hashlib.sha256(r.model_dump_json().encode("utf-8")).hexdigest()
                for r in records
            ]
            merkle_root = compute_merkle_root(leaf_hashes)
        except Exception as exc:
            logger.warning("Canonical Merkle root computation failed: %s", exc)
            merkle_root = None

    # 5. Overall Gate Status
    if opa_decision == "ERROR":
        gate_status = "ERROR"
    elif all_grounded and opa_decision == "ALLOW":
        gate_status = "ACCEPTED"
    else:
        gate_status = "BLOCKED"

    resp = ReviewerHydrationResponse(
        run_id=run_id,
        schema_version=START_SCHEMA_VERSION,
        model_name=submission.model_name,
        gate_status=gate_status,
        is_grounded=is_grounded,
        all_grounded=all_grounded,
        hydrated_findings=hydrated_findings,
        opa_policy_decision=opa_decision,  # type: ignore[arg-type]
        opa_reasons=opa_reasons,
        governance_disposition=governance_disp,  # type: ignore[arg-type]
        attestation_seal_merkle_root=merkle_root,
        attestation_timestamp=time.time(),
    )

    return APIResponseEnvelope(
        success=True,
        run_id=run_id,
        data=resp.model_dump(),
    )
