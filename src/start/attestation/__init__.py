"""Attestation layer — the part that makes a StART review checkable.

Four mechanisms, each addressing a claim that validation tooling routinely
makes and rarely substantiates.

===========================  =============================================
Claim commonly made          Mechanism that tests it
===========================  =============================================
"the LLM only rephrases,     ``invariance`` — dual-path narrative
 it doesn't compute"          invariance attestation
"we only send aggregates"    ``disclosure`` — policy-derived evidence
                              envelope with an egress check
"the results are             ``replay`` — chain replay and cross-run
 reproducible"                metric comparison
"this report is the one      ``seal`` — Merkle commitment over plan,
 that was signed off"         evidence head, attestations and profile
===========================  =============================================

The common design principle: a property nobody can check is a property nobody
should be asked to believe. Each mechanism turns an assurance into an artefact
that a third party can recompute from what was archived.

Everything here is standard-library only, so an archived review can be verified
years later on a machine with nothing but Python — which is precisely when
verification matters and precisely when the original environment is gone.

Typical use::

    from start.attestation import attest_narrative_invariance, build_seal

    att = attest_narrative_invariance(
        section="discriminatory_power",
        deterministic_narrative=det_text,
        model_narrative=llm_text,
        evidence=evidence_records,
    )
    if not att.invariant:
        for d in att.blocking_divergences():
            print(d.kind, d.detail)

    seal = build_seal(review_id="R-2026-0042", plan=plan.as_dict(), ...)
    print(seal.seal_string())
"""

from start.attestation.claims import (
    BindingResult,
    Claim,
    bind_claims,
    extract_claims,
    flatten_evidence_values,
)
from start.attestation.disclosure import (
    POLICIES,
    DisclosureEnvelope,
    DisclosurePolicy,
    DisclosureViolation,
    build_envelope,
    policy_for,
    verify_prompt_covered,
)
from start.attestation.invariance import (
    ATTESTATION_VERSION,
    Divergence,
    InvarianceAttestation,
    attest_narrative_invariance,
)
from start.attestation.replay import (
    ChainVerdict,
    ReplayComparison,
    compare_ledgers,
    replay_ledger,
)
from start.attestation.seal import (
    LEAF_ORDER,
    SEAL_VERSION,
    ReviewSeal,
    SealLeaf,
    build_seal,
    inclusion_proof,
    merkle_root,
    verify_inclusion,
    verify_seal,
)

__all__ = [
    # claims
    "Claim",
    "BindingResult",
    "extract_claims",
    "bind_claims",
    "flatten_evidence_values",
    # invariance
    "Divergence",
    "InvarianceAttestation",
    "attest_narrative_invariance",
    "ATTESTATION_VERSION",
    # disclosure
    "DisclosurePolicy",
    "DisclosureEnvelope",
    "DisclosureViolation",
    "POLICIES",
    "policy_for",
    "build_envelope",
    "verify_prompt_covered",
    # replay
    "ChainVerdict",
    "ReplayComparison",
    "replay_ledger",
    "compare_ledgers",
    # seal
    "ReviewSeal",
    "SealLeaf",
    "build_seal",
    "verify_seal",
    "merkle_root",
    "inclusion_proof",
    "verify_inclusion",
    "LEAF_ORDER",
    "SEAL_VERSION",
]
