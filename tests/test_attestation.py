"""Attestation layer: claims, invariance, disclosure, replay, seals.

Each mechanism exists to make a routinely-asserted property checkable. These
tests check the checker — including, importantly, that it does not fire on
honest rewording, because a control with false positives gets switched off.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from start.attestation import (
    POLICIES,
    DisclosureViolation,
    attest_narrative_invariance,
    bind_claims,
    build_envelope,
    build_seal,
    compare_ledgers,
    extract_claims,
    flatten_evidence_values,
    merkle_root,
    replay_ledger,
    verify_inclusion,
    verify_prompt_covered,
    verify_seal,
)

EVIDENCE = [
    {
        "evidence_id": "EV-7f3a1c8b0d21",
        "test_id": "supervised.cohort_metrics_comparison",
        "test_name": "Cohort metric comparison",
        "status": "warn",
        "metrics": {"train_auc": 0.8421, "test_auc": 0.7714, "gap": 0.0707},
        "thresholds": [{"metric": "gap", "warn": 0.05, "fail": 0.10}],
        "interpretation": "Holdout degradation exceeds the warn threshold.",
    }
]

DETERMINISTIC = (
    "Train AUC is 0.8421 against test AUC 0.7714 [EV-7f3a1c8b0d21]. The gap of 0.0707 "
    "exceeds the warn threshold of 0.05."
)


# --------------------------------------------------------------------------- #
# Claim extraction
# --------------------------------------------------------------------------- #
def test_numbers_at_sentence_end_are_extracted() -> None:
    """Regression: a trailing full stop used to swallow the claim before it."""
    claims = extract_claims("The gap was 0.05. Accuracy reached 0.913.")
    assert [c.surface for c in claims] == ["0.05", "0.913"]


def test_structural_and_citation_numbers_are_not_claims() -> None:
    text = "Per SR 11-7 and BCBS 239, phase 2 of the 2024 review found 12.4% lift."
    assert [c.surface for c in extract_claims(text)] == ["12.4%"]


def test_dotted_version_strings_are_not_claims() -> None:
    assert extract_claims("Running version 1.2.3 of the engine.") == []


def test_percent_and_proportion_are_the_same_claim() -> None:
    claim = extract_claims("a gap of 7.07%")[0]
    assert claim.normalised_value() == pytest.approx(0.0707)


def test_thousands_separators_parse() -> None:
    assert extract_claims("1,234,567 exposures")[0].value == 1234567.0


def test_binding_finds_values_nested_in_lists() -> None:
    """Regression: threshold values live under a list and were being missed."""
    values = flatten_evidence_values(EVIDENCE)
    assert 0.05 in values.values()
    result = bind_claims(extract_claims(DETERMINISTIC), values)
    assert result.unbound_count == 0
    assert result.grounding_rate == 1.0


# --------------------------------------------------------------------------- #
# Narrative invariance
# --------------------------------------------------------------------------- #
def _attest(model_narrative: str):
    return attest_narrative_invariance(
        section="performance",
        deterministic_narrative=DETERMINISTIC,
        model_narrative=model_narrative,
        evidence=EVIDENCE,
    )


def test_faithful_rewording_is_invariant() -> None:
    """The check must survive the model writing a better sentence."""
    result = _attest(
        "Discrimination falls from 0.8421 in training to 0.7714 on the holdout "
        "[EV-7f3a1c8b0d21]; that 0.0707 spread sits above the 0.05 tolerance."
    )
    assert result.invariant
    assert not result.blocking_divergences()


def test_percentage_rendering_is_invariant() -> None:
    """Rendering 0.0707 as 7.07% is a formatting choice, not a divergence."""
    result = _attest(
        "Training reached 0.8421 and holdout 0.7714 [EV-7f3a1c8b0d21]. The 7.07% gap "
        "exceeds the 0.05 threshold."
    )
    assert result.invariant, [d.detail for d in result.blocking_divergences()]


def test_invented_figure_is_caught() -> None:
    result = _attest(DETERMINISTIC + " Overall accuracy was 0.913.")
    assert not result.invariant
    assert result.unbound_claim_rate > 0
    assert any(d.kind in {"unbound", "contradiction"} for d in result.blocking_divergences())


def test_corrupted_transcription_is_reported_as_contradiction() -> None:
    """A near-miss is diagnosable; reporting it as a bare invention is not."""
    result = _attest(
        "Train AUC is 0.8421 against test AUC 0.7914 [EV-7f3a1c8b0d21]. The gap of 0.0707 "
        "exceeds the warn threshold of 0.05."
    )
    assert not result.invariant
    contradictions = [d for d in result.divergences if d.kind == "contradiction"]
    assert contradictions
    assert contradictions[0].deterministic_value == pytest.approx(0.7714)
    assert contradictions[0].model_value == pytest.approx(0.7914)


def test_omission_is_reported_but_does_not_block() -> None:
    """Dropping a figure is a finding; inventing one is a defect. Different severity."""
    result = _attest("Train AUC is 0.8421 [EV-7f3a1c8b0d21]. Performance is acceptable.")
    assert result.invariant, "omissions alone must not block sealing"
    assert any(d.kind == "omission" for d in result.divergences)


def test_attestation_hash_is_stable_and_discriminating() -> None:
    a = _attest(DETERMINISTIC)
    b = _attest(DETERMINISTIC)
    assert a.attestation_hash() == b.attestation_hash()
    assert a.attestation_hash() != _attest(DETERMINISTIC + " Also 0.913.").attestation_hash()


def test_empty_narratives_are_trivially_invariant() -> None:
    assert attest_narrative_invariance(
        section="s", deterministic_narrative="", model_narrative="", evidence=EVIDENCE
    ).invariant


# --------------------------------------------------------------------------- #
# Disclosure
# --------------------------------------------------------------------------- #
def test_policies_narrow_monotonically() -> None:
    sizes = [
        len(build_envelope(EVIDENCE, policy=POLICIES[p]).projected)
        for p in ("public_demo", "restricted", "minimal")
    ]
    assert sizes[0] >= sizes[1] >= sizes[2]
    assert sizes[2] == 0, "the minimal policy must project no numeric content at all"


def test_thresholds_survive_the_restricted_projection() -> None:
    """Regression: list indices broke the allow-list globs and silently withheld."""
    envelope = build_envelope(EVIDENCE, policy=POLICIES["restricted"])
    assert "0.05" in envelope.numeric_surface


def test_denied_paths_abort_rather_than_being_dropped() -> None:
    """A raw row reaching the disclosure layer is an upstream bug worth surfacing."""
    with pytest.raises(DisclosureViolation, match="denied by disclosure policy"):
        build_envelope(
            [{"metrics": {"a": 1.0}, "raw_rows": [[1, 2, 3]]}], policy=POLICIES["restricted"]
        )


def test_egress_check_refuses_uncovered_numbers() -> None:
    envelope = build_envelope(EVIDENCE, policy=POLICIES["restricted"])
    verify_prompt_covered("The gap of 0.0707 exceeds 0.05.", envelope)
    with pytest.raises(DisclosureViolation, match="absent from the disclosure envelope"):
        verify_prompt_covered("Account 4471982 holds 128455.30.", envelope)


def test_envelope_hash_changes_with_policy() -> None:
    a = build_envelope(EVIDENCE, policy=POLICIES["public_demo"])
    b = build_envelope(EVIDENCE, policy=POLICIES["restricted"])
    assert a.envelope_hash() != b.envelope_hash()


def test_envelope_evidence_record_omits_the_projected_content() -> None:
    """What was disclosed is auditable; the log is not a second copy of it."""
    record = build_envelope(EVIDENCE, policy=POLICIES["restricted"]).as_evidence()
    assert "projected" not in record
    assert record["envelope_hash"] and record["projected_field_count"] > 0


# --------------------------------------------------------------------------- #
# Seals
# --------------------------------------------------------------------------- #
def _seal():
    return build_seal(
        review_id="R-1",
        plan={"scope": "a"},
        policy={"v": 1},
        evidence_head="ab" * 32,
        attestations=[{"invariant": True}],
        profile={"profile": "public_demo"},
        environment={"python": "3.12"},
        controls={"ratio": 0.5},
        created_utc="2026-08-18T00:00:00+00:00",
    )


def test_seal_verifies_and_is_reproducible() -> None:
    seal = _seal()
    assert verify_seal(seal.manifest(), seal.seal_string())["verified"]
    assert _seal().seal_string() == seal.seal_string()


def test_tampering_is_localised_to_the_leaf() -> None:
    """'Something changed' is not actionable; naming the leaf is."""
    seal = _seal()
    manifest = seal.manifest()
    manifest["payloads"]["plan"] = {"scope": "tampered"}
    result = verify_seal(manifest, seal.seal_string())
    assert not result["verified"]
    assert result["mismatched_leaves"] == ["plan"]
    assert "Other leaves are intact" in result["reason"]


def test_absent_components_still_commit() -> None:
    """An incomplete review must not collide with a complete one."""
    partial = build_seal(review_id="R-1", plan={"scope": "a"}, created_utc="2026-08-18T00:00:00+00:00")
    assert partial.seal_string() != _seal().seal_string()


def test_manifest_without_payloads_cannot_be_verified() -> None:
    seal = _seal()
    manifest = seal.as_dict()  # hashes only
    result = verify_seal(manifest, seal.seal_string())
    assert not result["verified"]
    assert "no payloads" in result["reason"]


def test_inclusion_proofs_verify_for_every_leaf() -> None:
    seal = _seal()
    for name in [leaf.name for leaf in seal.leaves]:
        proof = seal.proof_for(name)
        assert verify_inclusion(proof["leaf_hash"], proof["proof"], proof["root"])


def test_odd_leaf_counts_do_not_collide() -> None:
    """Promoting an odd node beats duplicating it; duplication creates collisions."""
    assert merkle_root(["aa" * 32, "bb" * 32, "cc" * 32]) != merkle_root(
        ["aa" * 32, "bb" * 32, "cc" * 32, "cc" * 32]
    )


def test_v1_seal_manifest_backward_compatibility() -> None:
    """Archived v1 seals (7 leaves) must verify identically under the new codebase."""
    from start.attestation.seal import LEAF_ORDER_V1, ReviewSeal, SealLeaf

    payloads = {
        "plan": {"scope": "v1_plan"},
        "policy": {"policy": "v1_policy"},
        "evidence_head": "11" * 32,
        "attestations": [{"section": "v1"}],
        "profile": {"profile": "public_demo"},
        "environment": {"python": "3.12"},
        "controls": {"threshold": 0.05},
    }
    leaves = tuple(SealLeaf(name=name, payload=payloads[name]) for name in LEAF_ORDER_V1)
    v1_seal = ReviewSeal(
        version="start-seal/1",
        review_id="R-V1-ARCHIVE",
        created_utc="2026-08-01T12:00:00+00:00",
        leaves=leaves,
    )
    v1_manifest = v1_seal.manifest()
    assert len(v1_manifest["leaves"]) == 7

    res = verify_seal(v1_manifest, v1_seal.seal_string())
    assert res["verified"] is True
    assert res["recomputed_seal"] == v1_seal.seal_string()


def test_v2_seal_manifest_backward_compatibility() -> None:
    """Archived v2 seals (8 leaves) must verify identically under start-seal/3."""
    from start.attestation.seal import LEAF_ORDER_V2, ReviewSeal, SealLeaf

    payloads = {
        "plan": {"scope": "v2_plan"},
        "policy": {"policy": "v2_policy"},
        "evidence_head": "22" * 32,
        "attestations": [{"section": "v2"}],
        "profile": {"profile": "public_demo"},
        "environment": {"python": "3.12"},
        "controls": {"threshold": 0.05},
        "adjudications": {"decisions": []},
    }
    leaves = tuple(SealLeaf(name=name, payload=payloads[name]) for name in LEAF_ORDER_V2)
    v2_seal = ReviewSeal(
        version="start-seal/2",
        review_id="R-V2-ARCHIVE",
        created_utc="2026-08-10T12:00:00+00:00",
        leaves=leaves,
    )
    v2_manifest = v2_seal.manifest()
    assert len(v2_manifest["leaves"]) == 8

    res = verify_seal(v2_manifest, v2_seal.seal_string())
    assert res["verified"] is True
    assert res["recomputed_seal"] == v2_seal.seal_string()


def test_v3_seal_manifest_includes_execution_path() -> None:
    """v3 seals (9 leaves) commit to the execution path."""
    from start.attestation.seal import build_seal

    seal = build_seal(
        review_id="R-V3-TEST",
        plan={"scope": "v3_plan"},
        execution_path={"path": [{"node": "start"}]},
        created_utc="2026-08-20T12:00:00+00:00",
    )
    manifest = seal.manifest()
    assert len(manifest["leaves"]) == 9
    assert manifest["leaves"][-1]["name"] == "execution_path"
    res = verify_seal(manifest, seal.seal_string())
    assert res["verified"] is True
    assert res["recomputed_seal"] == seal.seal_string()


# --------------------------------------------------------------------------- #
# Replay
# --------------------------------------------------------------------------- #
def _write_ledger(path: Path, records: list[dict]) -> None:
    from hashlib import sha256

    def canonical(obj):
        return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)

    prev = "0" * 64
    with path.open("w", encoding="utf-8") as handle:
        for index, record in enumerate(records):
            record_hash = sha256(canonical(record).encode()).hexdigest()
            entry_hash = sha256((prev + record_hash).encode()).hexdigest()
            handle.write(
                json.dumps(
                    {
                        "index": index,
                        "prev_hash": prev,
                        "record_hash": record_hash,
                        "entry_hash": entry_hash,
                        "record": record,
                    }
                )
                + "\n"
            )
            prev = entry_hash


def test_intact_ledger_replays(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    _write_ledger(path, [{"test_id": "a", "metrics": {"x": 1.0}}, {"test_id": "b", "metrics": {"x": 2.0}}])
    verdict = replay_ledger(path)
    assert verdict.intact and verdict.entries == 2


def test_edited_record_is_detected_as_content_tampering(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    _write_ledger(path, [{"test_id": "a", "metrics": {"x": 1.0}}])
    lines = path.read_text(encoding="utf-8").splitlines()
    entry = json.loads(lines[0])
    entry["record"]["metrics"]["x"] = 9.0
    path.write_text(json.dumps(entry) + "\n", encoding="utf-8")

    verdict = replay_ledger(path)
    assert not verdict.intact
    assert verdict.divergence_kind == "content"
    assert verdict.first_divergence_index == 0


def test_missing_ledger_is_trivially_intact(tmp_path: Path) -> None:
    assert replay_ledger(tmp_path / "absent.jsonl").intact


def test_reruns_agree_ignoring_volatile_fields(tmp_path: Path) -> None:
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    _write_ledger(a, [{"test_id": "t", "run_id": "run-1", "metrics": {"auc": 0.8421}}])
    _write_ledger(b, [{"test_id": "t", "run_id": "run-2", "metrics": {"auc": 0.8421}}])
    assert compare_ledgers(a, b).reproducible


def test_drifted_metric_names_the_field(tmp_path: Path) -> None:
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    _write_ledger(a, [{"test_id": "t", "metrics": {"auc": 0.8421}}])
    _write_ledger(b, [{"test_id": "t", "metrics": {"auc": 0.8100}}])
    comparison = compare_ledgers(a, b)
    assert not comparison.reproducible
    assert comparison.drifted[0]["field"].endswith("metrics.auc")
