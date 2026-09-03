"""Review seals.

A validation report is a PDF. A PDF asserts whatever it was told to assert, and
by the time anyone questions it — a supervisory exam, an incident post-mortem,
an internal audit two years later — the environment that produced it is gone.

A seal is a single short string that commits to everything the review rested
on: the scope that was planned, the policy in force, the head of the evidence
chain, every attestation that was run, and the containment regime the review
ran under. Paste it into the memo. Anyone holding the artefacts can recompute
it; if a single leaf changed, the root changes, and the seal no longer verifies.

The structure is an ordered Merkle tree rather than a flat hash for a specific
reason: when verification fails, a flat hash says only "something changed",
while an inclusion proof identifies *which* leaf. In a dispute about a review,
the difference between "the evidence was altered" and "the plan hash does not
match, the evidence is intact" is the whole argument.

Leaves are ordered and named, so the tree is reproducible and a proof is
interpretable:

    0  plan            scope that was agreed
    1  policy          thresholds in force
    2  evidence_head   head of the tamper-evident ledger chain
    3  attestations    invariance / disclosure / replay results
    4  profile         egress regime the review ran under
    5  environment     runtime, versions, seeds
    6  controls        framework coverage claimed

Standard library only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC
from hashlib import sha256
from pathlib import Path
from typing import Any

__all__ = [
    "SEAL_VERSION",
    "LEAF_ORDER",
    "LEAF_ORDER_V1",
    "LEAF_ORDER_V2",
    "LEAF_ORDER_V3",
    "SealLeaf",
    "ReviewSeal",
    "merkle_root",
    "inclusion_proof",
    "verify_inclusion",
    "build_seal",
    "verify_seal",
]

SEAL_VERSION = "start-seal/3"

LEAF_ORDER_V1: tuple[str, ...] = (
    "plan",
    "policy",
    "evidence_head",
    "attestations",
    "profile",
    "environment",
    "controls",
)

LEAF_ORDER_V2: tuple[str, ...] = (
    "plan",
    "policy",
    "evidence_head",
    "attestations",
    "profile",
    "environment",
    "controls",
    "adjudications",
)

LEAF_ORDER_V3: tuple[str, ...] = (
    "plan",
    "policy",
    "evidence_head",
    "attestations",
    "profile",
    "environment",
    "controls",
    "adjudications",
    "execution_path",
)

LEAF_ORDER: tuple[str, ...] = LEAF_ORDER_V3


def _h(data: bytes) -> str:
    return sha256(data).hexdigest()


def _leaf_hash(name: str, payload: Any) -> str:
    """Domain-separated leaf hash.

    The ``0x00`` prefix distinguishes leaves from internal nodes, which is what
    stops a second-preimage attack that re-presents an internal node as a leaf.
    """
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return _h(b"\x00" + name.encode("utf-8") + b"\x1f" + body.encode("utf-8"))


def _node_hash(left: str, right: str) -> str:
    return _h(b"\x01" + bytes.fromhex(left) + bytes.fromhex(right))


@dataclass(frozen=True)
class SealLeaf:
    name: str
    payload: Any

    def leaf_hash(self) -> str:
        return _leaf_hash(self.name, self.payload)


def merkle_root(leaf_hashes: list[str]) -> str:
    """Root of an ordered Merkle tree.

    An odd node at any level is promoted rather than duplicated. Duplicating is
    the more common implementation and is the source of the classic
    duplicate-leaf ambiguity, where two different trees produce the same root.
    """
    if not leaf_hashes:
        return "0" * 64
    level = list(leaf_hashes)
    while len(level) > 1:
        nxt: list[str] = []
        for i in range(0, len(level) - 1, 2):
            nxt.append(_node_hash(level[i], level[i + 1]))
        if len(level) % 2:
            nxt.append(level[-1])
        level = nxt
    return level[0]


def inclusion_proof(leaf_hashes: list[str], index: int) -> list[dict[str, str]]:
    """Sibling path proving one leaf's membership in the root.

    The odd-node promotion in :func:`merkle_root` means a promoted node has no
    sibling at that level and therefore contributes no proof step. Getting this
    wrong produces proofs that verify for even-sized trees and silently fail for
    odd ones — which is why every leaf is exercised in the tests, not just the
    first.
    """
    if not 0 <= index < len(leaf_hashes):
        raise IndexError(f"leaf index {index} out of range for {len(leaf_hashes)} leaves")

    proof: list[dict[str, str]] = []
    level = list(leaf_hashes)
    idx = index

    while len(level) > 1:
        pair_count = len(level) // 2
        nxt = [_node_hash(level[2 * i], level[2 * i + 1]) for i in range(pair_count)]
        has_odd_tail = len(level) % 2 == 1
        if has_odd_tail:
            nxt.append(level[-1])

        if has_odd_tail and idx == len(level) - 1:
            idx = len(nxt) - 1  # promoted unchanged; no sibling to record
        else:
            sibling = idx ^ 1
            proof.append({"side": "right" if idx % 2 == 0 else "left", "hash": level[sibling]})
            idx = idx // 2

        level = nxt

    return proof


def verify_inclusion(leaf_hash: str, proof: list[dict[str, str]], root: str) -> bool:
    current = leaf_hash
    for step in proof:
        if step["side"] == "right":
            current = _node_hash(current, step["hash"])
        else:
            current = _node_hash(step["hash"], current)
    return current == root


@dataclass(frozen=True)
class ReviewSeal:
    """A verifiable commitment to everything a review rested on."""

    version: str
    review_id: str
    created_utc: str
    leaves: tuple[SealLeaf, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def leaf_hashes(self) -> list[str]:
        return [leaf.leaf_hash() for leaf in self.leaves]

    def root(self) -> str:
        return merkle_root(self.leaf_hashes())

    def seal_string(self) -> str:
        """The short string a human pastes into a memo.

        Truncated to 32 hex characters (128 bits) — long enough that collision
        is not a practical concern, short enough to survive being retyped from a
        printed page.
        """
        return f"{self.version}:{self.review_id}:{self.root()[:32]}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "review_id": self.review_id,
            "enterprise_run_id": self.metadata.get("enterprise_run_id", self.review_id),
            "inner_run_id": self.metadata.get("inner_run_id", ""),
            "created_utc": self.created_utc,
            "root": self.root(),
            "seal": self.seal_string(),
            "leaves": [
                {"index": i, "name": leaf.name, "leaf_hash": leaf.leaf_hash()}
                for i, leaf in enumerate(self.leaves)
            ],
            "metadata": dict(sorted(self.metadata.items())),
        }

    def manifest(self) -> dict[str, Any]:
        """Full manifest including payloads — this is what gets archived."""
        body = self.as_dict()
        body["payloads"] = {leaf.name: leaf.payload for leaf in self.leaves}
        return body

    def proof_for(self, leaf_name: str) -> dict[str, Any]:
        names = [leaf.name for leaf in self.leaves]
        if leaf_name not in names:
            raise KeyError(f"No leaf named {leaf_name!r}. Leaves: {', '.join(names)}")
        index = names.index(leaf_name)
        return {
            "leaf_name": leaf_name,
            "leaf_index": index,
            "leaf_hash": self.leaf_hashes()[index],
            "proof": inclusion_proof(self.leaf_hashes(), index),
            "root": self.root(),
        }


def build_seal(
    *,
    review_id: str,
    plan: Any = None,
    policy: Any = None,
    evidence_head: Any = None,
    attestations: Any = None,
    profile: Any = None,
    environment: Any = None,
    controls: Any = None,
    adjudications: Any = None,
    execution_path: Any = None,
    created_utc: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> ReviewSeal:
    """Assemble a seal.

    Absent components are committed as ``None`` rather than omitted, so a review
    that skipped a component produces a *different* root than one that included
    it. Omitting the leaf entirely would let an incomplete review and a complete
    one collide.
    """
    from datetime import datetime

    payloads = {
        "plan": plan,
        "policy": policy,
        "evidence_head": evidence_head,
        "attestations": attestations,
        "profile": profile,
        "environment": environment,
        "controls": controls,
        "adjudications": adjudications,
        "execution_path": execution_path,
    }
    leaves = tuple(SealLeaf(name=name, payload=payloads[name]) for name in LEAF_ORDER)

    return ReviewSeal(
        version=SEAL_VERSION,
        review_id=review_id,
        created_utc=created_utc or datetime.now(UTC).isoformat(timespec="seconds"),
        leaves=leaves,
        metadata=dict(metadata or {}),
    )


@dataclass
class SealCheckResult:
    passed: bool
    label: str
    detail: str = ""
    blocking: bool = True


def validate_seal_preconditions(
    *,
    review_id: str,
    ledger_records: list[Any],
    evidence_head: Any,
    adjudications: Any,
    attestations: Any,
    agent_mode: str = "deterministic",
    critic_verdict: str = "PASSED",
) -> tuple[bool, list[SealCheckResult]]:
    """Verify cryptographic preconditions before emitting a seal (A6).

    Checks 1-4 are blocking (SEAL WITHHELD):
    1. At least one evidence record carries this review's enterprise_run_id (or review_id).
    2. The evidence_head resolves to a record from this run.
    3. The adjudications leaf is populated and not an empty-payload constant.
    4. If agent_mode == 'llm', at least one narrative attestation exists.

    Check 5 (critic verdict) is recorded but non-blocking in v4.0.2 (Amendment 1).
    TODO(v4.1.0): Make critic verdict blocking once agent narratives reliably cite evidence.
    """
    checks: list[SealCheckResult] = []

    # 1. Evidence records for this run
    matching_records = [
        r
        for r in ledger_records
        if getattr(r, "enterprise_run_id", None) == review_id
        or getattr(r, "run_id", None) == review_id
        or (isinstance(r, dict) and (r.get("enterprise_run_id") == review_id or r.get("run_id") == review_id))
    ]
    if matching_records:
        checks.append(
            SealCheckResult(
                True, f"evidence chain contains {len(matching_records)} record(s) for {review_id}"
            )
        )
    else:
        checks.append(SealCheckResult(False, f"evidence chain contains 0 records for {review_id}"))

    # 2. evidence_head resolves to this run
    head_ok = False
    if evidence_head is not None:
        head_run = getattr(evidence_head, "enterprise_run_id", None) or getattr(evidence_head, "run_id", None)
        if not head_run and isinstance(evidence_head, dict):
            head_run = evidence_head.get("enterprise_run_id") or evidence_head.get("run_id")
        if (head_run == review_id or head_run is None) and matching_records:
            head_ok = True
    if head_ok:
        checks.append(SealCheckResult(True, f"evidence_head resolves to record from {review_id}"))
    else:
        if evidence_head:
            head_id = getattr(evidence_head, "evidence_id", "") or (
                evidence_head.get("evidence_id", "") if isinstance(evidence_head, dict) else "none"
            )
        else:
            head_id = "none"
        checks.append(SealCheckResult(False, f"evidence_head {head_id or 'none'} belongs to a different run"))

    # 3. adjudications leaf populated
    adj_populated = False
    if adjudications:
        if isinstance(adjudications, dict):
            n_dec = len(adjudications.get("decisions", []))
            n_chal = len(adjudications.get("challenges", []))
            if n_dec > 0 or n_chal > 0:
                adj_populated = True
                checks.append(
                    SealCheckResult(
                        True, f"adjudications leaf populated ({n_dec} decisions, {n_chal} challenges)"
                    )
                )
        elif isinstance(adjudications, (list, tuple)) and len(adjudications) > 0:
            adj_populated = True
            checks.append(SealCheckResult(True, f"adjudications leaf populated ({len(adjudications)} items)"))
    if not adj_populated:
        checks.append(SealCheckResult(False, "adjudications leaf is empty or static constant"))

    # 4. LLM narrative attestations
    if agent_mode == "llm":
        n_att = len(attestations) if isinstance(attestations, (list, tuple, dict)) and attestations else 0
        if n_att > 0:
            checks.append(SealCheckResult(True, f"agent_mode=llm with {n_att} narrative attestation(s)"))
        else:
            checks.append(SealCheckResult(False, "agent_mode=llm with 0 narrative attestations"))
    else:
        checks.append(SealCheckResult(True, "agent_mode=deterministic (narrative attestation optional)"))

    # 5. Critic verdict (non-blocking in v4.0.2 per Amendment 1)
    # TODO(v4.1.0): Make critic verdict blocking once agent narratives cite evidence reliably.
    if critic_verdict in ("PASSED", "passed", "ok"):
        checks.append(SealCheckResult(True, "evidence critique: PASSED", blocking=False))
    else:
        checks.append(
            SealCheckResult(
                True,
                f"evidence critique: {critic_verdict} (advisory in v4.0.2)",
                detail="non-blocking",
                blocking=False,
            )
        )

    all_blocking_passed = all(c.passed for c in checks if c.blocking)
    return all_blocking_passed, checks


def persist_seal_manifest(seal: ReviewSeal, output_root: str | Path) -> Path:
    """Write full seal manifest and update the global seal index (Amendment 3)."""
    from pathlib import Path

    root_path = Path(output_root)
    seal_dir = root_path / "seals" / seal.review_id
    seal_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = seal_dir / "seal_manifest.json"
    manifest_data = seal.manifest()
    manifest_path.write_text(json.dumps(manifest_data, indent=2, default=str))

    # Maintain index: start_output/seals/index.json
    index_path = root_path / "seals" / "index.json"
    index_data: dict[str, Any] = {}
    if index_path.exists():
        try:
            index_data = json.loads(index_path.read_text())
        except Exception:
            index_data = {}
    index_data[seal.seal_string()] = {
        "manifest_path": str(manifest_path),
        "enterprise_run_id": seal.review_id,
        "inner_run_id": seal.metadata.get("inner_run_id", ""),
        "created_utc": seal.created_utc,
        "root": seal.root(),
    }
    index_path.write_text(json.dumps(index_data, indent=2))
    return manifest_path


def verify_seal(manifest: dict[str, Any], expected_seal: str | None = None) -> dict[str, Any]:
    """Recompute a seal from an archived manifest.

    Returns a structured verdict rather than a bare boolean, because "it does
    not verify" is not actionable and "the ``plan`` leaf does not match, all
    other leaves are intact" is.
    """
    payloads = manifest.get("payloads")
    if payloads is None:
        return {
            "verified": False,
            "reason": "manifest contains no payloads; only the recorded hashes are present, "
            "so the seal cannot be recomputed independently",
            "mismatched_leaves": [],
        }

    version = manifest.get("version", SEAL_VERSION)
    if version == "start-seal/1":
        leaf_order = LEAF_ORDER_V1
    elif version == "start-seal/2":
        leaf_order = LEAF_ORDER_V2
    else:
        leaf_order = LEAF_ORDER_V3

    recomputed = ReviewSeal(
        version=version,
        review_id=manifest.get("review_id", ""),
        created_utc=manifest.get("created_utc", ""),
        leaves=tuple(SealLeaf(name=name, payload=payloads.get(name)) for name in leaf_order),
        metadata=manifest.get("metadata", {}) or {},
    )

    recorded = {row["name"]: row["leaf_hash"] for row in manifest.get("leaves", [])}
    computed = {leaf.name: leaf.leaf_hash() for leaf in recomputed.leaves}
    mismatched = sorted(name for name, digest in computed.items() if recorded.get(name) not in (None, digest))

    root_ok = recomputed.root() == manifest.get("root")
    seal_ok = True
    if expected_seal:
        seal_ok = recomputed.seal_string() == expected_seal.strip()

    verified = root_ok and seal_ok and not mismatched
    if verified:
        reason = "all leaves, Merkle root and seal string recomputed identically"
    elif mismatched:
        reason = f"leaf content changed after sealing: {', '.join(mismatched)}. Other leaves are intact."
    elif not root_ok:
        reason = "leaf hashes match individually but the recorded root does not; the leaf "
        "ordering or the recorded root was altered"
    else:
        reason = "recomputed seal string does not match the seal presented"

    return {
        "verified": verified,
        "reason": reason,
        "recomputed_seal": recomputed.seal_string(),
        "recomputed_root": recomputed.root(),
        "recorded_root": manifest.get("root"),
        "mismatched_leaves": mismatched,
    }
