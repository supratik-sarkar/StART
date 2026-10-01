"""Private deterministic reviewer twin for zero-cost terminal rehearsals.

This provider is deliberately local and fixture-like: it performs no network,
credential, or hosted-model operations.  Structured replies reference only
EvidenceRecord IDs and metric paths present in the current prompt.
"""

from __future__ import annotations

import hashlib
import json
import re

from start.providers.base import LLMProvider, ProviderResult, ProviderUsage

_CHECKPOINT_RE = re.compile(r"^Checkpoint:\s*(.+)$", re.MULTILINE)
_SEMANTIC_CHECKPOINT_RE = re.compile(r"^Semantic Checkpoint ID:\s*([^\s]+)\s*$", re.MULTILINE)
_INTERACTION_ACTION_RE = re.compile(r"^Interaction Action:\s*([A-Za-z_]+)\s*$", re.MULTILINE)
_QUESTION_RE = re.compile(r'^Question:\s*"(.+)"\s*$', re.MULTILINE)
_EVIDENCE_BLOCK_RE = re.compile(
    r"^- \[(EV-[^\]]+)\] Test:.*?(?=^- \[EV-|\Z)",
    re.MULTILINE | re.DOTALL,
)
_PATH_RE = re.compile(r"^\s*\*\s+((?:metrics|params)\.[A-Za-z0-9_.-]+)\s+\(value:", re.MULTILINE)


class OfflineDemoTwinProvider(LLMProvider):
    """Semantic, evidence-aware, zero-egress rehearsal provider."""

    name = "offline_demo_twin"
    model = "deterministic-semantic-fixture-v1"
    network_enabled = False
    hosted_calls = 0

    @property
    def available(self) -> bool:
        return True

    def complete(self, system: str, user: str, *, output_token_budget: int = 1024) -> str:
        return self._response(system, user)

    def complete_result(
        self, system: str, user: str, *, output_token_budget: int = 1024
    ) -> ProviderResult:
        text = self._response(system, user)
        response_id = "offline-" + hashlib.sha256(user.encode()).hexdigest()[:12]
        self.last_response_id = response_id
        self.last_input_tokens = 0
        self.last_output_tokens = 0
        self.last_reasoning_tokens = 0
        return ProviderResult(
            text=text,
            provider=self.name,
            model=self.model,
            response_id=response_id,
            status="completed",
            usage=ProviderUsage(),
            latency_seconds=0.0,
            api_surface="offline_fixture",
            max_output_tokens=output_token_budget,
            output_item_types=["deterministic_fixture"],
            content_part_types=["text"],
        )

    def _response(self, system: str, user: str) -> str:
        checkpoint_match = _CHECKPOINT_RE.search(user)
        checkpoint = checkpoint_match.group(1).strip() if checkpoint_match else "review"
        semantic_match = _SEMANTIC_CHECKPOINT_RE.search(user)
        semantic_checkpoint_id = semantic_match.group(1).strip() if semantic_match else ""
        action_match = _INTERACTION_ACTION_RE.search(user)
        interaction_action = action_match.group(1).strip().upper() if action_match else ""
        question_match = _QUESTION_RE.search(user)
        question = question_match.group(1).strip() if question_match else ""
        structured = "StructuredReviewerResponse" in system
        if not structured:
            return self._plain_response(checkpoint, user)

        # The one intentional rejection exercise is confined to the registered
        # degradable synthesis checkpoint and its intended informational query.
        # A title, invocation ordinal, domain, or Q action alone cannot trigger it.
        intentional_synthesis_query = (
            semantic_checkpoint_id == "market.cross_analytical_synthesis"
            and interaction_action == "QUESTION"
            and "final evidence" in question.lower()
        )
        if intentional_synthesis_query:
            return json.dumps(
                {
                    "findings": [
                        {
                            "finding_id": "F-DEMO-REJECTION",
                            "finding_type": "OBSERVED_EVIDENCE",
                            "conclusion": "Intentional invalid reference for the evidence-gate rehearsal.",
                            "evidence_refs": [
                                {
                                    "evidence_id": "EV-NOT-IN-CHECKPOINT",
                                    "metric_path": "metrics.not_admissible",
                                }
                            ],
                            "criterion_status": "EVIDENCE_ONLY",
                            "unresolved_reason": None,
                        }
                    ],
                    "overall_assessment": "This fixture is expected to be rejected by grounding.",
                }
            )

        # Select an Evidence ID and path from the same formatted record block.
        # This preserves the exact per-record admissibility relation.
        grounded_ref: tuple[str, str] | None = None
        for block_match in _EVIDENCE_BLOCK_RE.finditer(user):
            paths = _PATH_RE.findall(block_match.group(0))
            if paths:
                grounded_ref = (block_match.group(1), paths[0])
                break
        conclusion = self._checkpoint_conclusion(semantic_checkpoint_id, checkpoint)
        if grounded_ref is not None:
            evidence_id, metric_path = grounded_ref
            finding = {
                "finding_id": "F-01",
                "finding_type": "OBSERVED_EVIDENCE",
                "conclusion": conclusion,
                "evidence_refs": [
                    {"evidence_id": evidence_id, "metric_path": metric_path}
                ],
                "criterion_status": "APPLICABLE",
                "unresolved_reason": None,
            }
        else:
            finding = {
                "finding_id": "F-01",
                "finding_type": "EVIDENCE_GAP",
                "conclusion": f"No citable metric is supplied for {checkpoint}; no quantitative claim is made.",
                "evidence_refs": [],
                "criterion_status": "ABSENT",
                "unresolved_reason": "No admissible metric path is available at this checkpoint.",
            }
        return json.dumps(
            {
                "findings": [finding],
                "overall_assessment": (
                    f"Offline rehearsal assessment for {checkpoint}; deterministic evidence remains authoritative."
                ),
            }
        )

    @staticmethod
    def _plain_response(checkpoint: str, user: str) -> str:
        quoted = re.search(r'A reviewer asked:\s*"([\s\S]*?)"\.', user)
        question = quoted.group(1).lower() if quoted else user.lower()
        text = f"{checkpoint} {question}".lower()
        if any(term in question for term in ("leakage", "train/test", "oos", "preprocess")):
            temporal = "predictive input modality: temporal_sequence" in user.lower()
            cohort_wording = (
                "disjoint train, test, and OOS sequence cohorts. Any learned preprocessing must be fit "
                "on the training cohort only; sequence identifiers and timestep metadata remain provenance "
                "and cannot enter the predictive tensor."
                if temporal
                else "the resolved train, test, and OOS split must remain disjoint. Any learned preprocessing "
                "must be fit on the training partition only; input fields must match the declared tabular contract."
            )
            return (
                f"Leakage validation requires {cohort_wording}"
            )
        if any(term in question for term in ("fake sequence", "reshape", "rank-3", "modality", "within-sequence")):
            return (
                "The modality is valid only when each sample contains genuine within-sequence temporal "
                "structure in a rank-3 (N,T,F) tensor. Reshaping independent tabular columns into timesteps "
                "would be a fake sequence and would invalidate recurrent-model use."
            )
        if any(term in text for term in ("lstm", "architecture", "recurrent")):
            return (
                "An LSTM is justified when ordered timesteps within each sample carry dependence that a "
                "recurrent state can use. It is invalidated by absent temporal dependence, one-step sequences, "
                "or evidence that the data are merely tabular fields relabeled as time."
            )
        return (
            f"Offline rehearsal response for {checkpoint}. Only evidence supplied to this checkpoint is "
            "admissible; deterministic engines retain numeric authority."
        )

    @staticmethod
    def _checkpoint_conclusion(semantic_checkpoint_id: str, checkpoint: str) -> str:
        phrases = {
            "market.portfolio_allocation": (
                "The registered portfolio evidence supports review of allocation and risk assumptions."
            ),
            "market.factor_attribution": (
                "The registered attribution evidence supports reconciliation-focused review."
            ),
            "market.var_backtesting": (
                "The registered tail-risk evidence supports exception and coverage review."
            ),
            "market.var_tail_risk": (
                "The registered tail-risk evidence supports exception and coverage review."
            ),
            "market.covariance_structure": (
                "The registered covariance evidence supports conditioning and missing-data review."
            ),
            "market.scenario_stress": (
                "The registered scenario evidence supports stress integrity review."
            ),
            "market.cross_analytical_synthesis": (
                "The registered cross-analytical evidence supports a bounded synthesis."
            ),
            "market.governance_signoff": (
                "The registered evidence supports a governance assessment without replacing deterministic policy."
            ),
        }
        return phrases.get(
            semantic_checkpoint_id,
            "The registered checkpoint evidence supports a bounded qualitative assessment.",
        )
