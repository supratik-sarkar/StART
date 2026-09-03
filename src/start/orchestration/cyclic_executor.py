"""Bounded-cycle graph executor.

What the executor guarantees
----------------------------

**Every cycle terminates.** Budgets are checked before a back-edge is taken, not after
it is regretted. A review cannot hang.

**A remediation that changes nothing is caught.** Routing back to Tuning and producing
identical parameters is an infinite loop wearing a disguise — it would burn the budget
and look like three honest attempts. The executor compares the remediation fingerprint
against the previous attempt and stops with a distinct outcome
(``no_change``) if the retry would repeat itself.

**Exhausting a budget is a named outcome**, not a generic failure. ``budget_exhausted``
is a citable governance finding: *the generalisation gap was addressed three times by
retuning and did not resolve*. That is a stronger statement than the metric alone, and
it is the statement a validator would write by hand.

**The path is evidence.** ``ExecutionPath`` records every node visit, the edge kind that
led there, and every remediation attempt with what changed and whether it helped. It
hashes canonically, so two runs that took different routes through the same graph are
visibly different reviews even when their conclusions agree.

What it deliberately does not do
--------------------------------

It does not choose the remediation. When Overfitting routes back to Tuning, *Tuning*
decides what to change — the executor only records what came back and whether it
differed. Putting the remedy in the executor would make the graph a search algorithm,
and a search algorithm is not a review.

It does not require LangGraph. Handlers are plain callables; a LangGraph backend can be
layered on by supplying different handlers, and nothing here imports it.

Standard library only.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any

from start.orchestration.review_graph_spec import (
    Edge,
    EdgeKind,
    NodeKind,
    ReviewGraph,
)

__all__ = [
    "NodeOutcome",
    "NodeResult",
    "RemediationAttempt",
    "ExecutionPath",
    "GraphExecutor",
    "CycleBudgetExceeded",
]


class CycleBudgetExceeded(RuntimeError):
    """Raised only when a caller asks for strict mode; normally recorded, not raised."""


#: What a node handler reports back. Anything else is treated as ``ok``.
class NodeOutcome:
    OK = "ok"
    FAIL = "fail"
    BLOCK = "block"
    SKIP = "skip"
    #: Checkpoint-specific: the reviewer challenged or asked, so re-enter this node.
    REENTER = "reenter"


@dataclass
class NodeResult:
    """What a handler returns."""

    outcome: str = NodeOutcome.OK
    detail: str = ""
    #: Free-form state merged into the run context and visible to later nodes.
    state: dict[str, Any] = field(default_factory=dict)
    #: Identifies *what this node did*, so a repeated remediation that changed
    #: nothing can be detected. For Tuning this would be the chosen hyperparameters.
    fingerprint: str = ""
    evidence_ids: tuple[str, ...] = field(default=())


@dataclass
class RemediationAttempt:
    """One traversal of a back-edge, and what came of it."""

    edge_id: str
    source: str
    target: str
    reason: str
    attempt: int
    budget: int
    #: `resolved` | `still_failing` | `budget_exhausted` | `no_change`
    outcome: str = "pending"
    changed: str = ""
    detail: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "edge_id": self.edge_id,
            "source": self.source,
            "target": self.target,
            "reason": self.reason,
            "attempt": self.attempt,
            "budget": self.budget,
            "outcome": self.outcome,
            "changed": self.changed,
            "detail": self.detail,
        }


@dataclass
class Visit:
    node_id: str
    via: str  # EdgeKind value, or "entry"
    outcome: str
    detail: str = ""
    seconds: float = 0.0


@dataclass
class ExecutionPath:
    """The route a review actually took. This is a sealed artefact."""

    graph_name: str
    graph_hash: str
    visits: list[Visit] = field(default_factory=list)
    remediations: list[RemediationAttempt] = field(default_factory=list)
    budgets: dict[str, int] = field(default_factory=dict)
    consumed: dict[str, int] = field(default_factory=dict)
    terminated_at: str = ""
    termination_reason: str = ""

    # -- summary ------------------------------------------------------------
    @property
    def node_sequence(self) -> list[str]:
        return [v.node_id for v in self.visits]

    def remediation_summary(self) -> dict[str, Any]:
        resolved = [r for r in self.remediations if r.outcome == "resolved"]
        exhausted = [r for r in self.remediations if r.outcome == "budget_exhausted"]
        no_change = [r for r in self.remediations if r.outcome == "no_change"]
        return {
            "attempts": len(self.remediations),
            "resolved": len(resolved),
            "budget_exhausted": len(exhausted),
            "no_change": len(no_change),
            "exhausted_edges": sorted({r.edge_id for r in exhausted}),
        }

    def governance_findings(self) -> list[dict[str, str]]:
        """Findings that only a cyclic execution can produce.

        A linear review can say the gap is 0.28. Only a review that tried to fix it
        can say that three attempts failed — which is the more useful sentence.
        """
        findings: list[dict[str, str]] = []
        for attempt in self.remediations:
            if attempt.outcome == "budget_exhausted":
                findings.append(
                    {
                        "kind": "remediation_exhausted",
                        "severity": "blocker",
                        "detail": (
                            f"{attempt.reason}: remediation via {attempt.target} was "
                            f"attempted {attempt.budget} time(s) and did not resolve. "
                            "The condition is not addressable by the routed remedy."
                        ),
                        "edge_id": attempt.edge_id,
                    }
                )
            elif attempt.outcome == "no_change":
                findings.append(
                    {
                        "kind": "remediation_ineffective",
                        "severity": "concern",
                        "detail": (
                            f"{attempt.reason}: remediation via {attempt.target} produced "
                            "an identical configuration, so the retry could not have "
                            "changed the outcome. Remediation stopped early rather than "
                            "consuming the remaining budget."
                        ),
                        "edge_id": attempt.edge_id,
                    }
                )
            elif attempt.outcome == "resolved":
                findings.append(
                    {
                        "kind": "remediation_succeeded",
                        "severity": "informational",
                        "detail": (
                            f"{attempt.reason}: resolved on attempt {attempt.attempt} "
                            f"of {attempt.budget}. Changed: {attempt.changed or 'unrecorded'}."
                        ),
                        "edge_id": attempt.edge_id,
                    }
                )
        return findings

    # -- identity -----------------------------------------------------------
    def as_dict(self) -> dict[str, Any]:
        return {
            "graph_name": self.graph_name,
            "graph_hash": self.graph_hash,
            "path": [
                {"node": v.node_id, "via": v.via, "outcome": v.outcome, "detail": v.detail}
                for v in self.visits
            ],
            "remediations": [r.as_dict() for r in self.remediations],
            "budgets": dict(sorted(self.budgets.items())),
            "consumed": dict(sorted(self.consumed.items())),
            "terminated_at": self.terminated_at,
            "termination_reason": self.termination_reason,
            "summary": self.remediation_summary(),
        }

    def path_hash(self) -> str:
        """Canonical hash of the route, excluding timings.

        Two runs with the same inputs should take the same route. A different path
        hash on identical inputs is a reproducibility signal that no metric comparison
        would catch — the numbers can agree while the review did something else.
        """
        canonical = {
            "graph_hash": self.graph_hash,
            "path": [(v.node_id, v.via, v.outcome) for v in self.visits],
            "remediations": [(r.edge_id, r.attempt, r.outcome) for r in self.remediations],
        }
        payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
        return sha256(payload.encode("utf-8")).hexdigest()

    def summary_lines(self) -> list[str]:
        summary = self.remediation_summary()
        lines = [
            f"  Execution path: {len(self.visits)} node visit(s), "
            f"{summary['attempts']} remediation attempt(s)",
            f"  Path hash: {self.path_hash()[:16]}…",
        ]
        if summary["attempts"]:
            lines.append(
                f"  Remediation: {summary['resolved']} resolved, "
                f"{summary['budget_exhausted']} exhausted, "
                f"{summary['no_change']} ineffective"
            )
            for attempt in self.remediations:
                marker = {
                    "resolved": "✓",
                    "still_failing": "·",
                    "budget_exhausted": "✗",
                    "no_change": "!",
                }.get(attempt.outcome, "·")
                lines.append(
                    f"    {marker} {attempt.source} → {attempt.target} "
                    f"(attempt {attempt.attempt}/{attempt.budget}): {attempt.outcome}"
                )
        if self.termination_reason:
            lines.append(f"  Terminated at {self.terminated_at}: {self.termination_reason}")
        return lines


NodeHandler = Callable[[str, dict[str, Any]], NodeResult]


class GraphExecutor:
    """Walks a :class:`ReviewGraph`, enforcing cycle budgets.

    ``handlers`` maps node id to a callable taking ``(node_id, context)`` and returning
    a :class:`NodeResult`. Nodes with no handler are visited and recorded as ``skip`` —
    a partially wired graph degrades to a traversal rather than an exception, which
    matters while stages are being migrated onto it one at a time.
    """

    def __init__(
        self,
        graph: ReviewGraph,
        handlers: dict[str, NodeHandler] | None = None,
        *,
        budgets: dict[str, int] | None = None,
        max_visits: int = 400,
        strict: bool = False,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        graph.validate()
        self.graph = graph
        self.handlers = handlers or {}
        self.budget_overrides = budgets or {}
        self.max_visits = max_visits
        self.strict = strict
        self.clock = clock

    # -- budgets ------------------------------------------------------------
    def _budget_for(self, edge: Edge) -> int:
        return int(self.budget_overrides.get(edge.edge_id, edge.budget or 0))

    # -- edge selection -----------------------------------------------------
    def _select_edge(
        self, node_id: str, result: NodeResult, context: dict[str, Any], path: ExecutionPath
    ) -> Edge | None:
        """Choose the outgoing edge, in a fixed priority order.

        Order matters and is deliberate: a hard stop beats a remediation, and a
        remediation beats ordinary progression. Otherwise a blocking defect could be
        quietly retried instead of stopping the review.
        """
        outgoing = self.graph.outgoing(node_id)

        if result.outcome == NodeOutcome.BLOCK:
            for edge in outgoing:
                if edge.kind is EdgeKind.HARD_STOP:
                    return edge

        if result.outcome == NodeOutcome.REENTER:
            for edge in outgoing:
                if edge.kind is EdgeKind.SELF_LOOP:
                    if path.consumed.get(edge.edge_id, 0) < self._budget_for(edge):
                        return edge
                    # Budget spent: fall through to advance rather than looping.
                    break

        if result.outcome == NodeOutcome.FAIL:
            for edge in outgoing:
                if edge.kind is EdgeKind.REMEDIATION:
                    return edge

        for edge in outgoing:
            if edge.kind in {EdgeKind.ADVANCE, EdgeKind.BRANCH}:
                condition = edge.condition
                if not condition:
                    return edge
                if self._condition_holds(condition, context):
                    return edge
        return None

    @staticmethod
    def _condition_holds(condition: str, context: dict[str, Any]) -> bool:
        """Evaluate a declarative edge condition against the run context.

        Deliberately not ``eval``. Conditions are simple flag references, and a graph
        description that can execute arbitrary code is a graph description that can be
        weaponised by anyone who can edit a config.
        """
        text = condition.strip().lower()
        if text.startswith("not "):
            return not bool(context.get(text[4:].strip()))
        if " or " in text:
            return any(bool(context.get(part.strip())) for part in text.split(" or "))
        if " and " in text:
            return all(bool(context.get(part.strip())) for part in text.split(" and "))
        # Unknown conditions default to permitting the edge: an unrecognised label
        # should not silently strand the review at a node with no way forward.
        return bool(context.get(text, True))

    # -- remediation --------------------------------------------------------
    def _record_remediation(
        self, edge: Edge, path: ExecutionPath, context: dict[str, Any], reason: str
    ) -> RemediationAttempt | None:
        """Attempt a back-edge, or refuse and explain why."""
        budget = self._budget_for(edge)
        used = path.consumed.get(edge.edge_id, 0)

        if used >= budget:
            attempt = RemediationAttempt(
                edge_id=edge.edge_id,
                source=edge.source,
                target=edge.target,
                reason=reason or edge.condition,
                attempt=used,
                budget=budget,
                outcome="budget_exhausted",
                detail=edge.rationale,
            )
            path.remediations.append(attempt)
            if self.strict:
                raise CycleBudgetExceeded(attempt.detail)
            return None

        path.consumed[edge.edge_id] = used + 1
        attempt = RemediationAttempt(
            edge_id=edge.edge_id,
            source=edge.source,
            target=edge.target,
            reason=reason or edge.condition,
            attempt=used + 1,
            budget=budget,
            outcome="still_failing",
            detail=edge.rationale,
        )
        path.remediations.append(attempt)
        context.setdefault("_remediation_history", []).append(attempt.edge_id)
        return attempt


    # -- fan-out ------------------------------------------------------------
    def _join_target(self, fanout_id: str) -> str | None:
        """The join every branch of this fan-out converges on."""
        branch_targets = [
            e.target for e in self.graph.outgoing(fanout_id) if e.kind is EdgeKind.BRANCH
        ]
        for branch in branch_targets:
            for edge in self.graph.outgoing(branch):
                if self.graph.node(edge.target).kind is NodeKind.JOIN:
                    return edge.target
        return None

    def _run_branches(
        self,
        fanout_id: str,
        context: dict[str, Any],
        path: ExecutionPath,
        fingerprints: dict[str, str],
        pending: dict[str, RemediationAttempt],
    ) -> Edge | None:
        """Run every branch of a fan-out, then decide whether any of them remediates.

        Branches are independent by construction — they are diagnostics over an
        already-fitted model — so the order does not affect their results. They are
        run sequentially here for determinism; a concurrent backend can replace this
        method without changing anything else.

        If several branches fail, the first failing branch in declaration order takes
        its remediation edge. The others are recorded as failures and will be seen
        again when the loop returns, so nothing is lost — but only one back-edge is
        taken at a time, which keeps the path linear and readable.
        """
        branches = [e for e in self.graph.outgoing(fanout_id) if e.kind is EdgeKind.BRANCH]
        failures: list[tuple[Edge, NodeResult]] = []

        for branch in branches:
            node = self.graph.node(branch.target)
            if node.condition and not self._condition_holds(node.condition, context):
                path.visits.append(
                    Visit(branch.target, EdgeKind.BRANCH.value, NodeOutcome.SKIP,
                          "condition not met")
                )
                continue

            handler = self.handlers.get(branch.target)
            started = self.clock()
            if handler is None:
                result = NodeResult(outcome=NodeOutcome.SKIP, detail="no handler wired")
            else:
                try:
                    result = handler(branch.target, context) or NodeResult()
                except Exception as exc:
                    result = NodeResult(
                        outcome=NodeOutcome.FAIL, detail=f"{type(exc).__name__}: {exc}"
                    )
            elapsed = self.clock() - started
            path.visits.append(
                Visit(branch.target, EdgeKind.BRANCH.value, result.outcome,
                      result.detail, round(elapsed, 6))
            )
            if result.state:
                context.update(result.state)

            if branch.target in pending:
                attempt = pending.pop(branch.target)
                if result.outcome == NodeOutcome.OK:
                    attempt.outcome = "resolved"
            if result.fingerprint:
                fingerprints[branch.target] = result.fingerprint

            if result.outcome == NodeOutcome.FAIL:
                failures.append((branch, result))

        for branch, result in failures:
            remediation = next(
                (
                    e
                    for e in self.graph.outgoing(branch.target)
                    if e.kind is EdgeKind.REMEDIATION
                ),
                None,
            )
            if remediation is None:
                continue
            attempt = self._record_remediation(remediation, path, context, result.detail)
            if attempt is None:
                continue  # budget spent; fall through to the join
            previous = fingerprints.get(remediation.target, "")
            if previous and previous == context.get(f"_last_fp_{remediation.target}"):
                attempt.outcome = "no_change"
                continue
            pending[remediation.target] = attempt
            pending[branch.target] = attempt
            return remediation
        return None

    # -- run ----------------------------------------------------------------
    def run(self, context: dict[str, Any] | None = None) -> ExecutionPath:
        context = dict(context or {})
        path = ExecutionPath(
            graph_name=self.graph.name,
            graph_hash=self.graph.graph_hash(),
            budgets={
                e.edge_id: self._budget_for(e)
                for e in self.graph.edges
                if e.budget is not None
            },
        )

        current = self.graph.entry
        via = "entry"
        #: node id -> fingerprint of its last result, for no-change detection.
        fingerprints: dict[str, str] = {}
        #: remediation attempt awaiting a verdict from its target node.
        pending: dict[str, RemediationAttempt] = {}

        for _ in range(self.max_visits):
            node = self.graph.node(current)

            if node.condition and not self._condition_holds(node.condition, context):
                path.visits.append(Visit(current, via, NodeOutcome.SKIP, "condition not met"))
                result = NodeResult(outcome=NodeOutcome.SKIP)
            else:
                handler = self.handlers.get(current)
                started = self.clock()
                if handler is None:
                    result = NodeResult(outcome=NodeOutcome.SKIP, detail="no handler wired")
                else:
                    try:
                        result = handler(current, context) or NodeResult()
                    except Exception as exc:  # a node failing must not lose the path
                        result = NodeResult(
                            outcome=NodeOutcome.FAIL,
                            detail=f"{type(exc).__name__}: {exc}",
                        )
                elapsed = self.clock() - started
                path.visits.append(
                    Visit(current, via, result.outcome, result.detail, round(elapsed, 6))
                )

            if result.state:
                context.update(result.state)

            # Did this node resolve a remediation that routed into it?
            if current in pending:
                attempt = pending.pop(current)
                previous = fingerprints.get(current, "")
                if result.fingerprint and result.fingerprint == previous:
                    attempt.outcome = "no_change"
                    attempt.changed = "nothing"
                else:
                    attempt.changed = result.fingerprint or "unrecorded"
            if result.fingerprint:
                fingerprints[current] = result.fingerprint

            if current in self.graph.exits:
                path.terminated_at = current
                path.termination_reason = "reached exit"
                break

            # Fan-out is executed as a unit. Taking a single branch edge here would
            # silently run one diagnostic and skip the rest — which is exactly what a
            # naive "pick the first matching edge" does, and it looks like success.
            if node.kind is NodeKind.FANOUT:
                branch_edge = self._run_branches(current, context, path, fingerprints, pending)
                if branch_edge is not None:
                    current, via = branch_edge.target, branch_edge.kind.value
                    continue
                join = next(
                    (e for e in self.graph.outgoing(current) if e.kind is EdgeKind.ADVANCE),
                    None,
                )
                target = join.target if join else self._join_target(current)
                if target is None:
                    path.terminated_at = current
                    path.termination_reason = "fan-out has no join"
                    break
                current, via = target, EdgeKind.ADVANCE.value
                continue

            edge = self._select_edge(current, result, context, path)
            if edge is None:
                path.terminated_at = current
                path.termination_reason = (
                    f"no outgoing edge satisfied from {current!r} "
                    f"with outcome {result.outcome!r}"
                )
                break

            if edge.kind is EdgeKind.REMEDIATION:
                attempt = self._record_remediation(edge, path, context, result.detail)
                if attempt is None:
                    # Budget spent. Continue forward instead of looping, so the
                    # review still reaches sign-off and reports the exhaustion.
                    forward = next(
                        (
                            e
                            for e in self.graph.outgoing(current)
                            if e.kind in {EdgeKind.ADVANCE, EdgeKind.BRANCH}
                        ),
                        None,
                    )
                    if forward is None:
                        path.terminated_at = current
                        path.termination_reason = "remediation budget exhausted, no forward edge"
                        break
                    current, via = forward.target, forward.kind.value
                    continue

                # A remediation whose previous attempt changed nothing is stopped
                # early rather than burning the rest of the budget on a repeat.
                repeated = [
                    r for r in path.remediations if r.edge_id == edge.edge_id and r.outcome == "no_change"
                ]
                if repeated:
                    attempt.outcome = "no_change"
                    forward = next(
                        (
                            e
                            for e in self.graph.outgoing(current)
                            if e.kind in {EdgeKind.ADVANCE, EdgeKind.BRANCH}
                        ),
                        None,
                    )
                    if forward is not None:
                        current, via = forward.target, forward.kind.value
                        continue

                pending[edge.target] = attempt

            elif edge.kind is EdgeKind.SELF_LOOP:
                path.consumed[edge.edge_id] = path.consumed.get(edge.edge_id, 0) + 1

            current, via = edge.target, edge.kind.value
        else:
            path.terminated_at = current
            path.termination_reason = f"visit limit {self.max_visits} reached"

        # Any remediation still pending never got a verdict from its target.
        for attempt in pending.values():
            if attempt.outcome == "still_failing":
                attempt.outcome = "unverified"

        # A remediation whose downstream check later passed is marked resolved.
        self._mark_resolved(path)
        return path

    @staticmethod
    def _mark_resolved(path: ExecutionPath) -> None:
        """Mark the FINAL attempt on an edge resolved if its source ended ``ok``.

        Only the last attempt can be the one that worked. Marking every attempt
        resolved because the condition eventually cleared would read as "three
        successful remediations" when the truth is "two failures and a success" —
        and the number of failed attempts is the part a reviewer needs.
        """
        # The final attempt on an edge is the one with the highest attempt number,
        # whatever its current outcome. An earlier attempt is never eligible.
        final_by_edge: dict[str, RemediationAttempt] = {}
        for attempt in path.remediations:
            existing = final_by_edge.get(attempt.edge_id)
            if existing is None or attempt.attempt >= existing.attempt:
                final_by_edge[attempt.edge_id] = attempt

        for attempt in final_by_edge.values():
            if attempt.outcome not in {"still_failing", "unverified"}:
                continue
            source_visits = [v for v in path.visits if v.node_id == attempt.source]
            if source_visits and source_visits[-1].outcome == NodeOutcome.OK:
                attempt.outcome = "resolved"
