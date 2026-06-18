"""ValidationAgent review checkpoint (v2.3.0 #7, #8).

Makes ValidationAgent a first-class, visible reviewer that runs BEFORE signoff
and presents the sensitivity review as a primary artifact:

- feature sensitivity ranking (by absolute max drift)
- shock analysis table (-30%..+30%)
- business interpretation of the most sensitive features
- feature-dependence / signoff impact

The reviewer can [A] accept, [Q] ask ValidationAgent (answered from sensitivity
evidence only), or [C] challenge a finding. Everything is recorded in the
review session so it flows into the transcript, dashboard, and signoff.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def _risk_phrase(risk: str) -> str:
    return {
        "high": "a large, potentially unstable dependence",
        "moderate": "a moderate dependence worth monitoring",
        "low": "a small dependence",
        "negligible": "no material dependence",
    }.get(risk, "an unquantified dependence")


def business_interpretation(rows: list[dict[str, Any]], top: int = 3) -> list[str]:
    """Plain-language interpretation of the most sensitive features (#8)."""
    by_feat: dict[str, dict[str, Any]] = {}
    for r in rows:
        f = r.get("feature")
        d = abs(float(r.get("drift", 0.0)))
        if f not in by_feat or d > by_feat[f]["drift"]:
            by_feat[f] = {"drift": d, "risk": r.get("risk_impact", "")}
    ranked = sorted(by_feat.items(), key=lambda kv: kv[1]["drift"], reverse=True)
    lines = []
    for f, info in ranked[:top]:
        lines.append(
            f"The model shows {_risk_phrase(info['risk'])} on '{f}' "
            f"(max metric drift {info['drift']:.4f} under +/-30% shocks). "
            f"If '{f}' shifts in production, expect a proportional change in "
            f"model output; monitor it for drift and data-quality issues."
        )
    return lines


def render_validation_review_rich(sensitivity: Any, console: Any) -> dict[str, Any]:
    """Render the full ValidationAgent sensitivity review with Rich tables.

    Returns a serializable summary for transcript/dashboard/notebook (#12)."""
    from start.review_tables import sensitivity_ranking_table, shock_table

    sd = sensitivity.to_dict() if hasattr(sensitivity, "to_dict") else sensitivity
    rows = sd.get("rows", [])

    console.print("\n[bold]ValidationAgent — validation review[/bold]")
    console.print(sensitivity_ranking_table(rows))
    console.print("")
    console.print(shock_table(rows))
    console.print("")
    interp = business_interpretation(rows)
    if interp:
        console.print("[bold]Business interpretation[/bold]")
        for line in interp:
            console.print(f"  • {line}")

    # feature-dependence / signoff impact note
    max_drift = sd.get("max_abs_drift")
    feat = sd.get("most_sensitive_feature")
    impact = "low feature dependence; no signoff concern from sensitivity."
    if max_drift is not None:
        if max_drift > 0.30:
            impact = (f"excessive dependence on '{feat}' (drift {max_drift:.4f}); "
                      "this blocks an unconditional READY.")
        elif max_drift > 0.15:
            impact = (f"elevated dependence on '{feat}' (drift {max_drift:.4f}); "
                      "signoff should be conditional.")
    console.print(f"\n[bold]Signoff impact:[/bold] {impact}")

    return {
        "ranking": [
            {"feature": f, "max_abs_drift": d}
            for f, d in _ranking(rows)
        ],
        "most_sensitive_feature": feat,
        "max_abs_drift": max_drift,
        "business_interpretation": interp,
        "signoff_impact": impact,
    }


def _ranking(rows: list[dict[str, Any]]) -> list[tuple[str, float]]:
    by_feat: dict[str, float] = {}
    for r in rows:
        f = r.get("feature")
        d = abs(float(r.get("drift", 0.0)))
        if f not in by_feat or d > by_feat[f]:
            by_feat[f] = d
    return sorted(by_feat.items(), key=lambda kv: kv[1], reverse=True)


def run_validation_checkpoint(
    sensitivity: Any,
    session: Any,
    evidence: Any,
    console: Any,
    *,
    interactive: bool,
    auto_accept: bool,
    llm: Any = None,
    llm_connected: bool = False,
    ask: Callable[[str], str] = input,
) -> dict[str, Any]:
    """Present the validation review and let the reviewer [A]/[Q]/[C] (#7)."""
    from start.agent_dialogue import AgentContext, ask_agent
    from start.review_session import Decision

    summary = render_validation_review_rich(sensitivity, console)

    ctx = AgentContext(
        agent="ValidationAgent",
        recommendation="proceed to signoff with noted sensitivity",
        reason=summary["signoff_impact"],
        risk_if_ignored="Unreviewed feature dependence can mask model fragility.",
        checkpoint="validation", evidence=evidence,
    )

    if auto_accept or not interactive:
        session.record_decision(Decision(
            key="validation", prompt="Accept validation review?",
            recommended="accept", user_value="accept", effective="accept",
            choice="auto_accept", rationale=summary["signoff_impact"],
        ))
        return summary

    while True:
        resp = (ask("\n  ValidationAgent review — [A]ccept / [Q] ask / [C]hallenge: ")
                or "").strip().lower()
        if resp in ("q", "ask", "?"):
            q = (ask("    Ask ValidationAgent: ") or "").strip()
            if q:
                ans = ask_agent("ValidationAgent", q, ctx, session,
                                llm=llm, llm_connected=llm_connected).answer
                console.print(f"    {ans}")
            continue
        if resp in ("c", "challenge"):
            q = (ask("    State your challenge: ") or "").strip()
            if q:
                # routed through ask_agent so it's recorded as a challenge
                ans = ask_agent("ValidationAgent", q, ctx, session,
                                llm=llm, llm_connected=llm_connected).answer
                console.print(f"    {ans}")
            continue
        # accept (default)
        session.record_decision(Decision(
            key="validation", prompt="Accept validation review?",
            recommended="accept", user_value="accept", effective="accept",
            choice="accept", rationale=summary["signoff_impact"],
        ))
        break
    return summary
