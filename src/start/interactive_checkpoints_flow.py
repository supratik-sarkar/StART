"""Interactive checkpoint flow (v2.2.0 items 1, 2, 5).

Runs the pre-execution review checkpoints — target confirmation, feature
engineering negotiation, and metric priority — each with the [Q] Ask-Agent
capability and full ReviewSession recording. Returns the resolved choices that
the orchestrator then executes, so user decisions actually drive the run
(not just the transcript).

The single architecture checkpoint already lives in ``interactive_review`` for
historical reasons; this module adds the rest of the committee touchpoints and
shares the same ``resolve_checkpoint`` + ``ask_agent`` primitives.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from start.agent_dialogue import AgentContext, ask_agent
from start.interactive_checkpoints import resolve_checkpoint
from start.review_session import Decision, ReviewSession


def _ask_factory(agent: str, ctx: AgentContext, session: ReviewSession,
                 llm: Any, llm_connected: bool) -> Callable[[str], str]:
    def _ask(question: str) -> str:
        return ask_agent(agent, question, ctx, session,
                         llm=llm, llm_connected=llm_connected).answer
    return _ask


def run_feature_engineering_checkpoints(
    fe_set: Any,
    session: ReviewSession,
    *,
    interactive: bool,
    auto_accept: bool,
    df: Any = None,
    target: str = "",
    already_weighted: bool = False,
    llm: Any = None,
    llm_connected: bool = False,
    ask: Callable[[str], str] = input,
    emit: Callable[[str], None] | None = None,
    evidence: Any = None,
    business_context: str = "",
    reviewer_clarification: str = "",
    task_type: str = "",
    model_name: str = "",
) -> dict[str, str]:
    """Negotiate each FE recommendation using method_options menus.

    Returns step -> action map. Decisions are recorded so downstream FE
    execution respects user selections and overrides.
    """
    import pandas as pd

    from start.modeling.method_options import (
        encoding_options,
        imbalance_options,
        imputation_options,
        outlier_options,
        scaling_options,
    )

    say = emit or (lambda _m: None)
    overrides: dict[str, str] = {}
    for rec in fe_set.applicable():
        ctx = AgentContext(
            agent="FeatureEngineeringAgent",
            recommendation=rec.recommendation,
            reason=rec.reason, risk_if_ignored=rec.risk_if_ignored,
            alternatives=None,
            dataset_summary="",
            checkpoint=f"fe:{rec.step}", evidence=evidence,
            business_context=business_context,
            reviewer_clarification=reviewer_clarification,
            task_type=task_type,
            model_name=model_name,
        )
        on_ask = _ask_factory("FeatureEngineeringAgent", ctx, session, llm, llm_connected)

        menu = None
        numeric_cols: list[str] = []
        if df is not None:
            numeric_cols = [c for c in df.columns if c != target and pd.api.types.is_numeric_dtype(df[c])]
            cat_cols = [c for c in df.columns if c != target and (df[c].dtype == "object" or isinstance(df[c].dtype, pd.CategoricalDtype))]
            if rec.step == "outliers" and numeric_cols:
                menu = outlier_options(df, numeric_cols, iqr_multiplier=1.5)
            elif rec.step == "imputation":
                menu = imputation_options(df)
            elif rec.step == "encoding" and cat_cols:
                menu = encoding_options(df, cat_cols)
            elif rec.step == "scaling" and numeric_cols:
                menu = scaling_options(df, numeric_cols)
            elif rec.step == "imbalance" and target in df.columns:
                menu = imbalance_options(df[target], already_weighted=already_weighted)

        rec_opt = menu.recommended_option() if menu is not None else None
        rec_key = rec_opt.key if rec_opt else "apply"

        action = rec_key
        choice = "accept"
        fe_rationale = rec.reason

        if auto_accept:
            choice = "auto_accept"
        elif interactive:
            current_agent = "FeatureEngineeringAgent"
            if menu is not None:
                say(f"  {current_agent} recommends: {rec.recommendation}")
                for line in menu.render_lines():
                    say(line)
                while True:
                    prompt_line = f"    [A] Accept  [1-{len(menu.options)}] Choose  [P] Plot  [Q] Ask {current_agent}: "
                    if ask is input:
                        from start.interactive_checkpoints import read_multiline_paste_normalized
                        resp_raw = read_multiline_paste_normalized(prompt_line)
                    else:
                        resp_raw = ask(prompt_line)

                    stripped = (resp_raw or "").strip()
                    resp = stripped.lower()
                    if not resp:
                        continue

                    if resp in ("q", "ask", "?") or resp.startswith("q ") or resp.startswith("q:"):
                        q_text = stripped[1:].strip() if len(stripped) > 1 else ""
                        if not q_text:
                            prompt_q = f"    Ask {current_agent}: "
                            q_text = (read_multiline_paste_normalized(prompt_q) if ask is input else ask(prompt_q)).strip()
                        if q_text and q_text.lower() not in ("q", "ask", "?"):
                            agent_answer = on_ask(q_text)
                            say(f"    {agent_answer}")
                        continue

                    if resp in ("c", "challenge") or resp.startswith("c ") or resp.startswith("c:"):
                        c_text = stripped[1:].strip() if len(stripped) > 1 else ""
                        if not c_text:
                            prompt_c = f"    Enter challenge to {current_agent}: "
                            c_text = (read_multiline_paste_normalized(prompt_c) if ask is input else ask(prompt_c)).strip()
                        if not c_text:
                            c_text = f"Why is the recommendation '{rec.recommendation}' appropriate?"
                        agent_answer = on_ask(c_text)
                        say(f"    {agent_answer}")
                        from start.governance.challenge_disposition import CONCESSION_PROMPT
                        c_ans = (read_multiline_paste_normalized(CONCESSION_PROMPT) if ask is input else ask(CONCESSION_PROMPT)).strip().lower()
                        if session and getattr(session, "challenges", None):
                            session.challenges[-1].conceded = c_ans in {"y", "yes"}
                            session.challenges[-1].changes_disposition = session.challenges[-1].conceded
                        continue

                    if resp in ("p", "plot"):
                        from start.reporting.figure_viewer import open_figure
                        from start.reporting.figures import plot_distribution_with_bounds
                        col_to_plot = numeric_cols[0] if numeric_cols else ""
                        if col_to_plot and col_to_plot in df.columns:
                            plot_path = plot_distribution_with_bounds(df, col_to_plot, methods=menu.options)
                            if plot_path:
                                say(f"    Opened distribution plot: {plot_path}")
                                open_figure(plot_path)
                        continue

                    if resp in ("a", "accept", "y", "yes"):
                        action = rec_key
                        choice = "accept"
                        fe_rationale = rec.reason
                        break

                    if resp.isdigit() and 1 <= int(resp) <= len(menu.options):
                        chosen_opt = menu.options[int(resp) - 1]
                        action = chosen_opt.key
                        if chosen_opt.key == "custom":
                            prompt_cust = "    Enter custom parameter value: "
                            cust_val = (read_multiline_paste_normalized(prompt_cust) if ask is input else ask(prompt_cust)).strip()
                            action = f"custom:{cust_val}"

                        if chosen_opt.recommended:
                            choice = "accept"
                            fe_rationale = rec.reason
                        else:
                            choice = "override"
                            prompt_rat = "    Reviewer rationale for choice: "
                            rat = (read_multiline_paste_normalized(prompt_rat) if ask is input else ask(prompt_rat)).strip()
                            fe_rationale = rat or f"Reviewer selected option {chosen_opt.label}"
                        break

                    say(f"    Please select A (accept), 1-{len(menu.options)}, P (plot), or Q (ask).")
            else:
                say(f"  FeatureEngineeringAgent recommends: {rec.recommendation}")
                say(f"    reason: {rec.reason}")
                while True:
                    current_agent = "FeatureEngineeringAgent"
                    if ask is input:
                        from rich.console import Console

                        from start.cli.view import get_styled_agent_name
                        from start.interactive_checkpoints import read_multiline_paste_normalized
                        c = Console()
                        c.print(f"    Apply {rec.step}? [Y]es / [n]o / [Q] ask ", end="")
                        c.print(get_styled_agent_name(current_agent), end=": ")
                        resp_raw = read_multiline_paste_normalized("")
                    else:
                        from start.cli.view import get_ansi_agent_name
                        resp_raw = ask(
                            f"    Apply {rec.step}? [Y]es / [n]o / [Q] ask {get_ansi_agent_name(current_agent)}: "
                        )

                    stripped = (resp_raw or "").strip()
                    resp = stripped.lower()
                    if not resp:
                        continue

                    if resp in ("q", "ask", "?") or resp.startswith("q ") or resp.startswith("q:"):
                        q_text = stripped[1:].strip() if len(stripped) > 1 else ""
                        if not q_text:
                            prompt_q = f"    Ask {current_agent}: "
                            q_text = (read_multiline_paste_normalized(prompt_q) if ask is input else ask(prompt_q)).strip()
                        if q_text and q_text.lower() not in ("q", "ask", "?"):
                            agent_answer = on_ask(q_text)
                            say(f"    {agent_answer}")
                        continue

                    if resp in ("c", "challenge") or resp.startswith("c ") or resp.startswith("c:"):
                        c_text = stripped[1:].strip() if len(stripped) > 1 else ""
                        if not c_text:
                            prompt_c = f"    Enter challenge to {current_agent}: "
                            c_text = (read_multiline_paste_normalized(prompt_c) if ask is input else ask(prompt_c)).strip()
                        if not c_text:
                            c_text = f"Why is the recommendation '{rec.recommendation}' for step '{rec.step}' appropriate?"
                        agent_answer = on_ask(c_text)
                        say(f"    {agent_answer}")
                        from start.governance.challenge_disposition import CONCESSION_PROMPT
                        c_ans = (read_multiline_paste_normalized(CONCESSION_PROMPT) if ask is input else ask(CONCESSION_PROMPT)).strip().lower()
                        if session and getattr(session, "challenges", None):
                            session.challenges[-1].conceded = c_ans in {"y", "yes"}
                            session.challenges[-1].changes_disposition = session.challenges[-1].conceded
                        continue

                    if resp in ("n", "no", "skip", "reject"):
                        action, choice = "skip", "reject"
                        fe_rationale = f"Reviewer chose {choice} for {rec.step}"
                    else:
                        action, choice = "apply", "accept"
                        fe_rationale = rec.reason
                    break

        overrides[rec.step] = action
        session.record_decision(Decision(
            key=f"fe:{rec.step}", prompt=f"Apply {rec.step}?",
            recommended=rec_key, user_value=action, effective=action,
            choice=choice, rationale=fe_rationale,
            agent_rationale=rec.reason, evidence_ids=[rec.evidence_id],
        ))
    return overrides


def run_metric_checkpoint(
    user_cost: str,
    recommended_cost: str,
    reason: str,
    session: ReviewSession,
    *,
    interactive: bool,
    auto_accept: bool,
    llm: Any = None,
    llm_connected: bool = False,
    ask: Callable[[str], str] = input,
    emit: Callable[[str], None] | None = None,
    evidence: Any = None,
    business_context: str = "",
    reviewer_clarification: str = "",
    task_type: str = "",
    model_name: str = "",
) -> str:
    """Negotiate the metric prioritization checkpoint. Returns effective choice."""
    say = emit or (lambda _m: None)
    ctx = AgentContext(
        agent="HyperparameterTuningAgent",
        recommendation=recommended_cost, reason=reason,
        risk_if_ignored="Wrong metric can optimize for the wrong error type.",
        alternatives=[{"family": "false_negatives"}, {"family": "false_positives"},
                      {"family": "balanced"}],
        dataset_summary="", checkpoint="metric_priority", evidence=evidence,
        business_context=business_context,
        reviewer_clarification=reviewer_clarification,
        task_type=task_type,
        model_name=model_name,
    )
    on_ask = _ask_factory("HyperparameterTuningAgent", ctx, session, llm, llm_connected)
    dec = resolve_checkpoint(
        "metric_priority", user_cost, recommended_cost, reason,
        explanation=reason, interactive=interactive, auto_accept=auto_accept,
        ask=ask, emit=say, on_ask=on_ask,
        llm=llm, session=session, ctx=ctx,
    )
    session.record_decision(Decision(
        key="metric_priority", prompt="Cost priority?",
        recommended=recommended_cost, user_value=dec.user_value,
        effective=dec.effective_value, choice=dec.choice,
        rationale=dec.rationale or reason,
        agent_rationale=reason,
    ))
    return dec.effective_value


def run_target_checkpoint(
    candidate_target: str,
    recommended_target: str,
    reason: str,
    session: ReviewSession,
    *,
    interactive: bool,
    auto_accept: bool,
    llm: Any = None,
    llm_connected: bool = False,
    ask: Callable[[str], str] = input,
    emit: Callable[[str], None] | None = None,
    evidence: Any = None,
) -> str:
    """Confirm / override the target (item 4). Returns the effective target."""
    say = emit or (lambda _m: None)
    ctx = AgentContext(
        agent="DatasetDiscoveryAgent", recommendation=recommended_target,
        reason=reason, risk_if_ignored="Wrong target invalidates the entire review.",
        alternatives=None, dataset_summary="", checkpoint="target", evidence=evidence,
    )
    on_ask = _ask_factory("DatasetDiscoveryAgent", ctx, session, llm, llm_connected)
    dec = resolve_checkpoint(
        "target", candidate_target, recommended_target, reason,
        explanation=reason, interactive=interactive, auto_accept=auto_accept,
        ask=ask, emit=say, on_ask=on_ask,
        llm=llm, session=session, ctx=ctx,
    )
    session.record_decision(Decision(
        key="target", prompt="Confirm target column?",
        recommended=recommended_target, user_value=dec.user_value,
        effective=dec.effective_value, choice=dec.choice,
        rationale=dec.rationale or reason,
        agent_rationale=reason,
    ))
    return dec.effective_value
