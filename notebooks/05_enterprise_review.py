# Databricks notebook source
# MAGIC %md
# MAGIC # StART — Enterprise Agentic Model Review (v2.0.0)
# MAGIC The audit-ready operating system: the review runs as explicit layers
# MAGIC (Data → Model → Validation → Governance → AI-Engineering → Evidence →
# MAGIC Reporting), each emitting status, runtime, findings, artifacts, and
# MAGIC evidence IDs. Produces an enterprise dashboard (`dashboard.html/.json/.md`),
# MAGIC governance findings, executable AI-engineering controls, and a LangGraph-
# MAGIC style review graph — all from one execution flow.
# MAGIC
# MAGIC Deterministic by default. Public providers (OpenAI/Anthropic/Grok) and the
# MAGIC enterprise gateway are strictly isolated — no crossover, no key sharing.

# COMMAND ----------
import os

WIDGETS = {
    "dataset_path": "",            # blank = built-in demo dataset
    "target_column": "",          # blank = discovery proposes
    "task_type": "",              # blank = inferred
    "split_strategy": "stratified",
    "architecture": "mlp",
    "activation": "relu",
    "agent_mode": "deterministic",  # deterministic | llm
    "enterprise_gateway": "no",    # yes | no — yes routes via enterprise gateway, skips public keys
    "provider": "none",           # none | openai | anthropic | grok (used only when enterprise_gateway=no)
    "enterprise_mode": "true",
    "governance_mode": "true",
    "run_dl": "true",
    "cnn_preset": "simple_cnn_small",  # small | medium | deep | custom
}
try:
    for k, v in WIDGETS.items():
        dbutils.widgets.text(k, v, k)  # type: ignore[name-defined]
    get = lambda k: dbutils.widgets.get(k)  # type: ignore[name-defined]  # noqa: E731
    ON_DATABRICKS = True
except NameError:
    get = lambda k: os.environ.get(f"START_NB_{k.upper()}", WIDGETS[k])  # noqa: E731
    ON_DATABRICKS = False

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1. CNN configuration (transparent, evidence-backed)
# MAGIC Every architecture choice — preset, conv blocks, channels, kernel, pooling,
# MAGIC dense size, dropout, parameter count — becomes evidence metadata.

# COMMAND ----------
from start.modeling.vision_models import config_from_preset, describe_cnn

cnn_preset = get("cnn_preset")
if cnn_preset == "custom":
    cfg = config_from_preset("simple_cnn_small", n_blocks=3, base_channels=24, kernel_size=5)
    cnn_descriptor = describe_cnn("simple_cnn", channels=3, image_size=32, n_classes=3, config=cfg)
else:
    preset_name = f"simple_cnn_{cnn_preset}" if not cnn_preset.startswith("simple_cnn") else cnn_preset
    cnn_descriptor = describe_cnn(preset_name, channels=3, image_size=32, n_classes=3)
print("CNN architecture (evidence metadata):")
for k, v in cnn_descriptor.items():
    print(f"  {k}: {v}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2. Load data + resolve provider (strict trust-domain separation)

# COMMAND ----------
target = get("target_column").strip() or None
path = get("dataset_path").strip()
if path:
    from start.data.loaders import load_any_tabular

    df = load_any_tabular(path)
else:
    from start.modeling.data import load_attrition_dataset

    df = load_attrition_dataset(seed=42)
    target = target or "attrition"

llm = None
trust_domain_label = "public"
# v2.3.0 gateway UX (notebook mirror): enterprise_gateway=yes forces the
# enterprise gateway and skips public providers/keys entirely. Otherwise the
# public provider widget applies. Public and enterprise never cross over.
if get("agent_mode") == "llm" and str(get("enterprise_gateway")).strip().lower() in ("yes", "y", "true"):
    provider = "enterprise_llm_gateway"
    trust_domain_label = "enterprise"
    from start.core.config import LLMConfig
    from start.providers.llm import get_llm_provider

    llm = get_llm_provider(LLMConfig(provider=provider), expected_domain="private")
    print(f"enterprise gateway selected (enterprise domain) | available: "
          f"{getattr(llm, 'available', False)} | no public keys requested")
    if not getattr(llm, "available", False):
        print("FALLBACK: enterprise package not present — deterministic review "
              "(no public provider is ever used as a fallback).")
elif get("agent_mode") == "llm" and get("provider") not in ("none", ""):
    provider = get("provider")
    from start.core.config import LLMConfig
    from start.providers.llm import get_llm_provider
    from start.providers.trust_domains import trust_domain

    domain = trust_domain(provider).value
    expected = domain if domain in ("public", "private") else None
    llm = get_llm_provider(LLMConfig(provider=provider), expected_domain=expected)
    print(f"provider {provider} ({domain} domain) | available: {getattr(llm, 'available', False)}")
else:
    print("Deterministic mode — no key required.")
print(f"dataset: {len(df)} rows x {df.shape[1]} columns | target: {target}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 3. Run the enterprise layered review (stages + layers stream visibly)

# COMMAND ----------
from start.modeling.enterprise_orchestrator import EnterpriseReviewOrchestrator


def show_stage(e):
    if e.status in ("complete", "skipped"):
        print(f"  [{e.status[0].upper()}] {e.stage.replace('_', ' ').title()}")


def show_layer(lr):
    if lr.status != "running":
        print(f"  ╞══ {lr.name:16s} {lr.status} {lr.runtime_seconds:.3f}s "
              f"findings={len(lr.findings)} artifacts={len(lr.artifacts)} "
              f"evidence={len(lr.evidence_ids)}")


def show_adapter(r):
    print(f"      [{r.status}] {r.adapter} ({r.runtime_seconds:.3f}s)")


orch = EnterpriseReviewOrchestrator(
    on_stage=show_stage, on_layer=show_layer, on_adapter=show_adapter
)
outcome = orch.run(
    df,
    user_target=target,
    split_strategy=get("split_strategy"),
    agent_mode=get("agent_mode"),
    llm=llm,
    output_root="start_output",
    run_dl=(get("run_dl").lower() == "true"),
    enterprise_mode=(get("enterprise_mode").lower() == "true"),
    cnn_config=cnn_descriptor,
    seed=42,
)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 4. Governance findings

# COMMAND ----------
summary = outcome.findings_register.summary()
print(f"findings: {summary['total']} "
      f"(Critical={summary['Critical']} High={summary['High']} "
      f"Medium={summary['Medium']} Low={summary['Low']})")
for f in outcome.findings_register.sorted()[:10]:
    print(f"  [{f.severity.value}/{f.materiality.value}] {f.risk_category}: {f.title}")
    print(f"      evidence: {f.evidence_ids} | {f.recommendation}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 5. AI-engineering controls (honest availability)

# COMMAND ----------
for row in outcome.ai_engineering.summary_rows():
    print(f"  [{row['status']:14s}] {row['adapter']:16s} {row['category']:14s} "
          f"art={row['artifacts']} ev={row['evidence']}")
print(f"\n{outcome.ai_engineering.available_count}/{outcome.ai_engineering.total} available")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 6. Audit package

# COMMAND ----------
print("dashboard:", outcome.dashboard_paths["html"])
print("dashboard json:", outcome.dashboard_paths["json"])
print("review graph:", outcome.graph_paths)
print("evidence critique:", "PASSED" if outcome.critique_ok else "FAILED")
print("\nLayer timeline:")
for lr in outcome.layers:
    print(f"  {lr.name:16s} {lr.status:9s} {lr.runtime_seconds:.3f}s")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 7. Visible review execution (v2.1.1): LLM activation, agent reasoning, control surface, artifacts
# MAGIC The same visibility the terminal shows — nothing important happens silently.

# COMMAND ----------
# Section A: LLM activation (provider / model / trust domain / endpoint / status)
if outcome.activation_report is not None:
    print(outcome.activation_report.render_terminal())

# COMMAND ----------
# Section K: agent reasoning traces (inputs / reasoning / decision / confidence / alternative)
import pandas as pd

if outcome.trace_log is not None and outcome.trace_log.traces:
    display(pd.DataFrame(outcome.trace_log.to_list()))

# COMMAND ----------
# Sections L/M: AI-engineering control surface (purpose / role / status / outputs / install)
display(pd.DataFrame(outcome.ai_engineering.control_surface()))

# COMMAND ----------
# Section N: artifact catalog (every generated artifact, discoverable)
if outcome.artifact_registry is not None:
    display(pd.DataFrame(outcome.artifact_registry.to_list()))

# COMMAND ----------
# MAGIC %md
# MAGIC ## Model execution (v2.1.1): split, metrics, training, explainability

# COMMAND ----------
# Sections D/G/I/J/K: train/test/OOS split, metrics-by-split, training, explainability
me = outcome.model_execution
if me is not None:
    print("Train/Test/OOS split:")
    display(pd.DataFrame(me.split_table))
    print("Metrics by split:")
    display(pd.DataFrame(me.metrics_by_split).T)
    print(f"Generalization gap (train - OOS): {me.generalization_gap}")
    print(f"Explainability method: {me.explainability_method}")
    display(pd.DataFrame(me.global_importance))
else:
    print("Model execution skipped (run_dl=False or non-tabular). Enable training to see"
          " split, metrics, training history, and explainability.")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Live committee (v2.2.0): agent roster, adapter panel, sensitivity, review journey

# COMMAND ----------
# Item 1: the review committee — the agents and their purposes
from start.agent_roster import render_agent_roster, roster_as_list

print(render_agent_roster())
display(pd.DataFrame(roster_as_list()))

# COMMAND ----------
# Item 9: AI-engineering environment (adapters) with status
display(pd.DataFrame(outcome.ai_engineering.control_surface()))

# COMMAND ----------
# Item 4: sensitivity analysis table (feature / shock % / baseline / shocked / delta / risk)
if outcome.sensitivity is not None:
    rows = outcome.sensitivity.to_dict()["rows"]
    display(pd.DataFrame(rows)[["feature", "shock", "baseline", "metric", "drift", "risk_impact"]])
else:
    print("No sensitivity run (enable run_dl on a tabular cohort).")

# COMMAND ----------
# Item 11: the review journey — decisions, overrides, and agent conversations.
# In a notebook run these come from a ReviewSession you build; the same engine
# powers the terminal's interactive checkpoints and the dashboard transcript.
from start.review_session import Decision, ReviewSession

session = ReviewSession(run_id=outcome.run_id)
# Example: record a decision + a question you would ask an agent.
session.record_decision(Decision(
    key="architecture", prompt="Model family?", recommended="mlp",
    user_value="mlp", effective="mlp", choice="accept",
    rationale="Small tabular dataset favors a simpler MLP.",
))
from start.agent_dialogue import AgentContext, ask_agent

ctx = AgentContext(agent="ArchitectureReviewAgent", recommendation="mlp",
                   reason="Small tabular dataset favors a simpler MLP.",
                   risk_if_ignored="Higher overfitting risk.",
                   alternatives=[{"family": "mlp"}, {"family": "wide_deep"}],
                   checkpoint="architecture")
ask_agent("ArchitectureReviewAgent", "Why not wide_deep?", ctx, session)
display(pd.DataFrame(session.to_dict()["conversations"]))

# COMMAND ----------
# Write the same transcript artifacts the terminal writes
from start.reporting.review_transcript import write_transcript

paths = write_transcript(session, "/tmp/start_notebook", outcome.run_id,
                         sensitivity=outcome.sensitivity)
print("Transcript written:", paths)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Evidence-driven review committee (v2.3.0)
# MAGIC The same evidence store, committee cards, challenge memory, ValidationAgent
# MAGIC review, and MRM signoff the terminal uses — no duplicate logic.

# COMMAND ----------
# Evidence store assembled from the run's real diagnostics (#1)
from start.evidence_store import EvidenceStore

ev = EvidenceStore.from_artifacts(
    model_exec=outcome.model_execution,
    sensitivity=outcome.sensitivity,
    tuning_run=outcome.tuning_run,
)
print("evidence available:", ev.has_any())

# COMMAND ----------
# Evidence-constrained Ask-Agent (#2): diagnostic questions are answered ONLY
# from artifacts. A question without evidence is refused, never fabricated.
from start.evidence_dialogue import answer_from_evidence

for q in ["Show the feature sensitivity drift for the top features",
          "What is the recall by split?",
          "Which features have the most missing values?"]:
    ea = answer_from_evidence(q, ev)
    print(f"Q: {q}")
    print(f"   grounded={ea.grounded} refused={ea.refused}")
    print("   " + ea.answer.splitlines()[0])

# COMMAND ----------
# Committee card (#1/#4): evidence -> recommendation -> alternatives -> risks
from start.committee_card import CommitteeCard, render_card_markdown

card = CommitteeCard(
    agent="ArchitectureReviewAgent", purpose="Model selection",
    evidence=[f"{ev.n_rows or 'n'} rows",
              f"max sensitivity drift {ev.max_abs_drift}"],
    recommendation="MLP",
    alternatives=["MLP (recommended)", "ResidualMLP", "WideDeep"],
    risks=["overfitting on small data with a heavier architecture"],
    artifacts_used=["data_statistics", "sensitivity_analysis"],
)
print(render_card_markdown(card))

# COMMAND ----------
# ValidationAgent sensitivity review (#7/#8): ranking + business interpretation
from start.validation_review import business_interpretation

if outcome.sensitivity is not None:
    rows = outcome.sensitivity.to_dict()["rows"]
    ranking = {}
    for r in rows:
        f, d = r["feature"], abs(r["drift"])
        ranking[f] = max(ranking.get(f, 0.0), d)
    ranked = sorted(ranking.items(), key=lambda kv: kv[1], reverse=True)
    display(pd.DataFrame(ranked, columns=["feature", "max_abs_drift"]))
    print("\nBusiness interpretation:")
    for line in business_interpretation(rows):
        print(" -", line)

# COMMAND ----------
# MRM-grade signoff (#11): performance, calibration, sensitivity, challenges
from start.mrm_signoff import evaluate_signoff, render_signoff_markdown

decision = evaluate_signoff(ev, session, primary_metric="auc_roc")
print(render_signoff_markdown(decision))

# COMMAND ----------
# Reviewer challenge log (#3): challenges persist with the evidence used
from start.review_session import Challenge

session.record_challenge(Challenge(
    text="Why is the top feature the most sensitive?",
    agent="ValidationAgent"))
session.close_challenge("Why is the top feature the most sensitive?",
                        response="Answered from sensitivity artifacts.",
                        evidence_used=["sensitivity_analysis.rows"])
display(pd.DataFrame(session.to_dict()["challenges"]))
print("challenge summary:", session.to_dict()["challenge_summary"])

# COMMAND ----------
# MAGIC %md
# MAGIC ## v2.3.1 polish surfaces (notebook parity)
# MAGIC Boxed committee roster, LLM activation, colored adapter inventory,
# MAGIC dataset/target transparency, and the decision ledger.

# COMMAND ----------
# #2 dataset + target transparency
print("Dataset:", "sklearn breast-cancer (demo)" if not OPTIONS["dataset_path"] else OPTIONS["dataset_path"])
print("  url: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic"
      if not OPTIONS["dataset_path"] else "")
print(f"  loaded: {len(df)} rows x {df.shape[1]} columns")
print("Target:", target, "(supplied by user)" if OPTIONS["target_column"] else "(inferred by discovery)")
_cands = [c for c in df.columns if c != target and df[c].dropna().nunique() == 2][:8]
print("  candidate targets:", ", ".join(_cands) if _cands else "—")

# COMMAND ----------
# #3 committee roster with colored agent names (roster + colors share the engine)
from start.agent_roster import AGENT_COLORS, roster_as_list

display(pd.DataFrame([
    {"agent": r["agent"], "purpose": r["purpose"], "color": AGENT_COLORS.get(r["agent"], "white")}
    for r in roster_as_list()
]))

# COMMAND ----------
# #4 AI engineering adapter inventory (status/purpose/runtime/artifacts/evidence)
display(pd.DataFrame(outcome.ai_engineering.control_surface())[
    ["adapter", "status", "purpose", "runtime_s", "artifacts", "evidence", "install_guidance"]
])

# COMMAND ----------
# #5 endpoint display (safe; never secrets) for the resolved provider
from start.providers.llm_activation import preflight_llm

_act = preflight_llm(provider if 'provider' in dir() else 'none', llm)
print("provider:", _act.provider, "| trust domain:", _act.trust_domain)
print("endpoint:", _act.endpoint, "| status:", _act.status)

# COMMAND ----------
# #8 review decision ledger (mirrors terminal/dashboard/transcript)
from start.review_tables import decision_ledger_markdown

print(decision_ledger_markdown(session.to_dict().get("decisions", [])))

# COMMAND ----------
# MAGIC %md
# MAGIC ## K-fold tuning (train-only, stratified) — v2.3.1 #7
# MAGIC Real stratified K-fold model selection over the TRAINING split only.
# MAGIC Test/OOS rows are never used for selection. The DL path remains
# MAGIC single-split validation and is labelled as such (not K-fold).

# COMMAND ----------
# K-fold tuning summary + per-fold metrics (mirrors terminal/dashboard/transcript)
from start.modeling.kfold_tuning import render_kfold_markdown

_kf = getattr(outcome, "kfold_tuning", None)
if _kf is not None:
    print(render_kfold_markdown(_kf))
    display(pd.DataFrame([f.to_dict() for f in _kf.best_fold_results]))
    print("artifacts:", [a.split("/")[-1] for a in _kf.artifacts])
else:
    print("K-fold tuning not run for this configuration "
          "(e.g. too few rows/features); DL tuning uses single-split validation.")
