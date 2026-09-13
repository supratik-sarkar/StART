# Databricks notebook source
# MAGIC %md
# MAGIC # StART — Model-Risk Review (the operating system)
# MAGIC Data-first review: discovery → target → task → split → feature
# MAGIC engineering → recommendation → execution → metrics → explainability →
# MAGIC sensitivity → robustness → evidence → agentic challenge/governance/
# MAGIC sign-off → AI-engineering stages → report. **Every stage is visible.**
# MAGIC
# MAGIC Runs identically locally (VS Code / Jupyter, or as a script) and on
# MAGIC Databricks. Deterministic mode is the default and needs no key; the LLM
# MAGIC (if selected) reasons only over the evidence bundle, never raw data.

# COMMAND ----------
import os

WIDGETS = {
    "dataset_path": "",          # blank = built-in demo dataset
    "target_column": "",         # blank = discovery proposes
    "split_strategy": "stratified",
    "architecture": "mlp",
    "activation": "relu",
    "agent_mode": "deterministic",
    "llm_provider": "none",
    "run_dl": "true",
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
# MAGIC ## 1. Load data (demo or your own)

# COMMAND ----------
dataset_path = get("dataset_path").strip()
target_column = get("target_column").strip() or None
if dataset_path:
    from start.data.loaders import load_any_tabular

    df = load_any_tabular(dataset_path)
else:
    from start.modeling.data import load_attrition_dataset

    df = load_attrition_dataset(seed=42)
    target_column = target_column or "attrition"
print(f"dataset: {len(df)} rows x {df.shape[1]} columns | target: {target_column}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2. Resolve LLM provider (deterministic by default)

# COMMAND ----------
agent_mode = get("agent_mode")
llm_provider = get("llm_provider")
llm = None
if agent_mode == "llm" and llm_provider not in ("none", ""):
    if ON_DATABRICKS and llm_provider not in ("enterprise_llm_gateway",):
        from start.providers.keys import resolve_key_databricks

        resolve_key_databricks(llm_provider, dbutils=dbutils, scope="start")  # type: ignore[name-defined]
    from start.core.config import LLMConfig
    from start.providers.llm import get_llm_provider

    llm = get_llm_provider(LLMConfig(provider=llm_provider))
    print(f"LLM provider: {llm_provider} (available: {getattr(llm, 'available', False)})")
else:
    print("Deterministic mode — no LLM, no key required.")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 3. Run the full review (every stage printed as it runs)

# COMMAND ----------
from start.modeling.review_orchestrator import ReviewOrchestrator


def show(event):
    mark = {"running": "·", "complete": "✓", "skipped": "—"}.get(event.status, " ")
    detail = f"  {event.detail}" if event.detail else ""
    print(f"  {mark} {event.stage.replace('_', ' ').title():28s} [{event.status}]{detail}")


orch = ReviewOrchestrator(on_stage=show)
outcome = orch.run(
    df,
    user_target=target_column,
    split_strategy=get("split_strategy"),
    agent_mode=agent_mode,
    llm=llm,
    output_root="start_output",
    run_dl=(get("run_dl").lower() == "true"),
    seed=42,
)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 4. Result summary

# COMMAND ----------
print(f"run: {outcome.run_id}")
print(f"task: {outcome.task_type} | modality: {outcome.modality} | recommended: {outcome.recommended_family}")
print(f"evidence records: {len(outcome.evidence)}")
print(f"evidence critique: {'PASSED' if outcome.agent_review.critique_ok else 'FAILED'}")
if outcome.cohort_metrics:
    for cohort, m in outcome.cohort_metrics.items():
        print(f"  {cohort}: AUC={m.get('auc_roc', float('nan')):.4f}")
print(f"\nsign-off: {outcome.agent_review.signoff}")
print(f"report: {outcome.report_path}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 5. AI-engineering stage surface (honest availability)

# COMMAND ----------
for s in outcome.ai_engineering:
    print(f"  {s.name:34s} [{s.status}]  {s.category}")
