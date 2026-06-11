# Databricks notebook source
# MAGIC %md
# MAGIC # StART Deep Learning Model Review
# MAGIC
# MAGIC This notebook is Databricks-compatible and also runnable locally in VS Code/Jupyter as a Python script.
# MAGIC It trains a laptop-safe MLP classifier, generates DL diagnostics, evidence, figures, and a proof-carrying report.

# COMMAND ----------

from __future__ import annotations

import os
from getpass import getpass
from pathlib import Path

from IPython.display import Image, Markdown, display

from start.modeling.deep_learning import run_deep_learning_review
from start.modeling.dl_training import torch_available

# COMMAND ----------

try:
    dbutils  # type: ignore[name-defined]
    IN_DATABRICKS = True
except NameError:
    dbutils = None  # type: ignore[assignment]
    IN_DATABRICKS = False

if IN_DATABRICKS:
    dbutils.widgets.dropdown("architecture", "mlp", ["mlp", "residual_mlp", "wide_deep"])
    dbutils.widgets.dropdown("agent_mode", "deterministic", ["deterministic", "llm"])
    dbutils.widgets.dropdown("llm_provider", "none", ["none", "openai", "anthropic", "enterprise_llm_gateway"])
    dbutils.widgets.text("secret_scope", "start")
    architecture = dbutils.widgets.get("architecture")
    agent_mode = dbutils.widgets.get("agent_mode")
    llm_provider = dbutils.widgets.get("llm_provider")
    secret_scope = dbutils.widgets.get("secret_scope")
else:
    architecture = os.environ.get("START_DL_ARCHITECTURE", "mlp")
    agent_mode = os.environ.get("START_AGENT_MODE", "deterministic")
    llm_provider = os.environ.get("START_LLM_PROVIDER", "none")
    secret_scope = os.environ.get("START_SECRET_SCOPE", "start")

print(f"Databricks runtime: {IN_DATABRICKS}")
print(f"torch available: {torch_available()}")
print(f"architecture: {architecture}")
print(f"agent_mode: {agent_mode}")
print(f"llm_provider: {llm_provider}")

# COMMAND ----------

if agent_mode == "llm" and llm_provider == "openai" and not os.environ.get("OPENAI_API_KEY"):
    if IN_DATABRICKS:
        try:
            os.environ["OPENAI_API_KEY"] = dbutils.secrets.get(secret_scope, "OPENAI_API_KEY")  # type: ignore[union-attr]
            print("OpenAI key source: Databricks secret scope")
        except Exception:
            print("OpenAI key unavailable in Databricks secrets; falling back to deterministic agent mode.")
            agent_mode = "deterministic"
    else:
        os.environ["OPENAI_API_KEY"] = getpass("Enter OPENAI_API_KEY for this session only: ")
        print("OpenAI key source: hidden local prompt/session env")

# COMMAND ----------

result = run_deep_learning_review(architecture=architecture, epochs=8, agent_mode=agent_mode)
print(f"Run ID: {result.run_id}")
print(f"Report: {result.report_path}")
print(f"Device: {result.device}")

# COMMAND ----------

display(Markdown("## Cohort metrics"))
display(result.metrics)

# COMMAND ----------

display(Markdown("## Evidence"))
for ev in result.evidence:
    display(Markdown(f"- **{ev.name}** — `{ev.status}` — {ev.summary} [{ev.evidence_id}]"))

# COMMAND ----------

display(Markdown("## Top Feature Attribution"))
display(result.attribution.head(20))

# COMMAND ----------

display(Markdown("## Sensitivity and Robustness"))
display(result.sensitivity)
display(result.noise_robustness)
display(result.masking_robustness)

# COMMAND ----------

display(Markdown("## Figures"))
for fig in result.figure_paths:
    p = Path(fig)
    display(Markdown(f"### {p.name}"))
    if p.exists():
        display(Image(filename=str(p)))
    else:
        display(Markdown(f"Figure not found: `{fig}`"))

# COMMAND ----------

display(Markdown("## Report Preview"))
report_path = Path(result.report_path)
if report_path.exists():
    display(Markdown(report_path.read_text()[:12000]))
else:
    display(Markdown("Report not found."))
