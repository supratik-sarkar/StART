# Databricks notebook source
# MAGIC %md
# MAGIC # StART — Deep Learning Model Review (agentic governance)
# MAGIC Real laptop-safe PyTorch model -> evidence pipeline (EV-DL-0001..0007)
# MAGIC -> dual-mode agent review -> figures -> proof-carrying report.
# MAGIC
# MAGIC Runs identically:
# MAGIC * **Locally** in VS Code / Jupyter (kernel: `Python (StART .venv-start)`),
# MAGIC   or as a script: `python notebooks/03_deep_learning_model_review.py`
# MAGIC * **On Databricks** with widgets for architecture, training, data source,
# MAGIC   agent mode, and secret-scope keys.
# MAGIC
# MAGIC The LLM (if enabled) reasons only over the evidence bundle and never sees
# MAGIC raw data. Default mode is deterministic and needs no key.

# COMMAND ----------
import os

WIDGET_TEXT = {
    "architecture": "mlp",            # mlp | leaky_relu_mlp | residual_mlp | wide_deep
    "epochs": "8",                    # laptop-safe: <= 10
    "batch_size": "128",              # laptop-safe: <= 128
    "learning_rate": "0.001",
    "target_column": "attrition",
    "secret_scope": "start",
}
WIDGET_DROPDOWN = {
    "dataset_source": ("demo", ["demo", "files", "spark_table"]),
    "agent_mode": ("deterministic", ["deterministic", "llm"]),
    "llm_provider": ("none", ["none", "openai", "anthropic", "enterprise_llm_gateway"]),
}
try:
    for key, default in WIDGET_TEXT.items():
        dbutils.widgets.text(key, default, key)  # type: ignore[name-defined]
    for key, (default, choices) in WIDGET_DROPDOWN.items():
        dbutils.widgets.dropdown(key, default, choices, key)  # type: ignore[name-defined]
    get_widget = lambda k: dbutils.widgets.get(k)  # type: ignore[name-defined]  # noqa: E731
    ON_DATABRICKS = True
except NameError:  # local VS Code / Jupyter / plain python
    _ALL = {**WIDGET_TEXT, **{k: v[0] for k, v in WIDGET_DROPDOWN.items()}}
    get_widget = lambda k: os.environ.get(f"START_NB_{k.upper()}", _ALL[k])  # noqa: E731
    ON_DATABRICKS = False

FAST = bool(os.environ.get("START_NB_FAST"))  # smoke mode: fewer epochs

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1. Environment and device

# COMMAND ----------
from start.modeling.deep_learning import captum_available, torch_available

print(f"Databricks runtime: {ON_DATABRICKS}")
print(f"torch available:    {torch_available()}")
print(f"captum available:   {captum_available()}")
if not torch_available():
    print("\nDeep learning requires the torch extra: pip install -e \".[torch]\"")
    raise SystemExit(0)

from start.modeling.deep_learning import resolve_torch_device

print(f"device (CUDA -> MPS -> CPU): {resolve_torch_device()}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2. Review options (from widgets / environment)

# COMMAND ----------
from start.modeling.dl_training import DLReviewOptions

agent_mode = get_widget("agent_mode")
llm_provider = get_widget("llm_provider")
opts = DLReviewOptions(
    architecture=get_widget("architecture"),
    epochs=3 if FAST else min(int(get_widget("epochs")), 10),
    batch_size=min(int(get_widget("batch_size")), 128),
    learning_rate=float(get_widget("learning_rate")),
    target_column=get_widget("target_column"),
    data_source="demo" if get_widget("dataset_source") == "demo" else get_widget("dataset_source"),
    agent_mode=agent_mode,
    llm_provider="" if llm_provider == "none" else llm_provider,
)
print(f"architecture={opts.architecture} | agent_mode={agent_mode} | provider={llm_provider}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 3. Secure LLM key (LLM mode only)
# MAGIC Local: hidden `getpass` prompt (session-only, never persisted/echoed).
# MAGIC Databricks: secret scope -> environment -> deterministic fallback. Only
# MAGIC the key *source* is ever printed.

# COMMAND ----------
if agent_mode == "llm" and llm_provider not in ("none", "enterprise_llm_gateway"):
    if ON_DATABRICKS:
        from start.providers.keys import resolve_key_databricks

        key_status = resolve_key_databricks(
            llm_provider, dbutils=dbutils, scope=get_widget("secret_scope")  # type: ignore[name-defined]
        )
    else:
        from start.providers.keys import ensure_provider_key

        key_status = ensure_provider_key(llm_provider, prompt_for_key=None)
    print(f"LLM key source for '{llm_provider}': {key_status.source}")
    if not key_status.ok:
        print("WARNING: no key available; falling back to deterministic mode explicitly.")
        opts.agent_mode = "deterministic"
        opts.llm_provider = ""

# COMMAND ----------
# MAGIC %md
# MAGIC ## 4. Run the full review (train -> evidence -> agent review -> figures -> report)

# COMMAND ----------
from start.modeling.dl_training import run_dl_review

result = run_dl_review(opts)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 5. Findings — evidence IDs, challenge memo, governance, sign-off

# COMMAND ----------
for rec in result.evidence:
    label = rec.artifacts.get("dl_evidence_label", rec.evidence_id)
    print(f"[{label}] {rec.status.value.upper():7s} {rec.test_name}")

ar = result.agent_review
print(f"\nAgent mode: {'llm-assisted (' + ar.llm_provider + ')' if ar.mode == 'llm' else 'deterministic'}")
print(f"Evidence critique status: {'PASSED' if ar.critique_ok else 'FAILED'}")
for note in ar.notes:
    print(f"NOTE: {note}")
for title, items in (
    ("Reviewer summary / plan", ar.review_plan),
    ("Challenger memo", ar.challenge_memo),
    ("Governance assessment", ar.governance),
):
    print(f"\n## {title}")
    for item in items:
        print(f"- {item}")
print(f"\n## Sign-off recommendation\n{ar.signoff}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 6. Figures and report

# COMMAND ----------
print("Figures:")
for name, path in sorted(result.figures.items()):
    print(f"- {name}: {path}")
print(f"\nReport: {result.report_path}")

# On Databricks, render figures inline, e.g.:
#   from PIL import Image
#   display(Image.open(result.figures["learning_curve"]))
