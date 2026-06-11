# Databricks notebook source
# MAGIC %md
# MAGIC # StART — Propensity model review (thin orchestration notebook)
# MAGIC Notebooks orchestrate; the `start` package computes. This notebook runs the
# MAGIC same propensity-style review as `start propensity-demo`: 60/20/20
# MAGIC train/test/OOS split, feature engineering checks, model fit, cohort metrics,
# MAGIC explainability with honest SHAP fallback, top-feature shock sensitivity,
# MAGIC and the proof-carrying evidence pipeline. No core tests are defined here.

# COMMAND ----------
# MAGIC %pip install git+https://github.com/supratik-sarkar/StART.git
# (On Databricks, prefer a cluster library or `%pip install /Workspace/Repos/.../StART`.)

# COMMAND ----------
from start.providers.compute import detect_device, is_databricks_runtime, mlflow_available

print("Databricks runtime:", is_databricks_runtime())
print("Detected device (CUDA→MPS→CPU):", detect_device().value)
print("MLFlow available:", mlflow_available())

# COMMAND ----------
# MAGIC %md
# MAGIC ## Widgets
# MAGIC Everything is selectable inline: data source, model family, tuning, CV
# MAGIC strategy, and the sensitivity cohort. Outside Databricks the defaults apply.

# COMMAND ----------
WIDGET_DEFAULTS = {
    "data_source": "demo",          # demo | spark_table | files
    "spark_table": "samples.demo.attrition",
    "target_column": "attrition",
    "model_family": "random_forest",  # random_forest | xgboost | lightgbm
    "tuning": "none",                 # none | grid | random | optuna
    "cv_folds": "holdout",            # holdout | 3 | 5
    "sensitivity_cohort": "test",     # test | oos | development
    "secret_scope": "start",          # Databricks secret scope holding LLM keys
}
WIDGET_DROPDOWNS = {
    "agent_mode": ("deterministic", ["deterministic", "llm"]),
    "llm_provider": (
        "none",
        ["none", "enterprise_llm_gateway", "openai", "anthropic", "huggingface", "hf_local"],
    ),
    "run_agent_review": ("yes", ["yes", "no"]),
}
try:
    for key, default in WIDGET_DEFAULTS.items():
        dbutils.widgets.text(key, default, key)  # type: ignore[name-defined]
    for key, (default, choices) in WIDGET_DROPDOWNS.items():
        dbutils.widgets.dropdown(key, default, choices, key)  # type: ignore[name-defined]
    get_widget = lambda key: dbutils.widgets.get(key)  # type: ignore[name-defined]  # noqa: E731
except NameError:  # outside Databricks
    _ALL_DEFAULTS = {**WIDGET_DEFAULTS, **{k: v[0] for k, v in WIDGET_DROPDOWNS.items()}}
    get_widget = lambda key: _ALL_DEFAULTS[key]  # noqa: E731

# COMMAND ----------
# MAGIC %md
# MAGIC ## Data
# MAGIC On Databricks, read your scored or raw cohort from a Delta table and convert
# MAGIC to pandas. Outside Databricks this falls back to the public sklearn dataset —
# MAGIC the exact same downstream pipeline runs either way.

# COMMAND ----------
from start.connectors import DemoConnector, SparkConnector

target_column = get_widget("target_column")
if get_widget("data_source") == "spark_table":
    try:
        connector = SparkConnector(
            get_widget("spark_table"), spark=spark, target_column=target_column  # type: ignore[name-defined]
        )
    except NameError:
        print("No Spark runtime detected; falling back to the public demo dataset.")
        connector = DemoConnector(target_column=target_column)
else:
    connector = DemoConnector(target_column=target_column)

bundle = connector.load_bundle()
print(bundle.source, "| train/test/oos:", len(bundle.train), len(bundle.test), len(bundle.oos))

# COMMAND ----------
# MAGIC %md
# MAGIC ## Run the review
# MAGIC Same workflow object as the CLI/example; non-interactive, safe defaults.
# MAGIC MLFlow logging activates automatically when available via the experiment
# MAGIC provider (configure `experiment.provider: mlflow` in your config for full
# MAGIC tracking-server integration).

# COMMAND ----------
from start.modeling.propensity import PropensityOptions, run_propensity_demo

# LLM key resolution (LLM mode only): Databricks secret scope first
# (dbutils.secrets.get(scope, KEY_NAME)), environment second, deterministic
# fallback with an explicit warning last. Secrets never use visible widgets
# and are never printed to notebook output.
agent_mode = get_widget("agent_mode")
llm_provider = get_widget("llm_provider")
if agent_mode == "llm" and llm_provider not in ("none", "enterprise_llm_gateway", "hf_local"):
    from start.providers.keys import resolve_key_databricks

    try:
        _dbutils = dbutils  # type: ignore[name-defined]
    except NameError:
        _dbutils = None
    key_status = resolve_key_databricks(
        llm_provider, dbutils=_dbutils, scope=get_widget("secret_scope")
    )
    print(f"LLM key source for '{llm_provider}': {key_status.source}")  # source only, never the key
    if not key_status.ok:
        print(
            "WARNING: no key found in the secret scope or environment; "
            "the run will fall back to deterministic mode explicitly."
        )

cv = get_widget("cv_folds")
opts = PropensityOptions(
    model=get_widget("model_family"),
    tuning=get_widget("tuning"),
    cv_folds=None if cv == "holdout" else int(cv),
    sensitivity_cohort=get_widget("sensitivity_cohort"),
    target_column=target_column,
    agent_mode=agent_mode,
    llm_provider="" if llm_provider == "none" else llm_provider,
)
result = run_propensity_demo(opts)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Agent review (inline)
# MAGIC Same dual-mode agent review as `start agent-review` in the terminal:
# MAGIC deterministic governance fallback, or LLM-assisted evidence-grounded
# MAGIC review through the configured provider (including the generic
# MAGIC `enterprise_llm_gateway` placeholder). Every claim cites evidence IDs;
# MAGIC unsupported LLM output is rejected by the EvidenceCriticAgent.

# COMMAND ----------
if get_widget("run_agent_review") == "yes" and result.agent_review is not None:
    ar = result.agent_review
    print(f"Agent mode: {'llm-assisted (' + ar.llm_provider + ')' if ar.mode == 'llm' else 'deterministic'}")
    print(f"Evidence critique status: {'PASSED' if ar.critique_ok else 'FAILED'}")
    for note in ar.notes:
        print(f"NOTE: {note}")
    for title, items in (
        ("Review plan", ar.review_plan),
        ("Suggested next tests", ar.suggested_tests),
        ("Model-risk findings", ar.findings),
        ("Challenge memo", ar.challenge_memo),
        ("Missing evidence", ar.missing_evidence),
        ("Governance assessment", ar.governance),
    ):
        print(f"\n## {title}")
        for item in items:
            print(f"- {item}")
    print(f"\n## Sign-off recommendation\n{ar.signoff}")

# COMMAND ----------
from start.reporting import render_markdown

print(render_markdown(result))
