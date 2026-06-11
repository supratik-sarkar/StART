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
}
try:
    for key, default in WIDGET_DEFAULTS.items():
        dbutils.widgets.text(key, default, key)  # type: ignore[name-defined]
    get_widget = lambda key: dbutils.widgets.get(key)  # type: ignore[name-defined]  # noqa: E731
except NameError:  # outside Databricks
    get_widget = lambda key: WIDGET_DEFAULTS[key]  # noqa: E731

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

cv = get_widget("cv_folds")
opts = PropensityOptions(
    model=get_widget("model_family"),
    tuning=get_widget("tuning"),
    cv_folds=None if cv == "holdout" else int(cv),
    sensitivity_cohort=get_widget("sensitivity_cohort"),
    target_column=target_column,
)
result = run_propensity_demo(opts)

# COMMAND ----------
from start.reporting import render_markdown

print(render_markdown(result))
