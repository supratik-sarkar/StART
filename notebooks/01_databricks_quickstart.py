# Databricks notebook source
# MAGIC %md
# MAGIC # StART on Databricks — thin orchestration notebook
# MAGIC Notebooks are orchestration layers only: all logic lives in the `start`
# MAGIC package. Install from the repo, detect the runtime, run the pipeline,
# MAGIC and (optionally) log to MLFlow.

# COMMAND ----------
# MAGIC %pip install git+https://github.com/supratik-sarkar/StART.git
# (On Databricks, prefer a cluster library or `%pip install /Workspace/Repos/.../StART` for repos.)

# COMMAND ----------
from start.providers.compute import detect_device, is_databricks_runtime, mlflow_available

print("Databricks runtime:", is_databricks_runtime())
print("Detected device (CUDA→MPS→CPU):", detect_device().value)
print("MLFlow available:", mlflow_available())

# COMMAND ----------
# MAGIC %md
# MAGIC ## Load data
# MAGIC On Databricks you would typically read a Delta table via Spark and convert
# MAGIC the (sampled) cohort to pandas for the deterministic engines. Locally this
# MAGIC notebook falls back to the toy generator — same code path, safe degradation.

# COMMAND ----------
try:
    spark  # noqa: B018  (defined by Databricks runtime)
    df = spark.table("samples.demo.propensity").limit(50_000).toPandas()  # type: ignore[name-defined]
except NameError:
    import sys
    sys.path.insert(0, "..")
    from examples.quickstart_local import make_toy_propensity

    df = make_toy_propensity()

# COMMAND ----------
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from start import build_context, load_config, run_review

config = load_config("../configs/databricks.yaml") if not is_databricks_runtime() else load_config("configs/databricks.yaml")
if not mlflow_available():
    config.experiment.provider = "local"  # safe degradation off-Databricks

train_df, test_df = train_test_split(df, test_size=0.3, random_state=config.seed, stratify=df["target"])
features = [c for c in df.columns if c != "target"]
model = LogisticRegression(max_iter=1000).fit(train_df[features].fillna(0), train_df["target"])
train_df = train_df.assign(score=model.predict_proba(train_df[features].fillna(0))[:, 1])
test_df = test_df.assign(score=model.predict_proba(test_df[features].fillna(0))[:, 1])

result = run_review(config, build_context(config, train_df, test_df, model=model))
print(result.run_id, "->", [(r.test_id, r.status.value) for r in result.evidence])

# COMMAND ----------
from start.reporting import render_markdown

print(render_markdown(result))
