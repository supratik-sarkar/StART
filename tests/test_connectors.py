from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from start.connectors import (
    DemoConnector,
    LocalFileConnector,
    PandasConnector,
    SnowflakeConnector,
    SparkDataFrameAdapter,
    load_local_file,
    resolve_connector,
)
from start.modeling.data import TARGET_COLUMN, load_attrition_dataset


@pytest.fixture(scope="module")
def demo_df() -> pd.DataFrame:
    return load_attrition_dataset(seed=3)


def _pyarrow_available() -> bool:
    try:
        import pyarrow  # noqa: F401

        return True
    except ImportError:
        return False


def test_local_file_roundtrip_csv_parquet_feather(tmp_path, demo_df):
    formats = [(".csv", lambda df, p: df.to_csv(p, index=False))]
    if _pyarrow_available():
        formats += [
            (".parquet", lambda df, p: df.to_parquet(p)),
            (".feather", lambda df, p: df.to_feather(p)),
        ]
    for suffix, writer in formats:
        path = tmp_path / f"data{suffix}"
        writer(demo_df, path)
        loaded = load_local_file(path)
        assert loaded.shape == demo_df.shape, suffix


def test_local_file_unsupported_suffix(tmp_path):
    bad = tmp_path / "data.xlsx"
    bad.write_text("x")
    with pytest.raises(ValueError, match="Unsupported file type"):
        load_local_file(bad)


def test_local_connector_single_file_auto_splits(tmp_path, demo_df):
    path = tmp_path / "all.csv"
    demo_df.to_csv(path, index=False)
    bundle = LocalFileConnector(path, target_column=TARGET_COLUMN, seed=1).load_bundle()
    assert bundle.test is not None and bundle.oos is not None
    total = len(bundle.train) + len(bundle.test) + len(bundle.oos)
    assert total == len(demo_df)
    assert abs(len(bundle.train) / total - 0.6) < 0.02
    assert any("60/20/20" in note for note in bundle.notes)


@pytest.mark.skipif(not _pyarrow_available(), reason="pyarrow required for parquet")
def test_local_connector_explicit_three_files(tmp_path, demo_df):
    parts = {
        "train": demo_df.iloc[:300],
        "test": demo_df.iloc[300:450],
        "oos": demo_df.iloc[450:],
    }
    paths = {}
    for name, frame in parts.items():
        paths[name] = tmp_path / f"{name}.parquet"
        frame.to_parquet(paths[name])
    bundle = LocalFileConnector(
        paths["train"], paths["test"], paths["oos"], target_column=TARGET_COLUMN
    ).load_bundle()
    assert len(bundle.train) == 300 and len(bundle.test) == 150
    assert bundle.notes == []  # no auto-split applied


def test_pandas_connector_passthrough_and_split(demo_df):
    bundle = PandasConnector(demo_df, target_column=TARGET_COLUMN, seed=2).load_bundle()
    assert bundle.source == "pandas:in-memory"
    assert bundle.oos is not None
    # explicit cohorts are passed through untouched
    b2 = PandasConnector(
        demo_df.iloc[:400], demo_df.iloc[400:500], demo_df.iloc[500:], target_column=TARGET_COLUMN
    ).load_bundle()
    assert len(b2.train) == 400 and len(b2.oos) == len(demo_df) - 500


def test_split_without_target_raises(demo_df):
    with pytest.raises(ValueError, match="target_column"):
        PandasConnector(demo_df.drop(columns=[TARGET_COLUMN])).load_bundle()


class _FakeSparkDF:
    """Duck-typed Spark DataFrame: limit().toPandas()."""

    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df
        self.limited: int | None = None

    def limit(self, n: int) -> _FakeSparkDF:
        self.limited = n
        return _FakeSparkDF(self._df.head(n))

    def toPandas(self) -> pd.DataFrame:  # noqa: N802 - Spark API name
        return self._df.copy()


def test_spark_adapter_converts_with_row_guard(demo_df):
    adapter = SparkDataFrameAdapter(max_rows=100)
    out = adapter.to_pandas(_FakeSparkDF(demo_df))
    assert len(out) == 100  # guard applied
    with pytest.raises(TypeError, match="not a Spark DataFrame"):
        adapter.to_pandas(object())
    with pytest.raises(ValueError, match="SparkSession"):
        adapter.to_pandas("some.table")  # string without a session


def test_snowflake_connector_guidance_when_driver_missing(monkeypatch):
    connector = SnowflakeConnector(database="DB", schema="SCH", table="T")
    assert 'FROM "DB"."SCH"."T"' in connector._build_query()
    try:
        import snowflake.connector  # noqa: F401

        pytest.skip("snowflake driver installed; missing-driver path not testable here")
    except ImportError:
        with pytest.raises(ImportError, match="snowflake-connector-python"):
            connector.load_bundle()


def test_snowflake_requires_coordinates_or_query():
    with pytest.raises(ValueError, match="query or"):
        SnowflakeConnector()._build_query()


def test_resolve_connector_modes(tmp_path, demo_df):
    from start.core.config import StartConfig

    cfg = StartConfig()
    assert isinstance(resolve_connector(cfg.data), DemoConnector)
    assert isinstance(resolve_connector(cfg.data, train_df=demo_df), PandasConnector)
    cfg.data.source = "files"
    path = tmp_path / "t.csv"
    demo_df.head(50).to_csv(path, index=False)
    cfg.data.dataset.train = str(path)
    assert isinstance(resolve_connector(cfg.data), LocalFileConnector)
    cfg.data.source = "snowflake"
    assert isinstance(resolve_connector(cfg.data), SnowflakeConnector)


def test_review_dataframes_first_class_api(tmp_path, monkeypatch, demo_df):
    """The pandas API runs the full review on in-memory frames, scoring with
    a user-supplied model, and produces evidence + ledger without any files."""
    monkeypatch.chdir(tmp_path)
    from sklearn.ensemble import RandomForestClassifier

    from start.modeling.data import SCORE_COLUMN, feature_columns, three_way_split
    from start.orchestration import review_dataframes

    train, test, oos = three_way_split(demo_df, TARGET_COLUMN, seed=0)
    features = feature_columns(train)
    model = RandomForestClassifier(n_estimators=60, random_state=0, n_jobs=-1)
    model.fit(train[features], train[TARGET_COLUMN])
    for frame in (train, test, oos):
        frame[SCORE_COLUMN] = model.predict_proba(frame[features])[:, 1]

    result = review_dataframes(
        train, test, oos, target_column=TARGET_COLUMN, model=model, seed=0
    )
    test_ids = {rec.test_id for rec in result.evidence}
    assert "supervised.cohort_metrics_comparison" in test_ids
    assert "xai.feature_sensitivity" in test_ids
    rec = next(r for r in result.evidence if r.test_id == "supervised.cohort_metrics_comparison")
    assert "oos_auc_roc" in rec.metrics  # OOS cohort flowed through ctx.extra
    assert (Path("start_output") / "ledger.jsonl").exists()
    assert result.critique is not None and result.critique.ok


def test_propensity_cli_with_user_files(tmp_path, monkeypatch, demo_df):
    from typer.testing import CliRunner

    from start.cli import app

    monkeypatch.chdir(tmp_path)
    renamed = demo_df.rename(columns={TARGET_COLUMN: "churned"})
    renamed.to_csv(tmp_path / "clients.csv", index=False)
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "propensity-demo",
            "--non-interactive",
            "--train",
            "clients.csv",
            "--target",
            "churned",
            "--seed",
            "7",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "files:clients.csv" in result.output
    assert (Path("start_output") / "ledger.jsonl").exists()
