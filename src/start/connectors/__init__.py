"""Universal data abstraction layer.

The public datasets are examples; user datasets are the product. Every
workflow consumes data through one interface — `DataConnector.load_bundle()`
returning a `DatasetBundle` — so the exact same review runs on:

  - demo:      public sklearn / synthetic datasets (default, demo-only path)
  - files:     local CSV / Parquet / Feather / Delta
  - pandas:    in-memory DataFrames (first-class Python API)
  - spark:     Spark DataFrames or table names (Databricks-friendly)
  - snowflake: generic, config-driven warehouse access (no vendor specifics
               beyond the public Snowflake connector; credentials via env)

If only a train source is supplied, a stratified 60/20/20 train/test/OOS
split is applied automatically. No framework internals need modification to
bring your own data.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

SUPPORTED_LOCAL_FORMATS = (".csv", ".parquet", ".pq", ".feather", ".ft")


@dataclass
class DatasetBundle:
    """Unified output of every connector: cohorts plus column metadata."""

    train: pd.DataFrame
    test: pd.DataFrame | None = None
    oos: pd.DataFrame | None = None
    source: str = "unknown"
    target_column: str | None = None
    score_column: str | None = None
    timestamp_column: str | None = None
    entity_id_column: str | None = None
    notes: list[str] = field(default_factory=list)

    def ensure_split(self, seed: int = 42) -> DatasetBundle:
        """Apply a stratified 60/20/20 split when only train was provided."""
        if self.test is not None:
            return self
        if self.target_column is None or self.target_column not in self.train.columns:
            raise ValueError(
                "A single dataset was provided without a usable target_column; "
                "cannot create a stratified train/test/OOS split."
            )
        from start.modeling.data import three_way_split

        train, test, oos = three_way_split(self.train, self.target_column, seed=seed)
        self.notes.append(
            "Single dataset supplied; applied stratified 60/20/20 train/test/OOS split."
        )
        return DatasetBundle(
            train=train,
            test=test,
            oos=oos,
            source=self.source,
            target_column=self.target_column,
            score_column=self.score_column,
            timestamp_column=self.timestamp_column,
            entity_id_column=self.entity_id_column,
            notes=self.notes,
        )


class DataConnector:
    """Base interface; subclasses implement load_bundle()."""

    name: str = "base"

    def load_bundle(self) -> DatasetBundle:  # pragma: no cover - interface
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Mode 1 — public demo datasets (the example path, never the product)
# --------------------------------------------------------------------------- #
class DemoConnector(DataConnector):
    name = "demo"

    def __init__(self, dataset: str = "attrition", seed: int = 42, **meta: Any) -> None:
        self.dataset = dataset
        self.seed = seed
        self.meta = meta

    def load_bundle(self) -> DatasetBundle:
        from start.modeling.data import TARGET_COLUMN, load_attrition_dataset

        if self.dataset == "synthetic":
            from start.modeling.data import _synthetic_fallback

            df = _synthetic_fallback(self.seed)
        else:
            df = load_attrition_dataset(seed=self.seed)
        bundle = DatasetBundle(
            train=df,
            source=f"demo:{self.dataset}",
            target_column=self.meta.get("target_column") or TARGET_COLUMN,
            timestamp_column=self.meta.get("timestamp_column"),
            entity_id_column=self.meta.get("entity_id_column"),
            notes=["Public demo dataset; substitute your own data via any connector."],
        )
        return bundle.ensure_split(self.seed)


# --------------------------------------------------------------------------- #
# Mode 2 — local files (CSV / Parquet / Feather / Delta)
# --------------------------------------------------------------------------- #
def load_local_file(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.is_dir():  # Delta tables are directories
        return _read_delta(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".feather", ".ft"}:
        return pd.read_feather(path)
    raise ValueError(
        f"Unsupported file type '{suffix}' for {path}. "
        f"Supported: {', '.join(SUPPORTED_LOCAL_FORMATS)} or a Delta table directory."
    )


def _read_delta(path: Path) -> pd.DataFrame:
    try:
        from deltalake import DeltaTable  # optional

        return DeltaTable(str(path)).to_pandas()
    except ImportError as exc:
        raise ImportError(
            "Reading Delta tables locally requires the 'deltalake' package "
            "(pip install deltalake); on Databricks use the spark connector instead."
        ) from exc


class LocalFileConnector(DataConnector):
    name = "files"

    def __init__(
        self,
        train: str | Path,
        test: str | Path | None = None,
        oos: str | Path | None = None,
        seed: int = 42,
        **meta: Any,
    ) -> None:
        self.train, self.test, self.oos = train, test, oos
        self.seed = seed
        self.meta = meta

    def load_bundle(self) -> DatasetBundle:
        bundle = DatasetBundle(
            train=load_local_file(self.train),
            test=load_local_file(self.test) if self.test else None,
            oos=load_local_file(self.oos) if self.oos else None,
            source=f"files:{self.train}",
            **{k: self.meta.get(k) for k in (
                "target_column", "score_column", "timestamp_column", "entity_id_column"
            )},
        )
        return bundle.ensure_split(self.seed)


# --------------------------------------------------------------------------- #
# Mode 3 — in-memory pandas DataFrames (first-class API)
# --------------------------------------------------------------------------- #
class PandasConnector(DataConnector):
    name = "pandas"

    def __init__(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame | None = None,
        oos_df: pd.DataFrame | None = None,
        seed: int = 42,
        **meta: Any,
    ) -> None:
        self.train_df, self.test_df, self.oos_df = train_df, test_df, oos_df
        self.seed = seed
        self.meta = meta

    def load_bundle(self) -> DatasetBundle:
        bundle = DatasetBundle(
            train=self.train_df,
            test=self.test_df,
            oos=self.oos_df,
            source="pandas:in-memory",
            **{k: self.meta.get(k) for k in (
                "target_column", "score_column", "timestamp_column", "entity_id_column"
            )},
        )
        return bundle.ensure_split(self.seed)


# --------------------------------------------------------------------------- #
# Mode 4 — Spark DataFrames / tables (Databricks-friendly)
# --------------------------------------------------------------------------- #
class SparkDataFrameAdapter:
    """Standardizes Spark -> pandas conversion with a row-limit guard so a
    notebook user can hand over `spark.table(...)` directly."""

    def __init__(self, max_rows: int = 1_000_000) -> None:
        self.max_rows = max_rows

    def to_pandas(self, obj: Any, spark: Any = None) -> pd.DataFrame:
        sdf = obj
        if isinstance(obj, str):  # table name or SQL
            if spark is None:
                raise ValueError("A SparkSession is required to resolve table/SQL strings.")
            sdf = spark.sql(obj) if obj.lstrip().lower().startswith("select") else spark.table(obj)
        if hasattr(sdf, "toPandas"):
            if hasattr(sdf, "limit"):
                sdf = sdf.limit(self.max_rows)
            return sdf.toPandas()
        raise TypeError(f"Object of type {type(obj).__name__} is not a Spark DataFrame.")


class SparkConnector(DataConnector):
    name = "spark"

    def __init__(
        self,
        train: Any,
        test: Any = None,
        oos: Any = None,
        spark: Any = None,
        max_rows: int = 1_000_000,
        seed: int = 42,
        **meta: Any,
    ) -> None:
        self.train, self.test, self.oos = train, test, oos
        self.spark = spark
        self.adapter = SparkDataFrameAdapter(max_rows)
        self.seed = seed
        self.meta = meta

    def load_bundle(self) -> DatasetBundle:
        convert = lambda obj: self.adapter.to_pandas(obj, self.spark) if obj is not None else None  # noqa: E731
        bundle = DatasetBundle(
            train=convert(self.train),
            test=convert(self.test),
            oos=convert(self.oos),
            source="spark",
            notes=[f"Spark sources converted to pandas (row guard: {self.adapter.max_rows})."],
            **{k: self.meta.get(k) for k in (
                "target_column", "score_column", "timestamp_column", "entity_id_column"
            )},
        )
        return bundle.ensure_split(self.seed)


# --------------------------------------------------------------------------- #
# Mode 5 — Snowflake (generic, config-driven; credentials via environment)
# --------------------------------------------------------------------------- #
class SnowflakeConnector(DataConnector):
    """Generic warehouse access through the public Snowflake connector.

    Connection parameters come from config (database/schema/table or query)
    and credentials from standard environment variables (SNOWFLAKE_ACCOUNT,
    SNOWFLAKE_USER, SNOWFLAKE_PASSWORD / SNOWFLAKE_AUTHENTICATOR, ...).
    No organization-specific endpoints, schemas, or policies live here.
    """

    name = "snowflake"

    def __init__(
        self,
        database: str | None = None,
        schema: str | None = None,
        table: str | None = None,
        query: str | None = None,
        warehouse: str | None = None,
        role: str | None = None,
        seed: int = 42,
        **meta: Any,
    ) -> None:
        self.database, self.schema, self.table = database, schema, table
        self.query, self.warehouse, self.role = query, warehouse, role
        self.seed = seed
        self.meta = meta

    def _build_query(self) -> str:
        if self.query:
            return self.query
        if not (self.database and self.schema and self.table):
            raise ValueError(
                "Snowflake source requires either an explicit query or "
                "database + schema + table in the data config."
            )
        return f'SELECT * FROM "{self.database}"."{self.schema}"."{self.table}"'

    def load_bundle(self) -> DatasetBundle:
        try:
            import snowflake.connector  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(
                "Snowflake access requires the optional driver: "
                'pip install "snowflake-connector-python[pandas]". '
                "Credentials are read from SNOWFLAKE_* environment variables."
            ) from exc

        conn = snowflake.connector.connect(
            account=os.environ.get("SNOWFLAKE_ACCOUNT"),
            user=os.environ.get("SNOWFLAKE_USER"),
            password=os.environ.get("SNOWFLAKE_PASSWORD"),
            authenticator=os.environ.get("SNOWFLAKE_AUTHENTICATOR", "snowflake"),
            warehouse=self.warehouse or os.environ.get("SNOWFLAKE_WAREHOUSE"),
            role=self.role or os.environ.get("SNOWFLAKE_ROLE"),
        )
        try:
            cursor = conn.cursor()
            cursor.execute(self._build_query())
            df = cursor.fetch_pandas_all()
        finally:
            conn.close()
        bundle = DatasetBundle(
            train=df,
            source=f"snowflake:{self.table or 'query'}",
            **{k: self.meta.get(k) for k in (
                "target_column", "score_column", "timestamp_column", "entity_id_column"
            )},
        )
        return bundle.ensure_split(self.seed)


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def resolve_connector(
    data_config: Any,
    *,
    train_df: pd.DataFrame | None = None,
    test_df: pd.DataFrame | None = None,
    oos_df: pd.DataFrame | None = None,
    spark: Any = None,
    seed: int = 42,
) -> DataConnector:
    """Build a connector from a StartConfig.data block (plus optional
    in-memory frames / Spark session supplied at runtime)."""
    meta = {
        "target_column": getattr(data_config, "target_column", None),
        "score_column": getattr(data_config, "score_column", None),
        "timestamp_column": getattr(data_config, "timestamp_column", None),
        "entity_id_column": getattr(data_config, "entity_id_column", None),
    }
    source = getattr(data_config, "source", "demo")
    ds = getattr(data_config, "dataset", None)
    if train_df is not None or source == "pandas":
        if train_df is None:
            raise ValueError("source 'pandas' requires DataFrames passed at runtime.")
        return PandasConnector(train_df, test_df, oos_df, seed=seed, **meta)
    if source == "demo":
        return DemoConnector(getattr(data_config, "demo_dataset", "attrition"), seed=seed, **meta)
    if source == "files":
        if ds is None or not ds.train:
            raise ValueError("source 'files' requires data.dataset.train in the config.")
        return LocalFileConnector(ds.train, ds.test, ds.oos, seed=seed, **meta)
    if source == "spark":
        if ds is None or not ds.train:
            raise ValueError("source 'spark' requires data.dataset.train (table or SQL).")
        return SparkConnector(
            ds.train, ds.test, ds.oos, spark=spark,
            max_rows=getattr(data_config, "spark_max_rows", 1_000_000), seed=seed, **meta,
        )
    if source == "snowflake":
        sf = getattr(data_config, "snowflake", None)
        return SnowflakeConnector(
            database=getattr(sf, "database", None),
            schema=getattr(sf, "db_schema", None),
            table=getattr(sf, "table", None),
            query=getattr(sf, "query", None),
            warehouse=getattr(sf, "warehouse", None),
            role=getattr(sf, "role", None),
            seed=seed,
            **meta,
        )
    raise ValueError(f"Unknown data source '{source}'.")
