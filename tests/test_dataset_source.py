from __future__ import annotations

import pandas as pd

from start.modeling.data import load_attrition_dataset
from start.modeling.dataset_source import (
    describe_custom_dataset,
    describe_demo_dataset,
    frame_hash,
    render_dataset_source_markdown,
)


def test_demo_dataset_has_public_url():
    df = load_attrition_dataset(seed=0)
    src = describe_demo_dataset(df, "attrition")
    assert src.kind in ("builtin_demo", "synthetic_fallback")
    if src.kind == "builtin_demo":
        assert src.public_url and src.public_url.startswith("https://")
        assert "breast" in src.name.lower()
        assert src.reason_selected and src.task_suitability
        assert src.loading_route == "sklearn.datasets.load_breast_cancer"


def test_frame_hash_stable_and_sensitive():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    h1 = frame_hash(df)
    h2 = frame_hash(df.copy())
    assert h1 == h2  # stable
    df2 = df.copy()
    df2.loc[0, "a"] = 99
    assert frame_hash(df2) != h1  # sensitive to content


def test_custom_dataset_provenance(tmp_path):
    df = pd.DataFrame({"x": [1, 2, 3], "y": [0, 1, 0]})
    p = tmp_path / "data.csv"
    df.to_csv(p, index=False)
    src = describe_custom_dataset(df, str(p), "y")
    assert src.kind == "custom"
    assert src.detected_format == "csv"
    assert "read_csv" in src.loading_route
    assert src.file_path == str(p)
    assert src.data_hash


def test_custom_format_detection(tmp_path):
    df = pd.DataFrame({"x": [1, 2]})
    for ext, expected in (("parquet", "read_parquet"), ("json", "read_json")):
        p = tmp_path / f"d.{ext}"
        p.write_text("x")  # content irrelevant for provenance
        src = describe_custom_dataset(df, str(p))
        assert src.detected_format == ext
        assert expected in src.loading_route


def test_markdown_includes_url_for_demo():
    df = load_attrition_dataset(seed=0)
    md = render_dataset_source_markdown(describe_demo_dataset(df, "attrition"))
    assert "### Dataset source" in md
    assert "Data hash" in md


def test_to_dict_complete():
    df = load_attrition_dataset(seed=0)
    d = describe_demo_dataset(df, "attrition").to_dict()
    for key in (
        "kind",
        "name",
        "public_url",
        "reason_selected",
        "task_suitability",
        "loading_route",
        "data_hash",
    ):
        assert key in d
