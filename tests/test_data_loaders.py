from __future__ import annotations

import pandas as pd
import pytest

from start.data.loaders import (
    SUPPORTED_TABULAR_FORMATS,
    discover_image_folder,
    load_any_tabular,
    sniff_format,
)


@pytest.fixture()
def frame() -> pd.DataFrame:
    return pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"], "target": [0, 1, 0]})


def test_csv_tsv_txt(tmp_path, frame):
    frame.to_csv(tmp_path / "a.csv", index=False)
    frame.to_csv(tmp_path / "a.tsv", sep="\t", index=False)
    frame.to_csv(tmp_path / "a.txt", sep="|", index=False)
    assert load_any_tabular(tmp_path / "a.csv").shape == (3, 3)
    assert load_any_tabular(tmp_path / "a.tsv").shape == (3, 3)
    assert load_any_tabular(tmp_path / "a.txt").shape == (3, 3)


def test_json_and_jsonl(tmp_path, frame):
    frame.to_json(tmp_path / "a.json")
    frame.to_json(tmp_path / "a.jsonl", orient="records", lines=True)
    assert set(load_any_tabular(tmp_path / "a.json").columns) == {"a", "b", "target"}
    assert load_any_tabular(tmp_path / "a.jsonl").shape == (3, 3)


def test_excel(tmp_path, frame):
    pytest.importorskip("openpyxl")
    frame.to_excel(tmp_path / "a.xlsx", index=False)
    assert load_any_tabular(tmp_path / "a.xlsx").shape == (3, 3)


def test_parquet_feather(tmp_path, frame):
    pytest.importorskip("pyarrow")
    frame.to_parquet(tmp_path / "a.parquet")
    frame.to_feather(tmp_path / "a.feather")
    assert load_any_tabular(tmp_path / "a.parquet").shape == (3, 3)
    assert load_any_tabular(tmp_path / "a.feather").shape == (3, 3)


def test_pickle_is_gated(tmp_path, frame):
    frame.to_pickle(tmp_path / "a.pkl")
    with pytest.raises(ValueError, match="executes arbitrary code"):
        load_any_tabular(tmp_path / "a.pkl")
    assert load_any_tabular(tmp_path / "a.pkl", allow_pickle=True).shape == (3, 3)


def test_unsupported_format(tmp_path):
    (tmp_path / "a.xyz").write_text("nope")
    with pytest.raises(ValueError, match="Unsupported tabular format"):
        load_any_tabular(tmp_path / "a.xyz")
    assert ".jsonl" in SUPPORTED_TABULAR_FORMATS


def test_image_folder_discovery(tmp_path):
    for cls in ("cat", "dog"):
        (tmp_path / cls).mkdir()
        for i in range(3):
            (tmp_path / cls / f"{cls}{i}.png").write_bytes(b"img")
    assert sniff_format(tmp_path) == "image_folder"
    manifest = discover_image_folder(tmp_path)
    assert manifest.shape == (6, 2)
    assert sorted(manifest["label"].unique()) == ["cat", "dog"]
    assert all(manifest["image_path"].str.endswith(".png"))


def test_image_folder_empty_raises(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="No images found"):
        discover_image_folder(tmp_path)


def test_sniff_tabular_and_unknown(tmp_path, frame):
    frame.to_csv(tmp_path / "a.csv", index=False)
    assert sniff_format(tmp_path / "a.csv") == "tabular"
    (tmp_path / "weird.xyz").write_text("x")
    assert sniff_format(tmp_path / "weird.xyz") == "unknown"


def test_connector_load_local_file_uses_new_loader(tmp_path, frame):
    from start.connectors import load_local_file

    frame.to_json(tmp_path / "a.jsonl", orient="records", lines=True)
    assert load_local_file(tmp_path / "a.jsonl").shape == (3, 3)
