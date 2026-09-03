from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from start.connectors import DemoConnector
from start.modeling.data import load_attrition_dataset
from start.modeling.deep_learning import TorchMLPClassifier
from start.modeling.model_execution import _stratified_split
from start.modeling.sequence_dl import SequenceClassifier
from start.modeling.tabular_dl import TabularDLClassifier
from start.modeling.vision_dl import VisionCNNClassifier

torch = pytest.importorskip("torch", reason="Interactive v2.4.0 tests require the [torch] extra")


def test_custom_splits_stratified_vs_random():
    df = load_attrition_dataset(seed=42)
    # 1. Stratified split
    splits_strat = _stratified_split(df, "attrition", (0.6, 0.2, 0.2), seed=42, stratify=True)
    assert len(splits_strat["train"]) + len(splits_strat["test"]) + len(splits_strat["oos"]) == len(df)

    # Check that positive rates are very close (stratified)
    tr_rate = splits_strat["train"]["attrition"].mean()
    te_rate = splits_strat["test"]["attrition"].mean()
    assert abs(tr_rate - te_rate) < 0.05

    # 2. Random split
    splits_rand = _stratified_split(df, "attrition", (0.6, 0.2, 0.2), seed=42, stratify=False)
    assert len(splits_rand["train"]) + len(splits_rand["test"]) + len(splits_rand["oos"]) == len(df)


def test_connector_target_renaming():
    # If the user selects a target name other than "attrition" (e.g. "decision_label"),
    # DemoConnector should automatically rename it in the loaded DataFrame and succeed.
    connector = DemoConnector(seed=42, target_column="decision_label")
    bundle = connector.load_bundle()
    assert bundle.target_column == "decision_label"
    assert "decision_label" in bundle.train.columns
    assert "attrition" not in bundle.train.columns


def test_torch_mlp_class_weight():
    df = load_attrition_dataset(seed=42)
    # Create highly imbalanced subset
    df_imbalanced = pd.concat(
        [
            df[df["attrition"] == 0].sample(200, random_state=42),
            df[df["attrition"] == 1].sample(10, random_state=42),
        ]
    )
    features = [c for c in df_imbalanced.columns if c != "attrition"]

    # Without class weights
    clf_no_weight = TorchMLPClassifier(epochs=5, random_state=42, class_weight=None)
    clf_no_weight.fit(df_imbalanced[features], df_imbalanced["attrition"])

    # With class weights
    clf_weighted = TorchMLPClassifier(epochs=5, random_state=42, class_weight="balanced")
    clf_weighted.fit(df_imbalanced[features], df_imbalanced["attrition"])

    assert clf_weighted.class_weight == "balanced"


def test_tabular_dl_class_weight():
    df = load_attrition_dataset(seed=42)
    features = [c for c in df.columns if c != "attrition"]

    # Binary
    clf_bin = TabularDLClassifier(
        task="binary_classification", epochs=3, random_state=42, class_weight="balanced"
    )
    clf_bin.fit(df[features], df["attrition"])
    assert clf_bin.class_weight == "balanced"

    # Multiclass
    df_multi = df.copy()
    df_multi["target_multi"] = np.random.choice([0, 1, 2], size=len(df))
    clf_multi = TabularDLClassifier(
        task="multiclass_classification", epochs=3, random_state=42, class_weight="balanced"
    )
    clf_multi.fit(df_multi[features], df_multi["target_multi"])
    assert clf_multi.class_weight == "balanced"


def test_sequence_and_vision_2d_reshaping():
    df = load_attrition_dataset(seed=42)
    features = [c for c in df.columns if c != "attrition"][:5]  # use small subset
    X = df[features].to_numpy()
    y = df["attrition"].to_numpy()

    # Sequence Classifier should accept 2D inputs and fit successfully
    clf_seq = SequenceClassifier(epochs=3, random_state=42, class_weight="balanced")
    clf_seq.fit(X, y)
    probs_seq = clf_seq.predict_proba(X)
    assert probs_seq.shape == (len(X), 2)

    # Vision Classifier should accept 2D inputs and fit successfully
    clf_vis = VisionCNNClassifier(epochs=3, random_state=42, class_weight="balanced")
    clf_vis.fit(X, y)
    probs_vis = clf_vis.predict_proba(X)
    assert probs_vis.shape == (len(X), 2)
