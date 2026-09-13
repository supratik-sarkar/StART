from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch", reason="Layer 7 requires the [torch] extra")

from start.modeling.vision_data import (  # noqa: E402
    discover_labels,
    generate_image_dataset,
    load_vision_demo,
    normalize_images,
)
from start.modeling.vision_dl import (  # noqa: E402
    VisionCNNClassifier,
    vision_explainability,
    vision_metrics,
    vision_robustness,
)
from start.modeling.vision_models import (  # noqa: E402
    CNN_PRESETS,
    build_vision_network,
    config_from_preset,
    torchvision_available,
)


@pytest.fixture(scope="module")
def vbundle():
    return load_vision_demo(n_classes=3, image_size=16, channels=3, seed=0)


def test_generate_image_dataset_shapes():
    X, y, names = generate_image_dataset(n_per_class=40, n_classes=3, image_size=16, channels=3, seed=0)
    assert X.shape == (120, 3, 16, 16)
    assert len(names) == 3
    assert set(np.unique(y)) == {0, 1, 2}
    assert X.min() >= 0.0 and X.max() <= 1.0


def test_split_images_stratified(vbundle):
    assert vbundle.shapes["train"][0] + vbundle.shapes["test"][0] + vbundle.shapes["oos"][0] == 450
    assert vbundle.n_classes == 3
    # each class present in train
    assert len(np.unique(vbundle.y_train)) == 3


def test_discover_labels(tmp_path):
    for cls in ("cat", "dog", "bird"):
        (tmp_path / cls).mkdir()
        (tmp_path / cls / "x.png").write_bytes(b"img")
    assert discover_labels(tmp_path) == ["bird", "cat", "dog"]
    with pytest.raises(ValueError, match="No class subdirectories"):
        empty = tmp_path / "empty"
        empty.mkdir()
        discover_labels(empty)


def test_normalize_images():
    X = np.ones((2, 3, 8, 8), dtype=np.float32)
    out = normalize_images(X, mean=0.5, std=0.5)
    assert np.allclose(out, 1.0)


@pytest.mark.parametrize("preset", list(CNN_PRESETS))
def test_each_cnn_preset_trains(preset, vbundle):
    clf = VisionCNNClassifier(architecture=preset, epochs=6, learning_rate=3e-3, random_state=0)
    clf.fit(vbundle.X_train, vbundle.y_train)
    acc = clf.score(vbundle.X_test, vbundle.y_test)
    assert acc > 0.7, f"{preset} should learn spatial patterns; acc {acc:.3f}"
    assert clf.architecture_stamp_["architecture"] == preset
    assert set(clf.predict(vbundle.X_test)) <= set(clf.classes_)


def test_configurable_simple_cnn():
    cfg = config_from_preset("simple_cnn_small", base_channels=8, kernel_size=5, pooling="avg")
    assert cfg.base_channels == 8 and cfg.kernel_size == 5 and cfg.pooling == "avg"
    net, stamp = build_vision_network("simple_cnn", channels=3, image_size=16, n_classes=3, config=cfg)
    assert stamp["architecture"] == "simple_cnn_small"
    assert stamp["base_channels"] == 8
    import torch

    out = net(torch.randn(2, 3, 16, 16))
    assert out.shape == (2, 3)


def test_unknown_preset_and_architecture():
    with pytest.raises(ValueError, match="Unknown CNN preset"):
        config_from_preset("simple_cnn_huge")
    with pytest.raises(ValueError, match="Unknown vision architecture"):
        build_vision_network("vit_giant", 3, 16, 3)


def test_laptop_safe_constraints():
    with pytest.raises(ValueError, match="epochs"):
        VisionCNNClassifier(epochs=50)
    with pytest.raises(ValueError, match="batch_size"):
        VisionCNNClassifier(batch_size=9999)


def test_vision_metrics_with_confusion_matrix(vbundle):
    clf = VisionCNNClassifier(architecture="simple_cnn_small", epochs=6, learning_rate=3e-3, random_state=0)
    clf.fit(vbundle.X_train, vbundle.y_train)
    m = vision_metrics(vbundle.y_oos, clf.predict_proba(vbundle.X_oos), clf.classes_)
    assert m["n_classes"] == 3
    assert len(m["confusion_matrix"]) == 3 and len(m["confusion_matrix"][0]) == 3
    assert 0.0 <= m["accuracy"] <= 1.0 and "f1_macro" in m


def test_vision_robustness_suite(vbundle):
    clf = VisionCNNClassifier(architecture="simple_cnn_small", epochs=6, learning_rate=3e-3, random_state=0)
    clf.fit(vbundle.X_train, vbundle.y_train)
    rob = vision_robustness(clf, vbundle.X_test, vbundle.y_test, seed=0)
    assert "baseline_accuracy" in rob
    assert {"noise", "blur", "crop_resize", "brightness"} <= set(rob)
    zero = next(r for r in rob["noise"] if r["noise"] == 0.0)
    assert zero["drift"] == 0.0
    assert len(rob["brightness"]) == 3


def test_vision_explainability_reports_method(vbundle):
    clf = VisionCNNClassifier(architecture="simple_cnn_small", epochs=6, learning_rate=3e-3, random_state=0)
    clf.fit(vbundle.X_train, vbundle.y_train)
    exp = vision_explainability(clf, vbundle.X_test, seed=0)
    assert exp["method"] in {"gradient_saliency", "captum_saliency"}
    assert exp["saliency_map_shape"] == [16, 16]
    assert len(exp["peak_location"]) == 2


def test_resnet18_availability_is_explicit():
    # resnet18 is an optional preset; it must report availability, not silently stub.
    if torchvision_available():
        net, stamp = build_vision_network("resnet18", channels=3, image_size=32, n_classes=3)
        assert stamp["architecture"] == "resnet18"
    else:
        with pytest.raises(ImportError, match="torchvision not installed"):
            build_vision_network("resnet18", channels=3, image_size=32, n_classes=3)


def test_describe_cnn_param_count_and_metadata():
    from start.modeling.vision_models import config_from_preset, describe_cnn

    for preset in ("simple_cnn_small", "simple_cnn_medium", "simple_cnn_deep"):
        d = describe_cnn(preset, channels=3, image_size=32, n_classes=3)
        assert d["architecture"] == preset
        assert d["param_count"] > 0
        assert len(d["conv_block_channels"]) == d["n_blocks"]
        for key in ("kernel_size", "pooling", "dropout", "dense", "channels", "image_size"):
            assert key in d

    # custom config is user-editable and reflected in the descriptor
    cfg = config_from_preset("simple_cnn_small", n_blocks=3, base_channels=24, kernel_size=5)
    d = describe_cnn("simple_cnn", channels=1, image_size=28, n_classes=5, config=cfg)
    assert d["conv_block_channels"] == [24, 48, 96]
    assert d["kernel_size"] == 5
    assert d["param_count"] > 0
