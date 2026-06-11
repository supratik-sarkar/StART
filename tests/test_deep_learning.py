from pathlib import Path

import pytest

from start.modeling.dl_training import torch_available

pytestmark = pytest.mark.skipif(not torch_available(), reason="torch not installed")


def test_dl_review_smoke(tmp_path: Path):
    from start.modeling.deep_learning import run_deep_learning_review

    result = run_deep_learning_review(output_root=tmp_path, epochs=2)
    assert result.metrics.shape[0] == 3
    assert set(result.metrics["cohort"]) == {"train", "test", "oos"}
    assert result.evidence
    assert Path(result.report_path).exists()
    assert all(Path(p).exists() for p in result.figure_paths)


def test_dl_architecture_scope():
    from start.modeling.deep_learning import build_classifier

    assert build_classifier("mlp")["architecture"] == "mlp"
    with pytest.raises(NotImplementedError):
        build_classifier("lstm")
