from pathlib import Path

import pytest

from start.modeling.dl_training import torch_available

pytestmark = pytest.mark.skipif(not torch_available(), reason="torch not installed")


def test_dl_explainability_and_sensitivity(tmp_path: Path):
    from start.modeling.deep_learning import run_deep_learning_review

    result = run_deep_learning_review(output_root=tmp_path, epochs=2)
    assert len(result.attribution) > 0
    assert result.attribution.iloc[0]["importance"] >= 0
    assert 0.0 in set(result.sensitivity["shock"])
    zero = result.sensitivity.loc[result.sensitivity["shock"] == 0.0, "auc_drift"].iloc[0]
    assert abs(zero) < 1e-9
