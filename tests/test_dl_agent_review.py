from pathlib import Path

import pytest

from start.modeling.dl_training import torch_available

pytestmark = pytest.mark.skipif(not torch_available(), reason="torch not installed")


def test_dl_report_contains_agentic_review_and_citations(tmp_path: Path):
    from start.modeling.deep_learning import run_deep_learning_review

    result = run_deep_learning_review(output_root=tmp_path, epochs=2, agent_mode="deterministic")
    report = Path(result.report_path).read_text()
    assert "Agentic review" in report
    assert "EV-DL-" in report
    assert "Evidence critique status: PASSED" in report
