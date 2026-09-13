from __future__ import annotations

import glob
import json
import tempfile

import pytest

pytest.importorskip("torch", reason="surfacing needs a full enterprise run")

from start.interactive_review import ReviewConfig, run_interactive_review  # noqa: E402


def _run():
    out_root = tempfile.mkdtemp()
    cfg = ReviewConfig(
        data_path=None, target="attrition", architecture_family="mlp",
        activation="relu", agent_mode="deterministic", llm_provider="none",
        run_dl=True, enterprise_mode=True, costlier_errors="false_negatives",
        non_interactive=True, accept_recommendations=True, output_root=out_root,
        seed=0,
    )
    out = run_interactive_review(cfg)
    return out_root, out


def test_dashboard_embeds_mrm_and_validation():
    out_root, _ = _run()
    md = open(glob.glob(f"{out_root}/dashboards/*/dashboard.md")[0]).read()
    assert "MRM Signoff Decision" in md
    assert "ValidationAgent Review" in md


def test_transcript_json_has_signoff_and_validation():
    out_root, _ = _run()
    d = json.load(open(glob.glob(f"{out_root}/transcripts/*/transcript.json")[0]))
    assert d.get("mrm_signoff") is not None
    assert d.get("validation_review") is not None
    assert d["mrm_signoff"]["verdict"] in ("READY", "READY WITH CONDITIONS", "NOT READY")


def test_transcript_md_has_sections():
    out_root, _ = _run()
    md = open(glob.glob(f"{out_root}/transcripts/*/transcript.md")[0]).read()
    assert "MRM signoff decision" in md
    assert "ValidationAgent review" in md


def test_session_carries_validation_and_signoff():
    _, out = _run()
    assert out.review_session.mrm_signoff is not None
    assert out.review_session.validation_review is not None
