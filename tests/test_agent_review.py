from __future__ import annotations

from pathlib import Path

import pytest

from start.agents.prompts import SYSTEM_PROMPT, build_evidence_bundle, build_section_prompt
from start.agents.review import load_run_records, run_agent_review
from start.core.schemas import EvidenceRecord, Status, TestResult
from start.providers.base import LLMProvider
from start.providers.llm import EnterpriseLLMGatewayProvider, NoLLMProvider


def _record(test_id: str, status: Status, metrics: dict | None = None) -> EvidenceRecord:
    return EvidenceRecord.from_result(
        TestResult(
            test_id=test_id,
            test_name=test_id,
            status=status,
            metrics=metrics or {"value_metric": 1.0},
            interpretation="synthetic interpretation.",
        ),
        model_id="m-test",
        dataset_id="d-test",
        run_id="RUN-agent",
        policy_hash="hash123",
    )


@pytest.fixture()
def records() -> list[EvidenceRecord]:
    return [
        _record(
            "supervised.cohort_metrics_comparison",
            Status.PASS,
            {"train_auc_roc": 0.99, "test_auc_roc": 0.95, "oos_auc_roc": 0.94},
        ),
        _record("preprocessing.feature_drift", Status.WARN, {"max_psi": 0.15}),
        _record("xai.feature_sensitivity", Status.PASS, {"cohort": "test", "max_abs_auc_drift": 0.01}),
    ]


class FakeLLM(LLMProvider):
    """Scriptable provider: returns queued responses in order."""

    name = "fake"

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls: list[tuple[str, str]] = []

    @property
    def available(self) -> bool:
        return True

    def complete(self, system: str, user: str, *, max_tokens: int = 1024) -> str:
        self.calls.append((system, user))
        return self._responses.pop(0) if self._responses else self._responses_exhausted()

    @staticmethod
    def _responses_exhausted() -> str:
        return "Evidence is missing for further commentary."


# --------------------------------------------------------------------------- #
# Provider interface
# --------------------------------------------------------------------------- #
def test_provider_generate_interface():
    fake = FakeLLM(["hello"])
    out = fake.generate("prompt body", system="sys", metadata={"max_tokens": 99})
    assert out == "hello"
    assert fake.calls[0][0] == "sys" and "prompt body" in fake.calls[0][1]
    # NoLLMProvider implements the same interface and is explicitly unavailable
    assert NoLLMProvider().available is False


def test_enterprise_gateway_fails_gracefully():
    gateway = EnterpriseLLMGatewayProvider()
    assert gateway.available is False  # never silently used
    with pytest.raises(NotImplementedError, match="private"):
        gateway.generate("anything")


# --------------------------------------------------------------------------- #
# Evidence-grounded prompting
# --------------------------------------------------------------------------- #
def test_evidence_bundle_contains_records_not_raw_data(records):
    bundle = build_evidence_bundle(records, policy_hash="hash123", dataset_profile="tabular")
    for rec in records:
        assert rec.evidence_id in bundle
    assert "policy_hash: hash123" in bundle
    assert "fails_or_errors: none" in bundle
    prompt = build_section_prompt("findings", bundle)
    assert "cite" in prompt.lower()
    assert "Do not invent" in SYSTEM_PROMPT
    with pytest.raises(ValueError, match="Unknown review section"):
        build_section_prompt("poetry", bundle)


# --------------------------------------------------------------------------- #
# Dual-mode review flow
# --------------------------------------------------------------------------- #
def test_deterministic_mode_needs_no_llm(records):
    review = run_agent_review(records, mode="deterministic", llm=None)
    assert review.mode == "deterministic" and review.llm_provider == "none"
    assert review.review_plan and review.signoff
    assert review.critique_ok
    assert review.rejected_sections == []


def test_llm_mode_with_none_provider_falls_back_explicitly(records):
    review = run_agent_review(records, mode="llm", llm=NoLLMProvider())
    assert review.mode == "deterministic"  # no silent LLM pretense
    assert any("WARNING" in note and "fell back" in note for note in review.notes)


def test_llm_output_accepted_when_fully_cited(records):
    ev = records[0].evidence_id
    cited = f"- Cohort AUC values are stable across train, test, and OOS. [{ev}]"
    fake = FakeLLM([cited] * 6)
    review = run_agent_review(records, mode="llm", llm=fake)
    assert review.mode == "llm" and review.llm_provider == "fake"
    assert review.rejected_sections == []
    assert review.findings == [f"Cohort AUC values are stable across train, test, and OOS. [{ev}]"]
    assert review.critique_ok


def test_llm_uncited_numeric_rejected_then_corrected(records):
    ev = records[0].evidence_id
    bad = "- The test AUC is 0.95, which is excellent."
    good = f"- The test AUC is 0.95, which is strong. [{ev}]"
    # one bad + one corrected answer per section
    fake = FakeLLM([bad, good] * 6)
    review = run_agent_review(records, mode="llm", llm=fake)
    assert review.rejected_sections == []
    # correction prompt was actually issued
    assert any("rejected by the evidence critic" in user for _, user in fake.calls)


def test_llm_unsupported_evidence_id_rejected_to_fallback(records):
    bogus = "- Drift is severe per [EV-000000000000] and demands escalation."
    fake = FakeLLM([bogus] * 12)  # fails first try AND retry for every section
    review = run_agent_review(records, mode="llm", llm=fake)
    assert set(review.rejected_sections) == {
        "review_plan",
        "findings",
        "challenge_memo",
        "missing_evidence",
        "suggested_tests",
        "signoff",
    }
    assert any("deterministic fallback" in note for note in review.notes)
    # fallback content is the deterministic one and still critique-clean
    assert review.critique_ok
    assert review.signoff.startswith("Reviewer recommendation")


def test_llm_hallucinated_signoff_blocked():
    breached = [_record("supervised.calibration", Status.FAIL)]
    fake = FakeLLM(["- This run is ready for sign-off and looks great."] * 12)
    review = run_agent_review(breached, mode="llm", llm=fake)
    assert "signoff" in review.rejected_sections
    assert "NOT READY" in review.signoff  # deterministic fallback verdict


# --------------------------------------------------------------------------- #
# Post-hoc loading + CLI + report disclosure
# --------------------------------------------------------------------------- #
@pytest.fixture()
def stored_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from start.modeling.data import load_attrition_dataset
    from start.orchestration import review_dataframes

    result = review_dataframes(load_attrition_dataset(seed=21), target_column="attrition", seed=21)
    return result, tmp_path


def test_load_run_records_latest_and_explicit(stored_run):
    result, _ = stored_run
    run_id, records = load_run_records("start_output", run_id="latest")
    assert run_id == result.run_id
    assert {r.evidence_id for r in records} == {r.evidence_id for r in result.evidence}
    with pytest.raises(ValueError, match="not found"):
        load_run_records("start_output", run_id="RUN-nonexistent")


def test_agent_review_cli_deterministic(stored_run):
    from typer.testing import CliRunner

    from start.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["agent-review", "--agent-mode", "deterministic"])
    assert result.exit_code == 0, result.output
    assert "Review plan" in result.output
    assert "Challenge memo" in result.output
    assert "Sign-off recommendation" in result.output

    plan = runner.invoke(app, ["signoff"])
    assert plan.exit_code == 0 and "Governance assessment" in plan.output

    llm_none = runner.invoke(app, ["agent-review", "--agent-mode", "llm", "--llm-provider", "none"])
    assert llm_none.exit_code == 0
    assert "WARNING" in llm_none.output  # explicit, never silent


def test_report_discloses_agent_mode(stored_run):
    result, tmp_path = stored_run
    from start.reporting import render_markdown

    md = render_markdown(result)
    assert "## Agent review" in md
    assert "Agent mode: deterministic" in md
    assert "Evidence critique status:" in md
    assert "### Sign-off recommendation" in md

    # llm-mode disclosure
    ev = result.evidence[0].evidence_id
    fake = FakeLLM([f"- Evidence reviewed in full. [{ev}]"] * 6)
    result.agent_review = run_agent_review(result.evidence, mode="llm", llm=fake)
    md_llm = render_markdown(result)
    assert "Agent mode: llm-assisted" in md_llm
    assert "LLM provider: fake" in md_llm


def test_notebooks_compile_without_databricks():
    import py_compile

    for nb in Path("notebooks").glob("*.py"):
        py_compile.compile(str(nb), doraise=True)
