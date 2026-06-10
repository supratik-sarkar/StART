from pathlib import Path

import yaml

from start.agents import EvidenceCriticAgent, NarrativeAgent
from start.core.config import StartConfig, load_config
from start.core.schemas import EvidenceRecord, Narrative, Status, TestResult
from start.orchestration.pipeline import build_context, run_review
from start.providers.llm import NoLLMProvider


def _evidence() -> list[EvidenceRecord]:
    return [
        EvidenceRecord.from_result(
            TestResult(
                test_id="t1",
                test_name="T1",
                metrics={"auc": 0.81},
                status=Status.PASS,
                interpretation="AUC is 0.81.",
            ),
            model_id="m",
            dataset_id="d",
            run_id="r",
            policy_hash="ph",
        )
    ]


def test_critic_blocks_uncited_quantitative_claim():
    critic = EvidenceCriticAgent()
    records = _evidence()
    bad = Narrative(run_id="r", summary="The model achieved AUC of 0.81 on holdout.")
    assert not critic.critique_narrative(bad, records).ok
    good = Narrative(
        run_id="r",
        summary=f"The model achieved AUC of 0.81 on holdout. [{records[0].evidence_id}]",
    )
    assert critic.critique_narrative(good, records).ok


def test_critic_blocks_unknown_citation():
    critic = EvidenceCriticAgent()
    fake = Narrative(run_id="r", summary="AUC is 0.99. [EV-deadbeef0000]")
    assert not critic.critique_narrative(fake, _evidence()).ok


def test_template_narrative_is_proof_carrying():
    records = _evidence()
    narrative = NarrativeAgent(NoLLMProvider()).generate("r", records)
    assert EvidenceCriticAgent().critique_narrative(narrative, records).ok


def test_full_pipeline_end_to_end(tmp_path, toy_frame, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("configs/policy").mkdir(parents=True)
    Path("configs/policy/default_policy.yaml").write_text(
        yaml.safe_dump({"name": "test", "version": "0.0.1"})
    )
    config = StartConfig()
    config.model.target_column = "target"
    config.model.score_column = "score"
    config.test_families.enabled = ["preprocessing", "supervised"]
    ctx = build_context(config, toy_frame.iloc[:350], toy_frame.iloc[350:])
    result = run_review(config, ctx)
    assert result.policy and result.policy.allowed
    assert len(result.evidence) >= 5
    assert result.narrative and result.critique
    assert all(rec.policy_hash for rec in result.evidence)
    assert all(rec.input_artifact_hash for rec in result.evidence)
    ledger_path = Path(config.output.root) / config.output.ledger_file
    assert ledger_path.exists()


def test_config_yaml_roundtrip(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump({"project_name": "x", "llm": {"provider": "none"}}))
    config = load_config(path)
    assert config.project_name == "x"
