import json

from start.core.schemas import EvidenceRecord, Status, TestResult
from start.evidence.ledger import EvidenceLedger


def _record(i: int) -> EvidenceRecord:
    return EvidenceRecord.from_result(
        TestResult(test_id=f"t{i}", test_name=f"T{i}", metrics={"v": i}, status=Status.PASS),
        model_id="m",
        dataset_id="d",
        run_id="r",
        input_artifact_hash="abc",
        policy_hash="ph",
    )


def test_ledger_chain_and_verify(tmp_path):
    ledger = EvidenceLedger(tmp_path / "ledger.jsonl", tmp_path / "store")
    for i in range(3):
        ledger.append(_record(i))
    assert ledger.verify()
    assert len(ledger.records()) == 3


def test_ledger_detects_tampering(tmp_path):
    ledger = EvidenceLedger(tmp_path / "ledger.jsonl", tmp_path / "store")
    for i in range(3):
        ledger.append(_record(i))
    lines = (tmp_path / "ledger.jsonl").read_text().splitlines()
    entry = json.loads(lines[1])
    entry["record"]["metrics"]["v"] = 999  # tamper
    lines[1] = json.dumps(entry)
    (tmp_path / "ledger.jsonl").write_text("\n".join(lines) + "\n")
    assert not ledger.verify()


def test_cache_hit_for_identical_invocation(tmp_path):
    ledger = EvidenceLedger(tmp_path / "ledger.jsonl", tmp_path / "store")
    rec = _record(1)
    ledger.append(rec)
    cached = ledger.store.cached(
        test_id="t1", input_artifact_hash="abc", params=rec.params, policy_hash="ph"
    )
    assert cached is not None and cached.metrics["v"] == 1
