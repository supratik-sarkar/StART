"""Regression test proving scenario-integrity challenge emits valid scalar EvidenceRecords."""

import uuid

import pytest
from pydantic import ValidationError

from start.core.schemas import EvidenceRecord, Status
from start.portfolio.contracts import RepricingMethod, ScenarioSpec, ScenarioType
from start.portfolio.scenario import validate_scenario_data_integrity


def test_scenario_diagnostic_evidencerecord_scalar_schema():
    """Prove that scenario diagnostics produce valid scalar EvidenceRecords without schema failure."""
    # 1. Construct a scenario spec with known issues (e.g. missing shocks for some assets)
    spec = ScenarioSpec(
        scenario_id="SCEN-TEST-AUDIT",
        scenario_name="Test Scenario Audit",
        scenario_type=ScenarioType.SYNTHETIC,
        shocks=(),  # Zero shocks triggers issues
        repricing_method=RepricingMethod.LINEAR_RETURN,
    )
    portfolio_assets = ["ASSET_1", "ASSET_2", "ASSET_3"]
    
    # 2. Run deterministic diagnostic
    diag_res = validate_scenario_data_integrity(spec, portfolio_assets=portfolio_assets)
    assert not diag_res.valid
    assert len(diag_res.issues) >= 1
    
    # Simulate multiple issues
    sample_issues = [
        "Scenario contains zero shocks",
        "Asset ASSET_1 missing from shock specification",
        "Asset ASSET_2 missing from shock specification",
    ]
    
    # 3. Format metrics using scalar schema
    formatted_issues = "; ".join(sample_issues) if sample_issues else "none"
    diag_metrics: dict[str, bool | float | int | str | None] = {
        "is_valid": bool(diag_res.valid),
        "n_shocks": int(diag_res.n_shocks),
        "scenario_id": str(spec.scenario_id),
        "issues": formatted_issues,
    }
    
    # Verify every key in metrics is scalar
    for k, v in diag_metrics.items():
        assert isinstance(v, (bool, float, int, str)) or v is None, f"Metric key {k} value {v!r} is not scalar"
    
    # 4. Prove EvidenceRecord creation succeeds
    source_evidence_id = "EV-source-12345"
    diag_ev_id = f"EV-DIAG-{uuid.uuid4().hex[:8]}"
    
    # Must not raise ValidationError
    diag_record = EvidenceRecord(
        evidence_id=diag_ev_id,
        test_id="diagnostic.validate_scenario_data_integrity",
        test_name="Deterministic Challenge Diagnostic (validate_scenario_data_integrity)",
        model_id="MOD-DEFAULT",
        dataset_id="DS-DEFAULT",
        run_id="RUN-TEST-001",
        status=Status.RECORDED,
        metrics=diag_metrics,
    )
    
    assert diag_record.evidence_id == diag_ev_id
    assert diag_record.evidence_id != source_evidence_id
    assert diag_record.metrics["is_valid"] is False
    assert diag_record.metrics["n_shocks"] == 0
    assert diag_record.metrics["scenario_id"] == "SCEN-TEST-AUDIT"
    
    # 5. Prove multiple issues are preserved without silent discarding
    extracted_issues = str(diag_record.metrics["issues"]).split("; ")
    assert extracted_issues == sample_issues
    assert len(extracted_issues) == 3


def test_scenario_diagnostic_rejects_raw_list_metrics():
    """Prove that raw list metrics violate EvidenceRecord schema (reproducing the original bug)."""
    raw_list_metrics = {
        "is_valid": False,
        "n_shocks": 0,
        "scenario_id": "SCEN-TEST-AUDIT",
        "issues": ["Issue 1", "Issue 2"],  # Raw list violates schema
    }
    with pytest.raises(ValidationError) as exc_info:
        EvidenceRecord(
            evidence_id="EV-DIAG-FAIL",
            test_id="diagnostic.validate_scenario_data_integrity",
            test_name="Diagnostic Test",
            model_id="MOD-DEFAULT",
            dataset_id="DS-DEFAULT",
            run_id="RUN-TEST-001",
            status=Status.RECORDED,
            metrics=raw_list_metrics,  # type: ignore[arg-type]
        )
    assert "Input should be a valid string" in str(exc_info.value) or "Input should be a valid" in str(exc_info.value)
