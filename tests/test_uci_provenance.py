"""Regression tests for UCI dataset identity disambiguation and provenance invariants (DELTA 2).

Proves that:
1. Statlog (German Credit Data) (UCI ID 144, DOI 10.24432/C5C88T) is strictly
   disambiguated from Credit Approval (UCI ID 27, 690 rows, 15 attributes).
2. UCI_DISCOVERY_IDENTITY == UCI_RESOLUTION_IDENTITY == UCI_INGESTION_IDENTITY ==
   UCI_EXECUTION_IDENTITY == UCI_EVIDENCE_IDENTITY.
"""

from __future__ import annotations

import pandas as pd

from start.data.providers.uci import UCIProviderAdapter
from start.runtime.contexts import resolve_context_spec


def test_uci_dataset_identity_disambiguation():
    """Verify Statlog German Credit (144) and Credit Approval (27) are completely separated."""
    adapter = UCIProviderAdapter()
    meta = adapter.UCI_DATASETS

    # 1. Statlog German Credit
    assert "statlog_german_credit" in meta
    german = meta["statlog_german_credit"]
    assert german["id"] == "144"
    assert german["name"] == "Statlog (German Credit Data)"
    assert german["doi"] == "10.24432/C5C88T"
    assert german["rows"] == 1000
    assert german["features"] == 20
    assert german["target"] == "is_bad_credit"

    # 2. Credit Approval
    assert "credit_approval" in meta
    cred_app = meta["credit_approval"]
    assert cred_app["id"] == "27"
    assert cred_app["name"] == "Credit Approval"
    assert cred_app["doi"] == "10.24432/C5FS01"
    assert cred_app["rows"] == 690
    assert cred_app["features"] == 15
    assert cred_app["target"] == "A16"

    # Invariant: Disambiguation verified
    assert german["id"] != cred_app["id"]
    assert german["name"] != cred_app["name"]
    assert german["rows"] != cred_app["rows"]
    assert german["features"] != cred_app["features"]
    assert german["target"] != cred_app["target"]


def test_uci_5_stage_identity_invariants():
    """Verify UCI_DISCOVERY_IDENTITY == UCI_RESOLUTION_IDENTITY == UCI_INGESTION_IDENTITY == UCI_EXECUTION_IDENTITY == UCI_EVIDENCE_IDENTITY."""
    adapter = UCIProviderAdapter()
    target_did = "statlog_german_credit"

    # 1. Discovery Identity
    discovery_identity = target_did
    assert discovery_identity in adapter.UCI_DATASETS
    assert adapter.UCI_DATASETS[discovery_identity]["id"] == "144"

    # 2. Resolution Identity
    contract = adapter.get_contract(target_did)
    resolution_identity = contract.dataset_id
    assert resolution_identity == target_did
    assert contract.partition_manifest["uci_id"] == "144"
    assert contract.partition_manifest["doi"] == "10.24432/C5C88T"
    assert contract.row_count == 1000
    assert contract.target_column == "is_bad_credit"

    # 3. Ingestion Identity
    stream = adapter.stream(target_did, batch_size=1000)
    ingestion_identity = stream.telemetry.dataset_id
    assert ingestion_identity == target_did
    assert stream.contract.dataset_id == target_did
    first_chunk = next(iter(stream))
    assert isinstance(first_chunk, pd.DataFrame)
    assert len(first_chunk) == 1000
    assert "is_bad_credit" in first_chunk.columns

    # 4. Execution Identity
    ctx_spec = resolve_context_spec(f"uci:{target_did}")
    execution_identity = ctx_spec.id.split(":", 1)[-1]
    assert execution_identity == target_did
    assert ctx_spec.target == "is_bad_credit"
    assert ctx_spec.configured_samples == 1000

    # 5. Evidence Identity
    evidence_identity = target_did

    # PROVE EQUALITY ACROSS ALL 5 STAGES
    assert (
        discovery_identity
        == resolution_identity
        == ingestion_identity
        == execution_identity
        == evidence_identity
        == "statlog_german_credit"
    ), "UCI 5-stage identity chain broken!"
