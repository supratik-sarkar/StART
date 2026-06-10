"""Evidence persistence: append-only chained ledger + content-addressed store.

Ledger lines have the shape::

    {"index": n, "prev_hash": "...", "record_hash": "...", "entry_hash": "...", "record": {...}}

where ``record_hash = sha256(canonical_json(record))`` and
``entry_hash = sha256(prev_hash + record_hash)``. Any mutation of a past
record breaks the chain, which :meth:`EvidenceLedger.verify` detects.

The content-addressed store writes each record to ``<sha256>.json`` and keeps
a cache index keyed by ``(test_id, input_artifact_hash, params_hash, policy_hash)``
so identical test invocations can be served from cache.
"""

from __future__ import annotations

import json
from pathlib import Path

from start.core.hashing import canonical_json, hash_obj, sha256_hex
from start.core.schemas import EvidenceRecord
from start.providers.base import EvidenceProvider

GENESIS = "0" * 64


class ContentAddressedStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._index_path = self.root / "cache_index.json"

    def _cache_key(self, record: EvidenceRecord) -> str:
        return hash_obj(
            {
                "test_id": record.test_id,
                "input_artifact_hash": record.input_artifact_hash,
                "params_hash": hash_obj(record.params),
                "policy_hash": record.policy_hash,
            }
        )

    def put(self, record: EvidenceRecord) -> str:
        payload = canonical_json(record.model_dump(mode="json"))
        content_hash = sha256_hex(payload)
        (self.root / f"{content_hash}.json").write_text(payload)
        index = self._read_index()
        index[self._cache_key(record)] = content_hash
        self._index_path.write_text(json.dumps(index, indent=2))
        return content_hash

    def get(self, content_hash: str) -> EvidenceRecord:
        payload = (self.root / f"{content_hash}.json").read_text()
        return EvidenceRecord.model_validate_json(payload)

    def cached(
        self,
        *,
        test_id: str,
        input_artifact_hash: str | None,
        params: dict,
        policy_hash: str | None,
    ) -> EvidenceRecord | None:
        """Return a previously computed record for an identical invocation."""
        if input_artifact_hash is None:
            return None  # no data hash -> no safe cache identity
        key = hash_obj(
            {
                "test_id": test_id,
                "input_artifact_hash": input_artifact_hash,
                "params_hash": hash_obj(params),
                "policy_hash": policy_hash,
            }
        )
        content_hash = self._read_index().get(key)
        return self.get(content_hash) if content_hash else None

    def _read_index(self) -> dict[str, str]:
        if self._index_path.exists():
            return json.loads(self._index_path.read_text())
        return {}


class EvidenceLedger(EvidenceProvider):
    name = "ledger"

    def __init__(self, ledger_path: str | Path, store_root: str | Path) -> None:
        self.path = Path(ledger_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.store = ContentAddressedStore(store_root)

    def _last_entry(self) -> dict | None:
        if not self.path.exists():
            return None
        last = None
        with self.path.open() as f:
            for line in f:
                if line.strip():
                    last = json.loads(line)
        return last

    def append(self, record: EvidenceRecord) -> str:
        record_payload = record.model_dump(mode="json")
        record_hash = self.store.put(record)
        last = self._last_entry()
        prev_hash = last["entry_hash"] if last else GENESIS
        index = (last["index"] + 1) if last else 0
        entry = {
            "index": index,
            "prev_hash": prev_hash,
            "record_hash": record_hash,
            "entry_hash": sha256_hex(prev_hash + record_hash),
            "record": record_payload,
        }
        with self.path.open("a") as f:
            f.write(json.dumps(entry, default=str) + "\n")
        return record_hash

    def verify(self) -> bool:
        prev_hash = GENESIS
        if not self.path.exists():
            return True
        with self.path.open() as f:
            for expected_index, line in enumerate(ln for ln in f if ln.strip()):
                entry = json.loads(line)
                if entry["index"] != expected_index:
                    return False
                if entry["prev_hash"] != prev_hash:
                    return False
                recomputed_record = sha256_hex(canonical_json(entry["record"]))
                if recomputed_record != entry["record_hash"]:
                    return False
                if sha256_hex(prev_hash + entry["record_hash"]) != entry["entry_hash"]:
                    return False
                prev_hash = entry["entry_hash"]
        return True

    def records(self) -> list[EvidenceRecord]:
        if not self.path.exists():
            return []
        out: list[EvidenceRecord] = []
        with self.path.open() as f:
            for line in f:
                if line.strip():
                    out.append(EvidenceRecord.model_validate(json.loads(line)["record"]))
        return out
