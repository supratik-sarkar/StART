import json
import os
import pathlib
import time
from typing import Any


class StepCheckpointer:
    def __init__(self, storage_dir: str = "~/.state_cache_v240"):
        self.storage_path = pathlib.Path(os.path.expanduser(storage_dir))
        self.storage_path.mkdir(parents=True, exist_ok=True)

    def _get_path(self, workflow_id: str) -> pathlib.Path:
        return self.storage_path / f"chk_{workflow_id}.json"

    def save_checkpoint(self, workflow_id: str, stage_name: str, payload: dict[str, Any]) -> str:
        data = {
            "workflow_id": workflow_id,
            "last_completed_stage": stage_name,
            "timestamp": time.time(),
            "payload": payload,
        }
        target_path = self._get_path(workflow_id)
        temp_path = target_path.with_suffix(".tmp")
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        os.replace(temp_path, target_path)
        return str(target_path)

    def load_checkpoint(self, workflow_id: str) -> dict[str, Any] | None:
        target_path = self._get_path(workflow_id)
        if not target_path.exists():
            return None
        with open(target_path, encoding="utf-8") as f:
            return json.load(f)

    def clear_checkpoint(self, workflow_id: str) -> None:
        target_path = self._get_path(workflow_id)
        if target_path.exists():
            target_path.unlink()
