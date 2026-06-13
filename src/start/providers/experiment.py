"""Experiment tracking providers."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path

from start.providers.base import ExperimentProvider


class LocalExperimentProvider(ExperimentProvider):
    """JSONL-backed experiment log; zero dependencies."""

    name = "local"

    def __init__(self, root: str | Path = "start_output/experiments") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _log(self, run_id: str, payload: dict) -> None:
        payload = {"ts": datetime.now(UTC).isoformat(), **payload}
        with (self.root / f"{run_id}.jsonl").open("a") as f:
            f.write(json.dumps(payload, default=str) + "\n")

    def start_run(self, run_name: str) -> str:
        run_id = f"RUN-{uuid.uuid4().hex[:10]}"
        self._log(run_id, {"event": "start", "run_name": run_name})
        return run_id

    def log_metrics(self, run_id: str, metrics: dict[str, float]) -> None:
        self._log(run_id, {"event": "metrics", "metrics": metrics})

    def log_artifact(self, run_id: str, path: str) -> None:
        self._log(run_id, {"event": "artifact", "path": path})

    def end_run(self, run_id: str) -> None:
        self._log(run_id, {"event": "end"})


class MLFlowExperimentProvider(ExperimentProvider):
    """Optional MLFlow tracking. Degrades to LocalExperimentProvider if
    mlflow is missing (see get_experiment_provider)."""

    name = "mlflow"

    def __init__(self, tracking_uri: str | None = None, experiment_name: str = "start-runs") -> None:
        import mlflow

        self._mlflow = mlflow
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

    def start_run(self, run_name: str) -> str:
        run = self._mlflow.start_run(run_name=run_name)
        return run.info.run_id

    def log_metrics(self, run_id: str, metrics: dict[str, float]) -> None:
        clean = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
        self._mlflow.log_metrics(clean)

    def log_artifact(self, run_id: str, path: str) -> None:
        self._mlflow.log_artifact(path)

    def end_run(self, run_id: str) -> None:
        self._mlflow.end_run()


def get_experiment_provider(
    provider: str, tracking_uri: str | None, experiment_name: str, output_root: str
) -> ExperimentProvider:
    if provider == "mlflow":
        try:
            return MLFlowExperimentProvider(tracking_uri, experiment_name)
        except ImportError:
            pass  # safe degradation
    return LocalExperimentProvider(Path(output_root) / "experiments")
