"""Generic public experiment tracking abstraction for StART.

Invariants:
- Optional integration: StART functions fully with or without MLflow.
- Tracking failure != evidence failure.
- Tracking calls never mutate or invalidate deterministic evidence records.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from enum import StrEnum
from typing import Any

logger = logging.getLogger(__name__)


class ExperimentTracker(ABC):
    """Abstract interface for optional experiment tracking."""

    @abstractmethod
    def log_metric(self, key: str, value: float, step: int = 0) -> None:
        """Record a scalar metric."""
        ...

    @abstractmethod
    def log_param(self, key: str, value: Any) -> None:
        """Record a configuration or hyperparameter."""
        ...

    @abstractmethod
    def log_artifact(self, local_path: str, artifact_path: str = "") -> None:
        """Record an artifact file path."""
        ...

    @abstractmethod
    def set_tag(self, key: str, value: str) -> None:
        """Set a metadata tag."""
        ...

    @abstractmethod
    def get_logged_metrics(self) -> dict[str, float]:
        """Retrieve all recorded metrics."""
        ...

    @abstractmethod
    def get_logged_params(self) -> dict[str, Any]:
        """Retrieve all recorded parameters."""
        ...

    @abstractmethod
    def get_logged_tags(self) -> dict[str, str]:
        """Retrieve all recorded tags."""
        ...


class NoOpExperimentTracker(ExperimentTracker):
    """Deterministic in-memory fallback tracker (default when external tracker is absent)."""

    def __init__(self) -> None:
        self._metrics: dict[str, float] = {}
        self._params: dict[str, Any] = {}
        self._artifacts: list[tuple[str, str]] = []
        self._tags: dict[str, str] = {}

    def log_metric(self, key: str, value: float, step: int = 0) -> None:
        self._metrics[key] = float(value)

    def log_param(self, key: str, value: Any) -> None:
        self._params[key] = value

    def log_artifact(self, local_path: str, artifact_path: str = "") -> None:
        self._artifacts.append((local_path, artifact_path))

    def set_tag(self, key: str, value: str) -> None:
        self._tags[key] = str(value)

    def get_logged_metrics(self) -> dict[str, float]:
        return dict(self._metrics)

    def get_logged_params(self) -> dict[str, Any]:
        return dict(self._params)

    def get_logged_tags(self) -> dict[str, str]:
        return dict(self._tags)


class TrackingStatus(StrEnum):
    """Execution status of an optional experiment tracker."""

    SUCCESS = "SUCCESS"
    DISABLED = "DISABLED"
    UNAVAILABLE = "UNAVAILABLE"
    FAILED_NON_BLOCKING = "FAILED_NON_BLOCKING"


class MLflowExperimentTracker(ExperimentTracker):
    """Optional MLflow experiment tracking adapter with resilient fallback and explicit status reporting."""

    def __init__(self, experiment_name: str = "StART_Market_Review") -> None:
        self._fallback = NoOpExperimentTracker()
        self._mlflow_available = False
        self._experiment_name = experiment_name
        self.last_error: str | None = None

        try:
            import mlflow

            self._mlflow = mlflow
            self._mlflow_available = True
            self.tracking_status = TrackingStatus.SUCCESS
        except ImportError:
            self._mlflow = None
            self.tracking_status = TrackingStatus.UNAVAILABLE
            logger.info("MLflow not installed; MLflowExperimentTracker status=UNAVAILABLE.")

    @property
    def is_available(self) -> bool:
        return self._mlflow_available

    def log_metric(self, key: str, value: float, step: int = 0) -> None:
        self._fallback.log_metric(key, value, step)
        if self._mlflow_available and self._mlflow is not None:
            try:
                self._mlflow.log_metric(key, float(value), step=step)
            except Exception as e:
                self.tracking_status = TrackingStatus.FAILED_NON_BLOCKING
                self.last_error = f"{type(e).__name__}: {e}"
                logger.warning(f"MLflow log_metric failed non-fatally: {e}")

    def log_param(self, key: str, value: Any) -> None:
        self._fallback.log_param(key, value)
        if self._mlflow_available and self._mlflow is not None:
            try:
                self._mlflow.log_param(key, value)
            except Exception as e:
                self.tracking_status = TrackingStatus.FAILED_NON_BLOCKING
                self.last_error = f"{type(e).__name__}: {e}"
                logger.warning(f"MLflow log_param failed non-fatally: {e}")

    def log_artifact(self, local_path: str, artifact_path: str = "") -> None:
        self._fallback.log_artifact(local_path, artifact_path)
        if self._mlflow_available and self._mlflow is not None:
            try:
                self._mlflow.log_artifact(local_path, artifact_path=artifact_path or None)
            except Exception as e:
                self.tracking_status = TrackingStatus.FAILED_NON_BLOCKING
                self.last_error = f"{type(e).__name__}: {e}"
                logger.warning(f"MLflow log_artifact failed non-fatally: {e}")

    def set_tag(self, key: str, value: str) -> None:
        self._fallback.set_tag(key, value)
        if self._mlflow_available and self._mlflow is not None:
            try:
                self._mlflow.set_tag(key, str(value))
            except Exception as e:
                self.tracking_status = TrackingStatus.FAILED_NON_BLOCKING
                self.last_error = f"{type(e).__name__}: {e}"
                logger.warning(f"MLflow set_tag failed non-fatally: {e}")

    def get_logged_metrics(self) -> dict[str, float]:
        return self._fallback.get_logged_metrics()

    def get_logged_params(self) -> dict[str, Any]:
        return self._fallback.get_logged_params()

    def get_logged_tags(self) -> dict[str, str]:
        return self._fallback.get_logged_tags()
