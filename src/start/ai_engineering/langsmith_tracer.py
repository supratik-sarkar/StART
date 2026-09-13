"""Fail-safe, profile-contained LangSmith tracer for StART reviews.

Strict containment & data isolation rules:
  1. Profile gating: in enterprise/airgapped profiles, telemetry egress is refused
     by default unless START_ALLOW_TELEMETRY_EGRESS=true.
  2. Envelope projection: only envelope-projected prompts (envelope.render()) and
     their completions may be sent to LangSmith. Tracing of raw evidence objects
     is strictly refused.
  3. Non-blocking & fault-tolerant: tracing errors are caught, logged once, and
     suppressed so reviews are never blocked, slowed, or altered by telemetry.
"""

from __future__ import annotations

import logging
import os
import uuid
from typing import Any

from start.runtime_profile import ProfileViolation, assert_sink_allowed

logger = logging.getLogger("start.ai_engineering.langsmith")

_CLIENT = None
_INIT_ATTEMPTED = False
_DISABLED_DUE_TO_ERROR = False


def is_langsmith_enabled(env: dict[str, str] | None = None) -> bool:
    """Return True if LangSmith tracing is permitted and configured."""
    global _DISABLED_DUE_TO_ERROR
    if _DISABLED_DUE_TO_ERROR:
        return False

    env_map = os.environ if env is None else env
    # Explicit off-switch
    if env_map.get("START_LANGSMITH_ENABLED", "").strip().lower() in {"0", "false", "no"}:
        return False

    # Check profile containment
    try:
        assert_sink_allowed("langsmith", env_map)
    except ProfileViolation:
        return False

    # Check key presence (presence only, never value)
    key = env_map.get("LANGSMITH_API_KEY", "").strip() or env_map.get("LANGCHAIN_API_KEY", "").strip()
    return bool(key)


def _get_client() -> Any:
    global _CLIENT, _INIT_ATTEMPTED, _DISABLED_DUE_TO_ERROR
    if _DISABLED_DUE_TO_ERROR:
        return None
    if _CLIENT is not None:
        return _CLIENT
    if _INIT_ATTEMPTED:
        return None

    _INIT_ATTEMPTED = True
    if not is_langsmith_enabled():
        return None

    try:
        from langsmith import Client

        project = (
            os.environ.get("LANGSMITH_PROJECT")
            or os.environ.get("LANGCHAIN_PROJECT")
            or "start-model-review"
        )
        os.environ["LANGCHAIN_PROJECT"] = project
        _CLIENT = Client()
        return _CLIENT
    except Exception as exc:
        _DISABLED_DUE_TO_ERROR = True
        logger.warning(
            "LangSmith client initialization failed; disabling tracing for this session. Error: %s",
            exc,
        )
        return None


class LangSmithTracer:
    """Manages hierarchical review run tracing (Review -> Agent -> LLM Call)."""

    def __init__(self, run_id: str | None = None) -> None:
        self.run_id = run_id or str(uuid.uuid4())
        self.client = _get_client()
        self.root_run_id: str | None = None

    def start_review(
        self,
        *,
        review_id: str,
        plan_hash: str = "",
        profile_manifest_hash: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> str | None:
        """Create the root run for a review session."""
        if not self.client:
            return None
        try:
            self.root_run_id = str(uuid.uuid4())
            meta = dict(metadata or {})
            meta.update(
                {
                    "review_id": review_id,
                    "plan_hash": plan_hash,
                    "profile_manifest_hash": profile_manifest_hash,
                }
            )
            self.client.create_run(
                name=f"StART Review [{review_id}]",
                run_type="chain",
                id=self.root_run_id,
                inputs={"review_id": review_id, "plan_hash": plan_hash},
                extra={"metadata": meta},
            )
            return self.root_run_id
        except Exception as exc:
            logger.warning("LangSmith root run creation failed: %s", exc)
            return None

    def start_agent(
        self,
        agent_name: str,
        *,
        stage: str = "",
        inputs: dict[str, Any] | None = None,
    ) -> str | None:
        """Create a child run for an individual agent execution."""
        if not self.client or not self.root_run_id:
            return None
        try:
            agent_run_id = str(uuid.uuid4())
            self.client.create_run(
                name=f"Agent: {agent_name}",
                run_type="agent",
                id=agent_run_id,
                parent_run_id=self.root_run_id,
                inputs=inputs or {"agent": agent_name, "stage": stage},
                extra={"metadata": {"agent": agent_name, "stage": stage}},
            )
            return agent_run_id
        except Exception as exc:
            logger.warning("LangSmith agent run creation failed: %s", exc)
            return None

    def end_agent(
        self,
        agent_run_id: str | None,
        *,
        outputs: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        """Complete a child agent run."""
        if not self.client or not agent_run_id:
            return
        try:
            self.client.update_run(
                run_id=agent_run_id,
                outputs=outputs or {},
                error=error,
            )
        except Exception as exc:
            logger.warning("LangSmith agent run update failed: %s", exc)

    def trace_llm_call(
        self,
        *,
        parent_run_id: str | None = None,
        envelope: Any,
        completion: str,
        latency_ms: float = 0.0,
        provider: str = "",
        model: str = "",
        token_usage: dict[str, int] | None = None,
        invariance_hash: str | None = None,
        invariance_verdict: str | None = None,
    ) -> str | None:
        """Record an LLM completion as a grandchild run.

        Strictly enforces that only envelope-rendered text is sent. If no envelope
        is provided, the call is refused and nothing is sent.
        """
        if not self.client:
            return None

        if envelope is None or not hasattr(envelope, "render") or not hasattr(envelope, "envelope_hash"):
            logger.warning("Refusing to trace LLM call: envelope is missing or does not support render().")
            return None

        try:
            rendered_prompt = envelope.render()
            env_hash = envelope.envelope_hash()
            policy_id = getattr(envelope, "policy_id", "")

            call_run_id = str(uuid.uuid4())
            parent = parent_run_id or self.root_run_id

            extra_meta = {
                "provider": provider,
                "model": model,
                "envelope_hash": env_hash,
                "policy_id": policy_id,
                "latency_ms": latency_ms,
            }
            if invariance_hash:
                extra_meta["invariance_hash"] = invariance_hash
            if invariance_verdict:
                extra_meta["invariance_verdict"] = invariance_verdict
            if token_usage:
                extra_meta["token_usage"] = token_usage

            self.client.create_run(
                name=f"LLM: {provider}/{model}" if model else f"LLM: {provider}",
                run_type="llm",
                id=call_run_id,
                parent_run_id=parent,
                inputs={"prompt": rendered_prompt},
                outputs={"completion": completion},
                extra={"metadata": extra_meta},
            )
            return call_run_id
        except Exception as exc:
            logger.warning("LangSmith LLM call tracing failed: %s", exc)
            return None

    def end_review(
        self,
        *,
        seal_string: str = "",
        outputs: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        """Close the root review run."""
        if not self.client or not self.root_run_id:
            return
        try:
            out = dict(outputs or {})
            if seal_string:
                out["seal"] = seal_string
            self.client.update_run(
                run_id=self.root_run_id,
                outputs=out,
                error=error,
            )
        except Exception as exc:
            logger.warning("LangSmith root run completion failed: %s", exc)
