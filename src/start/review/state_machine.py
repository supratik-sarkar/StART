"""Checkpoint State Machine and Exception Taxonomy for StART Review Control-Plane.

Defines the explicit state transitions, invariant checks, and typed exception taxonomy
required for robust, proof-carrying interactive review.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from start.review.multiline_input import ReviewCancelled

__all__ = [
    "ReviewCancelled",
    "ProviderInvocationError",
    "GroundingValidationError",
    "GroundingRepairError",
    "InvalidStateTransitionError",
    "CheckpointState",
    "CheckpointStateMachine",
]


class ProviderInvocationError(Exception):
    """Raised when an external or local LLM provider request fails."""

    def __init__(
        self,
        message: str,
        *,
        provider: str = "",
        model: str = "",
        original_exc: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.model = model
        self.original_exc = original_exc


class GroundingValidationError(Exception):
    """Raised when reviewer claims fail deterministic evidence grounding."""

    def __init__(
        self,
        message: str,
        *,
        unbound_claims: list[dict[str, Any]] | None = None,
        reason_code: str = "",
    ) -> None:
        super().__init__(message)
        self.unbound_claims = unbound_claims or []
        self.reason_code = reason_code


class GroundingRepairError(Exception):
    """Raised when an automated grounding repair attempt fails."""

    def __init__(
        self,
        message: str,
        *,
        attempt: int = 1,
        remaining_unbound: list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(message)
        self.attempt = attempt
        self.remaining_unbound = remaining_unbound or []


class InvalidStateTransitionError(RuntimeError):
    """Raised when an illegal transition is attempted in the Checkpoint State Machine."""


class CheckpointState(StrEnum):
    """Explicit lifecycle states for a domain review checkpoint."""

    READY = "READY"
    PROVIDER_CALL = "PROVIDER_CALL"
    PROVIDER_RESPONSE = "PROVIDER_RESPONSE"
    GROUNDING_VALIDATE = "GROUNDING_VALIDATE"
    GROUNDING_REPAIR = "GROUNDING_REPAIR"
    VERIFIED = "VERIFIED"
    FALLBACK_OFFERED = "FALLBACK_OFFERED"
    DETERMINISTIC_FALLBACK = "DETERMINISTIC_FALLBACK"
    CANCELLED = "CANCELLED"
    COMPLETED = "COMPLETED"


# Legal transitions map
_VALID_TRANSITIONS: dict[CheckpointState, set[CheckpointState]] = {
    CheckpointState.READY: {
        CheckpointState.PROVIDER_CALL,
        CheckpointState.DETERMINISTIC_FALLBACK,
        CheckpointState.COMPLETED,
        CheckpointState.CANCELLED,
    },
    CheckpointState.PROVIDER_CALL: {
        CheckpointState.PROVIDER_RESPONSE,
        CheckpointState.FALLBACK_OFFERED,
        CheckpointState.CANCELLED,
    },
    CheckpointState.PROVIDER_RESPONSE: {
        CheckpointState.GROUNDING_VALIDATE,
        CheckpointState.FALLBACK_OFFERED,
        CheckpointState.CANCELLED,
    },
    CheckpointState.GROUNDING_VALIDATE: {
        CheckpointState.VERIFIED,
        CheckpointState.GROUNDING_REPAIR,
        CheckpointState.FALLBACK_OFFERED,
        CheckpointState.CANCELLED,
    },
    CheckpointState.GROUNDING_REPAIR: {
        CheckpointState.VERIFIED,
        CheckpointState.FALLBACK_OFFERED,
        CheckpointState.CANCELLED,
    },
    CheckpointState.VERIFIED: {
        CheckpointState.COMPLETED,
    },
    CheckpointState.FALLBACK_OFFERED: {
        CheckpointState.DETERMINISTIC_FALLBACK,
        CheckpointState.CANCELLED,
    },
    CheckpointState.DETERMINISTIC_FALLBACK: {
        CheckpointState.COMPLETED,
    },
    CheckpointState.CANCELLED: set(),  # Terminal state: NO outgoing transitions
    CheckpointState.COMPLETED: set(),  # Terminal state for checkpoint: NO outgoing transitions
}


class CheckpointStateMachine:
    """Deterministic state machine governing a single checkpoint's lifecycle."""

    def __init__(self, checkpoint_title: str) -> None:
        self.checkpoint_title = checkpoint_title
        self._current_state = CheckpointState.READY
        self._history: list[CheckpointState] = [CheckpointState.READY]
        self._terminal_decision: str | None = None
        self._repair_attempts: int = 0
        self._fallback_offers: int = 0

    @property
    def current_state(self) -> CheckpointState:
        return self._current_state

    @property
    def history(self) -> tuple[CheckpointState, ...]:
        return tuple(self._history)

    @property
    def repair_attempts(self) -> int:
        return self._repair_attempts

    @property
    def fallback_offers(self) -> int:
        return self._fallback_offers

    @property
    def is_terminal(self) -> bool:
        return self._current_state in (CheckpointState.CANCELLED, CheckpointState.COMPLETED)

    def transition(self, new_state: CheckpointState) -> None:
        """Attempt to transition to a new state, validating against transition rules."""
        if self._current_state == CheckpointState.CANCELLED:
            raise InvalidStateTransitionError(
                f"Cannot transition from terminal state CANCELLED to {new_state}."
            )

        if self._current_state == CheckpointState.COMPLETED:
            raise InvalidStateTransitionError(
                f"Cannot transition from terminal state COMPLETED to {new_state}."
            )

        allowed = _VALID_TRANSITIONS.get(self._current_state, set())
        if new_state not in allowed:
            raise InvalidStateTransitionError(
                f"Illegal checkpoint state transition: {self._current_state} -> {new_state}."
            )

        if new_state == CheckpointState.GROUNDING_REPAIR:
            self._repair_attempts += 1
            if self._repair_attempts > 1:
                raise InvalidStateTransitionError(
                    f"Invariant violated: At most 1 grounding repair attempt is allowed "
                    f"(attempt {self._repair_attempts})."
                )

        if new_state == CheckpointState.FALLBACK_OFFERED:
            self._fallback_offers += 1
            if self._fallback_offers > 1:
                raise InvalidStateTransitionError(
                    f"Invariant violated: At most 1 fallback menu offer is allowed "
                    f"(offer {self._fallback_offers})."
                )

        self._current_state = new_state
        self._history.append(new_state)

    def record_decision(self, decision: str) -> None:
        """Record final terminal decision, ensuring no failure branch produces multiple decisions."""
        if self._terminal_decision is not None and self._terminal_decision != decision:
            raise InvalidStateTransitionError(
                f"Invariant violated: Terminal decision already recorded as '{self._terminal_decision}', "
                f"cannot record '{decision}'."
            )
        self._terminal_decision = decision
