"""StART web transport, SSE, and institutional presentation package."""

try:
    from start.web.app import app, create_app
except ImportError:
    app = None  # type: ignore
    create_app = None  # type: ignore

from start.web.schemas import (
    START_SCHEMA_VERSION,
    START_VERSION,
    ReviewerHydrationResponse,
    RunRequest,
    RunStatusResponse,
    SSEEnvelope,
    WebReviewerSubmission,
)

__all__ = [
    "app",
    "create_app",
    "START_SCHEMA_VERSION",
    "START_VERSION",
    "RunRequest",
    "RunStatusResponse",
    "SSEEnvelope",
    "WebReviewerSubmission",
    "ReviewerHydrationResponse",
]
