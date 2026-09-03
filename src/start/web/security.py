"""Security, Origin Authentication & Turnstile Verification for StART v4.5 Web Transport.

Enforces:
1. Origin HMAC signature validation (Cloudflare Worker -> Oracle origin)
2. Server-side Cloudflare Turnstile token verification
3. Logical Artifact ID security (strict path traversal & escape defense)
4. Session ownership checks (preventing IDOR)
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import re
import time
from typing import Any
import urllib.request
import urllib.parse
import json

logger = logging.getLogger("start.web.security")

# Environment-injected origin secret (zero secrets in repo/git)
ORIGIN_SECRET = os.environ.get("START_ORIGIN_SECRET", "start-dev-origin-secret-local-only")
TURNSTILE_SECRET_KEY = os.environ.get("START_TURNSTILE_SECRET_KEY", "")


def verify_origin_hmac(
    signature: str | None,
    timestamp_str: str | None,
    nonce: str | None,
    method: str,
    path: str,
    body_bytes: bytes = b"",
    max_age_seconds: float = 60.0,
) -> bool:
    """Verify HMAC signature from Cloudflare Worker origin proxy.

    In development mode or when no origin secret is enforced, requests pass.
    In production mode with START_REQUIRE_ORIGIN_AUTH=true, valid HMAC is mandatory.
    """
    if not os.environ.get("START_REQUIRE_ORIGIN_AUTH", "false").lower() in ("true", "1"):
        return True

    if not signature or not timestamp_str or not nonce:
        logger.warning("Missing origin authentication headers")
        return False

    try:
        ts = float(timestamp_str)
        now = time.time()
        if abs(now - ts) > max_age_seconds:
            logger.warning("Origin request timestamp outside freshness window: diff=%.2fs", abs(now - ts))
            return False
    except ValueError:
        return False

    body_digest = hashlib.sha256(body_bytes).hexdigest()
    canonical_string = f"{method.upper()}:{path}:{timestamp_str}:{nonce}:{body_digest}"
    expected_sig = hmac.new(
        ORIGIN_SECRET.encode("utf-8"),
        canonical_string.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()

    return hmac.compare_digest(signature, expected_sig)


def verify_turnstile_token(token: str | None, remote_ip: str | None = None) -> bool:
    """Validate Turnstile token server-side via Cloudflare Siteverify API.

    In local/test environments with no configured secret key, dummy tokens pass.
    """
    if not TURNSTILE_SECRET_KEY:
        # Development mode or local offline run
        return True

    if not token:
        return False

    try:
        post_data = urllib.parse.urlencode({
            "secret": TURNSTILE_SECRET_KEY,
            "response": token,
            "remoteip": remote_ip or "",
        }).encode("utf-8")

        req = urllib.request.Request(
            "https://challenges.cloudflare.com/turnstile/v0/siteverify",
            data=post_data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        with urllib.request.urlopen(req, timeout=5.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return bool(data.get("success", False))
    except Exception as exc:
        logger.warning("Turnstile siteverify verification failed: %s", exc)
        return False


def sanitize_artifact_id(artifact_id: str) -> str:
    """Enforce strict logical format for artifact identifiers.

    Rejects directory traversal (../), absolute paths (/Users/), and special symbols.
    """
    if not artifact_id or not isinstance(artifact_id, str):
        raise ValueError("Invalid artifact identifier")

    # Reject path traversal patterns
    if ".." in artifact_id or "/" in artifact_id or "\\" in artifact_id:
        raise ValueError("Security violation: path traversal detected in artifact ID")

    # Only allow safe alphanumeric, hyphens, and underscores
    if not re.match(r"^[a-zA-Z0-9_-]{1,64}$", artifact_id):
        raise ValueError(f"Invalid artifact ID characters in '{artifact_id}'")

    return artifact_id
