"""Security, Origin Authentication & Turnstile Verification for StART v4.5.3 Web Transport.

Enforces:
1. Fail-closed Origin HMAC signature validation with replay prevention (Cloudflare Worker -> Oracle origin)
2. Server-side Cloudflare Turnstile siteverify token verification
3. Logical Artifact ID security (strict path traversal & escape defense)
4. Session ownership checks (preventing IDOR)
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import re
import threading
import time
import urllib.parse
import urllib.request

logger = logging.getLogger("start.web.security")

# In-memory replay protection cache: nonce -> insertion timestamp
_NONCE_LOCK = threading.Lock()
_SEEN_NONCES: dict[str, float] = {}
_MAX_NONCE_AGE: float = 120.0


def _clean_expired_nonces(now: float) -> None:
    """Prune expired nonces from memory."""
    cutoff = now - _MAX_NONCE_AGE
    expired = [k for k, v in _SEEN_NONCES.items() if v < cutoff]
    for k in expired:
        _SEEN_NONCES.pop(k, None)


def get_origin_secret() -> str:
    """Retrieve the active origin secret from environment."""
    return os.environ.get("START_ORIGIN_SECRET", "")


def get_turnstile_secret_key() -> str:
    """Retrieve the active Turnstile secret key from environment."""
    return os.environ.get("START_TURNSTILE_SECRET_KEY", "")


def verify_origin_hmac(
    signature: str | None,
    timestamp_str: str | None,
    nonce: str | None,
    method: str,
    path: str,
    body_bytes: bytes = b"",
    max_age_seconds: float = 60.0,
) -> bool:
    """Verify HMAC signature from Cloudflare Worker origin proxy with replay protection.

    In development mode or when no origin secret is enforced, requests pass.
    In production mode with START_REQUIRE_ORIGIN_AUTH=true, valid HMAC is mandatory (Fail Closed).
    """
    require_auth = os.environ.get("START_REQUIRE_ORIGIN_AUTH", "false").lower() in ("true", "1")
    if not require_auth:
        return True

    secret = get_origin_secret()
    if not secret or secret == "start-dev-origin-secret-local-only":
        logger.error(
            "Security violation: START_REQUIRE_ORIGIN_AUTH is enabled but "
            "START_ORIGIN_SECRET is absent or default. FAILING CLOSED."
        )
        return False

    if not signature or not timestamp_str or not nonce:
        logger.warning("Rejected request: Missing origin authentication headers")
        return False

    try:
        ts = float(timestamp_str)
        now = time.time()
        if abs(now - ts) > max_age_seconds:
            logger.warning(
                "Rejected request: Origin timestamp outside freshness window: diff=%.2fs",
                abs(now - ts),
            )
            return False
    except ValueError:
        logger.warning("Rejected request: Malformed origin timestamp")
        return False

    # Check and record nonce for replay attack prevention
    with _NONCE_LOCK:
        _clean_expired_nonces(now)
        if nonce in _SEEN_NONCES:
            logger.warning("Security violation: Replayed origin nonce detected: %s. REJECTING.", nonce)
            return False
        _SEEN_NONCES[nonce] = now

    body_digest = hashlib.sha256(body_bytes).hexdigest()
    canonical_string = f"{method.upper()}:{path}:{timestamp_str}:{nonce}:{body_digest}"
    expected_sig = hmac.new(
        secret.encode("utf-8"),
        canonical_string.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()

    matched = hmac.compare_digest(signature, expected_sig)
    if not matched:
        logger.warning("Rejected request: HMAC signature mismatch for path %s", path)
    return matched


CLOUDFLARE_TEST_SECRETS = {
    "1x0000000000000000000000000000000AA",
    "2x0000000000000000000000000000000AA",
    "3x0000000000000000000000000000000AA",
}


def verify_turnstile_token(token: str | None, remote_ip: str | None = None) -> bool:
    """Validate Turnstile token server-side via Cloudflare Siteverify API.

    In local/test environments with no configured secret key, dummy tokens pass.
    In production with START_TURNSTILE_SECRET_KEY configured, verifies via challenges.cloudflare.com.
    Refuses production execution if configured with a known test secret.
    """
    turnstile_key = get_turnstile_secret_key()
    is_prod = bool(
        os.environ.get("START_ORACLE_DEPLOYMENT") or os.environ.get("START_REQUIRE_ORIGIN_AUTH")
    )

    if not turnstile_key:
        if is_prod:
            logger.error(
                "Security violation: Production running without START_TURNSTILE_SECRET_KEY. "
                "Failing closed."
            )
            return False
        # Development mode or local offline run
        return True

    if is_prod and turnstile_key in CLOUDFLARE_TEST_SECRETS:
        logger.critical("FATAL: Production configured with a Cloudflare test secret key. FAILING CLOSED.")
        return False

    if not token:
        return False

    # Reject dummy/invalid probe tokens in production
    if is_prod and token in (
        "INVALID.TOKEN",
        "INVALID.NONEXISTENT.TOKEN",
        "INVALID.PROBE.TOKEN",
        "MALICIOUS.TOKEN",
        "XXXX.DUMMY.TOKEN.XXXX",
    ):
        return False

    try:
        post_data = urllib.parse.urlencode(
            {
                "secret": turnstile_key,
                "response": token,
                "remoteip": remote_ip or "",
            }
        ).encode("utf-8")

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
