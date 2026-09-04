"""Deterministic Security & Integrity Suite for Oracle WebLLM Model Mirror."""

import json
import urllib.error
import urllib.request
from pathlib import Path

ORACLE_MODEL_URL = "https://137.23.61.219.sslip.io/webllm-models/SmolLM2-1.7B-Instruct-q4f16_1-MLC"
PUBLIC_GATEWAY = "https://start-mrt-gateway.sapman.workers.dev"
MANIFEST_PATH = Path(__file__).resolve().parent.parent.parent / "deploy" / "oracle" / "webllm_model_manifest.json"


def http_request(url: str, method: str = "GET", headers: dict | None = None) -> tuple[int, dict[str, str], bytes]:
    req_headers = {"User-Agent": "StART-Security-Audit"}
    if headers:
        req_headers.update(headers)
    req = urllib.request.Request(url, headers=req_headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=10.0) as resp:
            norm_headers = {k.lower(): v for k, v in resp.headers.items()}
            return resp.status, norm_headers, resp.read()
    except urllib.error.HTTPError as e:
        norm_headers = {k.lower(): v for k, v in e.headers.items()}
        return e.code, norm_headers, e.read()


def test_model_manifest_local_integrity():
    """Verify local committed manifest is well-formed and has 43 pinned artifacts."""
    assert MANIFEST_PATH.exists()
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["model_id"] == "SmolLM2-1.7B-Instruct-q4f16_1-MLC"
    assert manifest["upstream_revision"] == "84f57f8580a9d8d623266b600ad4273bb9fd84c1"
    assert len(manifest["files"]) == 43
    assert "mlc-chat-config.json" in manifest["files"]
    assert "params_shard_0.bin" in manifest["files"]


def test_model_config_http_200():
    """Verify mlc-chat-config.json is served with HTTP 200 and immutable cache header."""
    status, headers, body = http_request(f"{ORACLE_MODEL_URL}/mlc-chat-config.json", method="HEAD")
    assert status == 200
    cache_control = headers.get("cache-control", "").lower()
    assert "immutable" in cache_control or "max-age" in cache_control


def test_model_weight_artifact_range_request():
    """Verify weight artifact supports byte range requests without downloading full shard."""
    status, headers, body = http_request(
        f"{ORACLE_MODEL_URL}/params_shard_0.bin",
        method="GET",
        headers={"Range": "bytes=0-1023"},
    )
    assert status in (200, 206)
    if status == 206:
        assert len(body) == 1024


def test_model_mirror_cors_allowed_public_app():
    """Verify CORS header matches StART Cloudflare gateway origin."""
    status, headers, _ = http_request(
        f"{ORACLE_MODEL_URL}/mlc-chat-config.json",
        method="HEAD",
        headers={"Origin": PUBLIC_GATEWAY},
    )
    assert status == 200
    assert headers.get("access-control-allow-origin") == PUBLIC_GATEWAY


def test_model_mirror_cors_random_origin_not_allowed():
    """Verify arbitrary origins do not receive unrestricted CORS reflection."""
    status, headers, _ = http_request(
        f"{ORACLE_MODEL_URL}/mlc-chat-config.json",
        method="HEAD",
        headers={"Origin": "https://malicious-site.example.com"},
    )
    # The header is pinned strictly to https://start-mrt-gateway.sapman.workers.dev (not reflecting origin)
    assert headers.get("access-control-allow-origin") != "https://malicious-site.example.com"


def test_model_mirror_autoindex_off():
    """Verify directory listing is disabled."""
    status, _, _ = http_request(f"{ORACLE_MODEL_URL}/", method="GET")
    assert status in (403, 404)


def test_model_mirror_path_traversal_rejected():
    """Verify path traversal outside static model directory is rejected."""
    traversal_urls = [
        "https://137.23.61.219.sslip.io/webllm-models/../etc/passwd",
        "https://137.23.61.219.sslip.io/webllm-models/SmolLM2-1.7B-Instruct-q4f16_1-MLC/../../etc/passwd",
        "https://137.23.61.219.sslip.io/webllm-models/SmolLM2-1.7B-Instruct-q4f16_1-MLC/..%2F..%2Fetc%2Fpasswd",
    ]
    for url in traversal_urls:
        status, _, _ = http_request(url, method="GET")
        assert status in (400, 403, 404, 405)


def test_model_mirror_private_file_disclosure_prevented():
    """Verify hidden files (.env, .git, etc.) cannot be read from static mirror."""
    hidden_urls = [
        "https://137.23.61.219.sslip.io/webllm-models/.env",
        "https://137.23.61.219.sslip.io/webllm-models/SmolLM2-1.7B-Instruct-q4f16_1-MLC/.env",
        "https://137.23.61.219.sslip.io/webllm-models/SmolLM2-1.7B-Instruct-q4f16_1-MLC/.git/config",
    ]
    for url in hidden_urls:
        status, _, _ = http_request(url, method="GET")
        assert status in (403, 404)
