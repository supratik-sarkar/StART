"""Frontend Static Bundle & Delivery Integrity Tests for StART v4.5."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from start.web.app import create_app

DIST_DIR = Path(__file__).resolve().parent.parent / "web" / "dist"


@pytest.fixture
def client() -> TestClient:
    app = create_app()
    return TestClient(app)


def test_index_html_delivery(client: TestClient) -> None:
    assert DIST_DIR.exists(), "web/dist must be built"
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["Content-Type"]
    assert '<div id="root"></div>' in resp.text
    assert "StART" in resp.text

    # Security Headers
    assert resp.headers["X-Content-Type-Options"] == "nosniff"
    assert resp.headers["X-Frame-Options"] == "SAMEORIGIN"
    assert "Content-Security-Policy" in resp.headers


def test_assets_integrity(client: TestClient) -> None:
    resp = client.get("/")
    assert resp.status_code == 200

    # Extract JS and CSS asset links from index.html
    scripts = re.findall(r'src="(/assets/[^"]+)"', resp.text)
    styles = re.findall(r'href="(/assets/[^"]+)"', resp.text)

    assert len(scripts) > 0, "Must include bundled scripts"
    assert len(styles) > 0, "Must include bundled stylesheets"

    for s in scripts:
        res_s = client.get(s)
        assert res_s.status_code == 200, f"Failed to fetch script {s}"
        assert len(res_s.content) > 100

    for st in styles:
        res_st = client.get(st)
        assert res_st.status_code == 200, f"Failed to fetch style {st}"
        assert len(res_st.content) > 100
