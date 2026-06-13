from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

from start.core.config import LLMConfig
from start.enterprise import EnterpriseLLMGatewayAdapter
from start.enterprise.llm_gateway import (
    SUPPORTED_METADATA,
    enterprise_package_available,
)
from start.providers.llm import get_llm_provider

REPO_ROOT = Path(__file__).resolve().parents[1]
FAKE_KEY = "sk-enterprise-FAKE-do-not-leak-4242"


def test_adapter_interface_matches_public_providers():
    adapter = EnterpriseLLMGatewayAdapter()
    assert adapter.provider_name == "enterprise_llm_gateway"
    # same signature as public providers: generate(prompt, *, system, metadata)
    assert hasattr(adapter, "generate") and hasattr(adapter, "available")
    assert set(SUPPORTED_METADATA) == {
        "run_id",
        "agent_name",
        "section_name",
        "max_tokens",
        "temperature",
        "model_name",
        "evidence_ids",
    }


def test_unavailable_when_package_absent():
    # The private package is not installed in the public/test environment.
    assert not enterprise_package_available()
    assert EnterpriseLLMGatewayAdapter().available() is False


def test_generate_raises_not_implemented_not_import_error():
    adapter = EnterpriseLLMGatewayAdapter()
    with pytest.raises(NotImplementedError):
        adapter.generate("evidence bundle only", system="sys", metadata={"run_id": "R"})
    # crucially NOT an ImportError — selecting enterprise must never crash
    try:
        adapter.generate("x")
    except ImportError:  # pragma: no cover - would be a defect
        pytest.fail("enterprise adapter must not raise ImportError when package is absent")
    except NotImplementedError:
        pass


def test_public_repo_does_not_require_enterprise_package():
    # No module under src/start imports the private package at top level.
    enterprise_pkg = "enterprise_package"
    src = REPO_ROOT / "src" / "start"
    offenders = []
    for path in src.rglob("*.py"):
        text = path.read_text()
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith(("import ", "from ")) and enterprise_pkg in stripped:
                offenders.append(str(path.relative_to(REPO_ROOT)))
    assert offenders == [], f"top-level import of private package found in: {offenders}"
    # And importing the adapter module never pulls the package in.
    assert "enterprise_package" not in sys.modules


def test_selecting_enterprise_provider_never_errors_and_degrades():
    # The factory must not raise and must degrade to deterministic (NoLLM).
    provider = get_llm_provider(LLMConfig(provider="enterprise_llm_gateway"))
    from start.providers.llm import NoLLMProvider

    assert isinstance(provider, NoLLMProvider)
    assert provider.available is False


def test_deterministic_fallback_still_works_with_enterprise_selected(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # Build a run, then drive the agent review in llm mode with the enterprise
    # provider unavailable -> explicit deterministic fallback, no crash.
    from start.agents.review import run_agent_review
    from start.modeling.data import load_attrition_dataset
    from start.orchestration import review_dataframes

    result = review_dataframes(load_attrition_dataset(seed=7), target_column="attrition", seed=7)
    enterprise = get_llm_provider(LLMConfig(provider="enterprise_llm_gateway"))
    review = run_agent_review(result.evidence, mode="llm", llm=enterprise)
    assert review.mode == "deterministic"  # explicit fallback
    assert any("deterministic" in n.lower() or "fell back" in n.lower() for n in review.notes)
    assert review.critique_ok


def test_available_flips_true_when_package_present(tmp_path, monkeypatch):
    # Simulate a firm environment by putting a fake package on sys.path.
    pkg_root = tmp_path / "fakefirm"
    (pkg_root / "enterprise_package").mkdir(parents=True)
    (pkg_root / "enterprise_package" / "__init__.py").write_text("VERSION = '1.0'\n")
    monkeypatch.syspath_prepend(str(pkg_root))
    importlib.invalidate_caches()
    assert enterprise_package_available() is True
    adapter = EnterpriseLLMGatewayAdapter()
    assert adapter.available() is True
    # generate now reaches the 'implement me' branch (still NotImplementedError)
    with pytest.raises(NotImplementedError, match="private adapter"):
        adapter.generate("bundle", metadata={"agent_name": "challenge"})


def test_env_var_overrides_package_name(tmp_path, monkeypatch):
    pkg_root = tmp_path / "firm2"
    (pkg_root / "ms_internal_llm").mkdir(parents=True)
    (pkg_root / "ms_internal_llm" / "__init__.py").write_text("")
    monkeypatch.syspath_prepend(str(pkg_root))
    monkeypatch.setenv("START_ENTERPRISE_PACKAGE", "ms_internal_llm")
    importlib.invalidate_caches()
    assert enterprise_package_available() is True


def test_no_secret_leakage_with_enterprise_selected(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", FAKE_KEY)
    from start.modeling.data import load_attrition_dataset
    from start.orchestration import review_dataframes

    result = review_dataframes(load_attrition_dataset(seed=9), target_column="attrition", seed=9)
    assert result.run_id  # run completed; now scan all artifacts for the key
    for artifact in Path(tmp_path).rglob("*"):
        if artifact.is_file() and artifact.suffix in {".md", ".jsonl", ".json", ".txt"}:
            assert FAKE_KEY not in artifact.read_text(errors="ignore"), artifact


def test_enterprise_readme_is_public_safe():
    readme = (REPO_ROOT / "src" / "start" / "enterprise" / "README.md").read_text().lower()
    # no firm name, endpoints, or credentials in the public adapter docs
    for forbidden in ("morgan stanley", "https://", "http://", "api_key=", "password"):
        assert forbidden not in readme, f"public-safety: '{forbidden}' must not appear"
