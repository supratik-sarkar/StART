"""Packaging contract.

These tests exist because of specific, previously observed failures, and each
one names the failure it prevents. A contract test with no remembered incident
behind it tends to get deleted the first time it is inconvenient.

The incidents:

* An ``observability`` extra declared Langfuse but not LangSmith or Phoenix,
  while the codebase shipped adapters for all three. A "full" install produced
  an adapter reporting ``not_installed`` forever.
* ``all = ["start-mrt[...]"]`` — a self-referencing extra. pip resolves it, but
  the behaviour is confusing enough that nobody could say what a full install
  contained.
* ``requirements.txt`` drifted away from ``pyproject.toml`` and became a second,
  contradictory source of truth.
* ``requires-python``, ruff's ``target-version`` and mypy's ``python_version``
  disagreed (3.10 / py310 / 3.10 against a 3.12 CI), so linting enforced a
  standard the project had already left behind.
"""

from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"

#: Unions of other extras. Excluded from the "no duplication" style checks
#: because duplication is exactly what they are for.
ALIAS_EXTRAS = {"everything", "all", "torch"}


@pytest.fixture(scope="module")
def meta() -> dict:
    with PYPROJECT.open("rb") as handle:
        return tomllib.load(handle)


@pytest.fixture(scope="module")
def extras(meta: dict) -> dict[str, list[str]]:
    return meta["project"].get("optional-dependencies", {})


def test_no_extra_references_the_package_itself(extras: dict[str, list[str]]) -> None:
    """No ``start-mrt[...]`` recursion anywhere in the extras."""
    offenders = {
        name: [spec for spec in specs if "start-mrt" in spec.replace("_", "-").lower()]
        for name, specs in extras.items()
    }
    offenders = {name: specs for name, specs in offenders.items() if specs}
    assert not offenders, (
        "Extras must declare real requirements, never the package itself. "
        f"Self-referencing extras: {offenders}"
    )


def test_everything_is_the_union_of_all_real_extras(extras: dict[str, list[str]]) -> None:
    """``everything`` must actually mean everything.

    This is the check that would have caught the missing Phoenix and LangSmith
    declarations at commit time rather than at demo time.
    """
    union: set[str] = set()
    for name, specs in extras.items():
        if name in ALIAS_EXTRAS:
            continue
        union |= set(specs)

    declared = set(extras["everything"])
    missing = union - declared
    assert not missing, (
        "The 'everything' extra is missing requirements declared elsewhere: "
        f"{sorted(missing)}. Add them, or the full install will silently lack them."
    )

    spurious = declared - union
    assert not spurious, (
        "The 'everything' extra declares requirements that belong to no other extra: "
        f"{sorted(spurious)}. Put them in a named extra first."
    )


def test_python_version_floor_is_consistent(meta: dict) -> None:
    """requires-python, ruff and mypy must agree on the supported version."""
    requires_python = meta["project"]["requires-python"]
    ruff_target = meta["tool"]["ruff"]["target-version"]
    mypy_version = meta["tool"]["mypy"]["python_version"]

    assert requires_python == ">=3.12", requires_python
    assert ruff_target == "py312", (
        f"ruff target-version is {ruff_target!r} but the project requires {requires_python}. "
        "Linting against an older target permits syntax the project has moved past."
    )
    assert mypy_version == "3.12", mypy_version


def test_requirements_txt_is_generated_and_current() -> None:
    """requirements.txt is a mirror, and mirrors must not lag."""
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "sync_requirements.py"), "--check"],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert result.returncode == 0, (
        "requirements.txt is out of sync with pyproject.toml. "
        "Run: python scripts/sync_requirements.py\n" + result.stdout + result.stderr
    )


def test_requirements_txt_is_marked_generated() -> None:
    text = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert "GENERATED FILE" in text and "DO NOT EDIT" in text, (
        "requirements.txt must announce that it is generated, or someone will edit it "
        "and reintroduce a second source of truth."
    )


def test_bootstrap_profiles_only_name_declared_extras(extras: dict[str, list[str]]) -> None:
    """The installer and the metadata cannot drift apart."""
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        import bootstrap  # type: ignore[import-not-found]
    finally:
        sys.path.pop(0)

    for profile_name, spec in bootstrap.PROFILES.items():
        unknown = [e for e in spec["extras"] if e not in extras]
        assert not unknown, (
            f"bootstrap profile {profile_name!r} names extras that pyproject.toml does not declare: {unknown}"
        )


def test_bootstrap_dry_run_needs_no_network() -> None:
    """``--dry-run`` must be usable offline; it is the reviewable path."""
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "bootstrap.py"), "--dry-run", "--profile", "minimal"],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert result.returncode == 0, result.stderr
    assert "Dry run" in result.stdout
    assert "numpy" in result.stdout


def test_single_installer_entry_point() -> None:
    """One installer. The shell wrapper must not carry its own logic."""
    scripts = ROOT / "scripts"
    assert (scripts / "bootstrap.py").exists()
    assert not (scripts / "install_dependencies.py").exists(), (
        "install_dependencies.py is a competing installer and was removed. scripts/bootstrap.py is canonical."
    )

    wrapper = (scripts / "bootstrap.sh").read_text(encoding="utf-8")
    assert "bootstrap.py" in wrapper, "bootstrap.sh must delegate to bootstrap.py"
    assert "pip install" not in wrapper, (
        "bootstrap.sh contains its own pip logic. Two installers is how macOS and Linux "
        "instructions drift apart; delegate instead."
    )


def test_project_version_and_package_version_agree(meta: dict) -> None:
    sys.path.insert(0, str(ROOT / "src"))
    try:
        import start
    finally:
        sys.path.pop(0)
    assert start.__version__ == meta["project"]["version"]
