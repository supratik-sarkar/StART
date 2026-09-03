#!/usr/bin/env python3
"""StART canonical installer.

    git clone https://github.com/start-project/start.git
    cd StART
    python scripts/bootstrap.py

That is the whole contract. A visitor should never have to work out whether
they need ``.[all]``, ``.[everything]``, ``.[ai-engineering]``,
``requirements.txt``, or some combination — the choice belongs here, not in
their head.

Design notes worth knowing before editing this file:

* **Standard library only.** It runs *before* anything is installed, so it
  cannot import anything that installation would provide. That includes TOML
  writers and rich console libraries. ``tomllib`` is standard from 3.11, so
  reading ``pyproject.toml`` is fine; writing it is not attempted.
* **pyproject.toml is the source of truth.** Extras are read from it at run
  time. Adding an extra there makes it available here with no edit. A profile
  naming an extra that does not exist is a hard error, so the two cannot drift
  apart silently.
* **Python packages and external executables are different things.** OPA and
  Promptfoo are binaries, not wheels. Reporting them as "missing dependencies"
  sends people to pip, where they will not find them. They are probed and
  reported separately.
* **``--dry-run`` must work with no network.** Being able to see exactly what
  would be installed, offline, is the difference between a reviewable installer
  and a magic one.

Exit codes: 0 success, 1 failure, 2 usage/environment error.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import venv
from pathlib import Path
from typing import Any

MIN_PYTHON = (3, 12)
DEFAULT_VENV = ".venv-start"
PACKAGE_NAME = "start-mrt"

# --------------------------------------------------------------------------- #
# Installation profiles
# --------------------------------------------------------------------------- #
# Each profile is a set of extras declared in pyproject.toml. Kept small and
# explicit: four choices a person can reason about, rather than twenty extras
# they must combine correctly.
PROFILES: dict[str, dict[str, Any]] = {
    "minimal": {
        "extras": [],
        "summary": "Core only. The deterministic engines, the risk core and the CLI.",
    },
    "demo": {
        "extras": ["dl", "trees", "tuning", "xai", "formats", "llm", "dev"],
        "summary": (
            "The public demonstration stack: modelling, explainability, the OpenAI and "
            "Anthropic client SDKs, and the test tooling. This is the default."
        ),
    },
    "enterprise": {
        "extras": [
            "dl",
            "trees",
            "tuning",
            "xai",
            "formats",
            "llm",
            "telemetry",
            "observability",
            "mcp",
            "evals",
            "guardrails",
            "orchestration",
            "dev",
        ],
        "summary": (
            "Everything the demo profile installs, plus the AI-engineering adapter "
            "backends. Note that the LLM client SDKs are included: they are the HTTP "
            "transport for an OpenAI-compatible internal gateway. Which endpoints may "
            "actually be reached is decided by START_PROFILE, not by what is installed."
        ),
    },
    "everything": {
        "extras": ["everything"],
        "summary": "Every declared extra, including the heavy optional stacks.",
    },
}

DEFAULT_PROFILE = "demo"

# Imports verified after installation. (module, label, profiles that require it)
IMPORT_CHECKS: list[tuple[str, str, tuple[str, ...]]] = [
    ("numpy", "NumPy", ("minimal", "demo", "enterprise", "everything")),
    ("pandas", "pandas", ("minimal", "demo", "enterprise", "everything")),
    ("scipy", "SciPy", ("minimal", "demo", "enterprise", "everything")),
    ("sklearn", "scikit-learn", ("minimal", "demo", "enterprise", "everything")),
    ("pydantic", "Pydantic", ("minimal", "demo", "enterprise", "everything")),
    ("typer", "Typer", ("minimal", "demo", "enterprise", "everything")),
    ("rich", "Rich", ("minimal", "demo", "enterprise", "everything")),
    ("torch", "PyTorch", ("demo", "enterprise", "everything")),
    ("openai", "OpenAI SDK", ("demo", "enterprise", "everything")),
    ("anthropic", "Anthropic SDK", ("demo", "enterprise", "everything")),
    ("shap", "SHAP", ("demo", "enterprise", "everything")),
    ("pytest", "pytest", ("demo", "enterprise", "everything")),
    ("opentelemetry.sdk", "OpenTelemetry", ("enterprise", "everything")),
    ("langfuse", "Langfuse", ("enterprise", "everything")),
    ("langsmith", "LangSmith", ("enterprise", "everything")),
    ("phoenix", "Arize Phoenix", ("enterprise", "everything")),
    ("mcp", "MCP SDK", ("enterprise", "everything")),
    ("deepeval", "DeepEval", ("enterprise", "everything")),
    ("nemoguardrails", "NeMo Guardrails", ("enterprise", "everything")),
    ("langgraph", "LangGraph", ("enterprise", "everything")),
    ("garak", "Garak", ("everything",)),
]

# Binaries, not wheels. Absence is reported, never treated as a failure.
EXTERNAL_TOOLS: list[tuple[str, str, str]] = [
    ("opa", "Open Policy Agent", "https://www.openpolicyagent.org/docs/latest/#running-opa"),
    ("promptfoo", "Promptfoo", "npm install -g promptfoo"),
    ("git", "Git", "https://git-scm.com/downloads"),
]


# --------------------------------------------------------------------------- #
# Small console helpers (no dependencies)
# --------------------------------------------------------------------------- #
def _supports_colour() -> bool:
    return sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


_C = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "red": "\033[31m",
    "cyan": "\033[36m",
}


def c(text: str, *styles: str) -> str:
    if not _supports_colour():
        return text
    return "".join(_C[s] for s in styles) + text + _C["reset"]


def heading(text: str) -> None:
    print()
    print(c(f"── {text} " + "─" * max(0, 68 - len(text)), "cyan", "bold"))


def ok(text: str) -> None:
    print(f"  {c('✓', 'green')} {text}")


def warn(text: str) -> None:
    print(f"  {c('!', 'yellow')} {text}")


def fail(text: str) -> None:
    print(f"  {c('✗', 'red')} {text}")


# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def check_python() -> None:
    if sys.version_info < MIN_PYTHON:
        required = ".".join(map(str, MIN_PYTHON))
        raise SystemExit(
            c(
                f"StART requires Python >= {required}; this interpreter is "
                f"{platform.python_version()} at {sys.executable}.\n"
                f"Install Python {required}+ and re-run with that interpreter, e.g.\n"
                f"    python{required} scripts/bootstrap.py",
                "red",
            )
        )


def read_declared_extras(pyproject: Path) -> dict[str, list[str]]:
    """Read ``[project.optional-dependencies]`` — the single source of truth."""
    import tomllib

    with pyproject.open("rb") as handle:
        data = tomllib.load(handle)
    return data.get("project", {}).get("optional-dependencies", {})


def venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def activation_hint(venv_dir: Path) -> list[str]:
    name = venv_dir.name
    return [
        f"macOS / Linux (bash, zsh):   source {name}/bin/activate",
        f"Windows PowerShell:          {name}\\Scripts\\Activate.ps1",
        f"Windows cmd.exe:             {name}\\Scripts\\activate.bat",
        f"fish:                        source {name}/bin/activate.fish",
    ]


def run(command: list[str], *, cwd: Path | None = None, check: bool = True) -> int:
    print(c(f"  $ {' '.join(command)}", "dim"))
    result = subprocess.run(command, cwd=str(cwd) if cwd else None)
    if check and result.returncode != 0:
        raise SystemExit(
            c(f"\nCommand failed with exit code {result.returncode}:\n  {' '.join(command)}", "red")
        )
    return result.returncode


def module_present(python_exe: Path, module: str) -> bool:
    code = f"import importlib.util as u, sys; sys.exit(0 if u.find_spec({module!r}) else 1)"
    return subprocess.run([str(python_exe), "-c", code], capture_output=True).returncode == 0


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bootstrap.py",
        description="Create a clean StART environment and install a supported dependency set.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Profiles:\n"
        + "\n".join(f"  {name:<12} {spec['summary']}" for name, spec in PROFILES.items()),
    )
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILES),
        default=DEFAULT_PROFILE,
        help=f"Dependency set to install. Default: {DEFAULT_PROFILE}",
    )
    parser.add_argument(
        "--venv",
        default=DEFAULT_VENV,
        help=f"Virtual environment directory name. Default: {DEFAULT_VENV}",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete an existing environment of the same name and rebuild from scratch.",
    )
    parser.add_argument(
        "--constraints",
        metavar="FILE",
        help="Pin versions using a pip constraints file (e.g. constraints.txt).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show exactly what would be installed and exit. Requires no network.",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip post-install import verification (not recommended).",
    )
    parser.add_argument(
        "--json-report",
        metavar="FILE",
        help="Write a machine-readable installation report to FILE.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    check_python()

    root = repo_root()
    pyproject = root / "pyproject.toml"
    if not pyproject.exists():
        raise SystemExit(
            c(f"pyproject.toml not found at {pyproject}. Run this from a StART checkout.", "red")
        )

    declared = read_declared_extras(pyproject)
    profile = PROFILES[args.profile]
    extras: list[str] = list(profile["extras"])

    missing = [e for e in extras if e not in declared]
    if missing:
        raise SystemExit(
            c(
                f"Profile '{args.profile}' names extra(s) not declared in pyproject.toml: "
                f"{', '.join(missing)}.\nDeclared extras: {', '.join(sorted(declared))}\n"
                "pyproject.toml is authoritative; fix it there or fix PROFILES here.",
                "red",
            )
        )

    venv_dir = root / args.venv
    target = f".[{','.join(extras)}]" if extras else "."

    heading("StART bootstrap")
    print(f"  repository      {root}")
    print(f"  interpreter     {platform.python_version()}  ({sys.executable})")
    print(f"  platform        {platform.system()} {platform.machine()}")
    print(f"  profile         {c(args.profile, 'bold')} — {profile['summary']}")
    print(f"  environment     {venv_dir}")
    print(f"  install target  {target}")
    if args.constraints:
        print(f"  constraints     {args.constraints}")

    if args.dry_run:
        heading("Dry run — nothing will be installed")
        print("  Packages that would be installed:\n")
        core = read_core_dependencies(pyproject)
        for spec in core:
            print(f"    {spec}   {c('(core)', 'dim')}")
        for extra in extras:
            for spec in declared[extra]:
                print(f"    {spec}   {c('(' + extra + ')', 'dim')}")
        total = len(core) + sum(len(declared[e]) for e in extras)
        print(f"\n  {total} requirement specifiers across core + {len(extras)} extra(s).")
        heading("Commands that would run")
        for line in [
            f"{sys.executable} -m venv {venv_dir}",
            f"{venv_python(venv_dir)} -m pip install --upgrade pip setuptools wheel",
            f"{venv_python(venv_dir)} -m pip install -e {target}",
            f"{venv_python(venv_dir)} -m pip check",
        ]:
            print(c(f"  $ {line}", "dim"))
        return 0

    # -- create the environment --------------------------------------------
    heading("Virtual environment")
    if venv_dir.exists():
        if args.recreate:
            warn(f"removing existing {venv_dir}")
            shutil.rmtree(venv_dir)
        else:
            raise SystemExit(
                c(
                    f"{venv_dir} already exists.\n"
                    "Pass --recreate to rebuild it, or --venv NAME to use a different name.\n\n"
                    "A pre-existing environment is not evidence that installation works: it may "
                    "carry packages installed by hand months ago. Validate against a fresh one.",
                    "yellow",
                )
            )
    venv.EnvBuilder(with_pip=True, clear=False, symlinks=os.name != "nt").create(venv_dir)
    ok(f"created {venv_dir}")

    python_exe = venv_python(venv_dir)
    if not python_exe.exists():
        raise SystemExit(c(f"Environment created but {python_exe} is missing.", "red"))

    # -- install ------------------------------------------------------------
    heading("Installing")
    run([str(python_exe), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
    install_cmd = [str(python_exe), "-m", "pip", "install", "-e", target]
    if args.constraints:
        install_cmd += ["-c", args.constraints]
    run(install_cmd, cwd=root)

    heading("Dependency resolution check")
    if run([str(python_exe), "-m", "pip", "check"], cwd=root, check=False) == 0:
        ok("pip check: no broken requirements")
    else:
        warn("pip check reported conflicts (shown above). Review before relying on this env.")

    # -- verify -------------------------------------------------------------
    report: dict[str, Any] = {
        "profile": args.profile,
        "extras": extras,
        "venv": str(venv_dir),
        "python": platform.python_version(),
        "platform": f"{platform.system()} {platform.machine()}",
        "imports": {},
        "external_tools": {},
    }
    verification_failed: list[str] = []

    if not args.no_verify:
        heading("Import verification")
        for module, label, profiles in IMPORT_CHECKS:
            required = args.profile in profiles
            present = module_present(python_exe, module)
            report["imports"][module] = {"label": label, "required": required, "present": present}
            if present:
                ok(f"{label:<22} {c('present', 'green')}")
            elif required:
                fail(f"{label:<22} {c('MISSING (required by this profile)', 'red')}")
                verification_failed.append(label)
            else:
                print(f"  {c('·', 'dim')} {label:<22} {c('not in this profile', 'dim')}")

        heading("External executables (not Python packages)")
        print(
            c(
                "  These are binaries. pip cannot install them, and their absence is not an\n"
                "  installation failure — StART's adapters report them separately.\n",
                "dim",
            )
        )
        for binary, label, hint in EXTERNAL_TOOLS:
            found = shutil.which(binary)
            report["external_tools"][binary] = {"label": label, "path": found}
            if found:
                ok(f"{label:<22} {found}")
            else:
                print(f"  {c('·', 'dim')} {label:<22} not on PATH  {c('(' + hint + ')', 'dim')}")

    # -- StART itself -------------------------------------------------------
    heading("StART")
    version_code = f"import importlib.metadata as m, sys; sys.stdout.write(m.version({PACKAGE_NAME!r}))"
    result = subprocess.run(
        [str(python_exe), "-c", version_code], capture_output=True, text=True, cwd=str(root)
    )
    if result.returncode == 0:
        report["start_version"] = result.stdout.strip()
        ok(f"start-mrt {result.stdout.strip()} installed")
    else:
        fail("start-mrt reported no version; the editable install did not take")
        verification_failed.append("start-mrt")

    profile_code = "from start.runtime_profile import profile_banner; print(profile_banner())"
    result = subprocess.run(
        [str(python_exe), "-c", profile_code], capture_output=True, text=True, cwd=str(root)
    )
    if result.returncode == 0:
        ok(result.stdout.strip())

    if args.json_report:
        Path(args.json_report).write_text(json.dumps(report, indent=2), encoding="utf-8")
        ok(f"report written to {args.json_report}")

    # -- next steps ---------------------------------------------------------
    if verification_failed:
        heading("Incomplete")
        fail(f"missing after install: {', '.join(verification_failed)}")
        print("\n  Re-run with --recreate. If it persists, the extra is declared but not")
        print("  installable on this platform — open an issue with the pip output above.")
        return 1

    heading("Ready")
    print("  Activate the environment:\n")
    for line in activation_hint(venv_dir):
        print(f"    {line}")
    print("\n  Then:\n")
    for cmd, desc in [
        ("start doctor", "environment, providers and egress profile"),
        ("start risk stripes", "the risk taxonomy this install supports"),
        ("start list-tests", "registered deterministic engines"),
        ("start review", "run a review on the bundled example"),
        ("python scripts/demo_flight.py", "the scripted end-to-end demonstration"),
    ]:
        print(f"    {cmd:<32} {c('# ' + desc, 'dim')}")
    print()
    return 0


def read_core_dependencies(pyproject: Path) -> list[str]:
    import tomllib

    with pyproject.open("rb") as handle:
        data = tomllib.load(handle)
    return list(data.get("project", {}).get("dependencies", []))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\ninterrupted")
        raise SystemExit(130) from None
