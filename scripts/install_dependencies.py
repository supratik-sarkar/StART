#!/usr/bin/env python3
"""Resilient dependency installer and audit utility for StART."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

# Define package groups
GROUPS = {
    "core": [
        "numpy", "pandas", "scipy", "scikit-learn", "pydantic", 
        "pydantic-settings", "pyyaml", "typer", "rich"
    ],
    "deep_learning": [
        "torch", "captum", "matplotlib", "pillow", "torchvision"
    ],
    "tree_models": [
        "xgboost", "lightgbm", "optuna", "shap"
    ],
    "file_formats": [
        "pyarrow", "openpyxl"
    ],
    "llm_provider": [
        "openai", "anthropic", "transformers", "huggingface-hub", 
        "accelerate", "sentence-transformers"
    ],
    "ai_engineering": [
        "opentelemetry-sdk", "opentelemetry-api", "langfuse", "mcp", 
        "garak", "deepeval", "nemoguardrails", "langgraph"
    ],
    "dev": [
        "pytest", "hypothesis", "ruff", "mypy", "pre-commit"
    ]
}


def normalize_package_name(name: str) -> str:
    """Normalize package name to lowercase and replace dashes with underscores."""
    return re.sub(r"[-_.]+", "_", name).lower()


def get_package_group(package_name: str) -> str:
    """Map a normalized package name to its corresponding group."""
    norm_name = normalize_package_name(package_name)
    for group, packages in GROUPS.items():
        if any(normalize_package_name(p) == norm_name for p in packages):
            return group
    return "other"


def evaluate_marker(marker_str: str) -> bool:
    """Safely evaluate PEP 508 environment markers."""
    import platform
    env = {
        "python_version": ".".join(map(str, sys.version_info[:2])),
        "python_full_version": ".".join(map(str, sys.version_info[:3])),
        "os_name": os.name,
        "sys_platform": sys.platform,
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "platform_python_implementation": platform.python_implementation(),
        "implementation_name": sys.implementation.name,
    }
    try:
        from packaging.markers import Marker
        return Marker(marker_str).evaluate()
    except ImportError:
        # Fallback simple evaluation using standard Python eval with clean environment
        try:
            expr = marker_str
            # Standard markers replace
            for k, v in env.items():
                expr = expr.replace(k, repr(v))
            # Safely evaluate
            return bool(eval(expr, {"__builtins__": {}}, {}))
        except Exception:
            return True


def parse_requirements_file(file_path: Path) -> list[dict[str, Any]]:
    """Parse requirements file returning a list of dicts with package details."""
    requirements = []
    if not file_path.exists():
        return requirements

    with open(file_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # Handle environment markers separated by ';'
            marker_str = ""
            if ";" in line:
                req_part, marker_part = line.split(";", 1)
                req_part = req_part.strip()
                marker_str = marker_part.strip()
            else:
                req_part = line

            # Extract package name and version specifiers
            # Regex to match package name and operators
            match = re.match(r"^([a-zA-Z0-9_\-\[\]]+)(.*)$", req_part)
            if not match:
                continue

            pkg_name = match.group(1).strip()
            version_spec = match.group(2).strip()

            requirements.append({
                "line": line,
                "name": pkg_name,
                "norm_name": normalize_package_name(pkg_name),
                "version_spec": version_spec,
                "marker": marker_str,
                "group": get_package_group(pkg_name)
            })

    return requirements


def run_pip_install(req_str: str, dry_run: bool = False) -> tuple[bool, str, str]:
    """Execute pip install for a package specifier."""
    cmd = [sys.executable, "-m", "pip", "install"]
    if dry_run:
        cmd.append("--dry-run")
    cmd.append(req_str)

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        return proc.returncode == 0, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Timeout expired during installation"
    except Exception as e:
        return False, "", str(e)


def main() -> int:
    parser = argparse.ArgumentParser(description="StART dependency manager and audit utility.")
    parser.add_argument("--requirements", type=str, default="requirements.txt", help="Path to requirements file.")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue installation if optional package fails.")
    parser.add_argument("--dry-run", action="store_true", help="Simulate installation without changes.")
    parser.add_argument("--firm-safe", action="store_true", help="Only install core packages; report others as excluded.")
    parser.add_argument("--output-json", type=str, default="start_output/dependency_audit.json", help="Path to write JSON audit report.")
    parser.add_argument("--output-csv", type=str, default="start_output/dependency_audit.csv", help="Path to write CSV audit report.")

    args = parser.parse_args()

    requirements_path = Path(args.requirements)
    if not requirements_path.exists():
        print(f"Error: Requirements file not found at {requirements_path}")
        return 1

    print(f"Parsing dependencies from {requirements_path}...")
    reqs = parse_requirements_file(requirements_path)

    # Initialize results containers
    installed = []
    satisfied = []
    skipped_marker = []
    failed = []
    excluded = []

    # Enforce firm-safe mode vs normal
    for req in reqs:
        pkg_name = req["name"]
        group = req["group"]
        
        # Check environment marker first
        if req["marker"] and not evaluate_marker(req["marker"]):
            skipped_marker.append({
                "package": pkg_name,
                "version_spec": req["version_spec"],
                "group": group,
                "reason": f"Skipped by environment marker: {req['marker']}"
            })
            continue

        # Exclude optional packages in firm-safe mode
        if args.firm_safe and group != "core":
            excluded.append({
                "package": pkg_name,
                "version_spec": req["version_spec"],
                "group": group,
                "reason": f"Excluded under firm-safe core profile (Group: {group})"
            })
            continue

        print(f"Processing {req['line']} (Group: {group})...")
        success, stdout, stderr = run_pip_install(req["line"], dry_run=args.dry_run)
        
        if success:
            if "Requirement already satisfied" in stdout:
                satisfied.append({
                    "package": pkg_name,
                    "version_spec": req["version_spec"],
                    "group": group
                })
            else:
                installed.append({
                    "package": pkg_name,
                    "version_spec": req["version_spec"],
                    "group": group
                })
        else:
            # Extract concise failure reason
            err_msg = stderr.strip().splitlines()
            reason = err_msg[-1] if err_msg else "Unknown installation error"
            # Remove any potentially sensitive directory paths or key/credential strings
            reason = re.sub(r"/Users/[a-zA-Z0-9_\-\.]+/", "/USER_HOME/", reason)
            
            failed.append({
                "package": pkg_name,
                "version_spec": req["version_spec"],
                "group": group,
                "reason": reason
            })
            
            # If mandatory core package fails, block release
            if group == "core":
                print(f"CRITICAL FAILURE: Mandatory core dependency '{pkg_name}' failed to install: {reason}")
                if not args.continue_on_error:
                    break

    # Build report
    report = {
        "python_version": sys.version,
        "dry_run": args.dry_run,
        "firm_safe": args.firm_safe,
        "installed": installed,
        "already_satisfied": satisfied,
        "skipped_environment_marker": skipped_marker,
        "failed": failed,
        "excluded": excluded,
    }

    # Run pip check to identify conflicts
    print("\nRunning 'pip check' for dependency conflict verification...")
    pip_check_proc = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True)
    report["pip_check_conflict"] = pip_check_proc.returncode != 0
    report["pip_check_output"] = pip_check_proc.stdout.strip()
    if pip_check_proc.returncode != 0:
        print("Dependency conflict warnings detected:")
        print(pip_check_proc.stdout.strip())
    else:
        print("No dependency conflicts detected.")

    # Write JSON report
    json_path = Path(args.output_json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(report, jf, indent=2)
    print(f"JSON audit report written to: {json_path}")

    # Write CSV report
    csv_path = Path(args.output_csv)
    with open(csv_path, "w", encoding="utf-8") as cf:
        cf.write("Package,Group,Status,VersionSpecifier,Message\n")
        for p in installed:
            cf.write(f"{p['package']},{p['group']},installed,{p['version_spec']},\n")
        for p in satisfied:
            cf.write(f"{p['package']},{p['group']},satisfied,{p['version_spec']},\n")
        for p in skipped_marker:
            cf.write(f"{p['package']},{p['group']},skipped_marker,{p['version_spec']},{p['reason']}\n")
        for p in failed:
            cf.write(f"{p['package']},{p['group']},failed,{p['version_spec']},{p['reason']}\n")
        for p in excluded:
            cf.write(f"{p['package']},{p['group']},excluded,{p['version_spec']},{p['reason']}\n")
    print(f"CSV audit report written to: {csv_path}")

    # Display final summary
    print("\n================== Dependency Installation Summary ==================")
    print(f"Successfully installed  : {len(installed)}")
    print(f"Already satisfied       : {len(satisfied)}")
    print(f"Skipped (markers)       : {len(skipped_marker)}")
    print(f"Failed                  : {len(failed)}")
    print(f"Excluded (firm-safe)    : {len(excluded)}")
    
    # Check if there were failed packages in the core group
    failed_core = [p for p in failed if p["group"] == "core"]
    if failed_core:
        print("\n[bold red]RELEASE GATE: FAILED[/bold red] - One or more mandatory core packages failed to install:")
        for p in failed_core:
            print(f"  - {p['package']}: {p['reason']}")
        return 1

    print("\nRELEASE GATE: PASSED (All mandatory core packages are verified/installed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
