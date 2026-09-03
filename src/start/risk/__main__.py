"""``python -m start.risk`` — the zero-dependency command line.

The full CLI (``start risk ...``) is nicer, but it needs Typer, and Typer needs
to have been installed. This entry point needs nothing but Python, which makes
it the one that still answers when an environment is half-built, locked down,
or being debugged.

    python -m start.risk stripes
    python -m start.risk objects
    python -m start.risk dimensions
    python -m start.risk egress
    python -m start.risk plan --stripe financial_crime --kind vendor_model --materiality high
"""

from __future__ import annotations

import json
import sys

from start.risk import (
    RiskObject,
    coverage_report,
    dimension,
    dimension_ids,
    object_kind,
    object_kind_ids,
    stripe,
    stripe_ids,
    synthesise_plan,
)

USAGE = """\
usage: python -m start.risk <command> [options]

  stripes                     list risk stripes
  objects                     list reviewable object kinds
  dimensions                  list review dimensions
  egress                      show the active containment regime
  plan     --stripe ID --kind ID [--materiality low|medium|high]
                              [--object-id ID] [--json]
  coverage --stripe ID [--examined a,b,c] [--json]

Install the full CLI for richer output:  start risk --help
"""


def _options(argv: list[str]) -> dict[str, str]:
    return dict(zip(argv[::2], argv[1::2], strict=False))


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help", "help"}:
        print(USAGE)
        return 0

    command, rest = argv[0], argv[1:]
    options = _options(rest)
    as_json = "--json" in rest

    try:
        if command == "stripes":
            if as_json:
                print(json.dumps([stripe(s).as_dict() for s in stripe_ids()], indent=2))
                return 0
            for stripe_id in stripe_ids():
                spec = stripe(stripe_id)
                print(f"{spec.id:<20} {spec.label}")
                print(f"{'':<20} mandatory: {', '.join(spec.mandatory_dimensions) or '-'}")
            return 0

        if command == "objects":
            for kind_id in object_kind_ids():
                spec = object_kind(kind_id)
                print(f"{spec.id:<28} {spec.label}")
            return 0

        if command == "dimensions":
            for dim_id in dimension_ids():
                dim = dimension(dim_id)
                print(f"[phase {dim.phase}] {dim.id:<30} {dim.question}")
            return 0

        if command == "egress":
            from start.runtime_profile import profile_manifest

            print(json.dumps(profile_manifest(), indent=2))
            return 0

        if command == "plan":
            if "--stripe" not in options or "--kind" not in options:
                print("plan requires --stripe and --kind", file=sys.stderr)
                print(USAGE, file=sys.stderr)
                return 2
            obj = RiskObject(
                object_id=options.get("--object-id", "OBJ-1"),
                kind=options["--kind"],
                materiality=options.get("--materiality", "medium"),
            )
            plan = synthesise_plan(stripe_id=options["--stripe"], obj=obj)
            if as_json:
                payload = plan.as_dict()
                payload["plan_hash"] = plan.plan_hash()
                print(json.dumps(payload, indent=2))
                return 0
            print("\n".join(plan.summary_lines()))
            if plan.substitutions:
                print("\nSubstituted (the obligation transfers, it does not lapse):")
                for sub in plan.substitutions:
                    print(f"  {sub['dimension']:<32} -> {', '.join(sub['burden_transferred_to'])}")
            if plan.excluded:
                print("\nExcluded (recorded with a reason, never silently skipped):")
                for row in plan.excluded:
                    print(f"  {row['dimension']:<32} {row['reason']}")
            print(f"\nplan hash  {plan.plan_hash()}")
            return 0

        if command == "coverage":
            spec = stripe(options.get("--stripe", "credit"))
            examined = {d.strip() for d in options.get("--examined", "").split(",") if d.strip()}
            report = coverage_report(spec.control_frameworks, examined)
            print(json.dumps(report, indent=2) if as_json else _render_coverage(report))
            return 0

    except (KeyError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(f"unknown command: {command}\n", file=sys.stderr)
    print(USAGE, file=sys.stderr)
    return 2


def _render_coverage(report: dict) -> str:
    lines = [
        f"mapping version {report['mapping_version']}",
        f"coverage        {report['expectations_covered']}/{report['expectations_total']} "
        f"({report['overall_coverage_ratio']:.0%})",
        "",
    ]
    for framework in report["frameworks"]:
        lines.append(
            f"{framework['framework_id']:<20} "
            f"{framework['expectations_covered']}/{framework['expectations_total']}  "
            f"{framework['label']}"
        )
        for row in framework["expectations"]:
            mark = "x" if row["covered"] else "."
            lines.append(
                f"  [{mark}] {row['expectation_id']:<28} missing: {', '.join(row['missing']) or '-'}"
            )
    if report["unmapped_frameworks"]:
        lines.append(f"\nunmapped (coverage not claimed): {', '.join(report['unmapped_frameworks'])}")
    lines.append(f"\n{report['caveat']}")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
