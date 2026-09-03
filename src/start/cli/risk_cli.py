"""``start risk`` and ``start attest`` — the stripe-agnostic CLI surface.

Kept in its own module rather than added to ``cli/main.py`` for two reasons.
The obvious one is that ``main.py`` is already a thousand lines. The more
important one is that everything reachable from here is standard-library only,
so these commands work in an environment where the modelling stack is absent or
broken — which is exactly when someone needs to ask "what does this review owe?"
or "does this seal still verify?".

When Typer itself is unavailable, the same queries are reachable through
``python -m start.risk``, which imports nothing outside the standard library.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer

from start.attestation import replay_ledger, verify_seal
from start.risk import (
    RiskObject,
    coverage_report,
    dimension_ids,
    object_kind,
    object_kind_ids,
    stripe,
    stripe_ids,
    synthesise_plan,
)
from start.risk.coverage import coverage_for_plan
from start.runtime_profile import egress_policy, profile_manifest

risk_app = typer.Typer(
    name="risk",
    help="Risk stripes, reviewable object kinds, and deterministic plan synthesis.",
    no_args_is_help=True,
)

attest_app = typer.Typer(
    name="attest",
    help="Verify seals, replay evidence ledgers, and inspect the egress profile.",
    no_args_is_help=True,
)


def _echo(text: str = "") -> None:
    typer.echo(text)


# --------------------------------------------------------------------------- #
# start risk
# --------------------------------------------------------------------------- #
@risk_app.command("stripes")
def list_stripes(
    as_json: bool = typer.Option(False, "--json", help="Emit machine-readable output."),
) -> None:
    """List the risk stripes this installation supports."""
    if as_json:
        _echo(json.dumps([stripe(s).as_dict() for s in stripe_ids()], indent=2))
        return

    _echo()
    _echo(f"  {len(stripe_ids())} risk stripes\n")
    for stripe_id in stripe_ids():
        spec = stripe(stripe_id)
        _echo(f"  {typer.style(spec.id, bold=True):<28} {spec.label}")
        _echo(f"  {'':<26} {spec.description}")
        _echo(f"  {'':<26} mandatory: {', '.join(spec.mandatory_dimensions) or '—'}")
        _echo()


@risk_app.command("objects")
def list_objects(
    as_json: bool = typer.Option(False, "--json", help="Emit machine-readable output."),
) -> None:
    """List the kinds of artefact StART can review.

    Note how few of these are machine-learning models. That is the point: a
    model inventory is mostly calculators, spreadsheets, rules and vendor
    components, and a review platform that only handles fitted estimators
    handles the minority of it.
    """
    if as_json:
        _echo(
            json.dumps(
                [
                    {
                        "id": object_kind(k).id,
                        "label": object_kind(k).label,
                        "description": object_kind(k).description,
                        "capabilities": object_kind(k).capabilities.as_dict(),
                        "always_required": list(object_kind(k).always_required),
                        "notes": object_kind(k).notes,
                    }
                    for k in object_kind_ids()
                ],
                indent=2,
            )
        )
        return

    _echo()
    _echo(f"  {len(object_kind_ids())} reviewable object kinds\n")
    for kind_id in object_kind_ids():
        spec = object_kind(kind_id)
        _echo(f"  {typer.style(spec.id, bold=True):<32} {spec.label}")
        _echo(f"  {'':<30} {spec.description}")
        if spec.notes:
            _echo(f"  {'':<30} {typer.style(spec.notes, dim=True)}")
        _echo()


@risk_app.command("dimensions")
def list_dimensions() -> None:
    """List the review dimensions — the questions a review must answer."""
    from start.risk import dimension

    _echo()
    current_phase = None
    for dim_id in dimension_ids():
        dim = dimension(dim_id)
        if dim.phase != current_phase:
            current_phase = dim.phase
            _echo(f"\n  ── phase {dim.phase} ──")
        _echo(f"  {typer.style(dim.id, bold=True):<36} {dim.label}")
        _echo(f"  {'':<34} {dim.question}")
    _echo()


@risk_app.command("plan")
def plan_command(
    stripe_id: str = typer.Option(..., "--stripe", help="Risk stripe (see: start risk stripes)."),
    kind: str = typer.Option(..., "--kind", help="Object kind (see: start risk objects)."),
    object_id: str = typer.Option("OBJ-1", "--object-id", help="Identifier for the artefact."),
    materiality: str = typer.Option("medium", "--materiality", help="low | medium | high"),
    show_tests: bool = typer.Option(
        False,
        "--show-tests",
        help="Show which registered tests could supply evidence for each dimension.",
    ),
    context: str = typer.Option(
        "tabular",
        "--context",
        help="Context type to filter candidate tests by (tabular | market | short_rate).",
    ),
    affects_individuals: bool = typer.Option(
        False,
        "--affects-individuals/--no-affects-individuals",
        help="Override: outputs bear on identifiable individuals.",
    ),
    as_json: bool = typer.Option(False, "--json", help="Emit the full plan as JSON."),
    out: Path | None = typer.Option(None, "--out", help="Write the plan JSON to a file."),
) -> None:
    """Synthesise a deterministic, hash-stable review plan."""
    overrides: dict[str, bool] = {}
    if affects_individuals:
        overrides["affects_individuals"] = True

    try:
        obj = RiskObject(
            object_id=object_id, kind=kind, materiality=materiality, capability_overrides=overrides
        )
        plan = synthesise_plan(stripe_id=stripe_id, obj=obj)
    except (KeyError, ValueError) as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from None

    payload = plan.as_dict()
    payload["plan_hash"] = plan.plan_hash()
    coverage = coverage_for_plan(plan, context_type=context) if show_tests else None
    if coverage is not None:
        payload["test_coverage"] = coverage.as_dict()

    if out:
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        typer.secho(f"plan written to {out}", fg=typer.colors.GREEN)
        if not as_json:
            return

    if as_json:
        _echo(json.dumps(payload, indent=2))
        return

    _echo()
    for line in plan.summary_lines():
        _echo(f"  {line}")

    if plan.substitutions:
        _echo("\n  Substituted — the obligation transfers rather than lapsing:")
        for sub in plan.substitutions:
            _echo(f"    {sub['dimension']:<34} → {', '.join(sub['burden_transferred_to'])}")
    if plan.excluded:
        _echo("\n  Excluded — recorded with a reason, never silently skipped:")
        for exc_row in plan.excluded:
            _echo(f"    {exc_row['dimension']:<34} {exc_row['reason']}")
    _echo(f"\n  plan hash  {plan.plan_hash()}")
    if coverage is not None:
        _echo()
        for line in coverage.summary_lines():
            _echo(line)
    _echo()


@risk_app.command("coverage")
def coverage_command(
    stripe_id: str = typer.Option(..., "--stripe", help="Risk stripe."),
    examined: str = typer.Option("", "--examined", help="Comma-separated dimensions that produced evidence."),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Report control-framework coverage from the dimensions actually examined."""
    try:
        spec = stripe(stripe_id)
    except KeyError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from None

    dims = {d.strip() for d in examined.split(",") if d.strip()}
    try:
        report = coverage_report(spec.control_frameworks, dims)
    except KeyError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from None

    if as_json:
        _echo(json.dumps(report, indent=2))
        return

    _echo()
    _echo(f"  mapping version {report['mapping_version']}")
    _echo(
        f"  coverage        {report['expectations_covered']}/{report['expectations_total']} "
        f"expectations ({report['overall_coverage_ratio']:.0%})\n"
    )
    for framework in report["frameworks"]:
        _echo(
            f"  {typer.style(framework['framework_id'], bold=True):<24} "
            f"{framework['expectations_covered']}/{framework['expectations_total']}  "
            f"{framework['label']}"
        )
        for row in framework["expectations"]:
            mark = typer.style("✓", fg=typer.colors.GREEN) if row["covered"] else typer.style("·", dim=True)
            _echo(f"    {mark} {row['expectation_id']:<28} missing: {', '.join(row['missing']) or '—'}")
        _echo()
    if report["unmapped_frameworks"]:
        _echo(
            typer.style(
                f"  unmapped frameworks (no encoded expectations, coverage not claimed): "
                f"{', '.join(report['unmapped_frameworks'])}",
                dim=True,
            )
        )
    _echo(typer.style(f"\n  {report['caveat']}", dim=True))
    _echo()


# --------------------------------------------------------------------------- #
# start attest
# --------------------------------------------------------------------------- #
@attest_app.command("verify-seal")
def verify_seal_command(
    manifest: Path = typer.Argument(..., help="Path to an archived seal manifest JSON."),
    seal: str = typer.Option("", "--seal", help="Seal string to check the manifest against."),
) -> None:
    """Recompute a seal from its archived manifest.

    Exits non-zero when verification fails, so this is usable as a gate.
    """
    if not manifest.exists():
        typer.secho(f"manifest not found: {manifest}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)

    data: dict[str, Any] = json.loads(manifest.read_text(encoding="utf-8"))
    result = verify_seal(data, seal or None)

    _echo()
    if result["verified"]:
        typer.secho(f"  ✓ {result['reason']}", fg=typer.colors.GREEN)
        _echo(f"    seal {result['recomputed_seal']}")
        _echo()
        return

    typer.secho(f"  ✗ {result['reason']}", fg=typer.colors.RED)
    if result["mismatched_leaves"]:
        _echo(f"    altered leaves: {', '.join(result['mismatched_leaves'])}")
    _echo(f"    recomputed {result['recomputed_root']}")
    _echo(f"    recorded   {result['recorded_root']}")
    _echo()
    raise typer.Exit(code=1)


@attest_app.command("replay")
def replay_command(
    ledger: Path = typer.Argument(..., help="Path to an evidence ledger (.jsonl)."),
    compare: Path | None = typer.Option(
        None, "--compare", help="A second ledger to compare against, for a reproducibility check."
    ),
) -> None:
    """Replay a ledger's hash chain and localise any divergence."""
    verdict = replay_ledger(ledger)
    _echo()
    colour = typer.colors.GREEN if verdict.intact else typer.colors.RED
    typer.secho(f"  {verdict.summary_line()}", fg=colour)
    if not verdict.intact:
        _echo(f"    kind {verdict.divergence_kind}")

    if compare:
        from start.attestation import compare_ledgers

        comparison = compare_ledgers(ledger, compare)
        colour = typer.colors.GREEN if comparison.reproducible else typer.colors.RED
        typer.secho(f"  {comparison.summary_line()}", fg=colour)
        for row in comparison.drifted[:10]:
            _echo(f"    {row['field']}: {row.get('original')} → {row.get('rerun')}")
    _echo()
    if not verdict.intact:
        raise typer.Exit(code=1)


@attest_app.command("egress")
def egress_command(
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Show the active containment regime: profile, permitted and refused providers."""
    if as_json:
        _echo(json.dumps(profile_manifest(), indent=2))
        return

    policy = egress_policy()
    manifest = profile_manifest()
    _echo()
    _echo(f"  runtime profile     {typer.style(policy.profile.value, bold=True)}")
    _echo(f"  permitted providers {', '.join(sorted(policy.allowed))}")
    _echo(f"  refused providers   {', '.join(sorted(policy.denied)) or '—'}")
    _echo(f"  gateway configured  {manifest['gateway_configured']}")
    _echo(f"  registered gateways {', '.join(manifest['registered_gateways']) or '—'}")
    if policy.overrides:
        typer.secho(f"  OVERRIDES ACTIVE    {', '.join(policy.overrides)}", fg=typer.colors.YELLOW)
    _echo(f"  manifest hash       {manifest['manifest_hash']}")
    _echo()
    _echo(typer.style(f"  {policy.rationale}", dim=True))
    _echo()


@attest_app.command("trace")
def trace_command(
    seal: str = typer.Argument(..., help="The seal string to trace back to evidence records."),
    output_root: Path = typer.Option(
        Path("start_output"), "--output-root", help="Root directory containing output and seals index."
    ),
    as_json: bool = typer.Option(False, "--json", help="Emit machine-readable output."),
) -> None:
    """Trace a seal string back to its manifest, run IDs, and evidence ledger records."""
    seal_clean = seal.strip()
    index_path = output_root / "seals" / "index.json"
    manifest_path: Path | None = None
    enterprise_run_id: str | None = None
    inner_run_id: str | None = None

    if index_path.exists():
        try:
            idx = json.loads(index_path.read_text())
            if seal_clean in idx:
                entry = idx[seal_clean]
                manifest_path = Path(entry.get("manifest_path", ""))
                enterprise_run_id = entry.get("enterprise_run_id")
                inner_run_id = entry.get("inner_run_id")
        except Exception:
            pass

    # If not in index, parse seal components: start-seal/2:<run_id>:<root_prefix>
    if not manifest_path or not manifest_path.exists():
        parts = seal_clean.split(":")
        if len(parts) >= 2:
            candidate_run = parts[1]
            candidate_manifest = output_root / "seals" / candidate_run / "seal_manifest.json"
            if candidate_manifest.exists():
                manifest_path = candidate_manifest
                enterprise_run_id = candidate_run

    if not manifest_path or not manifest_path.exists():
        # Fallback: search all seal_manifest.json files under output_root / "seals"
        for p in output_root.glob("seals/**/seal_manifest.json"):
            try:
                m = json.loads(p.read_text())
                if m.get("seal") == seal_clean or (m.get("root") and m.get("root", "").startswith(parts[-1])):
                    manifest_path = p
                    enterprise_run_id = m.get("enterprise_run_id") or m.get("review_id")
                    inner_run_id = m.get("inner_run_id")
                    break
            except Exception:
                continue

    if not manifest_path or not manifest_path.exists():
        typer.secho(
            f"Could not resolve seal {seal_clean} to any manifest under {output_root}.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)

    manifest_data = json.loads(manifest_path.read_text())
    enterprise_run_id = (
        enterprise_run_id or manifest_data.get("enterprise_run_id") or manifest_data.get("review_id")
    )
    inner_run_id = (
        inner_run_id
        or manifest_data.get("inner_run_id")
        or manifest_data.get("metadata", {}).get("inner_run_id", "")
    )

    # Load ledger records
    from start.evidence.ledger import EvidenceLedger

    ledger_path = output_root / "ledger.jsonl"
    store_root = output_root / "evidence_store"
    ledger = EvidenceLedger(ledger_path, store_root)
    records = ledger.records_for_run(enterprise_run_id) if enterprise_run_id else []

    if as_json:
        out = {
            "seal": seal_clean,
            "manifest_path": str(manifest_path),
            "enterprise_run_id": enterprise_run_id,
            "inner_run_id": inner_run_id,
            "created_utc": manifest_data.get("created_utc"),
            "root": manifest_data.get("root"),
            "evidence_count": len(records),
            "evidence_records": [r.model_dump(mode="json") for r in records],
        }
        _echo(json.dumps(out, indent=2, default=str))
        return

    _echo()
    typer.secho(f"  Seal Trace: {seal_clean}", bold=True)
    _echo(f"  Enterprise Run ID : {typer.style(str(enterprise_run_id), bold=True)}")
    _echo(f"  Inner Pipeline ID : {inner_run_id or '—'}")
    _echo(f"  Manifest Path     : {manifest_path}")
    _echo(f"  Created UTC       : {manifest_data.get('created_utc')}")
    _echo(f"  Merkle Root       : {manifest_data.get('root')}")
    _echo()

    # Adjudications leaf summary
    adjudications = manifest_data.get("payloads", {}).get("adjudications") or {}
    if isinstance(adjudications, dict):
        n_dec = len(adjudications.get("decisions", []))
        n_over = len(adjudications.get("overrides", []))
        n_chal = len(adjudications.get("challenges", []))
        _echo(f"  Adjudications     : {n_dec} decisions ({n_over} overrides), {n_chal} challenges")

    # Attestations leaf summary
    attestations = manifest_data.get("payloads", {}).get("attestations") or []
    if isinstance(attestations, list) and attestations:
        _echo(f"  Attestations      : {len(attestations)} narrative invariance checks")

    _echo()
    typer.secho(f"  Linked Evidence Records ({len(records)} found):", bold=True)
    if not records:
        typer.secho("    (no records found in ledger for this run ID)", fg=typer.colors.YELLOW)
    for r in records:
        if r.status == "pass":
            status_color = typer.colors.GREEN
        elif r.status in ("warn", "recorded", "informational"):
            status_color = typer.colors.YELLOW
        else:
            status_color = typer.colors.RED
        st_badge = typer.style(r.status.value, fg=status_color)
        _echo(f"    - {typer.style(r.evidence_id, bold=True)} [{st_badge}] {r.test_name} ({r.test_id})")
        if r.interpretation:
            _echo(f"        {typer.style(r.interpretation, dim=True)}")
        if r.metrics:
            metrics_str = ", ".join(f"{k}={v}" for k, v in list(r.metrics.items())[:5])
            _echo(f"        metrics: {metrics_str}")
    _echo()
