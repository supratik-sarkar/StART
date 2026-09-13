"""Rich renderers for the 8 terminal presentation panels.

Standardized presentation layer across StART CLI workflows:
  1. Run header & environment state
  2. Agent activity stream
  3. Performance & metrics panel
  4. Evidence ledger view
  5. Regulatory governance & MRM alignment
  6. Narrative invariance attestation
  7. Merkle review seal & leaf breakdown
  8. AI-engineering adapter table
"""

from __future__ import annotations

from typing import Any

from rich import box
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from start.cli.theme import format_agent_badge, format_status_glyph


def render_run_header(
    *,
    review_id: str,
    target: str = "",
    task_type: str = "binary_classification",
    dataset_shape: tuple[int, int] | str = "",
    profile: str = "public_demo",
    policy_id: str = "public_demo",
    seed: int = 42,
    mode: str = "deterministic",
    provider: str | None = None,
) -> Panel:
    """Panel 1: Review Run Header."""
    table = Table(box=box.SIMPLE, show_header=False, expand=True)
    table.add_column("Key", style="bold cyan", width=18)
    table.add_column("Val", style="white", ratio=1)
    table.add_column("Key2", style="bold cyan", width=18)
    table.add_column("Val2", style="white", ratio=1)

    if isinstance(dataset_shape, (tuple, list)):
        shape_str = f"{dataset_shape[0]:,} rows × {dataset_shape[1]} cols"
    else:
        shape_str = str(dataset_shape)

    table.add_row(
        "Review ID", f"[bold white]{review_id}[/bold white]",
        "Runtime Profile", f"[bold green]{profile}[/bold green]",
    )
    table.add_row(
        "Target Column", f"[yellow]{target or 'inferred'}[/yellow]",
        "Disclosure Policy", f"[blue]{policy_id}[/blue]",
    )
    table.add_row(
        "Task Type", f"{task_type or 'binary_classification'}",
        "Random Seed", f"{seed}",
    )
    table.add_row(
        "Agent Mode", f"{mode} ({provider})",
        "Dataset Shape", shape_str,
    )

    return Panel(
        table,
        title="[bold white]StART Model Review Execution[/bold white]",
        border_style="bright_blue",
        padding=(0, 1),
    )


def render_agent_stream(events: list[dict[str, Any]]) -> Panel:
    """Panel 2: Streaming Agent Activity & Handoffs."""
    table = Table(box=box.SIMPLE, expand=True)
    table.add_column("Agent", width=10)
    table.add_column("St", width=3, justify="center")
    table.add_column("Activity / Stage", style="bold white", width=24)
    table.add_column("Detail / Findings", style="dim", ratio=1)
    table.add_column("Time", width=8, justify="right", style="cyan")

    for ev in events:
        agent_name = ev.get("agent", "System")
        badge = format_agent_badge(agent_name)
        st = ev.get("status", "running")
        glyph = format_status_glyph(st)
        stage = ev.get("stage") or ev.get("activity", "")
        detail = ev.get("detail", "")
        sec = ev.get("runtime_seconds")
        runtime = f"{sec:.2f}s" if isinstance(sec, (int, float)) else ""
        table.add_row(badge, glyph, stage, detail, runtime)

    return Panel(
        table,
        title="[bold white]Agent Activity Stream[/bold white]",
        border_style="cyan",
        padding=(0, 1),
    )


def render_metrics_panel(metrics_by_split: dict[str, dict[str, float]]) -> Panel:
    """Panel 3: Performance, Generalization & Boundary Metrics."""
    table = Table(box=box.ROUNDED, expand=True)
    table.add_column("Cohort / Split", style="bold white", width=18)
    table.add_column("AUC", justify="right", width=10)
    table.add_column("ECE (Calib)", justify="right", width=12)
    table.add_column("Brier Score", justify="right", width=12)
    table.add_column("F1 Score", justify="right", width=10)
    table.add_column("Status", justify="center", width=10)

    for split_name, m in metrics_by_split.items():
        auc = f"{m.get('test_auc', m.get('auc', 0.0)):.4f}"
        ece = f"{m.get('ece', 0.0):.4f}"
        brier = f"{m.get('brier', 0.0):.4f}"
        f1 = f"{m.get('f1', 0.0):.4f}"
        auc_val = m.get("test_auc", m.get("auc", 0.0))
        st = "[bold green]PASS[/bold green]" if auc_val >= 0.70 else "[bold yellow]WARN[/bold yellow]"
        table.add_row(split_name.upper(), auc, ece, brier, f1, st)

    return Panel(
        table,
        title="[bold white]Cohort Performance & Boundary Breakdown[/bold white]",
        border_style="green",
    )


def render_ledger_view(evidence_records: list[dict[str, Any]]) -> Panel:
    """Panel 4: Evidence Ledger & Hash-Chain Records."""
    table = Table(box=box.SIMPLE, expand=True)
    table.add_column("Evidence ID", style="bold cyan", width=14)
    table.add_column("St", width=3, justify="center")
    table.add_column("Test ID / Name", style="white", width=28)
    table.add_column("Deterministic Metrics", style="dim", ratio=1)
    table.add_column("Chain Commitment", width=18, justify="right", style="dim")

    for rec in evidence_records:
        ev_id = rec.get("evidence_id", "")
        status = rec.get("status", "pass")
        glyph = format_status_glyph(status)
        test_name = rec.get("test_name") or rec.get("test_id", "")
        metrics = rec.get("metrics", {})
        if isinstance(metrics, dict):
            m_str = ", ".join(f"{k}={v}" for k, v in metrics.items())
        else:
            m_str = str(metrics)
        chain_hash = rec.get("entry_hash", "")[:14] + "…" if rec.get("entry_hash") else "[dim]sealed[/dim]"
        table.add_row(ev_id, glyph, test_name, m_str, chain_hash)

    return Panel(
        table,
        title="[bold white]Evidence Ledger & Cryptographic Chain[/bold white]",
        border_style="cyan",
    )


def render_governance_panel(
    tier: str = "Tier-1",
    controls: list[dict[str, Any]] | None = None,
) -> Panel:
    """Panel 5: MRM Regulatory Governance & Framework Alignment."""
    table = Table(box=box.ROUNDED, expand=True)
    table.add_column("Regulatory Control / Dimension", style="bold white", width=32)
    table.add_column("Guidance Ref", style="cyan", width=14)
    table.add_column("Status", justify="center", width=10)
    table.add_column("Findings & Rationale", style="dim", ratio=1)

    ctrls = controls or []
    for c in ctrls:
        name = c.get("name", "")
        ref = c.get("reference", "SR 11-7")
        status = c.get("status", "pass")
        glyph = format_status_glyph(status)
        notes = c.get("notes", "")
        table.add_row(name, ref, glyph, notes)

    disclaimer = Text(
        "\nRegulatory Notice: StART provides deterministic mathematical verification and automated "
        "control execution. Formal sign-off authority remains exclusively with human Model Risk "
        "Officers under SR 11-7 principles.",
        style="dim italic",
    )
    grid = Table.grid()
    grid.add_row(table)
    grid.add_row(disclaimer)
    return Panel(
        grid,
        title=f"[bold white]MRM Regulatory Governance & Framework Alignment ({tier})[/bold white]",
        border_style="yellow",
    )


def render_attestation_panel(attestation: Any) -> Panel:
    """Panel 6: Narrative Invariance Attestation Panel."""
    table = Table(box=box.ROUNDED, expand=True)
    table.add_column("Metric / Field", style="bold cyan", width=22)
    table.add_column("Value / Attestation State", style="white", ratio=1)

    inv = getattr(attestation, "invariant", True)
    if inv:
        verdict_text = "[bold green]INVARIANT (PASS)[/bold green]"
    else:
        verdict_text = "[bold red]DIVERGENT (FAIL)[/bold red]"
    prov = getattr(attestation, "provider_name", "deterministic")
    n_path = getattr(attestation, "narration_path", "deterministic_only")
    path_glyph = format_status_glyph(n_path)

    mb = getattr(attestation, "model_binding", {})
    bound = mb.get("bound_claims", 0)
    total = mb.get("total_claims", 0)
    unbound_rate = getattr(attestation, "unbound_claim_rate", 0.0)

    table.add_row("Invariance Verdict", verdict_text)
    table.add_row("Narration Path", Text.assemble(path_glyph, f" {n_path} ({prov})"))
    table.add_row(
        "Claim Grounding",
        f"{bound}/{total} quantitative claims bound to evidence ({unbound_rate * 100:.1f}% unbound)",
    )
    table.add_row("Attestation Hash", f"[dim]{getattr(attestation, 'attestation_hash', lambda: '')()}[/dim]")

    divergences = getattr(attestation, "divergences", ())
    if divergences:
        div_table = Table(box=box.SIMPLE, expand=True)
        div_table.add_column("Severity", width=10)
        div_table.add_column("Kind", width=14)
        div_table.add_column("Detail", style="dim", ratio=1)
        for d in divergences:
            sev_style = "bold red" if getattr(d, "severity", "") == "high" else "yellow"
            div_table.add_row(
                f"[{sev_style}]{getattr(d, 'severity', '')}[/{sev_style}]",
                getattr(d, "kind", ""),
                getattr(d, "detail", ""),
            )

        grid = Table.grid()
        grid.add_row(table)
        grid.add_row(Text("\nDivergence Records:", style="bold white"))
        grid.add_row(div_table)
        return Panel(
            grid,
            title="[bold white]Narrative Invariance Attestation[/bold white]",
            border_style="green" if inv else "red",
        )

    return Panel(
        table,
        title="[bold white]Narrative Invariance Attestation[/bold white]",
        border_style="green" if inv else "red",
    )


def render_seal_panel(seal: Any, critic_verdict: str = "PASSED") -> Panel:
    """Panel 7: Merkle Seal & Leaf Breakdown Panel."""
    table = Table(box=box.SIMPLE, expand=True)
    table.add_column("#", width=3, justify="right", style="dim")
    table.add_column("Leaf Name", style="bold white", width=18)
    table.add_column("Leaf Hash (SHA-256)", style="cyan", ratio=1)
    table.add_column("Commitment", width=14, justify="center")

    leaves = getattr(seal, "leaves", ())
    for i, leaf in enumerate(leaves):
        h = leaf.leaf_hash()
        table.add_row(str(i + 1), leaf.name, f"{h[:16]}…{h[-8:]}", "[green]committed ✓[/green]")

    seal_str = getattr(seal, "seal_string", lambda: "start-seal/2:R-UNKNOWN:0000")()
    root_str = getattr(seal, "root", lambda: "0" * 64)()

    summary = Table.grid(expand=True)
    summary.add_row(f"[bold white]Merkle Root:[/bold white] [bold cyan]{root_str}[/bold cyan]")
    short_seal = f"{seal_str[:60]}…" if len(seal_str) > 60 else seal_str
    summary.add_row(f"[bold white]Seal String:[/bold white] [bold green]{short_seal}[/bold green]")
    
    # Critic verdict display (advisory in v4.0.2 per Amendment 1)
    if critic_verdict in ("PASSED", "passed", "ok"):
        crit_text = "[bold green]PASSED ✓[/bold green]"
    else:
        crit_text = f"[bold yellow]{critic_verdict} (advisory in v4.0.2)[/bold yellow]"
    summary.add_row(f"[bold white]Critic Gate:[/bold white] {crit_text}")

    grid = Table.grid()
    grid.add_row(table)
    grid.add_row(Text(""))
    grid.add_row(summary)

    version_str = getattr(seal, "version", "start-seal/2")
    return Panel(
        grid,
        title=f"[bold white]Cryptographic Review Seal ({version_str})[/bold white]",
        border_style="bright_magenta",
    )


def render_adapters_table(adapters: list[dict[str, Any]]) -> Panel:
    """Panel 8: AI-Engineering & Observability Adapter Matrix (D8)."""
    table = Table(box=box.ROUNDED, expand=True)
    table.add_column("Adapter", style="bold white", width=18)
    table.add_column("Category", style="cyan", width=14)
    table.add_column("Status", width=24)
    table.add_column("Egress Sink Type", width=16)
    table.add_column("Role & Governance Function", style="dim", ratio=1)

    for a in adapters:
        st = str(a.get("status", "not_installed")).lower()
        eg_st = str(a.get("egress_status", st)).lower()
        if eg_st in ("active", "connected"):
            st_text = "[bold green]● active[/bold green]"
        elif eg_st in ("available", "installed"):
            st_text = "[green]● available[/green]"
        elif eg_st in ("available_not_configured", "not_configured"):
            st_text = "[yellow]○ not configured[/yellow]"
        elif eg_st in ("not_wired", "unwired"):
            st_text = "[yellow]○ not wired[/yellow]"
        elif eg_st == "blocked_by_profile":
            st_text = "[bold red]⊘ blocked by profile[/bold red]"
        elif eg_st in ("error", "failed"):
            st_text = "[bold red]✗ error[/bold red]"
        else:
            st_text = "[dim]○ not installed[/dim]"

        is_saas = a.get("name") in ("LangSmith", "Phoenix", "OpenAI", "Anthropic", "Google Gemini")
        egress = "Third-Party SaaS" if is_saas else "Local / Isolated"
        table.add_row(
            a.get("name", ""),
            a.get("category", ""),
            st_text,
            egress,
            a.get("role", a.get("purpose", "")),
        )

    return Panel(
        table,
        title="[bold white]AI-Engineering & Observability Control Surface[/bold white]",
        border_style="blue",
    )
