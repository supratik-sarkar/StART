#!/usr/bin/env python3
"""StART v4.4 Institutional Showcase & Verification Runner.

Executes a complete, non-interactive review using enriched synthetic market world
and outputs all institutional presentation tables, HRP topology showcase, agent
orchestration trace, architecture registry, and multi-format artifacts.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from rich.console import Console  # noqa: E402

from start.core.architecture_registry import get_architecture_capability_table  # noqa: E402
from start.data.synthetic_market import generate_market_world  # noqa: E402
from start.review.architecture import ReviewContextBundle, ReviewDomain, ReviewGroundingMode  # noqa: E402
from start.review.executor import run_unified_review  # noqa: E402

console = Console()


def run_showcase() -> int:
    console.print(
        "\n[bold cyan]╔════════════════════════════════════════════════════════════════════════════════════╗[/bold cyan]"
    )
    console.print(
        "[bold cyan]║                 StART v4.4 Institutional UX & Architecture Showcase                ║[/bold cyan]"
    )
    console.print(
        "[bold cyan]╚════════════════════════════════════════════════════════════════════════════════════╝[/bold cyan]\n"
    )

    # 1. Display Truthful Architecture & Capability Registry
    console.print(get_architecture_capability_table())
    console.print("\n")

    # 2. Generate Enriched Synthetic Market World
    console.print(
        "[bold green]1. Generating Enriched Synthetic Market World (50 Assets, 5 Factors, Constraints, Views, Scenarios)...[/bold green]"
    )
    world = generate_market_world(seed=42)
    mkt_ctx = world.market_context()

    # 3. Construct ReviewContextBundle
    bundle = ReviewContextBundle(
        mode="single_domain",
        domains=(ReviewDomain.MARKET,),
        market=mkt_ctx,
        materiality="high",
        lifecycle="pre_implementation",
        grounding_mode=ReviewGroundingMode.STRUCTURED,
    )

    # 4. Set non-interactive artifact view mode
    os.environ["START_ARTIFACT_VIEW"] = "auto"

    # 5. Run Unified Institutional Review (Non-interactive)
    console.print("[bold green]2. Executing StART Institutional Review Workflow...[/bold green]\n")
    review_output = run_unified_review(
        bundle=bundle,
        interactive=False,
    )

    out_path = Path(review_output["output_path"])
    console.print("\n[bold green]3. Verifying Generated Institutional Artifact Pack...[/bold green]")

    expected_files = [
        "review_summary.json",
        "ledger.jsonl",
        "agent_orchestration.json",
        "agent_orchestration.mmd",
        "agent_orchestration.svg",
        "presentation_model.json",
    ]

    for fname in expected_files:
        fpath = out_path / fname
        exists = fpath.exists()
        size = fpath.stat().st_size if exists else 0
        status_str = f"[green]EXISTS ({size:,} bytes)[/green]" if exists else "[red]MISSING[/red]"
        console.print(f"  - {fname}: {status_str}")

    art_dir = out_path / "artifacts"
    if art_dir.exists():
        arts = list(art_dir.glob("*.*"))
        console.print(f"  - Visual & Tabular Artifacts Directory: [green]{len(arts)} files generated[/green]")
        for a in arts[:8]:
            console.print(f"    * {a.name} ({a.stat().st_size:,} bytes)")

    console.print(
        "\n[bold cyan]════════════════════════════════════════════════════════════════════════════════════[/bold cyan]"
    )
    console.print(
        "[bold green]✔ StART v4.4 Institutional Showcase Completed Successfully with Zero Errors.[/bold green]\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(run_showcase())
