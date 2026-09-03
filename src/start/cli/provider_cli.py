"""StART CLI provider diagnostics and non-interactive probes.

Commands:
  start provider probe   Non-interactive live connectivity and contract probe.
"""

from __future__ import annotations

import time
from typing import Any

import typer
from rich.console import Console

from start.providers import keys as keys_module
from start.providers.llm import (
    AnthropicProvider,
    DeepSeekProvider,
    GeminiProvider,
    GrokProvider,
    OpenAIProvider,
)

provider_app = typer.Typer(
    name="provider",
    help="LLM provider inspection, capability diagnostics, and connectivity probes.",
    no_args_is_help=True,
)
console = Console()

CANONICAL_PROBE_MODELS: dict[str, str] = {
    "openai": "gpt-5-mini",
    "anthropic": "claude-sonnet-4-5",
    "deepseek": "deepseek-chat",
    "gemini": "gemini-2.0-flash",
    "grok": "grok-2-latest",
}

EXIT_SUCCESS = 0
EXIT_PROVIDER_FAILURE = 1
EXIT_MISSING_CREDENTIAL = 2


def run_provider_probe(
    provider: str,
    model: str | None = None,
    prompt: str = "Reply with exactly: START_PROVIDER_OK",
    system: str = "You are a StART provider connectivity probe. Reply with exactly: START_PROVIDER_OK",
    output_token_budget: int = 512,
) -> int:
    """Execute a strictly non-interactive, finite provider smoke probe.

    Exit codes:
      0: Success (live provider completed and responded)
      1: Provider failure (live request error, contract error, or bad response)
      2: Missing credential (no API key in env or Keychain; live request not attempted)
    """
    prov = provider.strip().lower()
    if prov not in keys_module.PROVIDER_KEY_ENV:
        console.print(f"[bold red]Error:[/bold red] Unknown provider '{prov}'.")
        valid = sorted(k for k in keys_module.PROVIDER_KEY_ENV if k != "none")
        console.print(f"Supported providers: {valid}")
        return EXIT_PROVIDER_FAILURE

    prov_display = keys_module.PROVIDER_DISPLAY_NAMES.get(prov, prov.title())
    target_model = model or CANONICAL_PROBE_MODELS.get(prov, "default")

    # 1. Resolve credential using canonical resolver (NEVER PROMPT)
    key_status = keys_module.ensure_provider_key(prov, prompt_for_key=False, interactive=False)

    if not key_status.ok:
        console.print(f"\n[bold]StART Provider Probe[/bold] — {prov_display}")
        console.print("────────────────────────────────────────")
        console.print(f"  Provider          : {prov_display}")
        console.print(f"  Model             : {target_model}")
        console.print("  Credential Source : [red]Missing[/red]")
        console.print("  Live Request      : [dim]NOT ATTEMPTED[/dim]")
        console.print("  Status            : [yellow]MISSING_CREDENTIAL (Exit 2)[/yellow]")
        return EXIT_MISSING_CREDENTIAL

    # 2. Instantiate actual provider adapter (NEVER use deterministic fallback)
    provider_inst: Any
    if prov == "openai":
        provider_inst = OpenAIProvider(model=target_model)
    elif prov == "anthropic":
        provider_inst = AnthropicProvider(model=target_model)
    elif prov == "deepseek":
        provider_inst = DeepSeekProvider(model=target_model)
    elif prov == "gemini":
        provider_inst = GeminiProvider(model=target_model)
    elif prov == "grok":
        provider_inst = GrokProvider(model=target_model)
    else:
        console.print(f"[bold red]Error:[/bold red] Unsupported probe provider '{prov}'.")
        return EXIT_PROVIDER_FAILURE

    # 3. Send tiny fixed probe prompt with hard timeout / latency capture
    console.print(f"\n[bold]StART Provider Probe[/bold] — {prov_display}")
    console.print("────────────────────────────────────────")
    console.print(f"  Provider          : {prov_display}")
    console.print(f"  Model             : {target_model}")
    console.print(f"  Credential Source : {key_status.source.title()}")

    t0 = time.perf_counter()
    try:
        response_text = provider_inst.complete(
            system=system,
            user=prompt,
            output_token_budget=output_token_budget,
        )
        latency = getattr(provider_inst, "last_latency_seconds", None) or (time.perf_counter() - t0)
        resp_id = getattr(provider_inst, "last_response_id", "") or "n/a"
        cleaned_response = (response_text or "").strip()

        console.print("  Live Request      : [green]SUCCESS[/green]")
        console.print(f"  Response ID       : {resp_id}")
        console.print(f"  Latency           : {latency:.4f}s")
        console.print(f"  Response Text     : {cleaned_response}")
        console.print("  Status            : [bold green]PASS (Exit 0)[/bold green]\n")
        return EXIT_SUCCESS
    except Exception as exc:
        latency = time.perf_counter() - t0
        safe_err = f"{type(exc).__name__}: {exc}"
        console.print("  Live Request      : [red]FAILED[/red]")
        console.print(f"  Latency           : {latency:.4f}s")
        console.print(f"  Error Detail      : [red]{safe_err}[/red]")
        console.print("  Status            : [bold red]FAIL (Exit 1)[/bold red]\n")
        return EXIT_PROVIDER_FAILURE


@provider_app.command("probe")
def probe_cmd(
    provider: str = typer.Option(
        ...,
        "--provider",
        "-p",
        help="Provider to probe: openai | anthropic | gemini | deepseek | grok",
    ),
    model: str = typer.Option(
        None,
        "--model",
        "-m",
        help="Model ID to probe (defaults to canonical reviewer model).",
    ),
    prompt: str = typer.Option(
        "Reply with exactly: START_PROVIDER_OK",
        "--prompt",
        help="Test prompt to send.",
    ),
    output_token_budget: int = typer.Option(
        512,
        "--output-token-budget",
        "-b",
        help="Output token budget.",
    ),
) -> None:
    """Run a non-interactive, finite live provider probe."""
    code = run_provider_probe(
        provider=provider,
        model=model,
        prompt=prompt,
        output_token_budget=output_token_budget,
    )
    if code != 0:
        raise typer.Exit(code=code)
