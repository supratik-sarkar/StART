"""StART CLI commands for managing persistent provider credentials in macOS Keychain.

Commands:
  start keys configure   Interactive setup for OpenAI, Anthropic, Gemini, DeepSeek, Grok.
  start keys status      View configuration status and resolution source.
  start keys delete      Remove a stored credential from Keychain.
"""

from __future__ import annotations

import getpass
import os
from collections.abc import Callable

import typer
from rich.console import Console
from rich.table import Table

from start.providers import keys as keys_module

keys_app = typer.Typer(
    name="keys",
    help="Manage LLM provider credentials stored securely in macOS Keychain.",
    no_args_is_help=True,
)
console = Console()

SUPPORTED_PROVIDERS: list[str] = ["openai", "anthropic", "gemini", "deepseek", "grok"]


def _configure_single_provider(
    provider: str,
    *,
    ask: Callable[[str], str] = input,
) -> bool:
    """Configure or update a single provider's key in macOS Keychain."""
    prov_display = keys_module.PROVIDER_DISPLAY_NAMES.get(provider, provider.title())
    has_existing = keys_module.keychain_has_key(provider)

    if has_existing:
        console.print(
            f"\n[bold]{prov_display}[/bold] credential already configured in macOS Keychain."
        )
        console.print("  [1] Keep existing (default)")
        console.print("  [2] Replace credential")
        console.print("  [3] Delete credential")
        console.print("  [4] Cancel")
        choice = (ask("  Select action [default: 1]: ") or "1").strip()

        if choice in ("1", "K", "k", ""):
            console.print(f"  Keeping existing {prov_display} credential.")
            return True
        elif choice in ("3", "D", "d"):
            prompt_msg = f"  Are you sure you want to delete {prov_display} credential? [y/N]: "
            confirm = (ask(prompt_msg) or "n").strip().lower()
            if confirm in ("y", "yes"):
                if keys_module.keychain_delete_key(provider):
                    console.print(
                        f"  [yellow]Deleted {prov_display} credential from macOS Keychain.[/yellow]"
                    )
                else:
                    console.print(f"  [red]Failed to delete {prov_display} credential.[/red]")
            return False
        elif choice in ("4", "C", "c"):
            console.print("  Cancelled.")
            return True
        elif choice not in ("2", "R", "r"):
            console.print("  Invalid choice. Keeping existing.")
            return True

    console.print(f"\n[bold]{prov_display}[/bold]")
    try:
        secret = getpass.getpass(f"  Enter {prov_display} API key (input hidden): ").strip()
    except (KeyboardInterrupt, EOFError):
        console.print("\n  Input cancelled.")
        return False

    if not secret:
        console.print("  [yellow]No key entered; skipping.[/yellow]")
        return False

    if keys_module.keychain_set_key(provider, secret):
        console.print("  [green]Stored securely in macOS Keychain ✓[/green]")
        return True
    else:
        console.print("  [red]Failed to store in macOS Keychain.[/red]")
        return False


@keys_app.command("configure")
@keys_app.command("config")
def configure_cmd() -> None:
    """Configure API credentials interactively in macOS Keychain."""
    if not keys_module.keychain_is_supported():
        console.print(
            "[red]Error: macOS Keychain is not available on this system.[/red]\n"
            "On non-macOS environments, set credentials via standard environment variables."
        )
        raise typer.Exit(code=1)

    console.print("\n[bold]StART Provider Credentials[/bold]")
    console.print("────────────────────────────────────────")
    console.print("  [1] OpenAI")
    console.print("  [2] Anthropic")
    console.print("  [3] Gemini")
    console.print("  [4] DeepSeek")
    console.print("  [5] Grok")
    console.print("  [6] Configure all")
    console.print("  [7] Exit")

    choice = (input("\nSelect option [1-7]: ") or "").strip()

    if choice == "1":
        _configure_single_provider("openai")
    elif choice == "2":
        _configure_single_provider("anthropic")
    elif choice == "3":
        _configure_single_provider("gemini")
    elif choice == "4":
        _configure_single_provider("deepseek")
    elif choice == "5":
        _configure_single_provider("grok")
    elif choice == "6":
        console.print("\n[bold]Configuring all supported providers...[/bold]")
        for p in SUPPORTED_PROVIDERS:
            _configure_single_provider(p)
    elif choice == "7":
        console.print("Exiting.")
        return
    else:
        console.print("[yellow]Invalid choice. Exiting.[/yellow]")
        return

    # Print summary status table
    console.print("\n[bold]Credential configuration summary:[/bold]")
    status_cmd()


@keys_app.command("status")
def status_cmd() -> None:
    """Show configuration status and resolution source for LLM providers."""
    table = Table(title="StART Provider Credentials", title_style="bold")
    table.add_column("Provider", style="bold")
    table.add_column("Configured", justify="center")
    table.add_column("Source")

    for provider in ("openai", "anthropic", "gemini", "deepseek", "grok"):
        prov_display = keys_module.PROVIDER_DISPLAY_NAMES.get(provider, provider.title())

        # Check resolution hierarchy per provider
        configured_source = None
        if provider == "gemini":
            if os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"):
                configured_source = "Environment"
        elif provider == "grok":
            if os.environ.get("XAI_API_KEY") or os.environ.get("GROK_API_KEY"):
                configured_source = "Environment"
        else:
            env_var = keys_module.PROVIDER_KEY_ENV.get(provider)
            if env_var and os.environ.get(env_var):
                configured_source = "Environment"

        if configured_source is None and keys_module.keychain_has_key(provider):
            configured_source = "Keychain"

        if configured_source:
            table.add_row(prov_display, "[green]✓[/green]", configured_source)
        else:
            table.add_row(prov_display, "[red]✗[/red]", "[dim]Missing[/dim]")

    console.print(table)


@keys_app.command("delete")
def delete_cmd(
    provider: str = typer.Argument(
        None,
        help="Provider to delete: openai | anthropic | gemini | deepseek | grok | all",
    )
) -> None:
    """Delete a stored provider credential from macOS Keychain."""
    if not keys_module.keychain_is_supported():
        console.print("[red]Error: macOS Keychain is not available on this system.[/red]")
        raise typer.Exit(code=1)

    target = (provider or "").strip().lower()
    if not target:
        console.print("[bold]Select provider credential to delete:[/bold]")
        console.print("  [1] OpenAI")
        console.print("  [2] Anthropic")
        console.print("  [3] Gemini")
        console.print("  [4] DeepSeek")
        console.print("  [5] Grok")
        console.print("  [6] All stored StART credentials")
        console.print("  [7] Cancel")
        sel = (input("\nSelect option [1-7]: ") or "").strip()
        map_sel = {
            "1": "openai",
            "2": "anthropic",
            "3": "gemini",
            "4": "deepseek",
            "5": "grok",
            "6": "all",
        }
        target = map_sel.get(sel, "")
        if not target or target == "7":
            console.print("Cancelled.")
            return

    if target == "all":
        confirm = (input("Delete ALL StART credentials from macOS Keychain? [y/N]: ") or "n").strip().lower()
        if confirm in ("y", "yes"):
            for p in list(keys_module.PROVIDER_KEY_ENV.keys()):
                if p not in ("none", "hf_local", "enterprise_llm_gateway"):
                    keys_module.keychain_delete_key(p)
            console.print("[green]All StART Keychain credentials deleted.[/green]")
        else:
            console.print("Cancelled.")
        return

    prov_display = keys_module.PROVIDER_DISPLAY_NAMES.get(target, target.title())
    if not keys_module.keychain_has_key(target):
        console.print(f"[yellow]No {prov_display} credential found in macOS Keychain.[/yellow]")
        return

    confirm = (input(f"Delete {prov_display} credential from macOS Keychain? [y/N]: ") or "n").strip().lower()
    if confirm in ("y", "yes"):
        if keys_module.keychain_delete_key(target):
            console.print(f"[green]Deleted {prov_display} credential from macOS Keychain.[/green]")
        else:
            console.print(f"[red]Failed to delete {prov_display} credential.[/red]")
    else:
        console.print("Cancelled.")
