"""v4.3.0 Interactive Review Wizard.

Guides the reviewer through:
  Review Mode -> Review Domain(s) -> Predictive Technology (if Predictive)
  -> Governance Metadata (Materiality, Lifecycle, Business Context, etc. with multiline input)
  -> Domain Data Setup -> Review Scope -> Plan Preview -> Execution
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TextIO, cast

from rich.console import Console

from start.data.synthetic_market import generate_market_world
from start.registry.market_contexts import MarketContext, PortfolioSpec
from start.review.applicability import build_plan_preview
from start.review.architecture import (
    DOMAIN_DESCRIPTIONS,
    DOMAIN_LABELS,
    LIFECYCLE_LABELS,
    MODE_LABELS,
    TECHNOLOGY_LABELS,
    TRADITIONAL_ML_MODELS,
    PredictiveTechnology,
    ReviewContextBundle,
    ReviewDomain,
    ReviewGroundingMode,
    ReviewLifecycle,
    ReviewMode,
    parse_domain_selection,
    requires_predictive_technology,
)
from start.review.multiline_input import ReviewCancelled, read_multiline_text

console = Console()


def _ask_choice(
    prompt: str,
    options: list[tuple[str, str, str]],
    default: str = "1",
    ask: Callable[[str], str] = input,
) -> str:
    """Prompt with numbered choices and return the selected key."""
    console.print(f"\n[bold]{prompt}[/bold]")
    for key, title, desc in options:
        console.print(f"  [{key}] [cyan]{title}[/cyan]")
        if desc:
            console.print(f"      [dim]{desc}[/dim]")
    try:
        raw = ask(f"Select option [default: {default}]: ").strip()
    except (EOFError, KeyboardInterrupt):
        return default
    return raw if raw else default


def run_review_wizard(
    ask: Callable[[str], str] = input,
    stream: TextIO | None = None,
    seed: int = 42,
    discovery_client: Any | None = None,
) -> dict[str, Any]:
    """Interactive wizard collecting review configuration and contexts."""
    console.print("\n[bold]StART — Standardized Agentic Reusable Tests[/bold]")
    console.print("[bold cyan]Interactive Model Risk Review Setup[/bold cyan]")
    console.print("────────────────────────────────────────────────────────────────────────────")

    # 1. Review Mode
    mode_choice = _ask_choice(
        "Select Review Mode:",
        [
            ("1", MODE_LABELS[ReviewMode.SINGLE_DOMAIN][0], MODE_LABELS[ReviewMode.SINGLE_DOMAIN][1]),
            ("2", MODE_LABELS[ReviewMode.CROSS_DOMAIN][0], MODE_LABELS[ReviewMode.CROSS_DOMAIN][1]),
        ],
        default="1",
        ask=ask,
    )
    mode = ReviewMode.CROSS_DOMAIN if mode_choice == "2" else ReviewMode.SINGLE_DOMAIN

    # 2. Review Domain(s)
    domain_options = [
        ("1", DOMAIN_LABELS[ReviewDomain.PREDICTIVE], DOMAIN_DESCRIPTIONS[ReviewDomain.PREDICTIVE]),
        ("2", DOMAIN_LABELS[ReviewDomain.MARKET], DOMAIN_DESCRIPTIONS[ReviewDomain.MARKET]),
        ("3", DOMAIN_LABELS[ReviewDomain.TREASURY], DOMAIN_DESCRIPTIONS[ReviewDomain.TREASURY]),
    ]

    if mode is ReviewMode.SINGLE_DOMAIN:
        d_choice = _ask_choice("Select Review Domain:", domain_options, default="1", ask=ask)
        domains = parse_domain_selection(d_choice, mode=ReviewMode.SINGLE_DOMAIN)
    else:
        console.print("\n[bold]Select two or more Review Domains:[/bold]")
        for key, title, desc in domain_options:
            console.print(f"  [{key}] [cyan]{title}[/cyan]")
            if desc:
                console.print(f"      [dim]{desc}[/dim]")
        while True:
            try:
                raw_d = ask(
                    "Enter selections separated by commas (e.g. 2,3 or 1,2,3) [default: 2,3]: "
                ).strip() or "2,3"
                domains = parse_domain_selection(raw_d, mode=ReviewMode.CROSS_DOMAIN)
                break
            except ValueError as e:
                console.print(f"  [red]Error:[/red] {e}. Please re-enter.")

    # 3. Predictive Technology (ONLY if Predictive domain selected)
    technology: PredictiveTechnology | None = None
    if requires_predictive_technology(domains):
        tech_choice = _ask_choice(
            "Select Predictive Modeling Technology:",
            [
                (
                    "1",
                    TECHNOLOGY_LABELS[PredictiveTechnology.TRADITIONAL_ML][0],
                    TECHNOLOGY_LABELS[PredictiveTechnology.TRADITIONAL_ML][1],
                ),
                (
                    "2",
                    TECHNOLOGY_LABELS[PredictiveTechnology.DEEP_LEARNING][0],
                    TECHNOLOGY_LABELS[PredictiveTechnology.DEEP_LEARNING][1],
                ),
            ],
            default="1",
            ask=ask,
        )
        technology = (
            PredictiveTechnology.DEEP_LEARNING
            if tech_choice == "2"
            else PredictiveTechnology.TRADITIONAL_ML
        )

    # 4. AI Reviewer Agent Backend
    backend_choice = _ask_choice(
        "Select AI Reviewer Agent Backend (LLM Mode):",
        [
            (
                "1",
                "None (Deterministic Rule-Based / Local Engines) (default)",
                "Fast, reproducible deterministic validation",
            ),
            (
                "2",
                "Enterprise LLM Gateway (Firm Environment)",
                "Secured enterprise gateway routing",
            ),
            (
                "3",
                "Public LLM Providers (OpenAI, Anthropic, Gemini, DeepSeek, Grok)",
                "Third-party APIs with live dialogue",
            ),
        ],
        default="1",
        ask=ask,
    )
    agent_mode = "deterministic"
    llm_provider = "none"
    llm_model = None
    llm_status = "DETERMINISTIC"
    llm_detail = "No LLM selected; deterministic engines only."

    if backend_choice == "2":
        agent_mode = "llm"
        llm_provider = "enterprise_llm_gateway"
        llm_model = "gateway-managed"
        llm_status = "CONFIGURED"
        llm_detail = "Enterprise gateway routing."
    elif backend_choice == "3":
        agent_mode = "llm"
        prov_choice = _ask_choice(
            "Select Public LLM Provider:",
            [
                ("1", "OpenAI", "gpt-4o-mini (default) / gpt-4o / o-series"),
                ("2", "Anthropic", "claude-3-5-sonnet / haiku"),
                ("3", "Gemini", "gemini-2.0-flash / gemini-1.5-flash"),
                ("4", "DeepSeek", "deepseek-chat / deepseek-reasoner"),
                ("5", "Grok", "grok-2-latest / grok-3-mini"),
            ],
            default="1",
            ask=ask,
        )
        prov_map = {"1": "openai", "2": "anthropic", "3": "gemini", "4": "deepseek", "5": "grok"}
        llm_provider = prov_map.get(prov_choice, "openai")
        provider_display_map = {
            "openai": "OpenAI",
            "anthropic": "Anthropic",
            "gemini": "Gemini",
            "deepseek": "DeepSeek",
            "grok": "Grok",
            "enterprise_llm_gateway": "Enterprise LLM Gateway",
            "none": "None",
        }
        prov_display = provider_display_map.get(llm_provider, llm_provider.title())

        # 4a. API Credential Preflight
        import os
        import sys

        from start.providers.keys import ensure_provider_key

        key_status = ensure_provider_key(llm_provider, prompt_for_key=False)
        if key_status.ok:
            console.print(f"  [green]{prov_display} credential: configured ✓[/green]")
            console.print(f"  [dim]Source: {key_status.source.title()}[/dim]")
        else:
            console.print(f"  [red]{prov_display} credential: missing ✗[/red]")
            if ask is input and sys.stdin.isatty():
                prompted_status = ensure_provider_key(llm_provider, prompt_for_key=True)
                if prompted_status.ok:
                    console.print(f"  [green]{prov_display} credential: configured ✓[/green]")
                else:
                    console.print(
                        "  [yellow]Proceeding without API key; "
                        "live calls will fail unless configured.[/yellow]"
                    )
            else:
                console.print(
                    "  [yellow]Proceeding without API key in non-interactive environment.[/yellow]"
                )

        # 4b. Model Discovery & Effective Default Resolution
        from start.core.config import load_config
        from start.providers.model_discovery import RealProviderModelDiscovery

        conf = load_config()
        configured_model = os.environ.get("START_LLM__MODEL") or (
            conf.llm.model if conf.llm.model else None
        )
        canonical_defaults = {
            "openai": "gpt-5-mini",
            "anthropic": "claude-sonnet-4-6",
            "gemini": "gemini-2.0-flash",
            "deepseek": "deepseek-chat",
            "grok": "grok-2-latest",
        }

        discovery = discovery_client if discovery_client is not None else RealProviderModelDiscovery()
        available_models: list[str] = []
        try:
            available_models = discovery.list_models(llm_provider)
        except Exception:
            available_models = []

        fallback_models: dict[str, list[str]] = {
            "openai": [
                "gpt-5-mini",
                "gpt-5",
                "gpt-4.1",
                "gpt-4.1-mini",
                "gpt-4o",
                "gpt-4o-mini",
                "o3-mini",
                "o1",
            ],
            "anthropic": [
                "claude-sonnet-4-6",
                "claude-3-7-sonnet-latest",
                "claude-3-5-sonnet-20241022",
                "claude-3-5-haiku-20241022",
            ],
            "gemini": ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"],
            "deepseek": ["deepseek-chat", "deepseek-reasoner"],
            "grok": ["grok-2-latest", "grok-3-mini", "grok-3"],
        }

        if not available_models:
            default_m = configured_model or canonical_defaults.get(llm_provider, "default-model")
            available_models = fallback_models.get(llm_provider, [default_m])
            console.print("  [dim]Model discovery unavailable; using configured supported models.[/dim]")

        # If model explicitly configured in config/env, ensure it is first
        if configured_model:
            if configured_model in available_models:
                available_models = [configured_model] + [m for m in available_models if m != configured_model]
            else:
                available_models = [configured_model] + available_models

        # Concise Reviewer Menu: top 7 primary models + optional extended list
        if len(available_models) > 7:
            primary_models = available_models[:7]
            model_opts = []
            for idx, m_id in enumerate(primary_models, 1):
                desc = "configured default" if idx == 1 else ""
                model_opts.append((str(idx), m_id, desc))
            model_opts.append((
                str(len(primary_models) + 1),
                "Show all compatible models...",
                f"View all {len(available_models)} models",
            ))

            m_choice = _ask_choice(f"Select {prov_display} Model:", model_opts, default="1", ask=ask)
            if m_choice == str(len(primary_models) + 1) or m_choice.lower() in ("all", "show all"):
                full_opts = [
                    (str(i), mid, "configured default" if i == 1 else "")
                    for i, mid in enumerate(available_models, 1)
                ]
                m_choice = _ask_choice(
                    f"Select {prov_display} Model (All Compatible):", full_opts, default="1", ask=ask
                )

            try:
                m_idx = int(m_choice) - 1
                if 0 <= m_idx < len(available_models):
                    llm_model = available_models[m_idx]
                else:
                    llm_model = available_models[0]
            except ValueError:
                if m_choice in available_models:
                    llm_model = m_choice
                else:
                    llm_model = available_models[0]
        else:
            model_opts = []
            for idx, m_id in enumerate(available_models, 1):
                desc = "configured default" if idx == 1 else ""
                model_opts.append((str(idx), m_id, desc))

            m_choice = _ask_choice(f"Select {prov_display} Model:", model_opts, default="1", ask=ask)
            try:
                m_idx = int(m_choice) - 1
                if 0 <= m_idx < len(available_models):
                    llm_model = available_models[m_idx]
                else:
                    llm_model = available_models[0]
            except ValueError:
                if m_choice in available_models:
                    llm_model = m_choice
                else:
                    llm_model = available_models[0]

        console.print("\n  [bold]AI Reviewer:[/bold] Public LLM")
        console.print(f"  [bold]Provider:[/bold]    {prov_display}")
        console.print(f"  [bold]Model:[/bold]       {llm_model}\n")

        # 4c. Live Backend Preflight
        try:
            from typing import Any as AnyType

            from start.core.config import LLMConfig
            from start.providers.llm import get_llm_provider
            from start.providers.llm_activation import preflight_llm

            cfg_llm = LLMConfig(provider=cast(AnyType, llm_provider), model=llm_model or "")
            prov_inst = get_llm_provider(cfg_llm)
            act_report = preflight_llm(llm_provider, prov_inst, probe=False)
            llm_status = act_report.status
            llm_detail = act_report.detail
        except Exception as exc:
            llm_status = "CONFIGURED"
            llm_detail = f"Configured {prov_display} ({exc})"

    from start.review.architecture import LLMReviewConfig

    backend_mode = "public" if backend_choice == "3" else (
        "enterprise" if backend_choice == "2" else "none"
    )
    llm_config = LLMReviewConfig(
        backend_mode=backend_mode,
        provider=llm_provider,
        model=llm_model,
        status=llm_status,
        detail=llm_detail,
    )

    # 5. Governance Metadata Setup Order
    mat_choice = _ask_choice(
        "Select Model Materiality:",
        [
            ("1", "High (Tier 1)", "Core risk or pricing framework"),
            ("2", "Medium (Tier 2)", "Operational / advisory model"),
            ("3", "Low (Tier 3)", "Low impact / experimental"),
        ],
        default="1",
        ask=ask,
    )
    materiality = {"1": "high", "2": "medium", "3": "low"}.get(mat_choice, "high")

    lifecycle_choice = _ask_choice(
        "Select Review Lifecycle:",
        [
            (
                "1",
                LIFECYCLE_LABELS[ReviewLifecycle.INITIAL_VALIDATION],
                "First comprehensive model review",
            ),
            (
                "2",
                LIFECYCLE_LABELS[ReviewLifecycle.PERIODIC_VALIDATION],
                "Scheduled annual/periodic validation",
            ),
            (
                "3",
                LIFECYCLE_LABELS[ReviewLifecycle.MATERIAL_MODEL_CHANGE],
                "Re-validation following material update",
            ),
            (
                "4",
                LIFECYCLE_LABELS[ReviewLifecycle.ONGOING_MONITORING],
                "Investigation of drift or performance anomaly",
            ),
            (
                "5",
                LIFECYCLE_LABELS[ReviewLifecycle.PRE_IMPLEMENTATION],
                "Pre-implementation design and architecture assessment",
            ),
        ],
        default="1",
        ask=ask,
    )
    lifecycle_map = {
        "1": ReviewLifecycle.INITIAL_VALIDATION,
        "2": ReviewLifecycle.PERIODIC_VALIDATION,
        "3": ReviewLifecycle.MATERIAL_MODEL_CHANGE,
        "4": ReviewLifecycle.ONGOING_MONITORING,
        "5": ReviewLifecycle.PRE_IMPLEMENTATION,
    }
    lifecycle = lifecycle_map.get(lifecycle_choice, ReviewLifecycle.INITIAL_VALIDATION)

    # Multiline text input for Governance
    console.print("\n[bold cyan]══════════════════ Governance Information ══════════════════[/bold cyan]")
    business_context = read_multiline_text(
        "Business Context", required=True, stream=stream, printer=console.print
    )
    reviewer_clarification = read_multiline_text(
        "Reviewer Clarification", required=False, stream=stream, printer=console.print
    )
    intended_use = read_multiline_text(
        "Intended Use / Decision Impact", required=False, stream=stream, printer=console.print
    )
    known_limitations = read_multiline_text(
        "Known Limitations / Reviewer Concerns", required=False, stream=stream, printer=console.print
    )

    grounding_mode = (
        ReviewGroundingMode.STRUCTURED
        if ReviewDomain.MARKET in domains
        else ReviewGroundingMode.LEGACY_FREEFORM
    )

    bundle = ReviewContextBundle(
        mode=mode,
        domains=domains,
        technology=technology,
        materiality=materiality,
        lifecycle=lifecycle,
        llm_config=llm_config,
        business_context=business_context,
        reviewer_clarification=reviewer_clarification,
        intended_use=intended_use,
        known_limitations=known_limitations,
        grounding_mode=grounding_mode,
    )

    predictive_config: dict[str, Any] = {
        "business_context": business_context,
        "clarification": reviewer_clarification,
        "intended_use": intended_use,
        "known_limitations": known_limitations,
        "agent_mode": agent_mode,
        "llm_provider": llm_provider,
        "llm_model": llm_model,
        "llm_config": llm_config,
        "materiality": materiality,
        "lifecycle": lifecycle,
        "mode": mode,
        "domains": domains,
        "technology": technology,
    }

    # 6. Domain Data Setup
    # A) Predictive Domain Data Setup
    if ReviewDomain.PREDICTIVE in domains:
        if technology is PredictiveTechnology.TRADITIONAL_ML:
            model_opts = [(str(i + 1), m, "") for i, m in enumerate(TRADITIONAL_ML_MODELS)]
            model_c = _ask_choice("Select Propensity Model:", model_opts, default="1", ask=ask)
            try:
                idx = int(model_c) - 1
                model_name = (
                    TRADITIONAL_ML_MODELS[idx]
                    if 0 <= idx < len(TRADITIONAL_ML_MODELS)
                    else "Random Forest"
                )
            except ValueError:
                model_name = "Random Forest"
            predictive_config["model"] = model_name.lower().replace(" ", "_")
            predictive_config["workflow_mode"] = "Propensity Suite"
            predictive_config["activation"] = "relu"
        else:
            nn_models = ["mlp", "rnn", "lstm", "cnn", "gru", "bi_lstm", "gnn", "dcn"]
            nn_opts = [
                ("1", "MLP", "Multilayer Perceptron"),
                ("2", "RNN", "Recurrent Neural Network"),
                ("3", "LSTM", "Long Short-Term Memory"),
                ("4", "CNN", "Convolutional 1D"),
                ("5", "GRU", "Gated Recurrent Unit"),
                ("6", "Bi-LSTM", "Bidirectional LSTM"),
                ("7", "GNN", "Graph Neural Network"),
                ("8", "DCN", "Wide & Deep / Deep & Cross Network"),
            ]
            nn_c = _ask_choice("Select Neural Network Architecture:", nn_opts, default="1", ask=ask)
            try:
                nidx = int(nn_c) - 1
                model_name = nn_models[nidx] if 0 <= nidx < len(nn_models) else "mlp"
            except ValueError:
                model_name = "mlp"
            predictive_config["model"] = model_name
            predictive_config["workflow_mode"] = "Deep Learning Suite"

            act_opts = [
                ("1", "ReLU", "Default standard rectified linear"),
                ("2", "LeakyReLU", "Small non-zero gradient for negative values"),
                ("3", "GELU", "Gaussian Error Linear Unit"),
                ("4", "Tanh", "Hyperbolic tangent"),
                ("5", "Sigmoid", "Logistic sigmoid"),
            ]
            act_c = _ask_choice("Select Activation Function:", act_opts, default="1", ask=ask)
            act_map = {"1": "relu", "2": "leaky_relu", "3": "gelu", "4": "tanh", "5": "sigmoid"}
            predictive_config["activation"] = act_map.get(act_c, "relu")

        # Dataset selection
        from start.data.selection import WIZARD_OPTIONS, resolve_wizard_choice

        ds_opts = [(key, text, "") for key, text in WIZARD_OPTIONS]
        ds_choice = _ask_choice("Select Predictive Dataset Source:", ds_opts, default="1", ask=ask)
        selection = resolve_wizard_choice(ds_choice, seed=seed)
        predictive_config["dataset_selection"] = selection
        predictive_config["target_column"] = selection.target_column or "is_fraud"
        predictive_config["split_strategy_name"] = "stratified"
        predictive_config["split_proportions"] = (0.60, 0.20, 0.20)
        predictive_config["stratify"] = True
        predictive_config["class_weight"] = "balanced"
        bundle.tabular = selection.frame

    # B) Market / Treasury Domain Data Setup
    if ReviewDomain.MARKET in domains or ReviewDomain.TREASURY in domains:
        market_ds_c = _ask_choice(
            "Select Market/Treasury Data Source:",
            [
                (
                    "1",
                    "Built-in Synthetic Market World (Recommended)",
                    "50 assets, 1000 daily periods, 5 factors, short-rate dynamics",
                ),
                (
                    "2",
                    "Local Market / Treasury Dataset",
                    "Load returns, prices, and risk factors from local CSV/Parquet",
                ),
                (
                    "3",
                    "Existing Prepared Context",
                    "Pass existing in-memory MarketContext / ShortRateContext",
                ),
            ],
            default="1",
            ask=ask,
        )

        if market_ds_c == "1" or market_ds_c not in ("2", "3"):
            world = generate_market_world(
                n_assets=50,
                n_periods=1000,
                n_factors=5,
                periods_per_year=252,
                seed=seed,
                include_short_rate=True,
                missing_rate=0.15,
            )
            renamed = {old: f"ASSET_{i + 1:03d}" for i, old in enumerate(world.returns.columns)}

            market_ctx = MarketContext(
                returns=world.returns.rename(columns=renamed),
                prices=world.prices.rename(columns=renamed),
                periods_per_year=world.periods_per_year,
                risk_free_rate=0.02,
                risk_free_frequency="annual",
                factor_returns=world.factor_returns,
                factor_exposures=world.factor_exposures.rename(index=renamed),
                pnl=world.pnl,
                hypothetical_pnl=world.hypothetical_pnl,
                var_series=world.var_series,
                var_confidence=world.var_confidence,
                portfolio=PortfolioSpec(
                    weights=world.weights.rename(renamed),
                    benchmark_weights=world.benchmark_weights.rename(renamed),
                ),
                seed=seed,
            )
            incomplete_ret = (
                world.incomplete_returns.rename(columns=renamed)
                if world.incomplete_returns is not None
                else world.returns.rename(columns=renamed)
            )
            incomplete_ctx = MarketContext(
                returns=incomplete_ret,
                periods_per_year=world.periods_per_year,
                portfolio=PortfolioSpec(weights=world.weights.rename(renamed)),
                seed=seed,
            )
            short_rate_ctx = world.short_rate_context()

            bundle.market = market_ctx
            bundle.short_rate = short_rate_ctx
            if bundle.tabular is None:
                bundle.tabular = incomplete_ctx

            # Display compact typed context summary
            console.print("\n[bold cyan]Context Summary:[/bold cyan]")
            if ReviewDomain.MARKET in domains:
                fp = market_ctx.fingerprint()[:16]
                n_a = market_ctx.returns.shape[1] if market_ctx.returns is not None else 0
                n_p = market_ctx.returns.shape[0] if market_ctx.returns is not None else 0
                n_f = market_ctx.factor_returns.shape[1] if market_ctx.factor_returns is not None else 0
                console.print(
                    f"  ✓ MarketContext: {n_a} assets x {n_p} periods, {n_f} factors (fp={fp}...)"
                )
            if (
                ReviewDomain.TREASURY in domains
                and short_rate_ctx is not None
                and short_rate_ctx.rates is not None
            ):
                n_obs = short_rate_ctx.rates.size
                dt = short_rate_ctx.dt
                console.print(f"  ✓ ShortRateContext: {n_obs} daily observations (dt={dt:.4f})")

    # 7. Review Scope
    scope_choice = _ask_choice(
        "Select Review Scope:",
        [
            (
                "1",
                "Full Recommended Review (All applicable registered surfaces)",
                "Execute full standard validation suite",
            ),
            (
                "2",
                "Customize",
                "Select specific test families to run",
            ),
        ],
        default="1",
        ask=ask,
    )
    custom_families: tuple[str, ...] | None = None
    if scope_choice == "2":
        console.print("\n[bold cyan]── Scope Customization ──[/bold cyan]")
        if ReviewDomain.MARKET in domains:
            m_fam_opts = [
                ("1", "Portfolio Risk & Optimization (portfolio)", "MVO, HRP, HERC, CVaR, Black-Litterman"),
                ("2", "Factor Attribution (attribution)", "Factor returns, exposures, Brinson, Carino"),
                ("3", "Traded Risk & VaR (traded_risk)", "Historical, Parametric, Kupiec, Christoffersen"),
                ("4", "Covariance Estimation (covariance)", "Empirical, Ledoit-Wolf, Regularized EM"),
                ("5", "All Market Families (Recommended)", "Run all 4 market test families"),
            ]
            fam_sel = _ask_choice("Select Market Test Families:", m_fam_opts, default="5", ask=ask)
            fam_map = {
                "1": ("portfolio",),
                "2": ("attribution",),
                "3": ("traded_risk",),
                "4": ("covariance",),
                "5": ("portfolio", "attribution", "traded_risk", "covariance"),
            }
            custom_families = fam_map.get(fam_sel, None)

            # Method Configuration & Sensitivity Analysis
            console.print("\n[bold cyan]── Method Configuration & Sensitivity Analysis ──[/bold cyan]")
            cov_opts = [
                ("1", "Ledoit-Wolf Shrinkage (Recommended)", "Optimal analytical shrinkage towards identity"),
                ("2", "Regularized EM Imputation", "Iterative conditional expectation for missing data"),
                ("3", "Sample Empirical Covariance", "Unadjusted pairwise complete sample covariance"),
                ("4", "Compare All Covariance Methods", "Run empirical, shrinkage, and Regularized EM"),
            ]
            cov_c = _ask_choice("Select Primary Covariance Estimator:", cov_opts, default="1", ask=ask)

            port_opts = [
                ("1", "Hierarchical Risk Parity (HRP)", "Clustering without matrix inversion"),
                ("2", "Mean-Variance Optimization (MVO)", "Markowitz quadratic programming with box bounds"),
                ("3", "Hierarchical Equal Risk (HERC)", "Equal risk bounding across dendrogram clusters"),
                ("4", "CVaR Optimization", "Linear program optimizing expected tail shortfall"),
                ("5", "All Implemented Optimizers (Recommended)", "Compare all valid portfolio models"),
            ]
            port_c = _ask_choice("Select Primary Portfolio Construction:", port_opts, default="5", ask=ask)

            predictive_config["custom_covariance_method"] = cov_c
            predictive_config["custom_portfolio_method"] = port_c

    # 8. Review Plan Preview
    preview = build_plan_preview(bundle, families=custom_families)
    console.print("\n" + preview.render() + "\n")

    proceed = (ask("Proceed to execute review? [Y/n]: ") or "Y").strip().lower()
    if proceed in ("n", "no"):
        console.print("[yellow]Review execution cancelled by reviewer.[/yellow]")
        raise ReviewCancelled("Cancelled at plan preview")

    return {
        "bundle": bundle,
        "preview": preview,
        "predictive_config": predictive_config,
        "interactive": True,
    }
