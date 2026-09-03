"""Uniform Rich-table renderers for the terminal review UI (v2.3.0 #9).

One place that builds the Rich tables used across the committee experience:
dataset discovery, feature-engineering diagnostics, sensitivity ranking, shock
tables, metrics, tuning, adapter inventory, challenge logs, and signoff
summaries. The terminal uses these exclusively — markdown tables are reserved
for dashboard.md / reports / transcripts.

Each builder returns a ``rich.table.Table`` (or panel) so callers just
``console.print(...)``. Builders never fabricate values; they render only what
they are given.
"""

from __future__ import annotations

from typing import Any

from start.review.tables import (
    build_artifact_catalog_table,
    build_attribution_table,
    build_barrier_table,
    build_covariance_table,
    build_governance_table,
    build_portfolio_table,
    build_predictive_table,
    build_preflight_data_summary_table,
    build_scenario_table,
    build_treasury_table,
    render_checkpoint_panel,
)

__all__ = [
    "build_artifact_catalog_table",
    "build_attribution_table",
    "build_barrier_table",
    "build_covariance_table",
    "build_governance_table",
    "build_portfolio_table",
    "build_predictive_table",
    "build_preflight_data_summary_table",
    "build_scenario_table",
    "build_treasury_table",
    "render_checkpoint_panel",
]


def _table(title: str, columns: list[str], header_style: str = "bold") -> Any:
    from rich.table import Table

    t = Table(title=title, title_style="bold", header_style=header_style,
              title_justify="left", show_lines=False)
    for c in columns:
        t.add_column(c)
    return t


def dataset_discovery_table(store: Any, candidate_targets: list[str],
                            split_props: tuple[float, float, float]) -> Any:
    """#5: dataset transparency before recommendations."""
    t = _table("DatasetDiscoveryAgent — dataset transparency", ["Field", "Value"])
    t.add_row("Detected target", str(store.target or "—"))
    t.add_row("Candidate targets", ", ".join(candidate_targets) if candidate_targets else "—")
    t.add_row("Rows", str(store.n_rows if store.n_rows is not None else "—"))
    t.add_row("Features", str(store.n_features if store.n_features is not None else "—"))
    t.add_row("Numeric / categorical",
              f"{store.n_numeric if store.n_numeric is not None else '—'} / "
              f"{store.n_categorical if store.n_categorical is not None else '—'}")
    if store.class_distribution:
        bal = ", ".join(f"{k}={v:.1%}" for k, v in store.class_distribution.items())
        t.add_row("Class balance", bal)
    n_missing = sum(1 for p in store.missingness.values() if p > 0)
    t.add_row("Columns with missing", str(n_missing))
    n_out = sum(1 for c in store.outliers.values() if c > 0)
    t.add_row("Columns with outliers", str(n_out))
    t.add_row("Leakage candidates",
              ", ".join(store.leakage_candidates) if store.leakage_candidates else "none")
    t.add_row("High-correlation pairs", str(len(store.correlations)))
    tr, te, oo = split_props
    t.add_row("Proposed split (train/test/OOS)",
              f"{tr:.0%} / {te:.0%} / {oo:.0%}")
    return t


def outlier_evidence_table(store: Any, n: int = 10) -> Any:
    """#6: outlier diagnostics with count, percent, rule, threshold."""
    t = _table("FeatureEngineeringAgent — outlier evidence",
               ["Feature", "Outliers", "Percent", "Rule", "Action"])
    items = store.top_outliers(n)
    total = store.n_rows or 0
    for it in items:
        col = it.label.split(":")[0]
        cnt = it.value
        pct = f"{(100.0 * cnt / total):.1f}%" if total else "—"
        t.add_row(col, str(cnt), pct, "IQR (1.5×)", "winsorize")
    return t, bool(items)


def correlation_evidence_table(store: Any, n: int = 10) -> Any:
    """#6: correlation pairs with coefficient and proposed drop."""
    t = _table("FeatureEngineeringAgent — correlation evidence",
               ["Pair", "Coefficient", "Proposed drop"])
    items = store.top_correlations(n)
    for it in items:
        v = it.value
        t.add_row(f"{v['a']} ~ {v['b']}", f"{float(v['r']):.3f}", v["b"])
    return t, bool(items)


def metrics_table(metrics_by_split: dict[str, dict[str, float]]) -> Any:
    """#9: metrics by split as a Rich table."""
    keys = ["auc_roc", "pr_auc", "accuracy", "precision", "recall", "f1",
            "specificity", "brier_score", "ece",
            "rmse", "mse", "mae", "r2", "mape"]
    present = [k for k in keys if any(k in m for m in metrics_by_split.values())]
    t = _table("Metrics by split", ["Split"] + present)
    for split, m in metrics_by_split.items():
        row = [split] + [
            f"{m[k]:.4f}" if isinstance(m.get(k), (int, float)) else "—" for k in present
        ]
        t.add_row(*row)
    return t


def tuning_table(trials: list[dict[str, Any]]) -> Any:
    """#9: tuning trials as a Rich table."""
    if not trials:
        return _table("Hyperparameter tuning trials", ["No trials"])
    first_p = trials[0].get("params", {})
    keys = list(first_p.keys())
    columns = ["Trial"] + keys + ["Val metric", "Status"]
    t = _table("Hyperparameter tuning trials", columns)
    for tr in trials:
        p = tr.get("params", {})
        row = [str(tr.get("trial"))] + [str(p.get(k, "")) for k in keys] + [
            f"{tr.get('validation_metric', float('nan')):.4f}",
            str(tr.get("status"))
        ]
        t.add_row(*row)
    return t



def importance_table(rows: list[dict[str, Any]], n: int = 20) -> Any:
    """#9: feature importance as a Rich table."""
    t = _table("Global feature importance", ["Rank", "Feature", "Importance", "Direction"])
    for r in rows[:n]:
        t.add_row(str(r.get("rank")), str(r.get("feature")),
                  str(r.get("importance")), str(r.get("direction", "")))
    return t


def sensitivity_ranking_table(rows: list[dict[str, Any]]) -> Any:
    """#4/#8: feature sensitivity ranking by absolute max drift."""
    # collapse to per-feature max |drift|
    by_feat: dict[str, float] = {}
    risk: dict[str, str] = {}
    for r in rows:
        feat_val = r.get("feature")
        if not feat_val:
            continue
        f = str(feat_val)
        d = abs(float(r.get("drift", 0.0)))
        if f not in by_feat or d > by_feat[f]:
            by_feat[f] = d
            risk[f] = str(r.get("risk_impact", ""))
    ranked = sorted(by_feat.items(), key=lambda kv: kv[1], reverse=True)
    t = _table("ValidationAgent — feature sensitivity ranking",
               ["Rank", "Feature", "Max |drift|", "Risk level"])
    for i, (f, d) in enumerate(ranked, 1):
        t.add_row(str(i), f, f"{d:.4f}", risk.get(f, ""))
    return t


def shock_table(rows: list[dict[str, Any]], features: list[str] | None = None) -> Any:
    """#8: shock table (-30%..+30%) per reviewed feature."""
    shocks = [-0.3, -0.2, -0.1, 0.1, 0.2, 0.3]
    cols = ["Feature"] + [f"{int(s * 100):+d}%" for s in shocks]
    t = _table("ValidationAgent — shock analysis (delta vs baseline)", cols)
    # group drift by feature and shock
    grid: dict[str, dict[int, float]] = {}
    for r in rows:
        feat_val = r.get("feature")
        if not feat_val:
            continue
        f = str(feat_val)
        s = int(round(float(r.get("shock", 0)) * 100))
        grid.setdefault(f, {})[s] = float(r.get("drift", 0.0))
    feats = features or list(grid.keys())
    for f in feats:
        row = [f] + [
            (f"{grid.get(f, {}).get(int(s * 100), 0.0):+.4f}") for s in shocks
        ]
        t.add_row(*row)
    return t


_ADAPTER_COLORS = {
    "OPA": "bright_red", "MCP Server": "bright_cyan", "MCP SDK": "cyan",
    "MCP Inspector": "bright_blue", "Langfuse": "bright_magenta",
    "OpenTelemetry": "bright_green", "Garak": "red", "Promptfoo": "yellow",
    "NeMo Guardrails": "bright_yellow", "DeepEval": "green",
    "LangSmith": "magenta", "Phoenix": "bright_cyan",
}


def _adapter_color(name: str) -> str:
    return _ADAPTER_COLORS.get(name, "white")


def adapter_inventory_table(control_surface: list[dict[str, Any]]) -> Any:
    """#4/#10: adapter transparency — colored name, status, purpose, runtime,
    artifacts, evidence, and an install/fallback note when unavailable."""
    t = _table("AI Engineering Environment",
               ["Adapter", "Status", "Purpose", "Runtime (s)", "Artifacts",
                "Evidence", "Install / fallback"])
    style = {"complete": "green", "available": "green",
             "not_installed": "yellow", "error": "red"}
    for r in control_surface:
        st = r.get("status", "")
        available = st in ("complete", "available")
        note = "" if available else (r.get("install_guidance", "") or "not installed")
        t.add_row(
            f"[bold {_adapter_color(str(r.get('adapter')))}]{r.get('adapter')}[/]",
            f"[{style.get(st, 'white')}]{st}[/]",
            str(r.get("purpose", ""))[:42],
            f"{r.get('runtime_s', 0):.3f}" if r.get("runtime_s") is not None else "—",
            str(r.get("artifacts", 0)),
            str(r.get("evidence", 0)),
            note[:40],
        )
    return t


def llm_activation_panel(activation: Any) -> Any:
    """#3/#5: boxed LLM activation view with endpoint + provider color."""
    from rich.panel import Panel
    from rich.table import Table

    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="bold", no_wrap=True)
    grid.add_column()
    prov = getattr(activation, "provider", None) or "none"
    prov_color = {"openai": "bright_green", "anthropic": "bright_magenta",
                  "grok": "bright_cyan", "gemini": "bright_blue",
                  "deepseek": "bright_blue", "enterprise_llm_gateway": "bright_yellow",
                  "none": "dim"}.get(prov, "white")
    grid.add_row("Provider", f"[{prov_color}]{prov}[/]")
    grid.add_row("Trust domain", str(getattr(activation, "trust_domain", "public")))
    grid.add_row("Status", str(getattr(activation, "status", "—")))
    grid.add_row("Endpoint", str(getattr(activation, "endpoint", "—")))
    if getattr(activation, "model", None):
        grid.add_row("Model", str(activation.model))
    _note = getattr(activation, "detail", None) or getattr(activation, "note", None)
    if _note:
        grid.add_row("Note", str(_note))
    return Panel(grid, title="[bold]LLM activation[/bold]",
                 border_style=prov_color if prov != "none" else "grey50",
                 title_align="left")


def challenge_log_table(challenges: list[dict[str, Any]]) -> Any:
    """#9: reviewer challenge log as a Rich table."""
    from start.cli.view import get_styled_agent_name
    t = _table("Reviewer challenge log",
               ["Status", "Agent", "Challenge", "Evidence used"])
    style = {"open": "yellow", "closed": "green", "unresolved": "red"}
    for c in challenges:
        st_val = str(c.get("status", "open"))
        t.add_row(
            f"[{style.get(st_val, 'white')}]{st_val}[/]",
            get_styled_agent_name(str(c.get("agent"))),
            str(c.get("text"))[:50],
            ", ".join(c.get("evidence_used", [])) or "—",
        )
    return t


def _impact_for(key: str, choice: str, effective: str, recommended: str = "") -> str:
    """Plain-language downstream impact of a decision (v2.3.1 #8)."""
    k = key.lower()
    if k == "architecture":
        is_override = (
            choice in ("override", "overridden", "keep", "modify", "reject")
            or (bool(recommended) and bool(effective) and effective != recommended)
        )
        if is_override:
            if recommended and effective != recommended:
                return f"trained user-selected {effective} (recommended: {recommended})"
            return f"trained user-selected {effective}"
        return f"trained recommended {effective}"
    if "correlation_pruning" in k:
        return ("kept all features (no pruning)" if choice == "reject"
                else "dropped highly-correlated features before training")
    if "outlier" in k:
        return ("outliers left untouched" if choice == "reject"
                else "outlier handling applied where wired")
    if "scaling" in k:
        return ("scaling skipped" if choice == "reject" else "scaling noted")
    if k in ("metric_priority", "metric"):
        return f"primary metric routed for '{effective}'"
    if k == "validation":
        return "validation review accepted into signoff"
    if k == "target":
        return f"review target = {effective}"
    return f"effective: {effective}"


def decision_ledger_table(decisions: list[dict[str, Any]]) -> Any:
    """#8: compact review decision ledger as a Rich table."""
    t = _table("Review decision ledger",
               ["Checkpoint", "Recommended", "User choice", "Status",
                "Evidence", "Execution impact"])
    status_style = {"accept": "green", "auto_accept": "green",
                    "keep": "yellow", "modify": "yellow", "reject": "red",
                    "non_interactive_keep": "green"}
    for d in decisions:
        choice = d.get("choice", "")
        rec_val = str(d.get("recommended", "")).strip()
        eff_val = str(d.get("effective", "")).strip()
        if rec_val and eff_val and rec_val == eff_val:
            status = "accepted"
            style_key = "accept"
        else:
            status = {"accept": "accepted", "auto_accept": "accepted",
                      "non_interactive_keep": "accepted", "keep": "overridden",
                      "modify": "overridden", "override": "overridden",
                      "reject": "rejected"}.get(choice, choice)
            style_key = choice
        ev = ", ".join(d.get("evidence_ids", []) or []) or "—"
        t.add_row(
            str(d.get("key")),
            str(d.get("recommended")),
            str(d.get("effective")),
            f"[{status_style.get(style_key, 'white')}]{status}[/]",
            ev[:24],
            _impact_for(
                d.get("key", ""),
                choice,
                str(d.get("effective", "")),
                recommended=str(d.get("recommended", "")),
            ),
        )
    return t


def kfold_table(kfold: Any) -> Any:
    """#7: per-fold metrics with mean/std and selected params (Rich)."""
    method = getattr(kfold, "method", "kfold")
    metric = getattr(kfold, "primary_metric", "metric")
    t = _table(
        f"K-fold tuning — {getattr(kfold, 'n_folds', '?')}-fold ({method}), "
        f"primary metric: {metric}",
        ["Fold", f"{metric} (best params)", "n_train", "n_val"])
    for f in getattr(kfold, "best_fold_results", []):
        t.add_row(str(f.fold), f"{f.metric:.4f}", str(f.n_train), str(f.n_val))
    t.add_section()
    t.add_row("mean", f"{getattr(kfold, 'best_mean_metric', 0):.4f}", "", "")
    t.add_row("std", f"{getattr(kfold, 'best_std_metric', 0):.4f}", "", "")
    t.add_row("params", str(getattr(kfold, "best_params", {})), "", "")
    return t


def decision_ledger_markdown(decisions: list[dict[str, Any]]) -> str:
    """#8: decision ledger for dashboard/transcript/notebook."""
    lines = ["### Review decision ledger", "",
             "| Checkpoint | Recommended | User choice | Status | Evidence | Execution impact |",
             "| --- | --- | --- | --- | --- | --- |"]
    for d in decisions:
        choice = d.get("choice", "")
        rec_val = str(d.get("recommended", "")).strip()
        eff_val = str(d.get("effective", "")).strip()
        if rec_val and eff_val and rec_val == eff_val:
            status = "accepted"
        else:
            status = {"accept": "accepted", "auto_accept": "accepted",
                      "non_interactive_keep": "accepted", "keep": "overridden",
                      "modify": "overridden", "override": "overridden",
                      "reject": "rejected"}.get(choice, choice)
        ev = ", ".join(d.get("evidence_ids", []) or []) or "—"
        impact = _impact_for(
            d.get("key", ""),
            choice,
            str(d.get("effective", "")),
            recommended=str(d.get("recommended", "")),
        )
        lines.append(
            f"| {d.get('key')} | {d.get('recommended')} | {d.get('effective')} "
            f"| {status} | {ev} | {impact} |"
        )
    return "\n".join(lines) + "\n"
