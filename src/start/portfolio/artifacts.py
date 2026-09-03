"""Typed artifact generators for institutional portfolio intelligence.

Core philosophy:
- Artifacts are backed by typed ArtifactRecord objects.
- Every artifact references the exact source test_id, EvidenceRecord IDs, and input data fingerprint.
- Non-empty evidence_ids provenance is strictly enforced.
- The underlying semantic payload is independently inspectable and hash-verified.
- Visual renderers produce inspectable SVG, Markdown, or JSON companion files.
"""

from __future__ import annotations

import hashlib
import html
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from start.portfolio.contracts import (
    ActiveRiskDecompositionResult,
    ActiveScenarioResult,
    BlackLittermanResult,
    BrinsonAttributionResult,
    CarinoLinkedAttributionResult,
    CovarianceComparisonResult,
    CovarianceDiagnostics,
    CVaROptimizationResult,
    DurationDiagnosticsResult,
    EfficientFrontierResult,
    FactorReturnAttributionResult,
    FactorRiskDecompositionResult,
    FactorRiskModelResult,
    HERCResult,
    HierarchicalTreeResult,
    MethodComparisonResult,
    PSDRepairResult,
    RebalanceDecision,
    ReverseStressResult,
    RiskContributionResult,
    RobustSensitivityResult,
    ScenarioResult,
    ScenarioSensitivityResult,
    ScenarioSetResult,
    TailBacktestResult,
    TailModelComparisonResult,
    TailRiskContributionResult,
    TailRiskEstimate,
    TailSeverityResult,
    TrackingErrorResult,
)


@dataclass(frozen=True)
class ArtifactSpec:
    """Canonical specification defining a deterministic artifact generator."""

    artifact_type: str
    title: str
    test_id: str
    evidence_ids: tuple[str, ...]
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class ArtifactRecord:
    """Instantiated, rendered artifact with cryptographic data provenance."""

    artifact_id: str
    spec: ArtifactSpec
    data_fingerprint: str
    semantic_payload: dict[str, Any]
    semantic_payload_hash: str
    semantic_hash_algorithm: str = "sha256"
    file_path: str | None = None
    rendering_format: str = "json"  # "svg" | "png" | "json" | "markdown" | "table"
    created_by_engine: str = "start.portfolio.artifacts"
    created_by_agent: str | None = None

    @property
    def evidence_ids(self) -> tuple[str, ...]:
        return self.spec.evidence_ids

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "artifact_type": self.spec.artifact_type,
            "title": self.spec.title,
            "test_id": self.spec.test_id,
            "evidence_ids": list(self.spec.evidence_ids),
            "data_fingerprint": self.data_fingerprint,
            "semantic_payload_hash": self.semantic_payload_hash,
            "semantic_hash_algorithm": self.semantic_hash_algorithm,
            "file_path": self.file_path,
            "rendering_format": self.rendering_format,
            "created_by_engine": self.created_by_engine,
            "created_by_agent": self.created_by_agent,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], spec: ArtifactSpec | None = None) -> ArtifactRecord:
        """Instantiate ArtifactRecord from dictionary, explicitly identifying legacy hash algorithms."""
        spec_obj = spec or ArtifactSpec(
            artifact_type=data.get("artifact_type", "unknown"),
            title=data.get("title", ""),
            test_id=data.get("test_id", ""),
            evidence_ids=tuple(data.get("evidence_ids", [])),
        )
        hash_val = data.get("semantic_payload_hash", "")
        algo = data.get("semantic_hash_algorithm") or parse_artifact_hash_algorithm(hash_val)
        return cls(
            artifact_id=data.get("artifact_id", ""),
            spec=spec_obj,
            data_fingerprint=data.get("data_fingerprint", ""),
            semantic_payload=data.get("semantic_payload", {}),
            semantic_payload_hash=hash_val,
            semantic_hash_algorithm=algo,
            file_path=data.get("file_path"),
            rendering_format=data.get("rendering_format", "json"),
            created_by_engine=data.get("created_by_engine", "start.portfolio.artifacts"),
            created_by_agent=data.get("created_by_agent"),
        )


def parse_artifact_hash_algorithm(hash_value: str) -> str:
    """Classify artifact semantic payload hash algorithm by length."""
    if len(hash_value) == 64:
        return "sha256"
    elif len(hash_value) == 32:
        return "md5_legacy"
    return "unknown"


def _hash_payload(payload: dict[str, Any]) -> str:
    serialized = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _validate_provenance(evidence_ids: tuple[str, ...], allow_empty: bool = False) -> None:
    if not evidence_ids and not allow_empty:
        raise ValueError(
            "Artifact must have explicit evidence_ids provenance; cannot create orphan artifact."
        )


# --------------------------------------------------------------------------- #
# 1. Dendrogram Artifact
# --------------------------------------------------------------------------- #
def render_dendrogram_artifact(
    tree_result: HierarchicalTreeResult,
    evidence_ids: tuple[str, ...],
    test_id: str = "portfolio.hierarchical_risk_parity.tree_topology",
    output_dir: Path | str | None = None,
    allow_empty_evidence: bool = False,
) -> ArtifactRecord:
    """Render a Hierarchical Dendrogram as an SVG vector image and semantic JSON tree."""
    _validate_provenance(evidence_ids, allow_empty=allow_empty_evidence)
    spec = ArtifactSpec(
        artifact_type="dendrogram",
        title="Hierarchical Risk Parity Dendrogram",
        test_id=test_id,
        evidence_ids=evidence_ids,
        parameters={"linkage_method": tree_result.linkage_method},
    )

    semantic_payload = {
        "assets": list(tree_result.assets),
        "linkage_method": tree_result.linkage_method,
        "leaf_order": list(tree_result.leaf_order),
        "quasi_diagonal_order": list(tree_result.quasi_diagonal_order),
        "linkage_matrix": tree_result.linkage_matrix,
        "cluster_tree": tree_result.cluster_tree,
        "cophenetic_correlation": tree_result.cophenetic_correlation,
    }
    payload_hash = _hash_payload(semantic_payload)
    artifact_id = f"ART-HRP-DENDROGRAM-{payload_hash[:8]}"

    file_path = None
    if output_dir:
        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        file_path = str(out_p / f"{artifact_id}.svg")
        svg_content = _generate_dendrogram_svg(tree_result)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(svg_content)
        # Write machine-readable JSON companion
        with open(str(out_p / f"{artifact_id}.json"), "w", encoding="utf-8") as f:
            json.dump(semantic_payload, f, indent=2)

    return ArtifactRecord(
        artifact_id=artifact_id,
        spec=spec,
        data_fingerprint=tree_result.covariance_fingerprint,
        semantic_payload=semantic_payload,
        semantic_payload_hash=payload_hash,
        file_path=file_path,
        rendering_format="svg" if file_path else "json",
    )


def _generate_dendrogram_svg(tree: HierarchicalTreeResult) -> str:
    assets = tree.quasi_diagonal_order
    n = len(assets)
    width = max(600, n * 60)
    height = 400

    lines = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
            f'width="{width}" height="{height}">'
        ),
        "  <style>",
        "    .title { font: bold 14px sans-serif; fill: #1e293b; }",
        "    .label { font: 12px sans-serif; fill: #475569; }",
        "    .link { fill: none; stroke: #2563eb; stroke-width: 2px; }",
        "  </style>",
        f'  <text x="20" y="30" class="title">HRP Dendrogram ({tree.linkage_method} linkage)</text>',
    ]
    step = (width - 100) / max(n, 1)
    for i, asset in enumerate(assets):
        x = 50 + i * step
        y = height - 40
        lines.append(f'  <text x="{x}" y="{y}" class="label" text-anchor="middle">{asset}</text>')
        lines.append(f'  <line x1="{x}" y1="{y - 15}" x2="{x}" y2="{y - 40}" class="link" />')

    lines.append("</svg>")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 2. Raw Correlation Heatmap Artifact
# --------------------------------------------------------------------------- #
def render_raw_correlation_artifact(
    corr_matrix: np.ndarray,
    assets: tuple[str, ...] | list[str],
    evidence_ids: tuple[str, ...],
    test_id: str = "portfolio.covariance_conditioning",
    output_dir: Path | str | None = None,
    allow_empty_evidence: bool = False,
) -> ArtifactRecord:
    """Render the Raw Asset Correlation Matrix."""
    _validate_provenance(evidence_ids, allow_empty=allow_empty_evidence)
    asset_list = list(assets)
    spec = ArtifactSpec(
        artifact_type="raw_correlation_heatmap",
        title="Asset Correlation Matrix",
        test_id=test_id,
        evidence_ids=evidence_ids,
        parameters={"assets": asset_list},
    )
    semantic_payload = {
        "assets": asset_list,
        "correlation_matrix": [[round(float(x), 6) for x in row] for row in corr_matrix],
    }
    payload_hash = _hash_payload(semantic_payload)
    artifact_id = f"ART-RAW-CORR-{payload_hash[:8]}"

    file_path = None
    if output_dir:
        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        file_path = str(out_p / f"{artifact_id}.md")
        df = pd.DataFrame(corr_matrix, index=asset_list, columns=asset_list)
        md = f"### Raw Asset Correlation Matrix\n\n```\n{df.to_string()}\n```\n"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(md)
        with open(str(out_p / f"{artifact_id}.json"), "w", encoding="utf-8") as f:
            json.dump(semantic_payload, f, indent=2)

    return ArtifactRecord(
        artifact_id=artifact_id,
        spec=spec,
        data_fingerprint=hashlib.sha256(corr_matrix.tobytes()).hexdigest()[:32],
        semantic_payload=semantic_payload,
        semantic_payload_hash=payload_hash,
        file_path=file_path,
        rendering_format="markdown" if file_path else "json",
    )


# --------------------------------------------------------------------------- #
# 3. Seriated Correlation Heatmap Artifact
# --------------------------------------------------------------------------- #
def render_seriated_correlation_artifact(
    corr_matrix: np.ndarray,
    ordered_assets: tuple[str, ...] | list[str],
    original_assets: tuple[str, ...] | list[str],
    evidence_ids: tuple[str, ...],
    test_id: str = "portfolio.hierarchical_risk_parity.tree_topology",
    output_dir: Path | str | None = None,
    allow_empty_evidence: bool = False,
) -> ArtifactRecord:
    """Render the Quasi-Diagonally Seriated Correlation Matrix."""
    _validate_provenance(evidence_ids, allow_empty=allow_empty_evidence)
    orig_list = list(original_assets)
    ord_list = list(ordered_assets)
    order_idx = [orig_list.index(a) for a in ord_list]
    seriated_mat = corr_matrix[np.ix_(order_idx, order_idx)]

    spec = ArtifactSpec(
        artifact_type="seriated_correlation_heatmap",
        title="Quasi-Diagonally Seriated Correlation Matrix",
        test_id=test_id,
        evidence_ids=evidence_ids,
        parameters={"ordered_assets": ord_list, "permutation": order_idx},
    )
    semantic_payload = {
        "ordered_assets": ord_list,
        "permutation": order_idx,
        "seriated_correlation_matrix": [[round(float(x), 6) for x in row] for row in seriated_mat],
    }
    payload_hash = _hash_payload(semantic_payload)
    artifact_id = f"ART-SERIATED-CORR-{payload_hash[:8]}"

    file_path = None
    if output_dir:
        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        file_path = str(out_p / f"{artifact_id}.md")
        df = pd.DataFrame(seriated_mat, index=ord_list, columns=ord_list)
        md = f"### Quasi-Diagonally Seriated Correlation Matrix\n\n```\n{df.to_string()}\n```\n"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(md)
        with open(str(out_p / f"{artifact_id}.json"), "w", encoding="utf-8") as f:
            json.dump(semantic_payload, f, indent=2)

    return ArtifactRecord(
        artifact_id=artifact_id,
        spec=spec,
        data_fingerprint=hashlib.sha256(corr_matrix.tobytes()).hexdigest()[:32],
        semantic_payload=semantic_payload,
        semantic_payload_hash=payload_hash,
        file_path=file_path,
        rendering_format="markdown" if file_path else "json",
    )


# --------------------------------------------------------------------------- #
# 4. Distance Matrix Artifact
# --------------------------------------------------------------------------- #
def render_distance_matrix_artifact(
    dist_matrix: np.ndarray,
    assets: tuple[str, ...] | list[str],
    evidence_ids: tuple[str, ...],
    distance_method: str = "correlation_distance",
    test_id: str = "portfolio.hierarchical_risk_parity.tree_topology",
    output_dir: Path | str | None = None,
    allow_empty_evidence: bool = False,
) -> ArtifactRecord:
    """Render the Pairwise Distance Matrix."""
    _validate_provenance(evidence_ids, allow_empty=allow_empty_evidence)
    asset_list = list(assets)
    spec = ArtifactSpec(
        artifact_type="distance_matrix",
        title="Pairwise Distance Matrix",
        test_id=test_id,
        evidence_ids=evidence_ids,
        parameters={"distance_method": distance_method, "assets": asset_list},
    )
    semantic_payload = {
        "assets": asset_list,
        "distance_method": distance_method,
        "distance_matrix": [[round(float(x), 6) for x in row] for row in dist_matrix],
    }
    payload_hash = _hash_payload(semantic_payload)
    artifact_id = f"ART-DISTANCE-MATRIX-{payload_hash[:8]}"

    file_path = None
    if output_dir:
        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        file_path = str(out_p / f"{artifact_id}.md")
        df = pd.DataFrame(dist_matrix, index=asset_list, columns=asset_list)
        md = f"### Pairwise Distance Matrix ({distance_method})\n\n```\n{df.to_string()}\n```\n"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(md)
        with open(str(out_p / f"{artifact_id}.json"), "w", encoding="utf-8") as f:
            json.dump(semantic_payload, f, indent=2)

    return ArtifactRecord(
        artifact_id=artifact_id,
        spec=spec,
        data_fingerprint=hashlib.sha256(dist_matrix.tobytes()).hexdigest()[:32],
        semantic_payload=semantic_payload,
        semantic_payload_hash=payload_hash,
        file_path=file_path,
        rendering_format="markdown" if file_path else "json",
    )


# --------------------------------------------------------------------------- #
# 5. Cluster Tree / Hierarchy Artifact
# --------------------------------------------------------------------------- #
def render_cluster_tree_artifact(
    tree_result: HierarchicalTreeResult,
    evidence_ids: tuple[str, ...],
    test_id: str = "portfolio.hierarchical_risk_parity.tree_topology",
    output_dir: Path | str | None = None,
    allow_empty_evidence: bool = False,
) -> ArtifactRecord:
    """Render the Cluster Tree Hierarchy Table."""
    _validate_provenance(evidence_ids, allow_empty=allow_empty_evidence)
    spec = ArtifactSpec(
        artifact_type="cluster_tree_table",
        title="Hierarchical Cluster Bisection Tree",
        test_id=test_id,
        evidence_ids=evidence_ids,
        parameters={"linkage_method": tree_result.linkage_method},
    )
    semantic_payload = {
        "assets": list(tree_result.assets),
        "linkage_method": tree_result.linkage_method,
        "quasi_diagonal_order": list(tree_result.quasi_diagonal_order),
        "splits": tree_result.cluster_tree.get("splits", []),
    }
    payload_hash = _hash_payload(semantic_payload)
    artifact_id = f"ART-CLUSTER-TREE-{payload_hash[:8]}"

    file_path = None
    if output_dir:
        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        file_path = str(out_p / f"{artifact_id}.md")
        splits = tree_result.cluster_tree.get("splits", [])
        rows = [
            "| Step | Left Cluster | Right Cluster | V_Left | V_Right | Alpha (Left Weight) |",
            "|---|---|---|---|---|---|",
        ]
        for idx, s in enumerate(splits):
            left_str = ", ".join(s["left"])
            right_str = ", ".join(s["right"])
            rows.append(
                f"| {idx + 1} | {left_str} | {right_str} | {s['v_left']:.4e} | "
                f"{s['v_right']:.4e} | {s['alpha']:.4f} |"
            )
        md = "### Hierarchical Cluster Bisection Tree\n\n" + "\n".join(rows) + "\n"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(md)
        with open(str(out_p / f"{artifact_id}.json"), "w", encoding="utf-8") as f:
            json.dump(semantic_payload, f, indent=2)

    return ArtifactRecord(
        artifact_id=artifact_id,
        spec=spec,
        data_fingerprint=tree_result.covariance_fingerprint,
        semantic_payload=semantic_payload,
        semantic_payload_hash=payload_hash,
        file_path=file_path,
        rendering_format="markdown" if file_path else "json",
    )


# --------------------------------------------------------------------------- #
# 6. Cluster Allocation / Waterfall Artifact
# --------------------------------------------------------------------------- #
def render_cluster_allocation_artifact(
    cluster_weights: dict[str, float],
    cluster_risk: dict[str, float],
    evidence_ids: tuple[str, ...],
    test_id: str = "portfolio.hierarchical_risk_parity",
    output_dir: Path | str | None = None,
    allow_empty_evidence: bool = False,
) -> ArtifactRecord:
    """Render Cluster Allocation and Risk Waterfall."""
    _validate_provenance(evidence_ids, allow_empty=allow_empty_evidence)
    spec = ArtifactSpec(
        artifact_type="cluster_allocation",
        title="Cluster Allocation and Risk Budgeting",
        test_id=test_id,
        evidence_ids=evidence_ids,
    )
    semantic_payload = {
        "cluster_weights": {k: round(float(v), 8) for k, v in cluster_weights.items()},
        "cluster_risk_contributions": {k: round(float(v), 8) for k, v in cluster_risk.items()},
    }
    payload_hash = _hash_payload(semantic_payload)
    artifact_id = f"ART-CLUSTER-ALLOC-{payload_hash[:8]}"

    file_path = None
    if output_dir:
        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        file_path = str(out_p / f"{artifact_id}.md")
        rows = [
            "| Cluster | Allocated Weight | Risk Contribution (%CR) |",
            "|---|---|---|",
        ]
        for c in cluster_weights:
            w = cluster_weights[c]
            r = cluster_risk.get(c, 0.0)
            rows.append(f"| {c} | {w:.4%} | {r:.4%} |")
        md = "### Cluster Allocation and Risk Budgeting\n\n" + "\n".join(rows) + "\n"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(md)
        with open(str(out_p / f"{artifact_id}.json"), "w", encoding="utf-8") as f:
            json.dump(semantic_payload, f, indent=2)

    return ArtifactRecord(
        artifact_id=artifact_id,
        spec=spec,
        data_fingerprint=payload_hash,
        semantic_payload=semantic_payload,
        semantic_payload_hash=payload_hash,
        file_path=file_path,
        rendering_format="markdown" if file_path else "json",
    )


# --------------------------------------------------------------------------- #
# 7. Asset Weights Artifact
# --------------------------------------------------------------------------- #
def render_asset_weights_artifact(
    weights: dict[str, float] | pd.Series,
    evidence_ids: tuple[str, ...],
    method_name: str = "HRP",
    test_id: str = "portfolio.hierarchical_risk_parity",
    output_dir: Path | str | None = None,
    allow_empty_evidence: bool = False,
) -> ArtifactRecord:
    """Render Asset Weights Table and Concentration Summary."""
    _validate_provenance(evidence_ids, allow_empty=allow_empty_evidence)
    w_dict = {
        str(k): float(v) for k, v in (weights.items() if hasattr(weights, "items") else enumerate(weights))
    }
    values = np.array(list(w_dict.values()), dtype=float)
    h = float(np.sum(values**2))
    eff_n = float(1.0 / h) if h > 1e-12 else 0.0

    spec = ArtifactSpec(
        artifact_type="asset_weights_table",
        title=f"{method_name} Asset Allocation Weights",
        test_id=test_id,
        evidence_ids=evidence_ids,
        parameters={"method": method_name},
    )
    semantic_payload = {
        "method": method_name,
        "weights": {k: round(float(v), 8) for k, v in w_dict.items()},
        "herfindahl": round(h, 8),
        "effective_n_positions": round(eff_n, 4),
        "max_weight": round(float(np.max(values)), 8) if len(values) else 0.0,
        "min_weight": round(float(np.min(values)), 8) if len(values) else 0.0,
        "weights_sum": round(float(np.sum(values)), 8),
    }
    payload_hash = _hash_payload(semantic_payload)
    artifact_id = f"ART-ASSET-WEIGHTS-{payload_hash[:8]}"

    file_path = None
    if output_dir:
        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        file_path = str(out_p / f"{artifact_id}.md")
        rows = [
            "| Asset | Weight |",
            "|---|---|",
        ]
        for a, w in w_dict.items():
            rows.append(f"| {a} | {w:.4%} |")
        rows.append(f"| **Effective Positions (1/H)** | **{eff_n:.2f}** |")
        md = f"### {method_name} Asset Allocation Weights\n\n" + "\n".join(rows) + "\n"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(md)
        with open(str(out_p / f"{artifact_id}.json"), "w", encoding="utf-8") as f:
            json.dump(semantic_payload, f, indent=2)

    return ArtifactRecord(
        artifact_id=artifact_id,
        spec=spec,
        data_fingerprint=payload_hash,
        semantic_payload=semantic_payload,
        semantic_payload_hash=payload_hash,
        file_path=file_path,
        rendering_format="markdown" if file_path else "json",
    )


# --------------------------------------------------------------------------- #
# 8. Asset Risk Contribution Waterfall Artifact
# --------------------------------------------------------------------------- #
def render_risk_contribution_artifact(
    rc: RiskContributionResult,
    assets: tuple[str, ...] | list[str],
    evidence_ids: tuple[str, ...],
    test_id: str = "portfolio.risk_statistics.euler_decomposition",
    output_dir: Path | str | None = None,
    allow_empty_evidence: bool = False,
) -> ArtifactRecord:
    """Render Euler Risk Contribution Waterfall."""
    _validate_provenance(evidence_ids, allow_empty=allow_empty_evidence)
    spec = ArtifactSpec(
        artifact_type="risk_contribution_waterfall",
        title="Euler Risk Contribution Waterfall",
        test_id=test_id,
        evidence_ids=evidence_ids,
    )
    semantic_payload = {
        "portfolio_volatility": rc.portfolio_volatility,
        "portfolio_variance": rc.portfolio_variance,
        "marginal_contributions": rc.marginal_contributions,
        "component_contributions": rc.component_contributions,
        "percentage_contributions": rc.percentage_contributions,
        "euler_reconciliation_error": rc.euler_reconciliation_error,
        "cluster_percentage_contributions": rc.cluster_percentage_contributions,
    }
    payload_hash = _hash_payload(semantic_payload)
    artifact_id = f"ART-RISK-CONTRIB-{payload_hash[:8]}"

    file_path = None
    if output_dir:
        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        file_path = str(out_p / f"{artifact_id}.md")
        rows = [
            "| Asset | Marginal Risk (MCR) | Component Risk (CR) | % Risk Contribution (%CR) |",
            "|---|---|---|---|",
        ]
        for a in assets:
            mcr = rc.marginal_contributions.get(a, 0.0)
            cr = rc.component_contributions.get(a, 0.0)
            pcr = rc.percentage_contributions.get(a, 0.0)
            rows.append(f"| {a} | {mcr:.6f} | {cr:.6f} | {pcr:.4%} |")
        rows.append(f"| **Total** | — | **{rc.portfolio_volatility:.6f}** | **100.00%** |")
        md = "### Euler Risk Contribution Breakdown\n\n" + "\n".join(rows) + "\n"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(md)
        with open(str(out_p / f"{artifact_id}.json"), "w", encoding="utf-8") as f:
            json.dump(semantic_payload, f, indent=2)

    return ArtifactRecord(
        artifact_id=artifact_id,
        spec=spec,
        data_fingerprint=payload_hash,
        semantic_payload=semantic_payload,
        semantic_payload_hash=payload_hash,
        file_path=file_path,
        rendering_format="markdown" if file_path else "json",
    )


# --------------------------------------------------------------------------- #
# 9. Cluster Risk Contribution Artifact
# --------------------------------------------------------------------------- #
def render_cluster_risk_artifact(
    rc: RiskContributionResult,
    evidence_ids: tuple[str, ...],
    test_id: str = "portfolio.risk_statistics.cluster_euler_decomposition",
    output_dir: Path | str | None = None,
    allow_empty_evidence: bool = False,
) -> ArtifactRecord:
    """Render Cluster-Level Euler Risk Contribution Breakdown."""
    _validate_provenance(evidence_ids, allow_empty=allow_empty_evidence)
    spec = ArtifactSpec(
        artifact_type="cluster_risk_contribution",
        title="Cluster-Level Euler Risk Contribution",
        test_id=test_id,
        evidence_ids=evidence_ids,
    )
    semantic_payload = {
        "cluster_contributions": rc.cluster_contributions,
        "cluster_percentage_contributions": rc.cluster_percentage_contributions,
        "portfolio_volatility": rc.portfolio_volatility,
        "euler_reconciliation_error": rc.euler_reconciliation_error,
    }
    payload_hash = _hash_payload(semantic_payload)
    artifact_id = f"ART-CLUSTER-RISK-{payload_hash[:8]}"

    file_path = None
    if output_dir:
        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        file_path = str(out_p / f"{artifact_id}.md")
        rows = [
            "| Cluster | Component Risk (CR) | % Risk Contribution (%CR) |",
            "|---|---|---|",
        ]
        for c, cr in rc.cluster_contributions.items():
            pcr = rc.cluster_percentage_contributions.get(c, 0.0)
            rows.append(f"| {c} | {cr:.6f} | {pcr:.4%} |")
        md = "### Cluster-Level Risk Contributions\n\n" + "\n".join(rows) + "\n"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(md)
        with open(str(out_p / f"{artifact_id}.json"), "w", encoding="utf-8") as f:
            json.dump(semantic_payload, f, indent=2)

    return ArtifactRecord(
        artifact_id=artifact_id,
        spec=spec,
        data_fingerprint=payload_hash,
        semantic_payload=semantic_payload,
        semantic_payload_hash=payload_hash,
        file_path=file_path,
        rendering_format="markdown" if file_path else "json",
    )


# --------------------------------------------------------------------------- #
# 10. Efficient Frontier Artifact
# --------------------------------------------------------------------------- #
def render_efficient_frontier_artifact(
    frontier_res: EfficientFrontierResult,
    evidence_ids: tuple[str, ...],
    test_id: str = "portfolio.mean_variance.efficient_frontier",
    output_dir: Path | str | None = None,
    allow_empty_evidence: bool = False,
) -> ArtifactRecord:
    """Render Parametric Efficient Frontier Curve and Reference Portfolio Overlays."""
    _validate_provenance(evidence_ids, allow_empty=allow_empty_evidence)
    spec = ArtifactSpec(
        artifact_type="efficient_frontier_plot",
        title="Parametric Efficient Frontier and Reference Overlays",
        test_id=test_id,
        evidence_ids=evidence_ids,
    )
    points_data = [
        {
            "label": p.label,
            "target_return": p.target_return,
            "expected_return_annualised": p.expected_return_annualised,
            "volatility_annualised": p.volatility_annualised,
            "sharpe_annualised": p.sharpe_annualised,
        }
        for p in frontier_res.frontier_points
    ]
    overlays = {
        "min_variance": {
            "expected_return": frontier_res.min_variance_point.expected_return_annualised,
            "volatility": frontier_res.min_variance_point.volatility_annualised,
            "sharpe": frontier_res.min_variance_point.sharpe_annualised,
        },
        "max_sharpe": {
            "expected_return": frontier_res.max_sharpe_point.expected_return_annualised,
            "volatility": frontier_res.max_sharpe_point.volatility_annualised,
            "sharpe": frontier_res.max_sharpe_point.sharpe_annualised,
        },
    }
    if frontier_res.equal_weight_point:
        overlays["equal_weight"] = {
            "expected_return": frontier_res.equal_weight_point.expected_return_annualised,
            "volatility": frontier_res.equal_weight_point.volatility_annualised,
            "sharpe": frontier_res.equal_weight_point.sharpe_annualised,
        }
    if frontier_res.erc_point:
        overlays["erc"] = {
            "expected_return": frontier_res.erc_point.expected_return_annualised,
            "volatility": frontier_res.erc_point.volatility_annualised,
            "sharpe": frontier_res.erc_point.sharpe_annualised,
        }
    if frontier_res.hrp_point:
        overlays["hrp"] = {
            "expected_return": frontier_res.hrp_point.expected_return_annualised,
            "volatility": frontier_res.hrp_point.volatility_annualised,
            "sharpe": frontier_res.hrp_point.sharpe_annualised,
        }

    semantic_payload = {
        "frontier_points": points_data,
        "overlays": overlays,
    }
    payload_hash = _hash_payload(semantic_payload)
    artifact_id = f"ART-FRONTIER-{payload_hash[:8]}"

    file_path = None
    if output_dir:
        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        file_path = str(out_p / f"{artifact_id}.svg")
        svg_content = _generate_frontier_svg(frontier_res)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(svg_content)
        with open(str(out_p / f"{artifact_id}.json"), "w", encoding="utf-8") as f:
            json.dump(semantic_payload, f, indent=2)

    return ArtifactRecord(
        artifact_id=artifact_id,
        spec=spec,
        data_fingerprint=payload_hash,
        semantic_payload=semantic_payload,
        semantic_payload_hash=payload_hash,
        file_path=file_path,
        rendering_format="svg" if file_path else "json",
    )


def _generate_frontier_svg(frontier: EfficientFrontierResult) -> str:
    width = 700
    height = 450
    lines = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
            f'width="{width}" height="{height}">'
        ),
        "  <style>",
        "    .title { font: bold 14px sans-serif; fill: #1e293b; }",
        "    .axis { stroke: #94a3b8; stroke-width: 1px; }",
        "    .curve { fill: none; stroke: #2563eb; stroke-width: 3px; }",
        "    .point { fill: #dc2626; stroke: #fff; stroke-width: 2px; }",
        "    .label { font: 11px sans-serif; fill: #334155; }",
        "  </style>",
        '  <text x="20" y="30" class="title">Parametric Efficient Frontier and Reference Overlays</text>',
        f'  <line x1="60" y1="{height - 50}" x2="{width - 30}" y2="{height - 50}" class="axis" />',
        f'  <line x1="60" y1="50" x2="60" y2="{height - 50}" class="axis" />',
        (
            f'  <text x="{width / 2}" y="{height - 15}" class="label" '
            f'text-anchor="middle">Annualized Volatility</text>'
        ),
        "</svg>",
    ]
    return "\n".join(lines)


# =========================================================================== #
# GATE-3 INSTITUTIONAL ARTIFACT RENDERERS
# =========================================================================== #
def render_bl_returns_artifact(
    bl: BlackLittermanResult,
    evidence_ids: tuple[str, ...],
    test_id: str = "portfolio.black_litterman",
    output_dir: Path | str | None = None,
    allow_empty_evidence: bool = False,
) -> ArtifactRecord:
    """Render Black-Litterman implied prior vs posterior expected returns comparison."""
    _validate_provenance(evidence_ids, allow_empty=allow_empty_evidence)
    spec = ArtifactSpec(
        artifact_type="bl_returns_comparison",
        title="Black-Litterman Prior vs Posterior Returns",
        test_id=test_id,
        evidence_ids=evidence_ids,
        parameters={"tau": bl.tau, "risk_aversion": bl.risk_aversion},
    )
    semantic_payload: dict[str, Any] = {
        "assets": list(bl.implied_returns.keys()),
        "implied_returns": bl.implied_returns,
        "posterior_returns": bl.posterior_returns,
        "view_residuals": bl.view_residuals,
        "view_uncertainties": bl.view_uncertainties,
        "tau": bl.tau,
        "risk_aversion": bl.risk_aversion,
    }
    payload_hash = _hash_payload(semantic_payload)
    artifact_id = f"ART-BL-RET-{payload_hash[:8]}"

    file_path = None
    if output_dir:
        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        file_path = str(out_p / f"{artifact_id}.json")
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(semantic_payload, f, indent=2)

    return ArtifactRecord(
        artifact_id=artifact_id,
        spec=spec,
        data_fingerprint=payload_hash,
        semantic_payload=semantic_payload,
        semantic_payload_hash=payload_hash,
        file_path=file_path,
        rendering_format="json",
    )


def render_bl_allocation_artifact(
    bl: BlackLittermanResult,
    evidence_ids: tuple[str, ...],
    test_id: str = "portfolio.black_litterman",
    output_dir: Path | str | None = None,
    allow_empty_evidence: bool = False,
) -> ArtifactRecord:
    """Render Black-Litterman prior market weights vs posterior allocation weights."""
    _validate_provenance(evidence_ids, allow_empty=allow_empty_evidence)
    spec = ArtifactSpec(
        artifact_type="bl_allocation_comparison",
        title="Black-Litterman Prior vs Posterior Weights",
        test_id=test_id,
        evidence_ids=evidence_ids,
        parameters={"tau": bl.tau, "turnover": bl.turnover_vs_prior},
    )
    semantic_payload: dict[str, Any] = {
        "assets": list(bl.posterior_weights.keys()),
        "prior_weights": bl.prior_weights,
        "posterior_weights": bl.posterior_weights,
        "turnover_vs_prior": bl.turnover_vs_prior,
        "posterior_volatility_annualised": bl.posterior_volatility_annualised,
        "posterior_sharpe_annualised": bl.posterior_sharpe_annualised,
    }
    payload_hash = _hash_payload(semantic_payload)
    artifact_id = f"ART-BL-ALLOC-{payload_hash[:8]}"

    file_path = None
    if output_dir:
        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        file_path = str(out_p / f"{artifact_id}.json")
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(semantic_payload, f, indent=2)

    return ArtifactRecord(
        artifact_id=artifact_id,
        spec=spec,
        data_fingerprint=payload_hash,
        semantic_payload=semantic_payload,
        semantic_payload_hash=payload_hash,
        file_path=file_path,
        rendering_format="json",
    )


def render_robust_mvo_sensitivity_artifact(
    sens: RobustSensitivityResult,
    evidence_ids: tuple[str, ...],
    test_id: str = "portfolio.mean_variance.robust_sensitivity",
    output_dir: Path | str | None = None,
    allow_empty_evidence: bool = False,
) -> ArtifactRecord:
    """Render Robust MVO sensitivity path across uncertainty radii."""
    _validate_provenance(evidence_ids, allow_empty=allow_empty_evidence)
    spec = ArtifactSpec(
        artifact_type="robust_mvo_sensitivity",
        title="Robust MVO Uncertainty Sensitivity Path",
        test_id=test_id,
        evidence_ids=evidence_ids,
        parameters={"radii": list(sens.radii_evaluated)},
    )
    points_data = [
        {
            "uncertainty_radius": pt.uncertainty_radius,
            "nominal_expected_return_annualised": pt.nominal_expected_return_annualised,
            "worst_case_expected_return_annualised": pt.worst_case_expected_return_annualised,
            "portfolio_volatility_annualised": pt.portfolio_volatility_annualised,
            "worst_case_sharpe_annualised": pt.worst_case_sharpe_annualised,
            "effective_n_positions": pt.effective_n_positions,
            "turnover_vs_prior": pt.turnover_vs_prior,
            "weights": pt.weights,
        }
        for pt in sens.points
    ]
    semantic_payload: dict[str, Any] = {
        "baseline_radius": sens.baseline_radius,
        "radii_evaluated": list(sens.radii_evaluated),
        "sensitivity_points": points_data,
    }
    payload_hash = _hash_payload(semantic_payload)
    artifact_id = f"ART-ROB-SENS-{payload_hash[:8]}"

    file_path = None
    if output_dir:
        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        file_path = str(out_p / f"{artifact_id}.json")
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(semantic_payload, f, indent=2)

    return ArtifactRecord(
        artifact_id=artifact_id,
        spec=spec,
        data_fingerprint=payload_hash,
        semantic_payload=semantic_payload,
        semantic_payload_hash=payload_hash,
        file_path=file_path,
        rendering_format="json",
    )


def render_cvar_tail_risk_artifact(
    cvar_res: CVaROptimizationResult,
    evidence_ids: tuple[str, ...],
    test_id: str = "portfolio.cvar_optimization",
    output_dir: Path | str | None = None,
    allow_empty_evidence: bool = False,
) -> ArtifactRecord:
    """Render CVaR optimization risk profile and tail diagnostics."""
    _validate_provenance(evidence_ids, allow_empty=allow_empty_evidence)
    spec = ArtifactSpec(
        artifact_type="cvar_tail_risk_profile",
        title="Rockafellar-Uryasev CVaR Tail Risk Profile",
        test_id=test_id,
        evidence_ids=evidence_ids,
        parameters={"confidence_level": cvar_res.confidence_level},
    )
    semantic_payload: dict[str, Any] = {
        "confidence_level": cvar_res.confidence_level,
        "cvar_at_scenario_horizon": cvar_res.cvar_at_scenario_horizon,
        "var_at_scenario_horizon": cvar_res.var_at_scenario_horizon,
        "cvar_annualised": cvar_res.cvar_annualised,
        "var_auxiliary_annualised": cvar_res.var_auxiliary_annualised,
        "cvar_periodic": cvar_res.cvar_at_scenario_horizon,
        "var_auxiliary_periodic": cvar_res.var_at_scenario_horizon,
        "scenario_horizon": cvar_res.scenario_horizon,
        "tail_scenario_count": cvar_res.tail_scenario_count,
        "n_scenarios": cvar_res.n_scenarios,
        "expected_return_periodic": cvar_res.expected_return_periodic,
        "expected_return_annualised": cvar_res.expected_return_annualised,
        "effective_n_positions": cvar_res.effective_n_positions,
        "weights": cvar_res.weights,
    }
    payload_hash = _hash_payload(semantic_payload)
    artifact_id = f"ART-CVAR-TAIL-{payload_hash[:8]}"

    file_path = None
    if output_dir:
        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        file_path = str(out_p / f"{artifact_id}.json")
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(semantic_payload, f, indent=2)

    return ArtifactRecord(
        artifact_id=artifact_id,
        spec=spec,
        data_fingerprint=payload_hash,
        semantic_payload=semantic_payload,
        semantic_payload_hash=payload_hash,
        file_path=file_path,
        rendering_format="json",
    )


def render_herc_hierarchy_artifact(
    herc_res: HERCResult,
    evidence_ids: tuple[str, ...],
    test_id: str = "portfolio.hierarchical_equal_risk_contribution",
    output_dir: Path | str | None = None,
    allow_empty_evidence: bool = False,
) -> ArtifactRecord:
    """Render HERC hierarchy and cluster risk parity allocation."""
    _validate_provenance(evidence_ids, allow_empty=allow_empty_evidence)
    spec = ArtifactSpec(
        artifact_type="herc_hierarchy_allocation",
        title="HERC Hierarchy & Cluster Risk Parity",
        test_id=test_id,
        evidence_ids=evidence_ids,
        parameters={
            "linkage_method": herc_res.tree_result.linkage_method,
            "risk_measure": herc_res.risk_measure,
        },
    )
    semantic_payload: dict[str, Any] = {
        "linkage_method": herc_res.tree_result.linkage_method,
        "quasi_diagonal_order": list(herc_res.tree_result.quasi_diagonal_order),
        "cluster_tree": herc_res.tree_result.cluster_tree,
        "weights": herc_res.weights,
        "percentage_risk_contributions": herc_res.percentage_risk_contributions,
        "effective_n_positions": herc_res.effective_n_positions,
        "portfolio_volatility_annualised": herc_res.portfolio_volatility_annualised,
    }
    payload_hash = _hash_payload(semantic_payload)
    artifact_id = f"ART-HERC-TREE-{payload_hash[:8]}"

    file_path = None
    if output_dir:
        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        file_path = str(out_p / f"{artifact_id}.json")
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(semantic_payload, f, indent=2)

    return ArtifactRecord(
        artifact_id=artifact_id,
        spec=spec,
        data_fingerprint=payload_hash,
        semantic_payload=semantic_payload,
        semantic_payload_hash=payload_hash,
        file_path=file_path,
        rendering_format="json",
    )


def render_tracking_error_artifact(
    te_res: TrackingErrorResult,
    evidence_ids: tuple[str, ...],
    test_id: str = "portfolio.tracking_error_constrained",
    output_dir: Path | str | None = None,
    allow_empty_evidence: bool = False,
) -> ArtifactRecord:
    """Render Tracking Error active weight positions and risk metrics."""
    _validate_provenance(evidence_ids, allow_empty=allow_empty_evidence)
    spec = ArtifactSpec(
        artifact_type="tracking_error_active_positions",
        title="Tracking Error Constrained Active Positioning",
        test_id=test_id,
        evidence_ids=evidence_ids,
        parameters={"tracking_error_annualised": te_res.tracking_error_annualised},
    )
    semantic_payload: dict[str, Any] = {
        "weights": te_res.weights,
        "benchmark_weights": te_res.benchmark_weights,
        "active_weights": te_res.active_weights,
        "tracking_error_annualised": te_res.tracking_error_annualised,
        "active_return_annualised": te_res.active_return_annualised,
        "information_ratio": te_res.information_ratio,
        "portfolio_volatility_annualised": te_res.portfolio_volatility_annualised,
    }
    payload_hash = _hash_payload(semantic_payload)
    artifact_id = f"ART-TE-POS-{payload_hash[:8]}"

    file_path = None
    if output_dir:
        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        file_path = str(out_p / f"{artifact_id}.json")
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(semantic_payload, f, indent=2)

    return ArtifactRecord(
        artifact_id=artifact_id,
        spec=spec,
        data_fingerprint=payload_hash,
        semantic_payload=semantic_payload,
        semantic_payload_hash=payload_hash,
        file_path=file_path,
        rendering_format="json",
    )


def render_rebalance_waterfall_artifact(
    reb: RebalanceDecision,
    evidence_ids: tuple[str, ...],
    test_id: str = "portfolio.rebalance.decision",
    output_dir: Path | str | None = None,
    allow_empty_evidence: bool = False,
) -> ArtifactRecord:
    """Render Rebalance trade weights, turnover, and cost waterfall."""
    _validate_provenance(evidence_ids, allow_empty=allow_empty_evidence)
    spec = ArtifactSpec(
        artifact_type="rebalance_cost_waterfall",
        title="Rebalance Trades & Transaction Cost Waterfall",
        test_id=test_id,
        evidence_ids=evidence_ids,
        parameters={"turnover": reb.turnover, "cost": reb.estimated_transaction_cost},
    )
    semantic_payload: dict[str, Any] = {
        "current_weights": reb.current_weights,
        "proposed_weights": reb.proposed_weights,
        "trade_weights": reb.trade_weights,
        "turnover": reb.turnover,
        "estimated_transaction_cost": reb.estimated_transaction_cost,
        "cost_provenance": reb.cost_provenance,
        "pre_trade_risk": reb.pre_trade_risk,
        "post_trade_risk": reb.post_trade_risk,
        "expected_return_gross_periodic": reb.expected_return_gross_periodic,
        "expected_return_gross_annualised": reb.expected_return_gross_annualised,
        "expected_return_net_periodic": reb.expected_return_net_periodic,
        "expected_return_net_annualised": reb.expected_return_net_annualised,
        "expected_return_gross": reb.expected_return_gross,
        "expected_return_net": reb.expected_return_net,
    }
    payload_hash = _hash_payload(semantic_payload)
    artifact_id = f"ART-REB-WF-{payload_hash[:8]}"

    file_path = None
    if output_dir:
        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        file_path = str(out_p / f"{artifact_id}.json")
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(semantic_payload, f, indent=2)

    return ArtifactRecord(
        artifact_id=artifact_id,
        spec=spec,
        data_fingerprint=payload_hash,
        semantic_payload=semantic_payload,
        semantic_payload_hash=payload_hash,
        file_path=file_path,
        rendering_format="json",
    )


def render_multi_method_comparison_artifact(
    comp: MethodComparisonResult,
    evidence_ids: tuple[str, ...],
    test_id: str = "portfolio.method_comparison",
    output_dir: Path | str | None = None,
    allow_empty_evidence: bool = False,
) -> ArtifactRecord:
    """Render comprehensive multi-method portfolio comparison matrix."""
    _validate_provenance(evidence_ids, allow_empty=allow_empty_evidence)
    spec = ArtifactSpec(
        artifact_type="multi_method_comparison_matrix",
        title="Multi-Method Institutional Portfolio Comparison Matrix",
        test_id=test_id,
        evidence_ids=evidence_ids,
        parameters={"methods": list(comp.methods)},
    )
    semantic_payload: dict[str, Any] = {
        "methods": list(comp.methods),
        "summary_table": comp.summary_table,
        "weights_matrix": comp.weights_matrix,
        "risk_contributions_matrix": comp.risk_contributions_matrix,
    }
    payload_hash = _hash_payload(semantic_payload)
    artifact_id = f"ART-COMP-MAT-{payload_hash[:8]}"

    file_path = None
    if output_dir:
        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        file_path = str(out_p / f"{artifact_id}.json")
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(semantic_payload, f, indent=2)

    return ArtifactRecord(
        artifact_id=artifact_id,
        spec=spec,
        data_fingerprint=payload_hash,
        semantic_payload=semantic_payload,
        semantic_payload_hash=payload_hash,
        file_path=file_path,
        rendering_format="json",
    )


# =========================================================================== #
# GATE 4: SVG VISUAL GENERATOR HELPERS
# =========================================================================== #
def _generate_matrix_heatmap_svg(
    matrix: list[list[float]],
    row_labels: list[str],
    col_labels: list[str],
    title: str,
    subtitle: str = "",
    fmt: str = ".4f",
    is_diverging: bool = False,
) -> str:
    """Generate institutional SVG heatmap visualization for 2D numerical matrices."""
    n_rows = len(row_labels)
    n_cols = len(col_labels)
    cell_w = 75
    cell_h = 42
    left_m = 110
    top_m = 90
    right_m = 50
    bottom_m = 50

    width = left_m + n_cols * cell_w + right_m
    height = top_m + n_rows * cell_h + bottom_m

    all_vals = [v for row in matrix for v in row if np.isfinite(v)]
    min_v = min(all_vals) if all_vals else 0.0
    max_v = max(all_vals) if all_vals else 1.0

    if is_diverging:
        abs_max = max(abs(min_v), abs(max_v), 1e-12)
    else:
        val_range = max(max_v - min_v, 1e-12)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">',
        "  <style>",
        '    .title { font: bold 15px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; fill: #0f172a; }',
        '    .subtitle { font: 12px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; fill: #64748b; }',
        '    .header { font: bold 11px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; fill: #334155; }',
        '    .cell-txt { font: 11px "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace; }',
        "  </style>",
        '  <rect width="100%" height="100%" fill="#ffffff" />',
        f'  <text x="24" y="32" class="title">{title}</text>',
    ]
    if subtitle:
        lines.append(f'  <text x="24" y="52" class="subtitle">{subtitle}</text>')

    # Column headers
    for j, c_lbl in enumerate(col_labels):
        x = left_m + j * cell_w + cell_w / 2.0
        y = top_m - 12
        lines.append(f'  <text x="{x}" y="{y}" class="header" text-anchor="middle">{c_lbl}</text>')

    # Rows and cells
    for i, r_lbl in enumerate(row_labels):
        y_row = top_m + i * cell_h + cell_h / 2.0 + 4
        lines.append(f'  <text x="{left_m - 12}" y="{y_row}" class="header" text-anchor="end">{r_lbl}</text>')

        for j, _c_lbl in enumerate(col_labels):
            val = matrix[i][j]
            cx = left_m + j * cell_w
            cy = top_m + i * cell_h

            if not np.isfinite(val):
                fill_color = "#f1f5f9"
                txt_color = "#94a3b8"
                txt_val = "NaN"
            else:
                if is_diverging:
                    norm = val / abs_max  # [-1, 1]
                    if norm >= 0:
                        t = min(1.0, norm)
                        r = int(248 + (185 - 248) * t)
                        g = int(250 + (28 - 250) * t)
                        b = int(252 + (28 - 252) * t)
                    else:
                        t = min(1.0, -norm)
                        r = int(248 + (29 - 248) * t)
                        g = int(250 + (78 - 250) * t)
                        b = int(252 + (216 - 252) * t)
                    txt_color = "#ffffff" if abs(norm) > 0.55 else "#0f172a"
                else:
                    norm = (val - min_v) / val_range  # [0, 1]
                    t = min(1.0, max(0.0, norm))
                    r = int(241 + (30 - 241) * t)
                    g = int(245 + (58 - 245) * t)
                    b = int(249 + (138 - 249) * t)
                    txt_color = "#ffffff" if norm > 0.55 else "#0f172a"

                fill_color = f"rgb({r},{g},{b})"
                txt_val = f"{val:{fmt}}"

            lines.append(
                f'  <rect x="{cx}" y="{cy}" width="{cell_w - 2}" height="{cell_h - 2}" rx="4" fill="{fill_color}" stroke="#cbd5e1" stroke-width="0.5" />'
            )
            lines.append(
                f'  <text x="{cx + cell_w / 2.0}" y="{cy + cell_h / 2.0 + 4}" class="cell-txt" fill="{txt_color}" text-anchor="middle">{txt_val}</text>'
            )

    lines.append("</svg>")
    return "\n".join(lines)


def _generate_eigenspectrum_svg(
    eigenvalues: list[float],
    title: str = "Covariance Matrix Eigenvalue Spectrum",
    subtitle: str = "",
) -> str:
    """Generate SVG eigenspectrum chart."""
    n = len(eigenvalues)
    width = max(550, n * 65 + 140)
    height = 360
    left_m = 60
    bottom_m = 50
    top_m = 80
    chart_w = width - left_m - 40
    chart_h = height - top_m - bottom_m

    max_eig = max(max(eigenvalues, default=1.0), 1e-6)
    min_eig = min(min(eigenvalues, default=0.0), 0.0)
    total_span = max(max_eig - min_eig, 1e-6)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">',
        "  <style>",
        '    .title { font: bold 15px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; fill: #0f172a; }',
        '    .subtitle { font: 12px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; fill: #64748b; }',
        '    .axis-lbl { font: 11px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; fill: #475569; }',
        '    .bar-val { font: 10px "SFMono-Regular", Consolas, monospace; fill: #0f172a; font-weight: bold; }',
        "  </style>",
        '  <rect width="100%" height="100%" fill="#ffffff" />',
        f'  <text x="24" y="32" class="title">{title}</text>',
    ]
    if subtitle:
        lines.append(f'  <text x="24" y="52" class="subtitle">{subtitle}</text>')

    # Zero line
    zero_y = top_m + chart_h * (max_eig / total_span)
    lines.append(
        f'  <line x1="{left_m}" y1="{zero_y}" x2="{left_m + chart_w}" y2="{zero_y}" stroke="#94a3b8" stroke-width="1" />'
    )

    bar_step = chart_w / max(n, 1)
    bar_w = min(45, bar_step * 0.7)

    for i, eig in enumerate(eigenvalues):
        bx = left_m + i * bar_step + (bar_step - bar_w) / 2.0
        if eig >= 0:
            bh = (eig / total_span) * chart_h
            by = zero_y - bh
            fill = "#3b82f6"
            txt_y = by - 6
        else:
            bh = (-eig / total_span) * chart_h
            by = zero_y
            fill = "#ef4444"
            txt_y = by + bh + 14

        lines.append(
            f'  <rect x="{bx}" y="{by}" width="{bar_w}" height="{max(1.0, bh)}" rx="3" fill="{fill}" />'
        )
        lines.append(
            f'  <text x="{bx + bar_w / 2.0}" y="{txt_y}" class="bar-val" text-anchor="middle">{eig:.4f}</text>'
        )
        lines.append(
            f'  <text x="{bx + bar_w / 2.0}" y="{height - 20}" class="axis-lbl" text-anchor="middle">&lambda;{i + 1}</text>'
        )

    lines.append("</svg>")
    return "\n".join(lines)


def _generate_waterfall_svg(
    items: list[tuple[str, float]],
    total_val: float,
    title: str,
    subtitle: str = "",
    unit: str = "",
) -> str:
    """Generate institutional financial waterfall chart SVG."""
    n = len(items) + 1
    width = max(600, n * 80 + 120)
    height = 380
    left_m = 70
    top_m = 85
    bottom_m = 65
    chart_w = width - left_m - 40
    chart_h = height - top_m - bottom_m

    # Compute running steps
    steps: list[tuple[str, float, float, float]] = []  # name, start, end, delta
    curr = 0.0
    for name, delta in items:
        nxt = curr + delta
        steps.append((name, curr, nxt, delta))
        curr = nxt

    all_y = [0.0, total_val] + [s[1] for s in steps] + [s[2] for s in steps]
    max_y = max(all_y) * 1.15 if max(all_y) > 0 else 0.01
    min_y = min(all_y) * 1.15 if min(all_y) < 0 else 0.0
    span_y = max(max_y - min_y, 1e-9)

    def to_svg_y(v: float) -> float:
        return top_m + chart_h * (1.0 - (v - min_y) / span_y)

    zero_y = to_svg_y(0.0)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">',
        "  <style>",
        '    .title { font: bold 15px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; fill: #0f172a; }',
        '    .subtitle { font: 12px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; fill: #64748b; }',
        '    .axis-lbl { font: 10px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; fill: #334155; }',
        '    .val-lbl { font: 10px "SFMono-Regular", Consolas, monospace; font-weight: bold; }',
        "  </style>",
        '  <rect width="100%" height="100%" fill="#ffffff" />',
        f'  <text x="24" y="32" class="title">{html.escape(str(title))}</text>',
    ]
    if subtitle:
        lines.append(f'  <text x="24" y="52" class="subtitle">{html.escape(str(subtitle))}</text>')

    # Zero axis line
    lines.append(
        f'  <line x1="{left_m}" y1="{zero_y}" x2="{left_m + chart_w}" y2="{zero_y}" stroke="#cbd5e1" stroke-dasharray="3,3" />'
    )

    step_w = chart_w / max(n, 1)
    bar_w = min(50, step_w * 0.7)

    for i, (name, y_start, y_end, delta) in enumerate(steps):
        bx = left_m + i * step_w + (step_w - bar_w) / 2.0
        sy = to_svg_y(max(y_start, y_end))
        ey = to_svg_y(min(y_start, y_end))
        bh = max(2.0, ey - sy)
        fill = "#16a34a" if delta >= 0 else "#dc2626"

        lines.append(f'  <rect x="{bx}" y="{sy}" width="{bar_w}" height="{bh}" rx="3" fill="{fill}" />')
        lbl_y = sy - 6 if delta >= 0 else ey + 14
        txt_fill = "#16a34a" if delta >= 0 else "#dc2626"
        lines.append(
            f'  <text x="{bx + bar_w / 2.0}" y="{lbl_y}" class="val-lbl" fill="{txt_fill}" text-anchor="middle">{delta:+.4f}{unit}</text>'
        )
        lines.append(
            f'  <text x="{bx + bar_w / 2.0}" y="{height - 25}" class="axis-lbl" text-anchor="middle">{html.escape(str(name))}</text>'
        )

    # Total bar
    tot_bx = left_m + len(steps) * step_w + (step_w - bar_w) / 2.0
    tot_sy = to_svg_y(max(0.0, total_val))
    tot_ey = to_svg_y(min(0.0, total_val))
    tot_bh = max(2.0, tot_ey - tot_sy)

    lines.append(
        f'  <rect x="{tot_bx}" y="{tot_sy}" width="{bar_w}" height="{tot_bh}" rx="3" fill="#2563eb" />'
    )
    tot_lbl_y = tot_sy - 6 if total_val >= 0 else tot_ey + 14
    lines.append(
        f'  <text x="{tot_bx + bar_w / 2.0}" y="{tot_lbl_y}" class="val-lbl" fill="#2563eb" text-anchor="middle">{total_val:.4f}{unit}</text>'
    )
    lines.append(
        f'  <text x="{tot_bx + bar_w / 2.0}" y="{height - 25}" class="axis-lbl" font-weight="bold" text-anchor="middle">TOTAL</text>'
    )

    lines.append("</svg>")
    return "\n".join(lines)


def _generate_bars_svg(
    categories: list[str],
    values: list[float],
    title: str,
    subtitle: str = "",
    fmt: str = ".4f",
) -> str:
    """Generate horizontal bar chart SVG."""
    n = len(categories)
    width = 580
    height = max(240, n * 45 + 110)
    left_m = 120
    top_m = 80
    right_m = 90
    chart_w = width - left_m - right_m

    max_v = max(max(values, default=1.0), 0.0)
    min_v = min(min(values, default=0.0), 0.0)
    span = max(max_v - min_v, 1e-9)

    zero_x = left_m + chart_w * (-min_v / span)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">',
        "  <style>",
        '    .title { font: bold 15px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; fill: #0f172a; }',
        '    .subtitle { font: 12px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; fill: #64748b; }',
        '    .cat-lbl { font: 11px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; fill: #334155; }',
        '    .bar-val { font: 10px "SFMono-Regular", Consolas, monospace; font-weight: bold; }',
        "  </style>",
        '  <rect width="100%" height="100%" fill="#ffffff" />',
        f'  <text x="24" y="32" class="title">{html.escape(str(title))}</text>',
    ]
    if subtitle:
        lines.append(f'  <text x="24" y="52" class="subtitle">{html.escape(str(subtitle))}</text>')

    lines.append(f'  <line x1="{zero_x}" y1="{top_m}" x2="{zero_x}" y2="{height - 30}" stroke="#cbd5e1" />')

    bar_h = 24
    row_step = (height - top_m - 30) / max(n, 1)

    for i, (cat, val) in enumerate(zip(categories, values, strict=False)):
        cy = top_m + i * row_step + (row_step - bar_h) / 2.0
        lines.append(
            f'  <text x="{left_m - 10}" y="{cy + bar_h / 2.0 + 4}" class="cat-lbl" text-anchor="end">{html.escape(str(cat))}</text>'
        )

        if val >= 0:
            bw = (val / span) * chart_w
            bx = zero_x
            fill = "#3b82f6"
            txt_x = bx + bw + 6
            txt_anchor = "start"
            txt_fill = "#1e40af"
        else:
            bw = (-val / span) * chart_w
            bx = zero_x - bw
            fill = "#ef4444"
            txt_x = bx - 6
            txt_anchor = "end"
            txt_fill = "#b91c1c"

        lines.append(
            f'  <rect x="{bx}" y="{cy}" width="{max(1.0, bw)}" height="{bar_h}" rx="3" fill="{fill}" />'
        )
        lines.append(
            f'  <text x="{txt_x}" y="{cy + bar_h / 2.0 + 4}" class="bar-val" fill="{txt_fill}" text-anchor="{txt_anchor}">{val:{fmt}}</text>'
        )

    lines.append("</svg>")
    return "\n".join(lines)


def _generate_dashboard_svg(
    metrics: list[tuple[str, str, str]],  # title, value, status/note
    title: str,
    subtitle: str = "",
) -> str:
    """Generate institutional metric card dashboard SVG."""
    n = len(metrics)
    cols = min(3, n)
    rows = math.ceil(n / cols)
    card_w = 175
    card_h = 95
    left_m = 24
    top_m = 80
    gap = 16

    width = left_m * 2 + cols * card_w + (cols - 1) * gap
    height = top_m + rows * card_h + (rows - 1) * gap + 24

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">',
        "  <style>",
        '    .title { font: bold 15px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; fill: #0f172a; }',
        '    .subtitle { font: 12px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; fill: #64748b; }',
        '    .card-title { font: 11px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; fill: #64748b; font-weight: 600; text-transform: uppercase; }',
        '    .card-val { font: bold 18px "SFMono-Regular", Consolas, monospace; fill: #0f172a; }',
        '    .card-note { font: 10px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; fill: #475569; }',
        "  </style>",
        '  <rect width="100%" height="100%" fill="#ffffff" />',
        f'  <text x="24" y="32" class="title">{html.escape(str(title))}</text>',
    ]
    if subtitle:
        lines.append(f'  <text x="24" y="52" class="subtitle">{html.escape(str(subtitle))}</text>')

    for idx, (m_title, m_val, m_note) in enumerate(metrics):
        r = idx // cols
        c = idx % cols
        x = left_m + c * (card_w + gap)
        y = top_m + r * (card_h + gap)

        lines.append(
            f'  <rect x="{x}" y="{y}" width="{card_w}" height="{card_h}" rx="6" fill="#f8fafc" stroke="#e2e8f0" stroke-width="1" />'
        )
        lines.append(
            f'  <text x="{x + 14}" y="{y + 24}" class="card-title">{html.escape(str(m_title))}</text>'
        )
        lines.append(f'  <text x="{x + 14}" y="{y + 54}" class="card-val">{html.escape(str(m_val))}</text>')
        lines.append(f'  <text x="{x + 14}" y="{y + 78}" class="card-note">{html.escape(str(m_note))}</text>')

    lines.append("</svg>")
    return "\n".join(lines)


def _generate_table_svg(
    headers: list[str],
    rows: list[list[str]],
    title: str,
    subtitle: str = "",
) -> str:
    """Generate institutional SVG table visualization."""
    col_w = 110
    row_h = 32
    top_m = 85
    left_m = 24
    width = left_m * 2 + len(headers) * col_w
    height = top_m + (len(rows) + 1) * row_h + 30

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">',
        "  <style>",
        '    .title { font: bold 15px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; fill: #0f172a; }',
        '    .subtitle { font: 12px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; fill: #64748b; }',
        '    .th { font: bold 11px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; fill: #ffffff; }',
        '    .td { font: 11px "SFMono-Regular", Consolas, monospace; fill: #1e293b; }',
        "  </style>",
        '  <rect width="100%" height="100%" fill="#ffffff" />',
        f'  <text x="24" y="32" class="title">{html.escape(str(title))}</text>',
    ]
    if subtitle:
        lines.append(f'  <text x="24" y="52" class="subtitle">{html.escape(str(subtitle))}</text>')

    # Header row
    lines.append(
        f'  <rect x="{left_m}" y="{top_m}" width="{len(headers) * col_w}" height="{row_h}" rx="4" fill="#1e293b" />'
    )
    for j, h in enumerate(headers):
        lines.append(
            f'  <text x="{left_m + j * col_w + col_w / 2.0}" y="{top_m + row_h / 2.0 + 4}" class="th" text-anchor="middle">{html.escape(str(h))}</text>'
        )

    # Table rows
    for i, row in enumerate(rows):
        ry = top_m + (i + 1) * row_h
        bg = "#f8fafc" if i % 2 == 0 else "#ffffff"
        lines.append(
            f'  <rect x="{left_m}" y="{ry}" width="{len(headers) * col_w}" height="{row_h}" fill="{bg}" stroke="#e2e8f0" stroke-width="0.5" />'
        )
        for j, cell in enumerate(row):
            lines.append(
                f'  <text x="{left_m + j * col_w + col_w / 2.0}" y="{ry + row_h / 2.0 + 4}" class="td" text-anchor="middle">{html.escape(str(cell))}</text>'
            )

    lines.append("</svg>")
    return "\n".join(lines)


# =========================================================================== #
# GATE 4: COVARIANCE, FACTOR RISK & ATTRIBUTION ARTIFACT RENDERERS
# =========================================================================== #
def render_covariance_diagnostics_artifact(
    diag: CovarianceDiagnostics,
    evidence_ids: tuple[str, ...],
    cov_matrix: list[list[float]] | None = None,
    assets: list[str] | tuple[str, ...] | None = None,
    test_id: str = "covariance.diagnostics",
    output_dir: Path | str | None = None,
    allow_empty_evidence: bool = False,
) -> ArtifactRecord:
    """Render Covariance Structural & Spectral Diagnostics artifact (SVG + JSON)."""
    _validate_provenance(evidence_ids, allow_empty=allow_empty_evidence)
    spec = ArtifactSpec(
        artifact_type="covariance_diagnostics",
        title="Covariance Structural & Spectral Diagnostics",
        test_id=test_id,
        evidence_ids=evidence_ids,
        parameters={"n_assets": diag.n_assets},
    )
    semantic_payload: dict[str, Any] = {
        "n_assets": diag.n_assets,
        "is_symmetric": diag.is_symmetric,
        "symmetry_error": diag.symmetry_error,
        "is_psd": diag.is_psd,
        "minimum_eigenvalue": diag.minimum_eigenvalue,
        "maximum_eigenvalue": diag.maximum_eigenvalue,
        "eigenvalue_spectrum": list(diag.eigenvalue_spectrum),
        "rank": diag.rank,
        "numerical_rank": diag.numerical_rank,
        "condition_number": diag.condition_number,
        "trace": diag.trace,
        "effective_rank": diag.effective_rank,
        "largest_eigenvalue_share": diag.largest_eigenvalue_share,
        "diagonal_positive": diag.diagonal_positive,
        "valid_correlation_conversion": diag.valid_correlation_conversion,
        "matrix_fingerprint": diag.matrix_fingerprint,
    }
    if cov_matrix is not None:
        semantic_payload["raw_covariance_matrix"] = cov_matrix

    payload_hash = _hash_payload(semantic_payload)
    artifact_id = f"ART-COV-DIAG-{payload_hash[:8]}"

    file_path = None
    if output_dir:
        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        file_path = str(out_p / f"{artifact_id}.svg")
        svg_content = _generate_eigenspectrum_svg(
            eigenvalues=list(diag.eigenvalue_spectrum),
            title="Covariance Matrix Eigenspectrum & Condition Diagnostics",
            subtitle=f"Assets: {diag.n_assets} | Condition Number: {diag.condition_number:.2e} | Effective Rank: {diag.effective_rank:.2f} | PSD: {diag.is_psd}",
        )
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(svg_content)
        with open(str(out_p / f"{artifact_id}.json"), "w", encoding="utf-8") as f:
            json.dump(semantic_payload, f, indent=2)

    return ArtifactRecord(
        artifact_id=artifact_id,
        spec=spec,
        data_fingerprint=payload_hash,
        semantic_payload=semantic_payload,
        semantic_payload_hash=payload_hash,
        file_path=file_path,
        rendering_format="svg" if file_path else "json",
    )


def render_raw_covariance_heatmap_artifact(
    cov_matrix: list[list[float]] | np.ndarray,
    assets: list[str] | tuple[str, ...],
    evidence_ids: tuple[str, ...],
    test_id: str = "covariance.raw_matrix",
    output_dir: Path | str | None = None,
    allow_empty_evidence: bool = False,
) -> ArtifactRecord:
    """Render Raw Covariance Heatmap (SVG + JSON)."""
    _validate_provenance(evidence_ids, allow_empty=allow_empty_evidence)
    mat_list = [list(r) for r in cov_matrix] if isinstance(cov_matrix, np.ndarray) else cov_matrix
    asset_list = list(assets)
    spec = ArtifactSpec(
        artifact_type="raw_covariance_heatmap",
        title="Asset Covariance Matrix Heatmap",
        test_id=test_id,
        evidence_ids=evidence_ids,
        parameters={"assets": asset_list},
    )
    semantic_payload = {
        "assets": asset_list,
        "covariance_matrix": mat_list,
    }
    payload_hash = _hash_payload(semantic_payload)
    artifact_id = f"ART-RAW-COV-{payload_hash[:8]}"

    file_path = None
    if output_dir:
        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        file_path = str(out_p / f"{artifact_id}.svg")
        svg_content = _generate_matrix_heatmap_svg(
            matrix=mat_list,
            row_labels=asset_list,
            col_labels=asset_list,
            title="Raw Asset Covariance Heatmap",
            subtitle=f"{len(asset_list)} Assets | Symmetric Pairwise Covariances",
            fmt=".6f",
            is_diverging=False,
        )
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(svg_content)
        with open(str(out_p / f"{artifact_id}.json"), "w", encoding="utf-8") as f:
            json.dump(semantic_payload, f, indent=2)

    return ArtifactRecord(
        artifact_id=artifact_id,
        spec=spec,
        data_fingerprint=payload_hash,
        semantic_payload=semantic_payload,
        semantic_payload_hash=payload_hash,
        file_path=file_path,
        rendering_format="svg" if file_path else "json",
    )


def render_psd_repair_artifact(
    repair: PSDRepairResult,
    evidence_ids: tuple[str, ...],
    raw_matrix: list[list[float]] | None = None,
    assets: list[str] | tuple[str, ...] | None = None,
    test_id: str = "covariance.psd_repair",
    output_dir: Path | str | None = None,
    allow_empty_evidence: bool = False,
) -> ArtifactRecord:
    """Render PSD Repair & Frobenius Distortion Audit artifact (SVG + JSON)."""
    _validate_provenance(evidence_ids, allow_empty=allow_empty_evidence)
    spec = ArtifactSpec(
        artifact_type="psd_repair_distortion",
        title="Covariance Numerical PSD Repair & Distortion Audit",
        test_id=test_id,
        evidence_ids=evidence_ids,
        parameters={"repair_method": str(repair.repair_method)},
    )
    semantic_payload: dict[str, Any] = {
        "repair_method": str(repair.repair_method),
        "original_minimum_eigenvalue": repair.original_minimum_eigenvalue,
        "repaired_minimum_eigenvalue": repair.repaired_minimum_eigenvalue,
        "frobenius_distortion": repair.frobenius_distortion,
        "relative_frobenius_distortion": repair.relative_frobenius_distortion,
        "maximum_element_change": repair.maximum_element_change,
        "diagonal_preserved": repair.diagonal_preserved,
        "iterations_used": repair.iterations_used,
        "converged": repair.converged,
        "matrix_fingerprint_before": repair.matrix_fingerprint_before,
        "matrix_fingerprint_after": repair.matrix_fingerprint_after,
        "repaired_matrix": repair.repaired_matrix,
        "pd_floor": repair.pd_floor,
    }
    if raw_matrix is not None:
        semantic_payload["raw_matrix"] = raw_matrix

    payload_hash = _hash_payload(semantic_payload)
    artifact_id = f"ART-PSD-REPAIR-{payload_hash[:8]}"

    file_path = None
    if output_dir:
        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        file_path = str(out_p / f"{artifact_id}.svg")
        metrics = [
            ("Method", str(repair.repair_method).split(".")[-1], "Intervention Type"),
            (
                "Frobenius Distortion",
                f"{repair.frobenius_distortion:.4f}",
                f"Rel: {repair.relative_frobenius_distortion:.2%}",
            ),
            ("Original Min Eigenvalue", f"{repair.original_minimum_eigenvalue:.4f}", "Non-PSD Input"),
            ("Repaired Min Eigenvalue", f"{repair.repaired_minimum_eigenvalue:.2e}", "Guaranteed PSD"),
            ("Iterations Used", str(repair.iterations_used), f"Converged: {repair.converged}"),
            ("Diagonal Preserved", str(repair.diagonal_preserved), "Exact Variance"),
        ]
        svg_content = _generate_dashboard_svg(
            metrics=metrics,
            title="PSD Repair Distortion & Eigenvalue Verification",
            subtitle=f"Method: {repair.repair_method} | Converged in {repair.iterations_used} iterations",
        )
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(svg_content)
        with open(str(out_p / f"{artifact_id}.json"), "w", encoding="utf-8") as f:
            json.dump(semantic_payload, f, indent=2)

    return ArtifactRecord(
        artifact_id=artifact_id,
        spec=spec,
        data_fingerprint=payload_hash,
        semantic_payload=semantic_payload,
        semantic_payload_hash=payload_hash,
        file_path=file_path,
        rendering_format="svg" if file_path else "json",
    )


def render_covariance_comparison_artifact(
    comp: CovarianceComparisonResult,
    evidence_ids: tuple[str, ...],
    test_id: str = "covariance.model_comparison",
    output_dir: Path | str | None = None,
    allow_empty_evidence: bool = False,
) -> ArtifactRecord:
    """Render Covariance Estimator Comparative Evaluation Matrix (SVG + JSON)."""
    _validate_provenance(evidence_ids, allow_empty=allow_empty_evidence)
    spec = ArtifactSpec(
        artifact_type="covariance_estimator_comparison",
        title="Covariance Estimator Comparative Evaluation Matrix",
        test_id=test_id,
        evidence_ids=evidence_ids,
        parameters={"estimators": list(comp.estimators_compared)},
    )
    semantic_payload: dict[str, Any] = {
        "estimators_compared": list(comp.estimators_compared),
        "asset_order": list(comp.asset_order),
        "pairwise_frobenius_distances": comp.pairwise_frobenius_distances,
        "pairwise_spectral_distances": comp.pairwise_spectral_distances,
        "portfolio_volatilities_annualised": comp.portfolio_volatilities_annualised,
    }
    payload_hash = _hash_payload(semantic_payload)
    artifact_id = f"ART-COV-COMP-{payload_hash[:8]}"

    file_path = None
    if output_dir:
        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        file_path = str(out_p / f"{artifact_id}.svg")
        # Build pairwise distance matrix for heatmap
        ests = list(comp.estimators_compared)
        matrix = [[0.0] * len(ests) for _ in range(len(ests))]
        for i in range(len(ests)):
            for j in range(len(ests)):
                if i == j:
                    matrix[i][j] = 0.0
                else:
                    k1 = f"{ests[i]}_vs_{ests[j]}"
                    k2 = f"{ests[j]}_vs_{ests[i]}"
                    matrix[i][j] = comp.pairwise_frobenius_distances.get(
                        k1, comp.pairwise_frobenius_distances.get(k2, 0.0)
                    )

        svg_content = _generate_matrix_heatmap_svg(
            matrix=matrix,
            row_labels=ests,
            col_labels=ests,
            title="Covariance Estimator Pairwise Frobenius Distances",
            subtitle=f"Estimators: {', '.join(ests)}",
            fmt=".4f",
            is_diverging=False,
        )
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(svg_content)
        with open(str(out_p / f"{artifact_id}.json"), "w", encoding="utf-8") as f:
            json.dump(semantic_payload, f, indent=2)

    return ArtifactRecord(
        artifact_id=artifact_id,
        spec=spec,
        data_fingerprint=payload_hash,
        semantic_payload=semantic_payload,
        semantic_payload_hash=payload_hash,
        file_path=file_path,
        rendering_format="svg" if file_path else "json",
    )


def render_factor_risk_model_artifact(
    frm: FactorRiskModelResult,
    evidence_ids: tuple[str, ...],
    test_id: str = "factor_risk.model",
    output_dir: Path | str | None = None,
    allow_empty_evidence: bool = False,
) -> ArtifactRecord:
    """Render Factor Risk Model Specification & Exposure Heatmap (SVG + JSON)."""
    _validate_provenance(evidence_ids, allow_empty=allow_empty_evidence)
    spec = ArtifactSpec(
        artifact_type="factor_risk_model",
        title="Linear Factor Risk Model Specification & Reconstructed Covariance",
        test_id=test_id,
        evidence_ids=evidence_ids,
        parameters={"n_assets": len(frm.asset_order), "n_factors": len(frm.factor_order)},
    )
    semantic_payload: dict[str, Any] = {
        "asset_order": list(frm.asset_order),
        "factor_order": list(frm.factor_order),
        "exposure_matrix": frm.exposure_matrix,
        "factor_covariance": frm.factor_covariance,
        "specific_variances": frm.specific_variances,
        "reconstructed_covariance": frm.reconstructed_covariance,
        "exposure_fingerprint": frm.exposure_fingerprint,
        "reconstructed_covariance_fingerprint": frm.reconstructed_covariance_fingerprint,
        "condition_number": frm.diagnostics.condition_number,
        "is_psd": frm.diagnostics.is_psd,
        "horizon": str(frm.horizon),
    }
    payload_hash = _hash_payload(semantic_payload)
    artifact_id = f"ART-FACTOR-MDL-{payload_hash[:8]}"

    file_path = None
    if output_dir:
        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        file_path = str(out_p / f"{artifact_id}.svg")
        svg_content = _generate_matrix_heatmap_svg(
            matrix=frm.exposure_matrix,
            row_labels=list(frm.asset_order),
            col_labels=list(frm.factor_order),
            title="Factor Exposure Matrix (B)",
            subtitle=f"{len(frm.asset_order)} Assets x {len(frm.factor_order)} Factors | Reconstructed Covariance Condition: {frm.diagnostics.condition_number:.2e}",
            fmt=".3f",
            is_diverging=True,
        )
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(svg_content)
        with open(str(out_p / f"{artifact_id}.json"), "w", encoding="utf-8") as f:
            json.dump(semantic_payload, f, indent=2)

    return ArtifactRecord(
        artifact_id=artifact_id,
        spec=spec,
        data_fingerprint=payload_hash,
        semantic_payload=semantic_payload,
        semantic_payload_hash=payload_hash,
        file_path=file_path,
        rendering_format="svg" if file_path else "json",
    )


def render_factor_risk_waterfall_artifact(
    frd: FactorRiskDecompositionResult,
    evidence_ids: tuple[str, ...],
    test_id: str = "factor_risk.decomposition",
    output_dir: Path | str | None = None,
    allow_empty_evidence: bool = False,
) -> ArtifactRecord:
    """Render Euler Factor Risk Component Waterfall artifact (SVG + JSON)."""
    _validate_provenance(evidence_ids, allow_empty=allow_empty_evidence)
    spec = ArtifactSpec(
        artifact_type="factor_risk_waterfall",
        title="Euler Factor Risk Variance Waterfall Decomposition",
        test_id=test_id,
        evidence_ids=evidence_ids,
        parameters={"periods_per_year": frd.periods_per_year},
    )
    semantic_payload: dict[str, Any] = {
        "portfolio_factor_exposures": frd.portfolio_factor_exposures,
        "systematic_variance_periodic": frd.systematic_variance_periodic,
        "specific_variance_periodic": frd.specific_variance_periodic,
        "total_variance_periodic": frd.total_variance_periodic,
        "portfolio_volatility_annualised": frd.portfolio_volatility_annualised,
        "systematic_volatility_annualised": frd.systematic_volatility_annualised,
        "specific_volatility_annualised": frd.specific_volatility_annualised,
        "systematic_variance_share": frd.systematic_variance_share,
        "specific_variance_share": frd.specific_variance_share,
        "factor_variance_contributions_periodic": frd.factor_variance_contributions_periodic,
        "factor_variance_shares": frd.factor_variance_shares,
        "asset_specific_variance_contributions": frd.asset_specific_variance_contributions,
        "euler_reconciliation_error": frd.euler_reconciliation_error,
        "horizon": str(frd.horizon),
    }
    payload_hash = _hash_payload(semantic_payload)
    artifact_id = f"ART-FACTOR-WF-{payload_hash[:8]}"

    file_path = None
    if output_dir:
        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        file_path = str(out_p / f"{artifact_id}.svg")
        items = list(frd.factor_variance_contributions_periodic.items()) + [
            ("Specific Risk", frd.specific_variance_periodic)
        ]
        svg_content = _generate_waterfall_svg(
            items=items,
            total_val=frd.total_variance_periodic,
            title="Euler Factor Risk Variance Waterfall Decomposition",
            subtitle=f"Total Volatility: {frd.portfolio_volatility_annualised:.2%} ann | Systematic: {frd.systematic_variance_share:.1%} | Specific: {frd.specific_variance_share:.1%}",
        )
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(svg_content)
        with open(str(out_p / f"{artifact_id}.json"), "w", encoding="utf-8") as f:
            json.dump(semantic_payload, f, indent=2)

    return ArtifactRecord(
        artifact_id=artifact_id,
        spec=spec,
        data_fingerprint=payload_hash,
        semantic_payload=semantic_payload,
        semantic_payload_hash=payload_hash,
        file_path=file_path,
        rendering_format="svg" if file_path else "json",
    )


def render_active_risk_decomposition_artifact(
    ard: ActiveRiskDecompositionResult,
    evidence_ids: tuple[str, ...],
    test_id: str = "factor_risk.active_decomposition",
    output_dir: Path | str | None = None,
    allow_empty_evidence: bool = False,
) -> ArtifactRecord:
    """Render Benchmark-Relative Active Risk (Tracking Error) Decomposition (SVG + JSON)."""
    _validate_provenance(evidence_ids, allow_empty=allow_empty_evidence)
    spec = ArtifactSpec(
        artifact_type="active_risk_decomposition",
        title="Benchmark-Relative Active Risk (Tracking Error) Decomposition",
        test_id=test_id,
        evidence_ids=evidence_ids,
        parameters={"periods_per_year": ard.periods_per_year},
    )
    semantic_payload: dict[str, Any] = {
        "active_weights": ard.active_weights,
        "active_factor_exposures": ard.active_factor_exposures,
        "factor_active_variance_periodic": ard.factor_active_variance_periodic,
        "specific_active_variance_periodic": ard.specific_active_variance_periodic,
        "total_active_variance_periodic": ard.total_active_variance_periodic,
        "tracking_error_annualised": ard.tracking_error_annualised,
        "factor_active_share": ard.factor_active_share,
        "specific_active_share": ard.specific_active_share,
        "active_factor_contributions_periodic": ard.active_factor_contributions_periodic,
        "asset_specific_active_contributions": ard.asset_specific_active_contributions,
        "reconciliation_error": ard.reconciliation_error,
        "horizon": str(ard.horizon),
    }
    payload_hash = _hash_payload(semantic_payload)
    artifact_id = f"ART-ACTIVE-RISK-{payload_hash[:8]}"

    file_path = None
    if output_dir:
        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        file_path = str(out_p / f"{artifact_id}.svg")
        items = list(ard.active_factor_contributions_periodic.items()) + [
            ("Active Specific", ard.specific_active_variance_periodic)
        ]
        svg_content = _generate_waterfall_svg(
            items=items,
            total_val=ard.total_active_variance_periodic,
            title="Benchmark Active Risk (Tracking Error) Decomposition",
            subtitle=f"Annualized Tracking Error: {ard.tracking_error_annualised:.2%} | Factor Share: {ard.factor_active_share:.1%} | Specific Share: {ard.specific_active_share:.1%}",
        )
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(svg_content)
        with open(str(out_p / f"{artifact_id}.json"), "w", encoding="utf-8") as f:
            json.dump(semantic_payload, f, indent=2)

    return ArtifactRecord(
        artifact_id=artifact_id,
        spec=spec,
        data_fingerprint=payload_hash,
        semantic_payload=semantic_payload,
        semantic_payload_hash=payload_hash,
        file_path=file_path,
        rendering_format="svg" if file_path else "json",
    )


def render_factor_return_attribution_artifact(
    fra: FactorReturnAttributionResult,
    evidence_ids: tuple[str, ...],
    test_id: str = "attribution.factor_performance",
    output_dir: Path | str | None = None,
    allow_empty_evidence: bool = False,
) -> ArtifactRecord:
    """Render Factor Return Performance Attribution artifact (SVG + JSON)."""
    _validate_provenance(evidence_ids, allow_empty=allow_empty_evidence)
    spec = ArtifactSpec(
        artifact_type="factor_return_attribution",
        title="Period-by-Period Factor Return Performance Attribution",
        test_id=test_id,
        evidence_ids=evidence_ids,
        parameters={"n_periods": fra.n_periods},
    )
    semantic_payload: dict[str, Any] = {
        "n_periods": fra.n_periods,
        "total_portfolio_return": fra.total_portfolio_return,
        "total_factor_contribution": fra.total_factor_contribution,
        "total_specific_contribution": fra.total_specific_contribution,
        "cumulative_factor_contributions": fra.cumulative_factor_contributions,
        "period_portfolio_returns": list(fra.period_portfolio_returns),
        "period_specific_contributions": list(fra.period_specific_contributions),
        "period_reconciliation_errors": list(fra.period_reconciliation_errors),
        "max_abs_reconciliation_error": fra.max_abs_reconciliation_error,
        "is_reconciled": fra.is_reconciled,
    }
    payload_hash = _hash_payload(semantic_payload)
    artifact_id = f"ART-FACTOR-ATTRIB-{payload_hash[:8]}"

    file_path = None
    if output_dir:
        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        file_path = str(out_p / f"{artifact_id}.svg")
        items = list(fra.cumulative_factor_contributions.items()) + [
            ("Specific Return", fra.total_specific_contribution)
        ]
        svg_content = _generate_waterfall_svg(
            items=items,
            total_val=fra.total_portfolio_return,
            title="Factor Return Performance Attribution Waterfall",
            subtitle=f"{fra.n_periods} Periods | Total Return: {fra.total_portfolio_return:.2%} | Factor: {fra.total_factor_contribution:.2%} | Specific: {fra.total_specific_contribution:.2%}",
        )
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(svg_content)
        with open(str(out_p / f"{artifact_id}.json"), "w", encoding="utf-8") as f:
            json.dump(semantic_payload, f, indent=2)

    return ArtifactRecord(
        artifact_id=artifact_id,
        spec=spec,
        data_fingerprint=payload_hash,
        semantic_payload=semantic_payload,
        semantic_payload_hash=payload_hash,
        file_path=file_path,
        rendering_format="svg" if file_path else "json",
    )


def render_brinson_attribution_artifact(
    ba: BrinsonAttributionResult,
    evidence_ids: tuple[str, ...],
    test_id: str = "attribution.brinson",
    output_dir: Path | str | None = None,
    allow_empty_evidence: bool = False,
) -> ArtifactRecord:
    """Render Brinson-Fachler Active Performance Attribution Waterfall artifact (SVG + JSON)."""
    _validate_provenance(evidence_ids, allow_empty=allow_empty_evidence)
    spec = ArtifactSpec(
        artifact_type="brinson_attribution_waterfall",
        title="Brinson-Fachler Active Return Performance Attribution Waterfall",
        test_id=test_id,
        evidence_ids=evidence_ids,
        parameters={"convention": ba.convention},
    )
    semantic_payload: dict[str, Any] = {
        "group_names": list(ba.group_names),
        "portfolio_group_weights": ba.portfolio_group_weights,
        "benchmark_group_weights": ba.benchmark_group_weights,
        "portfolio_group_returns": ba.portfolio_group_returns,
        "benchmark_group_returns": ba.benchmark_group_returns,
        "total_portfolio_return": ba.total_portfolio_return,
        "total_benchmark_return": ba.total_benchmark_return,
        "total_active_return": ba.total_active_return,
        "allocation_effects": ba.allocation_effects,
        "selection_effects": ba.selection_effects,
        "interaction_effects": ba.interaction_effects,
        "total_allocation_effect": ba.total_allocation_effect,
        "total_selection_effect": ba.total_selection_effect,
        "total_interaction_effect": ba.total_interaction_effect,
        "reconciliation_error": ba.reconciliation_error,
        "is_reconciled": ba.is_reconciled,
    }
    payload_hash = _hash_payload(semantic_payload)
    artifact_id = f"ART-BRINSON-{payload_hash[:8]}"

    file_path = None
    if output_dir:
        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        file_path = str(out_p / f"{artifact_id}.svg")
        items = [
            ("Allocation", ba.total_allocation_effect),
            ("Selection", ba.total_selection_effect),
            ("Interaction", ba.total_interaction_effect),
        ]
        svg_content = _generate_waterfall_svg(
            items=items,
            total_val=ba.total_active_return,
            title="Brinson-Fachler Active Return Attribution Waterfall",
            subtitle=f"Portfolio: {ba.total_portfolio_return:.2%} | Benchmark: {ba.total_benchmark_return:.2%} | Active: {ba.total_active_return:.2%}",
        )
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(svg_content)
        with open(str(out_p / f"{artifact_id}.json"), "w", encoding="utf-8") as f:
            json.dump(semantic_payload, f, indent=2)

    return ArtifactRecord(
        artifact_id=artifact_id,
        spec=spec,
        data_fingerprint=payload_hash,
        semantic_payload=semantic_payload,
        semantic_payload_hash=payload_hash,
        file_path=file_path,
        rendering_format="svg" if file_path else "json",
    )


def render_carino_linking_artifact(
    ca: CarinoLinkedAttributionResult,
    evidence_ids: tuple[str, ...],
    test_id: str = "attribution.multi_period_linking",
    output_dir: Path | str | None = None,
    allow_empty_evidence: bool = False,
) -> ArtifactRecord:
    """Render Carino Logarithmic Multi-Period Attribution Linking artifact (SVG + JSON)."""
    _validate_provenance(evidence_ids, allow_empty=allow_empty_evidence)
    spec = ArtifactSpec(
        artifact_type="carino_linking_waterfall",
        title="Carino Logarithmic Multi-Period Active Attribution Linking",
        test_id=test_id,
        evidence_ids=evidence_ids,
        parameters={"n_periods": ca.n_periods},
    )
    semantic_payload: dict[str, Any] = {
        "n_periods": ca.n_periods,
        "group_names": list(ca.group_names),
        "total_portfolio_return_geometric": ca.total_portfolio_return_geometric,
        "total_benchmark_return_geometric": ca.total_benchmark_return_geometric,
        "total_active_return_geometric": ca.total_active_return_geometric,
        "linked_allocation_effects": ca.linked_allocation_effects,
        "linked_selection_effects": ca.linked_selection_effects,
        "linked_interaction_effects": ca.linked_interaction_effects,
        "total_linked_allocation": ca.total_linked_allocation,
        "total_linked_selection": ca.total_linked_selection,
        "total_linked_interaction": ca.total_linked_interaction,
        "period_linking_coefficients": list(ca.period_linking_coefficients),
        "benchmark_linking_coefficient": ca.benchmark_linking_coefficient,
        "reconciliation_error": ca.reconciliation_error,
        "is_reconciled": ca.is_reconciled,
    }
    payload_hash = _hash_payload(semantic_payload)
    artifact_id = f"ART-CARINO-{payload_hash[:8]}"

    file_path = None
    if output_dir:
        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        file_path = str(out_p / f"{artifact_id}.svg")
        items = [
            ("Linked Allocation", ca.total_linked_allocation),
            ("Linked Selection", ca.total_linked_selection),
            ("Linked Interaction", ca.total_linked_interaction),
        ]
        svg_content = _generate_waterfall_svg(
            items=items,
            total_val=ca.total_active_return_geometric,
            title="Carino Logarithmic Multi-Period Active Attribution Linking",
            subtitle=f"{ca.n_periods} Periods Geometric Linking | Portfolio: {ca.total_portfolio_return_geometric:.2%} | Benchmark: {ca.total_benchmark_return_geometric:.2%} | Active: {ca.total_active_return_geometric:.2%}",
        )
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(svg_content)
        with open(str(out_p / f"{artifact_id}.json"), "w", encoding="utf-8") as f:
            json.dump(semantic_payload, f, indent=2)

    return ArtifactRecord(
        artifact_id=artifact_id,
        spec=spec,
        data_fingerprint=payload_hash,
        semantic_payload=semantic_payload,
        semantic_payload_hash=payload_hash,
        file_path=file_path,
        rendering_format="svg" if file_path else "json",
    )


# =========================================================================== #
# GATE 5: TAIL RISK, EXPECTED SHORTFALL & BACKTESTING ARTIFACTS
# =========================================================================== #
def render_tail_loss_distribution_artifact(
    estimate: TailRiskEstimate,
    evidence_ids: tuple[str, ...],
    output_dir: str | Path | None = None,
    test_id: str = "traded_risk.expected_shortfall",
) -> ArtifactRecord:
    """Render loss distribution summary with exact finite-sample VaR and Expected Shortfall."""
    _validate_provenance(evidence_ids)
    spec = ArtifactSpec(
        artifact_type="tail_loss_distribution",
        title=f"Loss Distribution Tail Risk ({estimate.method.upper()} @ {estimate.confidence:.1%})",
        test_id=test_id,
        evidence_ids=evidence_ids,
        parameters={"confidence": estimate.confidence, "method": str(estimate.method)},
    )
    semantic_payload: dict[str, Any] = {
        "method": str(estimate.method),
        "confidence": estimate.confidence,
        "sign_convention": str(estimate.sign_convention),
        "var": estimate.var,
        "es": estimate.es,
        "n_observations": estimate.n_observations,
        "tail_observations_count": estimate.tail_observations_count,
        "tail_fraction": estimate.tail_fraction,
        "boundary_weight": estimate.boundary_weight,
        "quantile_method": estimate.quantile_method,
        "horizon": str(estimate.horizon),
        "parameters": estimate.parameters,
        "converged": estimate.converged,
        "limitations": list(estimate.limitations),
        "data_fingerprint": estimate.data_fingerprint,
    }
    payload_hash = _hash_payload(semantic_payload)
    artifact_id = f"ART-TAIL-DIST-{payload_hash[:8]}"

    file_path = None
    if output_dir:
        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        file_path = str(out_p / f"{artifact_id}.svg")
        cards = [
            ("Method", str(estimate.method).upper(), "Model Specification"),
            ("Confidence", f"{estimate.confidence:.1%}", "alpha_var parameter"),
            ("VaR (Loss Mag)", f"{estimate.var:.4f}", "Positive Loss Magnitude"),
            ("Expected Shortfall", f"{estimate.es:.4f}", "Exact Finite-Sample Tail Avg"),
            (
                "Tail Obs Count",
                f"{estimate.tail_observations_count} (q={estimate.parameters.get('q_tail_mass', 0):.2f})",
                "Tail support",
            ),
            ("Boundary Weight", f"{estimate.boundary_weight:.4f}", "Fractional weight"),
        ]
        svg_content = _generate_dashboard_svg(
            metrics=cards,
            title=f"Loss Distribution Tail Risk ({estimate.method.upper()})",
            subtitle=f"Confidence: {estimate.confidence:.1%} | VaR: {estimate.var:.4f} | ES: {estimate.es:.4f} | Observations: {estimate.n_observations}",
        )
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(svg_content)
        with open(str(out_p / f"{artifact_id}.json"), "w", encoding="utf-8") as f:
            json.dump(semantic_payload, f, indent=2)

    return ArtifactRecord(
        artifact_id=artifact_id,
        spec=spec,
        data_fingerprint=payload_hash,
        semantic_payload=semantic_payload,
        semantic_payload_hash=payload_hash,
        file_path=file_path,
        rendering_format="svg" if file_path else "json",
    )


def render_var_pnl_timeline_artifact(
    backtest: TailBacktestResult,
    evidence_ids: tuple[str, ...],
    output_dir: str | Path | None = None,
    test_id: str = "traded_risk.var_exceptions",
) -> ArtifactRecord:
    """Render VaR forecast vs realized P&L exception tracking timeline."""
    _validate_provenance(evidence_ids)
    spec = ArtifactSpec(
        artifact_type="var_pnl_timeline",
        title="VaR Forecast vs Realized P&L Exception Timeline",
        test_id=test_id,
        evidence_ids=evidence_ids,
        parameters={"pnl_source": backtest.pnl_source, "confidence": backtest.var_confidence},
    )
    semantic_payload: dict[str, Any] = {
        "pnl_source": backtest.pnl_source,
        "var_confidence": backtest.var_confidence,
        "test_significance": backtest.test_significance,
        "n_observations": backtest.n_observations,
        "n_exceptions": backtest.n_exceptions,
        "exception_rate": backtest.exception_rate,
        "expected_probability": backtest.expected_probability,
        "expected_exceptions": backtest.expected_exceptions,
        "exception_dates": list(backtest.exception_dates),
        "indicators_count": len(backtest.indicators),
        "exception_convention": backtest.exception_convention,
        "data_fingerprint": backtest.data_fingerprint,
    }
    payload_hash = _hash_payload(semantic_payload)
    artifact_id = f"ART-VAR-TIMELINE-{payload_hash[:8]}"

    file_path = None
    if output_dir:
        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        file_path = str(out_p / f"{artifact_id}.svg")
        cards = [
            ("Aligned Obs", f"{backtest.n_observations:,}", "Out-of-sample sample"),
            ("Exceptions", f"{backtest.n_exceptions}", "Realized breaches"),
            ("Realized Rate", f"{backtest.exception_rate:.2%}", "Observed frequency"),
            ("Expected Rate", f"{backtest.expected_probability:.2%}", "1 - alpha_var"),
            ("Expected Count", f"{backtest.expected_exceptions:.1f}", "Target count"),
            ("P&L Source", backtest.pnl_source.upper(), "P&L provenance"),
        ]
        svg_content = _generate_dashboard_svg(
            metrics=cards,
            title="VaR Forecast vs Realized P&L Exception Diagnostics",
            subtitle=f"{backtest.n_exceptions} exception(s) in {backtest.n_observations:,} observations @ {backtest.var_confidence:.1%} VaR Confidence ({backtest.pnl_source} P&L)",
        )
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(svg_content)
        with open(str(out_p / f"{artifact_id}.json"), "w", encoding="utf-8") as f:
            json.dump(semantic_payload, f, indent=2)

    return ArtifactRecord(
        artifact_id=artifact_id,
        spec=spec,
        data_fingerprint=payload_hash,
        semantic_payload=semantic_payload,
        semantic_payload_hash=payload_hash,
        file_path=file_path,
        rendering_format="svg" if file_path else "json",
    )


def render_exception_transition_artifact(
    backtest: TailBacktestResult,
    evidence_ids: tuple[str, ...],
    output_dir: str | Path | None = None,
    test_id: str = "traded_risk.var_christoffersen_independence",
) -> ArtifactRecord:
    """Render 2x2 first-order Markov exception transition matrix heatmap."""
    _validate_provenance(evidence_ids)
    spec = ArtifactSpec(
        artifact_type="exception_transition_matrix",
        title="First-Order Markov Exception Transition Matrix",
        test_id=test_id,
        evidence_ids=evidence_ids,
        parameters={"transition_counts": list(backtest.transition_counts)},
    )
    n00, n01, n10, n11 = backtest.transition_counts
    mat = [[float(n00), float(n01)], [float(n10), float(n11)]]
    labels = ["No Exception (0)", "Exception (1)"]

    semantic_payload: dict[str, Any] = {
        "n00": n00,
        "n01": n01,
        "n10": n10,
        "n11": n11,
        "pi_01": backtest.pi_01,
        "pi_11": backtest.pi_11,
        "has_zero_transition_cell": backtest.has_zero_transition_cell,
        "christoffersen_lr": backtest.christoffersen_lr,
        "christoffersen_p_value": backtest.christoffersen_p_value,
        "christoffersen_rejected": backtest.christoffersen_rejected,
        "data_fingerprint": backtest.data_fingerprint,
    }
    payload_hash = _hash_payload(semantic_payload)
    artifact_id = f"ART-TAIL-TRANS-{payload_hash[:8]}"

    file_path = None
    if output_dir:
        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        file_path = str(out_p / f"{artifact_id}.svg")
        svg_content = _generate_matrix_heatmap_svg(
            matrix=mat,
            row_labels=labels,
            col_labels=labels,
            title="First-Order Markov Exception Transition Matrix",
            subtitle=f"n00={n00}, n01={n01}, n10={n10}, n11={n11} | pi01={backtest.pi_01 if backtest.pi_01 is not None else 0.0:.2%}, pi11={backtest.pi_11 if backtest.pi_11 is not None else 0.0:.2%}",
        )
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(svg_content)
        with open(str(out_p / f"{artifact_id}.json"), "w", encoding="utf-8") as f:
            json.dump(semantic_payload, f, indent=2)

    return ArtifactRecord(
        artifact_id=artifact_id,
        spec=spec,
        data_fingerprint=payload_hash,
        semantic_payload=semantic_payload,
        semantic_payload_hash=payload_hash,
        file_path=file_path,
        rendering_format="svg" if file_path else "json",
    )


def render_duration_diagnostics_artifact(
    durations: DurationDiagnosticsResult,
    evidence_ids: tuple[str, ...],
    output_dir: str | Path | None = None,
    test_id: str = "traded_risk.exception_durations",
) -> ArtifactRecord:
    """Render descriptive inter-exception duration and clustering diagnostics."""
    _validate_provenance(evidence_ids)
    spec = ArtifactSpec(
        artifact_type="duration_diagnostics",
        title="Inter-Exception Duration & Clustering Diagnostics",
        test_id=test_id,
        evidence_ids=evidence_ids,
        parameters={"n_durations": durations.n_durations},
    )
    semantic_payload: dict[str, Any] = {
        "n_durations": durations.n_durations,
        "mean_duration": durations.mean_duration,
        "median_duration": durations.median_duration,
        "min_duration": durations.min_duration,
        "max_duration": durations.max_duration,
        "duration_std": durations.duration_std,
        "max_run_length": durations.max_run_length,
        "durations": list(durations.durations),
        "data_fingerprint": durations.data_fingerprint,
    }
    payload_hash = _hash_payload(semantic_payload)
    artifact_id = f"ART-TAIL-DURATION-{payload_hash[:8]}"

    file_path = None
    if output_dir:
        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        file_path = str(out_p / f"{artifact_id}.svg")
        cards = [
            ("Duration Count", f"{durations.n_durations}", "Interval count"),
            ("Mean Duration", f"{durations.mean_duration:.1f} days", "Average interval"),
            ("Median Duration", f"{durations.median_duration:.1f} days", "50th percentile"),
            ("Min Duration", f"{durations.min_duration} days", "Minimum spacing"),
            ("Max Duration", f"{durations.max_duration} days", "Maximum spacing"),
            ("Max Run Length", f"{durations.max_run_length} day(s)", "Consecutive run"),
        ]
        svg_content = _generate_dashboard_svg(
            metrics=cards,
            title="Inter-Exception Duration & Clustering Diagnostics",
            subtitle=f"{durations.n_durations} inter-exception duration intervals | Std Dev: {durations.duration_std:.2f} days",
        )
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(svg_content)
        with open(str(out_p / f"{artifact_id}.json"), "w", encoding="utf-8") as f:
            json.dump(semantic_payload, f, indent=2)

    return ArtifactRecord(
        artifact_id=artifact_id,
        spec=spec,
        data_fingerprint=payload_hash,
        semantic_payload=semantic_payload,
        semantic_payload_hash=payload_hash,
        file_path=file_path,
        rendering_format="svg" if file_path else "json",
    )


def render_tail_severity_artifact(
    severity: TailSeverityResult,
    evidence_ids: tuple[str, ...],
    output_dir: str | Path | None = None,
    test_id: str = "traded_risk.tail_severity",
) -> ArtifactRecord:
    """Render tail exceedance severity and magnitude analysis."""
    _validate_provenance(evidence_ids)
    spec = ArtifactSpec(
        artifact_type="tail_severity",
        title="Tail Exceedance Loss Severity Diagnostics",
        test_id=test_id,
        evidence_ids=evidence_ids,
        parameters={"n_exceptions": severity.n_exceptions},
    )
    semantic_payload: dict[str, Any] = {
        "n_exceptions": severity.n_exceptions,
        "mean_absolute_exceedance": severity.mean_absolute_exceedance,
        "median_absolute_exceedance": severity.median_absolute_exceedance,
        "max_absolute_exceedance": severity.max_absolute_exceedance,
        "total_tail_exceedance_loss": severity.total_tail_exceedance_loss,
        "mean_normalized_exceedance": severity.mean_normalized_exceedance,
        "max_normalized_exceedance": severity.max_normalized_exceedance,
        "mean_relative_exceedance": severity.mean_relative_exceedance,
        "max_relative_exceedance": severity.max_relative_exceedance,
        "data_fingerprint": severity.data_fingerprint,
    }
    payload_hash = _hash_payload(semantic_payload)
    artifact_id = f"ART-TAIL-SEVERITY-{payload_hash[:8]}"

    file_path = None
    if output_dir:
        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        file_path = str(out_p / f"{artifact_id}.svg")
        cards = [
            ("Exception Count", f"{severity.n_exceptions}", "Breaches"),
            ("Mean Exceedance (L-VaR)", f"{severity.mean_absolute_exceedance:.4f}", "Average excess loss"),
            ("Median Exceedance", f"{severity.median_absolute_exceedance:.4f}", "50th percentile excess"),
            ("Max Exceedance", f"{severity.max_absolute_exceedance:.4f}", "Worst excess loss"),
            ("Total Excess Loss", f"{severity.total_tail_exceedance_loss:.4f}", "Cumulative tail loss"),
            (
                "Max Ratio (L/VaR)",
                f"{severity.max_normalized_exceedance:.2f}x"
                if severity.max_normalized_exceedance is not None
                else "N/A",
                "Worst relative severity",
            ),
        ]
        svg_content = _generate_dashboard_svg(
            metrics=cards,
            title="Tail Exceedance Loss Severity Diagnostics",
            subtitle=f"{severity.n_exceptions} exception events | Total Excess Loss: {severity.total_tail_exceedance_loss:.4f}",
        )
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(svg_content)
        with open(str(out_p / f"{artifact_id}.json"), "w", encoding="utf-8") as f:
            json.dump(semantic_payload, f, indent=2)

    return ArtifactRecord(
        artifact_id=artifact_id,
        spec=spec,
        data_fingerprint=payload_hash,
        semantic_payload=semantic_payload,
        semantic_payload_hash=payload_hash,
        file_path=file_path,
        rendering_format="svg" if file_path else "json",
    )


def render_backtest_summary_artifact(
    backtest: TailBacktestResult,
    evidence_ids: tuple[str, ...],
    output_dir: str | Path | None = None,
    test_id: str = "traded_risk.var_conditional_coverage",
) -> ArtifactRecord:
    """Render comprehensive backtest statistical summary (Kupiec, Christoffersen, Conditional Coverage)."""
    _validate_provenance(evidence_ids)
    spec = ArtifactSpec(
        artifact_type="backtest_statistical_summary",
        title="Comprehensive Out-of-Sample VaR Backtest Summary",
        test_id=test_id,
        evidence_ids=evidence_ids,
        parameters={
            "var_confidence": backtest.var_confidence,
            "test_significance": backtest.test_significance,
        },
    )
    semantic_payload: dict[str, Any] = {
        "pnl_source": backtest.pnl_source,
        "var_confidence": backtest.var_confidence,
        "test_significance": backtest.test_significance,
        "n_observations": backtest.n_observations,
        "n_exceptions": backtest.n_exceptions,
        "exception_rate": backtest.exception_rate,
        "expected_exceptions": backtest.expected_exceptions,
        "kupiec_lr": backtest.kupiec_lr,
        "kupiec_p_value": backtest.kupiec_p_value,
        "kupiec_rejected": backtest.kupiec_rejected,
        "christoffersen_lr": backtest.christoffersen_lr,
        "christoffersen_p_value": backtest.christoffersen_p_value,
        "christoffersen_rejected": backtest.christoffersen_rejected,
        "conditional_coverage_lr": backtest.conditional_coverage_lr,
        "conditional_coverage_p_value": backtest.conditional_coverage_p_value,
        "conditional_coverage_rejected": backtest.conditional_coverage_rejected,
        "data_fingerprint": backtest.data_fingerprint,
    }
    payload_hash = _hash_payload(semantic_payload)
    artifact_id = f"ART-TAIL-BACKTEST-{payload_hash[:8]}"

    file_path = None
    if output_dir:
        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        file_path = str(out_p / f"{artifact_id}.svg")
        cards = [
            (
                "Kupiec POF",
                f"LR={backtest.kupiec_lr:.3f} (p={backtest.kupiec_p_value:.4f})",
                "REJECT" if backtest.kupiec_rejected else "DO NOT REJECT",
            ),
            (
                "Christoffersen Ind",
                f"LR={backtest.christoffersen_lr:.3f} (p={backtest.christoffersen_p_value:.4f})",
                "REJECT" if backtest.christoffersen_rejected else "DO NOT REJECT",
            ),
            (
                "Conditional Coverage",
                f"LR={backtest.conditional_coverage_lr:.3f} (p={backtest.conditional_coverage_p_value:.4f})",
                "REJECT" if backtest.conditional_coverage_rejected else "DO NOT REJECT",
            ),
            (
                "Exceptions",
                f"{backtest.n_exceptions} / {backtest.n_observations:,}",
                f"Observed Rate: {backtest.exception_rate:.2%}",
            ),
            (
                "Significance Level",
                f"gamma_test = {backtest.test_significance:.2%}",
                "Test critical threshold",
            ),
            (
                "VaR Confidence",
                f"alpha_var = {backtest.var_confidence:.2%}",
                f"Expected Rate: {backtest.expected_probability:.2%}",
            ),
        ]
        svg_content = _generate_dashboard_svg(
            metrics=cards,
            title="Comprehensive Out-of-Sample VaR Backtest Summary",
            subtitle=f"VaR Confidence: {backtest.var_confidence:.1%} | Test Significance: {backtest.test_significance:.1%} | Observations: {backtest.n_observations:,}",
        )
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(svg_content)
        with open(str(out_p / f"{artifact_id}.json"), "w", encoding="utf-8") as f:
            json.dump(semantic_payload, f, indent=2)

    return ArtifactRecord(
        artifact_id=artifact_id,
        spec=spec,
        data_fingerprint=payload_hash,
        semantic_payload=semantic_payload,
        semantic_payload_hash=payload_hash,
        file_path=file_path,
        rendering_format="svg" if file_path else "json",
    )


def render_tail_comparison_artifact(
    comparison: TailModelComparisonResult,
    evidence_ids: tuple[str, ...],
    output_dir: str | Path | None = None,
    test_id: str = "traded_risk.var_es_comparison",
) -> ArtifactRecord:
    """Render multi-method tail model comparison table."""
    _validate_provenance(evidence_ids)
    spec = ArtifactSpec(
        artifact_type="tail_model_comparison",
        title="Multi-Method Tail Risk Model Comparison",
        test_id=test_id,
        evidence_ids=evidence_ids,
        parameters={"confidence": comparison.confidence, "models": list(comparison.models_compared)},
    )
    headers = ["Model", "VaR (Loss Mag)", "Expected Shortfall", "ES / VaR Ratio"]
    rows = []
    for m in comparison.models_compared:
        v = comparison.var_values.get(m, 0.0)
        e = comparison.es_values.get(m, 0.0)
        r = comparison.es_to_var_ratios.get(m, 1.0)
        rows.append([m.upper(), f"{v:.4f}", f"{e:.4f}", f"{r:.3f}x"])

    semantic_payload: dict[str, Any] = {
        "models_compared": list(comparison.models_compared),
        "confidence": comparison.confidence,
        "var_values": comparison.var_values,
        "es_values": comparison.es_values,
        "es_to_var_ratios": comparison.es_to_var_ratios,
        "data_fingerprint": comparison.data_fingerprint,
    }
    payload_hash = _hash_payload(semantic_payload)
    artifact_id = f"ART-TAIL-COMPARE-{payload_hash[:8]}"

    file_path = None
    if output_dir:
        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        file_path = str(out_p / f"{artifact_id}.svg")
        svg_content = _generate_table_svg(
            headers=headers,
            rows=rows,
            title="Multi-Method Tail Risk Model Comparison",
            subtitle=f"Confidence alpha = {comparison.confidence:.1%} | Comparative Analysis Across Estimators",
        )
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(svg_content)
        with open(str(out_p / f"{artifact_id}.json"), "w", encoding="utf-8") as f:
            json.dump(semantic_payload, f, indent=2)

    return ArtifactRecord(
        artifact_id=artifact_id,
        spec=spec,
        data_fingerprint=payload_hash,
        semantic_payload=semantic_payload,
        semantic_payload_hash=payload_hash,
        file_path=file_path,
        rendering_format="svg" if file_path else "json",
    )


def render_tail_risk_contribution_artifact(
    contrib: TailRiskContributionResult,
    evidence_ids: tuple[str, ...],
    output_dir: str | Path | None = None,
    test_id: str = "traded_risk.es_contribution",
) -> ArtifactRecord:
    """Render component asset risk contributions to VaR and Expected Shortfall."""
    _validate_provenance(evidence_ids)
    spec = ArtifactSpec(
        artifact_type="tail_risk_contribution",
        title=f"Component Tail Risk Contributions ({contrib.method.upper()})",
        test_id=test_id,
        evidence_ids=evidence_ids,
        parameters={"confidence": contrib.confidence, "method": contrib.method},
    )
    headers = ["Asset", "Component VaR", "% VaR", "Component ES", "% ES"]
    rows = []
    assets = list(contrib.component_es.keys())
    for a in assets:
        c_var = contrib.component_var.get(a, float("nan"))
        p_var = contrib.percentage_var_contributions.get(a, float("nan"))
        c_es = contrib.component_es.get(a, 0.0)
        p_es = contrib.percentage_es_contributions.get(a, 0.0)
        rows.append(
            [
                a,
                f"{c_var:.4f}" if math.isfinite(c_var) else "DEFERRED",
                f"{p_var:.1%}" if math.isfinite(p_var) else "DEFERRED",
                f"{c_es:.4f}",
                f"{p_es:.1%}",
            ]
        )

    semantic_payload: dict[str, Any] = {
        "method": contrib.method,
        "confidence": contrib.confidence,
        "portfolio_var": contrib.portfolio_var,
        "portfolio_es": contrib.portfolio_es,
        "component_var": contrib.component_var,
        "component_es": contrib.component_es,
        "percentage_var_contributions": contrib.percentage_var_contributions,
        "percentage_es_contributions": contrib.percentage_es_contributions,
        "var_reconciliation_error": contrib.var_reconciliation_error,
        "es_reconciliation_error": contrib.es_reconciliation_error,
        "data_fingerprint": contrib.data_fingerprint,
    }
    payload_hash = _hash_payload(semantic_payload)
    artifact_id = f"ART-TAIL-CONTRIB-{payload_hash[:8]}"

    file_path = None
    if output_dir:
        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        file_path = str(out_p / f"{artifact_id}.svg")
        svg_content = _generate_table_svg(
            headers=headers,
            rows=rows,
            title=f"Component Tail Risk Contributions ({contrib.method.upper()})",
            subtitle=f"Portfolio VaR: {contrib.portfolio_var:.4f} | Portfolio ES: {contrib.portfolio_es:.4f} | Reconciliation Error: {contrib.es_reconciliation_error:.2e}",
        )
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(svg_content)
        with open(str(out_p / f"{artifact_id}.json"), "w", encoding="utf-8") as f:
            json.dump(semantic_payload, f, indent=2)

    return ArtifactRecord(
        artifact_id=artifact_id,
        spec=spec,
        data_fingerprint=payload_hash,
        semantic_payload=semantic_payload,
        semantic_payload_hash=payload_hash,
        file_path=file_path,
        rendering_format="svg" if file_path else "json",
    )


# =========================================================================== #
# GATE 6: SCENARIO, STRESS & REVERSE-STRESS ARTIFACT GENERATORS
# =========================================================================== #


def render_scenario_pnl_waterfall_artifact(
    res: ScenarioResult,
    evidence_ids: tuple[str, ...],
    output_dir: str | Path | None = None,
) -> ArtifactRecord:
    """Render scenario P&L / return waterfall and summary dashboard."""
    spec = ArtifactSpec(
        artifact_type="scenario_pnl_waterfall",
        title=f"Scenario P&L Waterfall: {res.scenario_id}",
        test_id=f"scenario.{res.repricing_method.lower()}",
        evidence_ids=evidence_ids,
        parameters={"scenario_id": res.scenario_id, "repricing_method": res.repricing_method},
    )
    headers = ["Component", "Type", "Scenario Return", "P&L Contribution", "% Share"]
    rows = []

    # Asset or factor contributions
    if res.asset_contributions:
        for a, c in res.asset_contributions.items():
            share = (c / res.scenario_return) if abs(res.scenario_return) > 1e-8 else 0.0
            pnl_c = (res.portfolio_value * c) if res.portfolio_value is not None else c
            rows.append(
                [
                    a,
                    "Asset Leg",
                    f"{c:.4f}",
                    f"{pnl_c:,.2f}" if res.portfolio_value else f"{pnl_c:.4f}",
                    f"{share:.1%}",
                ]
            )
    elif res.factor_contributions:
        for f, c in res.factor_contributions.items():
            share = (c / res.scenario_return) if abs(res.scenario_return) > 1e-8 else 0.0
            pnl_c = (res.portfolio_value * c) if res.portfolio_value is not None else c
            rows.append(
                [
                    f,
                    "Factor Leg",
                    f"{c:.4f}",
                    f"{pnl_c:,.2f}" if res.portfolio_value else f"{pnl_c:.4f}",
                    f"{share:.1%}",
                ]
            )

    total_pnl_str = (
        f"{res.scenario_pnl:,.2f}" if res.scenario_pnl is not None else f"{res.scenario_return:.4f}"
    )
    rows.append(["TOTAL", "Portfolio", f"{res.scenario_return:.4f}", total_pnl_str, "100.0%"])

    semantic_payload: dict[str, Any] = {
        "scenario_id": res.scenario_id,
        "scenario_type": res.scenario_type,
        "repricing_method": res.repricing_method,
        "scenario_return": res.scenario_return,
        "scenario_loss": res.scenario_loss,
        "scenario_pnl": res.scenario_pnl,
        "scenario_monetary_loss": res.scenario_monetary_loss,
        "asset_contributions": res.asset_contributions,
        "factor_contributions": res.factor_contributions,
        "reconciliation_error": res.reconciliation_error,
        "data_fingerprint": res.data_fingerprint,
    }
    payload_hash = _hash_payload(semantic_payload)
    artifact_id = f"ART-SCEN-WATERFALL-{payload_hash[:8]}"

    file_path = None
    if output_dir:
        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        file_path = str(out_p / f"{artifact_id}.svg")
        svg_content = _generate_table_svg(
            headers=headers,
            rows=rows,
            title=f"Scenario P&L Waterfall: {res.scenario_id} ({res.repricing_method})",
            subtitle=f"Scenario Return: {res.scenario_return:.4f} | Canonical Loss: {res.scenario_loss:.4f} | Recon Error: {res.reconciliation_error:.2e}",
        )
        with open(file_path, "w", encoding="utf-8") as fp:
            fp.write(svg_content)
        with open(str(out_p / f"{artifact_id}.json"), "w", encoding="utf-8") as fp:
            json.dump(semantic_payload, fp, indent=2)

    return ArtifactRecord(
        artifact_id=artifact_id,
        spec=spec,
        data_fingerprint=payload_hash,
        semantic_payload=semantic_payload,
        semantic_payload_hash=payload_hash,
        file_path=file_path,
        rendering_format="svg" if file_path else "json",
    )


def render_scenario_asset_contribution_artifact(
    res: ScenarioResult,
    evidence_ids: tuple[str, ...],
    output_dir: str | Path | None = None,
) -> ArtifactRecord:
    """Render asset-level stress contribution breakdown table."""
    spec = ArtifactSpec(
        artifact_type="scenario_asset_contribution",
        title=f"Scenario Asset Contributions: {res.scenario_id}",
        test_id="scenario.linear_return",
        evidence_ids=evidence_ids,
        parameters={"scenario_id": res.scenario_id},
    )
    headers = ["Asset", "Contribution", "Percentage Share", "Cumulative Sum"]
    rows = []
    cum = 0.0
    for a, c in res.asset_contributions.items():
        cum += c
        share = (c / res.scenario_return) if abs(res.scenario_return) > 1e-8 else 0.0
        rows.append([a, f"{c:.4f}", f"{share:.1%}", f"{cum:.4f}"])

    semantic_payload = {
        "scenario_id": res.scenario_id,
        "asset_contributions": res.asset_contributions,
        "scenario_return": res.scenario_return,
        "scenario_loss": res.scenario_loss,
        "data_fingerprint": res.data_fingerprint,
    }
    payload_hash = _hash_payload(semantic_payload)
    artifact_id = f"ART-SCEN-ASSET-{payload_hash[:8]}"

    file_path = None
    if output_dir:
        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        file_path = str(out_p / f"{artifact_id}.svg")
        svg_content = _generate_table_svg(
            headers=headers,
            rows=rows,
            title=f"Asset Contributions: {res.scenario_id}",
            subtitle=f"Total Return: {res.scenario_return:.4f} | Total Assets: {len(res.asset_contributions)}",
        )
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(svg_content)
        with open(str(out_p / f"{artifact_id}.json"), "w", encoding="utf-8") as f:
            json.dump(semantic_payload, f, indent=2)

    return ArtifactRecord(
        artifact_id=artifact_id,
        spec=spec,
        data_fingerprint=payload_hash,
        semantic_payload=semantic_payload,
        semantic_payload_hash=payload_hash,
        file_path=file_path,
        rendering_format="svg" if file_path else "json",
    )


def render_scenario_factor_contribution_artifact(
    res: ScenarioResult,
    evidence_ids: tuple[str, ...],
    output_dir: str | Path | None = None,
) -> ArtifactRecord:
    """Render factor stress contribution breakdown table."""
    spec = ArtifactSpec(
        artifact_type="scenario_factor_contribution",
        title=f"Scenario Factor Contributions: {res.scenario_id}",
        test_id="scenario.factor_linear",
        evidence_ids=evidence_ids,
        parameters={"scenario_id": res.scenario_id},
    )
    headers = ["Factor", "Contribution", "Percentage Share", "Specific Residual"]
    rows = []
    for f, c in res.factor_contributions.items():
        share = (c / res.scenario_return) if abs(res.scenario_return) > 1e-8 else 0.0
        rows.append([f, f"{c:.4f}", f"{share:.1%}", f"{res.specific_contribution or 0.0:.4f}"])

    semantic_payload = {
        "scenario_id": res.scenario_id,
        "factor_contributions": res.factor_contributions,
        "specific_contribution": res.specific_contribution,
        "scenario_return": res.scenario_return,
        "scenario_loss": res.scenario_loss,
        "data_fingerprint": res.data_fingerprint,
    }
    payload_hash = _hash_payload(semantic_payload)
    artifact_id = f"ART-SCEN-FACTOR-{payload_hash[:8]}"

    file_path = None
    if output_dir:
        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        file_path = str(out_p / f"{artifact_id}.svg")
        svg_content = _generate_table_svg(
            headers=headers,
            rows=rows,
            title=f"Factor Risk Contributions: {res.scenario_id}",
            subtitle=f"Factor Stress Return: {res.scenario_return:.4f} | Specific Component: {res.specific_contribution or 0.0:.4f}",
        )
        with open(file_path, "w", encoding="utf-8") as fp:
            fp.write(svg_content)
        with open(str(out_p / f"{artifact_id}.json"), "w", encoding="utf-8") as fp:
            json.dump(semantic_payload, fp, indent=2)

    return ArtifactRecord(
        artifact_id=artifact_id,
        spec=spec,
        data_fingerprint=payload_hash,
        semantic_payload=semantic_payload,
        semantic_payload_hash=payload_hash,
        file_path=file_path,
        rendering_format="svg" if file_path else "json",
    )


def render_scenario_active_comparison_artifact(
    act_res: ActiveScenarioResult,
    evidence_ids: tuple[str, ...],
    output_dir: str | Path | None = None,
) -> ArtifactRecord:
    """Render portfolio vs benchmark active stress comparison table."""
    spec = ArtifactSpec(
        artifact_type="scenario_active_comparison",
        title=f"Active Scenario Comparison: {act_res.scenario_id}",
        test_id="scenario.active_stress",
        evidence_ids=evidence_ids,
        parameters={"scenario_id": act_res.scenario_id},
    )
    headers = ["Portfolio Metric", "Portfolio", "Benchmark", "Active (Port - Bmk)"]
    rows = [
        [
            "Scenario Return",
            f"{act_res.portfolio_return:.4f}",
            f"{act_res.benchmark_return:.4f}",
            f"{act_res.active_return:.4f}",
        ],
        [
            "Canonical Loss",
            f"{act_res.portfolio_loss:.4f}",
            f"{act_res.benchmark_loss:.4f}",
            f"{act_res.active_loss:.4f}",
        ],
        ["Reconciliation Error", "-", "-", f"{act_res.reconciliation_error:.2e}"],
    ]

    semantic_payload = {
        "scenario_id": act_res.scenario_id,
        "portfolio_return": act_res.portfolio_return,
        "benchmark_return": act_res.benchmark_return,
        "active_return": act_res.active_return,
        "portfolio_loss": act_res.portfolio_loss,
        "benchmark_loss": act_res.benchmark_loss,
        "active_loss": act_res.active_loss,
        "reconciliation_error": act_res.reconciliation_error,
        "data_fingerprint": act_res.data_fingerprint,
    }
    payload_hash = _hash_payload(semantic_payload)
    artifact_id = f"ART-SCEN-ACTIVE-{payload_hash[:8]}"

    file_path = None
    if output_dir:
        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        file_path = str(out_p / f"{artifact_id}.svg")
        svg_content = _generate_table_svg(
            headers=headers,
            rows=rows,
            title=f"Portfolio vs Benchmark Stress: {act_res.scenario_id}",
            subtitle=f"Active Return: {act_res.active_return:.4f} | Reconciliation: {act_res.reconciliation_error:.2e}",
        )
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(svg_content)
        with open(str(out_p / f"{artifact_id}.json"), "w", encoding="utf-8") as f:
            json.dump(semantic_payload, f, indent=2)

    return ArtifactRecord(
        artifact_id=artifact_id,
        spec=spec,
        data_fingerprint=payload_hash,
        semantic_payload=semantic_payload,
        semantic_payload_hash=payload_hash,
        file_path=file_path,
        rendering_format="svg" if file_path else "json",
    )


def render_scenario_group_heatmap_artifact(
    scenario_id: str,
    group_contributions: dict[str, float],
    partition_contract: str,
    evidence_ids: tuple[str, ...],
    output_dir: str | Path | None = None,
) -> ArtifactRecord:
    """Render group/sector scenario stress decomposition table."""
    spec = ArtifactSpec(
        artifact_type="scenario_group_heatmap",
        title=f"Group Stress Decomposition: {scenario_id}",
        test_id="scenario.group_decomposition",
        evidence_ids=evidence_ids,
        parameters={"scenario_id": scenario_id, "partition_contract": partition_contract},
    )
    headers = ["Analytical Group / Sector", "Contribution", "% Share", "Partition Contract"]
    total_g = sum(group_contributions.values())
    rows = []
    is_exhaustive = str(partition_contract).upper() in ("EXHAUSTIVE_PARTITION",)
    for g, c in group_contributions.items():
        share_str = f"{(c / total_g):.1%}" if (is_exhaustive and abs(total_g) > 1e-8) else "N/A (Overlapping)"
        rows.append([g, f"{c:.4f}", share_str, partition_contract])
    if is_exhaustive:
        rows.append(["TOTAL", f"{total_g:.4f}", "100.0%", partition_contract])
    else:
        rows.append(["NON-ADDITIVE SUM", f"{total_g:.4f}", "N/A (Overlapping View)", partition_contract])

    semantic_payload = {
        "scenario_id": scenario_id,
        "group_contributions": group_contributions,
        "partition_contract": partition_contract,
        "total_group_contribution": total_g,
    }
    payload_hash = _hash_payload(semantic_payload)
    artifact_id = f"ART-SCEN-GROUP-{payload_hash[:8]}"

    file_path = None
    if output_dir:
        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        file_path = str(out_p / f"{artifact_id}.svg")
        svg_content = _generate_table_svg(
            headers=headers,
            rows=rows,
            title=f"Group Stress Decomposition: {scenario_id}",
            subtitle=f"Partition Contract: {partition_contract} | Total Groups: {len(group_contributions)}",
        )
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(svg_content)
        with open(str(out_p / f"{artifact_id}.json"), "w", encoding="utf-8") as f:
            json.dump(semantic_payload, f, indent=2)

    return ArtifactRecord(
        artifact_id=artifact_id,
        spec=spec,
        data_fingerprint=payload_hash,
        semantic_payload=semantic_payload,
        semantic_payload_hash=payload_hash,
        file_path=file_path,
        rendering_format="svg" if file_path else "json",
    )


def render_scenario_set_ranking_artifact(
    set_res: ScenarioSetResult,
    evidence_ids: tuple[str, ...],
    output_dir: str | Path | None = None,
) -> ArtifactRecord:
    """Render multi-scenario comparative loss rankings table."""
    spec = ArtifactSpec(
        artifact_type="scenario_set_ranking",
        title="Multi-Scenario Loss Rankings",
        test_id="scenario.set_comparison",
        evidence_ids=evidence_ids,
        parameters={
            "ranking_metric": set_res.ranking_metric,
            "n_scenarios": len(set_res.scenarios_evaluated),
        },
    )
    headers = ["Rank", "Scenario ID", "Repricing Method", "Canonical Loss", "Return", "Classification"]
    rows = []
    for rank, sid in enumerate(set_res.loss_rankings, 1):
        loss_val = set_res.scenario_losses[sid]
        ret_val = set_res.scenario_returns[sid]
        meth = set_res.method_disclosures.get(sid, "LINEAR_RETURN")
        tag = (
            "WORST LOSS"
            if sid == set_res.worst_scenario_id
            else ("BEST RESULT" if sid == set_res.best_scenario_id else "INTERMEDIATE")
        )
        rows.append([str(rank), sid, meth, f"{loss_val:.4f}", f"{ret_val:.4f}", tag])

    semantic_payload = {
        "scenarios_evaluated": set_res.scenarios_evaluated,
        "loss_rankings": set_res.loss_rankings,
        "worst_scenario_id": set_res.worst_scenario_id,
        "best_scenario_id": set_res.best_scenario_id,
        "worst_scenario_loss": set_res.worst_scenario_loss,
        "best_scenario_loss": set_res.best_scenario_loss,
        "scenario_losses": set_res.scenario_losses,
        "scenario_returns": set_res.scenario_returns,
        "method_disclosures": set_res.method_disclosures,
        "data_fingerprint": set_res.data_fingerprint,
    }
    payload_hash = _hash_payload(semantic_payload)
    artifact_id = f"ART-SCEN-RANKING-{payload_hash[:8]}"

    file_path = None
    if output_dir:
        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        file_path = str(out_p / f"{artifact_id}.svg")
        svg_content = _generate_table_svg(
            headers=headers,
            rows=rows,
            title="Multi-Scenario Comparative Loss Rankings",
            subtitle=f"Worst: '{set_res.worst_scenario_id}' ({set_res.worst_scenario_loss:.4f}) | Best: '{set_res.best_scenario_id}' ({set_res.best_scenario_loss:.4f})",
        )
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(svg_content)
        with open(str(out_p / f"{artifact_id}.json"), "w", encoding="utf-8") as f:
            json.dump(semantic_payload, f, indent=2)

    return ArtifactRecord(
        artifact_id=artifact_id,
        spec=spec,
        data_fingerprint=payload_hash,
        semantic_payload=semantic_payload,
        semantic_payload_hash=payload_hash,
        file_path=file_path,
        rendering_format="svg" if file_path else "json",
    )


def render_scenario_sensitivity_curve_artifact(
    sens_res: ScenarioSensitivityResult,
    evidence_ids: tuple[str, ...],
    output_dir: str | Path | None = None,
) -> ArtifactRecord:
    """Render scenario sensitivity sweep response curve table."""
    spec = ArtifactSpec(
        artifact_type="scenario_sensitivity_curve",
        title=f"Scenario Sensitivity Curve: {sens_res.risk_factor_id}",
        test_id="scenario.sensitivity_grid",
        evidence_ids=evidence_ids,
        parameters={"risk_factor_id": sens_res.risk_factor_id, "n_points": len(sens_res.grid_points)},
    )
    headers = ["Multiplier", "Raw Shock", "Normalized Shock", "Portfolio Loss", "Portfolio Return"]
    rows = []
    for pt in sens_res.grid_points:
        rows.append(
            [
                f"x{pt.shock_multiplier:.2f}",
                f"{pt.raw_shock_value:.4f}",
                f"{pt.normalized_shock_value:.4f}",
                f"{pt.portfolio_loss:.4f}",
                f"{pt.portfolio_return:.4f}",
            ]
        )

    semantic_payload = {
        "risk_factor_id": sens_res.risk_factor_id,
        "base_loss": sens_res.base_loss,
        "max_loss": sens_res.max_loss,
        "min_loss": sens_res.min_loss,
        "grid_points": [
            {
                "multiplier": pt.shock_multiplier,
                "raw_shock": pt.raw_shock_value,
                "norm_shock": pt.normalized_shock_value,
                "loss": pt.portfolio_loss,
                "return": pt.portfolio_return,
            }
            for pt in sens_res.grid_points
        ],
        "data_fingerprint": sens_res.data_fingerprint,
    }
    payload_hash = _hash_payload(semantic_payload)
    artifact_id = f"ART-SCEN-SENSITIVITY-{payload_hash[:8]}"

    file_path = None
    if output_dir:
        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        file_path = str(out_p / f"{artifact_id}.svg")
        svg_content = _generate_table_svg(
            headers=headers,
            rows=rows,
            title=f"Sensitivity Response Curve: {sens_res.risk_factor_id}",
            subtitle=f"Base Loss: {sens_res.base_loss:.4f} | Max Loss: {sens_res.max_loss:.4f} | Min Loss: {sens_res.min_loss:.4f}",
        )
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(svg_content)
        with open(str(out_p / f"{artifact_id}.json"), "w", encoding="utf-8") as f:
            json.dump(semantic_payload, f, indent=2)

    return ArtifactRecord(
        artifact_id=artifact_id,
        spec=spec,
        data_fingerprint=payload_hash,
        semantic_payload=semantic_payload,
        semantic_payload_hash=payload_hash,
        file_path=file_path,
        rendering_format="svg" if file_path else "json",
    )


def render_reverse_stress_profile_artifact(
    rev_res: ReverseStressResult,
    evidence_ids: tuple[str, ...],
    output_dir: str | Path | None = None,
) -> ArtifactRecord:
    """Render reverse-stress minimum shock profile and optimization summary."""
    spec = ArtifactSpec(
        artifact_type="reverse_stress_profile",
        title=f"Minimum Shock Reverse Stress Profile ({rev_res.distance_norm})",
        test_id="scenario.reverse_stress",
        evidence_ids=evidence_ids,
        parameters={"target_loss": rev_res.target_loss, "distance_norm": rev_res.distance_norm},
    )
    headers = ["Risk Factor", "Required Shock", "Distance Contribution", "Status"]
    rows = []
    for rf, s in rev_res.shock_vector.items():
        rows.append([rf, f"{s:.4f}", f"{abs(s):.4f}", "SOLVED"])
    rows.append(["TARGET LOSS", f"{rev_res.target_loss:.4f}", "-", f"Achieved: {rev_res.achieved_loss:.4f}"])
    rows.append(
        [
            "NORM DISTANCE",
            f"{rev_res.distance:.4f}",
            "-",
            f"Norm: {rev_res.distance_norm} ({rev_res.solver_status})",
        ]
    )

    semantic_payload = {
        "target_loss": rev_res.target_loss,
        "achieved_loss": rev_res.achieved_loss,
        "loss_gap": rev_res.loss_gap,
        "distance": rev_res.distance,
        "distance_norm": rev_res.distance_norm,
        "shock_vector": rev_res.shock_vector,
        "solver_status": rev_res.solver_status,
        "converged": rev_res.converged,
        "is_closed_form": rev_res.is_closed_form,
        "data_fingerprint": rev_res.data_fingerprint,
    }
    payload_hash = _hash_payload(semantic_payload)
    artifact_id = f"ART-REV-STRESS-{payload_hash[:8]}"

    file_path = None
    if output_dir:
        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        file_path = str(out_p / f"{artifact_id}.svg")
        svg_content = _generate_table_svg(
            headers=headers,
            rows=rows,
            title=f"Minimum Shock Reverse Stress ({rev_res.distance_norm})",
            subtitle=f"Target Loss: {rev_res.target_loss:.4f} | Achieved: {rev_res.achieved_loss:.4f} | Status: {rev_res.solver_status}",
        )
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(svg_content)
        with open(str(out_p / f"{artifact_id}.json"), "w", encoding="utf-8") as f:
            json.dump(semantic_payload, f, indent=2)

    return ArtifactRecord(
        artifact_id=artifact_id,
        spec=spec,
        data_fingerprint=payload_hash,
        semantic_payload=semantic_payload,
        semantic_payload_hash=payload_hash,
        file_path=file_path,
        rendering_format="svg" if file_path else "json",
    )
