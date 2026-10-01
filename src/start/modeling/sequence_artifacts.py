"""Current-run scientific artifacts for rank-3 sequence classification.

Renderers consume outputs already produced by ``sequence_saliency`` and
``sequence_robustness``.  They do not call a model or recompute attribution.
"""

from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path
from typing import Any


def render_temporal_contract_artifact(
    *,
    n_sequences: int,
    timesteps: int,
    feature_names: list[str],
    run_id: str,
    model_id: str,
    output_dir: Path,
) -> dict[str, Any]:
    payload = {
        "run_id": run_id,
        "model_id": model_id,
        "input_shape": [n_sequences, timesteps, len(feature_names)],
        "n_sequences": n_sequences,
        "timesteps": timesteps,
        "feature_names": feature_names,
        "rank": 3,
        "task": "binary_sequence_classification",
        "forecasting": False,
    }
    rows = [
        ("N", f"{n_sequences:,} sequences"),
        ("T", f"{timesteps} timesteps"),
        ("F", f"{len(feature_names)} features"),
        ("Input", f"({n_sequences}, {timesteps}, {len(feature_names)}) rank-3"),
        ("Task", "binary sequence classification"),
        ("Semantics", "independent sequences; not forecasting"),
        ("Sample axis", "not chronological future periods"),
    ]
    return _write_card_artifact(
        payload=payload,
        rows=rows,
        title="Temporal Sequence Input Contract",
        prefix="TEMPORAL-CONTRACT",
        output_dir=output_dir,
    )


def render_input_gradient_saliency_artifact(
    *,
    saliency: dict[str, Any],
    feature_names: list[str],
    run_id: str,
    model_id: str,
    output_dir: Path,
) -> dict[str, Any]:
    if saliency.get("algorithm") != "INPUT_GRADIENT":
        raise ValueError("temporal saliency renderer only accepts INPUT_GRADIENT output")
    matrix = saliency.get("per_timestep_feature")
    if not isinstance(matrix, list) or not matrix:
        raise ValueError("INPUT_GRADIENT output lacks per_timestep_feature values")
    if any(len(row) != len(feature_names) for row in matrix):
        raise ValueError("temporal saliency matrix does not match feature names")
    payload = {
        **saliency,
        "run_id": run_id,
        "model_id": model_id,
        "feature_names": feature_names,
        "signed": False,
        "units": "absolute input gradient magnitude",
        "limitation": "Input-gradient magnitude is local sensitivity; it is not causal attribution.",
    }
    digest = _payload_hash(payload)
    output_dir.mkdir(parents=True, exist_ok=True)
    svg_path = output_dir / f"ART-TEMPORAL-SALIENCY-{digest[:8]}.svg"
    json_path = svg_path.with_suffix(".json")
    svg_path.write_text(
        _heatmap_svg(
            matrix=matrix,
            row_labels=[f"t-{len(matrix) - 1 - i}" for i in range(len(matrix))],
            col_labels=feature_names,
            title="Temporal Input-Gradient Saliency",
            subtitle=(
                f"Run {run_id} | Model {model_id} | target: {saliency.get('attributed_output')} | "
                f"mean absolute gradient, n={saliency.get('n_samples')}"
            ),
        ),
        encoding="utf-8",
    )
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return _entry(
        artifact_id=f"ART-TEMPORAL-SALIENCY-{digest[:8]}",
        title="Temporal Input-Gradient Saliency",
        test_id="deep_learning.explainability_diagnostics",
        run_id=run_id,
        file_path=svg_path,
        companion=json_path,
        payload_hash=digest,
    )


def render_temporal_robustness_artifact(
    *,
    robustness: dict[str, Any],
    run_id: str,
    model_id: str,
    output_dir: Path,
) -> dict[str, Any]:
    baseline_auc = robustness.get("baseline_auc")
    if not isinstance(baseline_auc, (int, float)):
        raise ValueError("temporal robustness output lacks a numeric baseline_auc")
    payload = {
        **robustness,
        "run_id": run_id,
        "model_id": model_id,
        "metric": "ROC-AUC drift from unperturbed baseline",
        "limitation": "Perturbation stability does not establish model correctness.",
    }
    digest = _payload_hash(payload)
    output_dir.mkdir(parents=True, exist_ok=True)
    svg_path = output_dir / f"ART-TEMPORAL-ROBUSTNESS-{digest[:8]}.svg"
    json_path = svg_path.with_suffix(".json")
    categories = [f"noise {row['noise']:.1%}" for row in robustness.get("noise", [])]
    values = [float(row["drift"]) for row in robustness.get("noise", [])]
    categories += [f"jitter ±{row['max_shift']}" for row in robustness.get("time_jitter", [])]
    values += [float(row["drift"]) for row in robustness.get("time_jitter", [])]
    svg_path.write_text(
        _bars_svg(
            categories,
            values,
            title="Temporal Robustness Perturbation Profile",
            subtitle=(
                f"Run {run_id} | Model {model_id} | baseline ROC-AUC "
                f"{float(baseline_auc):.4f} | stability ≠ correctness"
            ),
        ),
        encoding="utf-8",
    )
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return _entry(
        artifact_id=f"ART-TEMPORAL-ROBUSTNESS-{digest[:8]}",
        title="Temporal Robustness Perturbation Profile",
        test_id="deep_learning.robustness_diagnostics",
        run_id=run_id,
        file_path=svg_path,
        companion=json_path,
        payload_hash=digest,
    )


def _write_card_artifact(
    *, payload: dict[str, Any], rows: list[tuple[str, str]], title: str, prefix: str, output_dir: Path
) -> dict[str, Any]:
    digest = _payload_hash(payload)
    output_dir.mkdir(parents=True, exist_ok=True)
    svg_path = output_dir / f"ART-{prefix}-{digest[:8]}.svg"
    json_path = svg_path.with_suffix(".json")
    width, height = 700, 100 + len(rows) * 48
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        f'<text x="24" y="35" font-family="sans-serif" font-size="17" font-weight="bold" fill="#0f172a">{html.escape(title)}</text>',
        f'<text x="24" y="57" font-family="sans-serif" font-size="11" fill="#64748b">Run {html.escape(str(payload["run_id"]))} | Model {html.escape(str(payload["model_id"]))}</text>',
    ]
    for index, (key, value) in enumerate(rows):
        y = 92 + index * 48
        parts.extend(
            [
                f'<rect x="24" y="{y - 24}" width="652" height="38" rx="5" fill="#f1f5f9"/>',
                f'<text x="40" y="{y}" font-family="sans-serif" font-size="12" font-weight="bold" fill="#334155">{html.escape(key)}</text>',
                f'<text x="180" y="{y}" font-family="monospace" font-size="12" fill="#0f172a">{html.escape(value)}</text>',
            ]
        )
    parts.append("</svg>")
    svg_path.write_text("\n".join(parts), encoding="utf-8")
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return _entry(
        artifact_id=f"ART-{prefix}-{digest[:8]}",
        title=title,
        test_id="deep_learning.sequence_contract",
        run_id=str(payload["run_id"]),
        file_path=svg_path,
        companion=json_path,
        payload_hash=digest,
    )


def _heatmap_svg(
    *, matrix: list[list[float]], row_labels: list[str], col_labels: list[str], title: str, subtitle: str
) -> str:
    rows, cols = len(matrix), len(col_labels)
    cell_w, cell_h = max(42, 540 // max(cols, 1)), max(16, min(30, 520 // max(rows, 1)))
    width, height = 150 + cols * cell_w, 125 + rows * cell_h
    vmax = max((float(v) for row in matrix for v in row), default=1.0) or 1.0
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        f'<text x="20" y="28" font-family="sans-serif" font-size="16" font-weight="bold" fill="#0f172a">{html.escape(title)}</text>',
        f'<text x="20" y="48" font-family="sans-serif" font-size="10" fill="#64748b">{html.escape(subtitle)}</text>',
    ]
    for j, label in enumerate(col_labels):
        parts.append(
            f'<text x="{130 + j * cell_w + cell_w / 2}" y="72" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#334155">{html.escape(label)}</text>'
        )
    for i, row in enumerate(matrix):
        y = 82 + i * cell_h
        parts.append(
            f'<text x="118" y="{y + cell_h * .68}" text-anchor="end" font-family="monospace" font-size="9" fill="#475569">{html.escape(row_labels[i])}</text>'
        )
        for j, value in enumerate(row):
            intensity = min(1.0, max(0.0, float(value) / vmax))
            blue = int(245 - intensity * 165)
            color = f"rgb({blue},{blue + 5},{255})"
            parts.append(
                f'<rect x="{130 + j * cell_w}" y="{y}" width="{cell_w - 1}" height="{cell_h - 1}" fill="{color}"/>'
            )
    parts.append("</svg>")
    return "\n".join(parts)


def _bars_svg(categories: list[str], values: list[float], *, title: str, subtitle: str) -> str:
    width, height = 760, max(280, 110 + len(categories) * 36)
    zero_x, chart_w = 385.0, 300.0
    scale = max((abs(v) for v in values), default=1.0) or 1.0
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        f'<text x="22" y="30" font-family="sans-serif" font-size="16" font-weight="bold" fill="#0f172a">{html.escape(title)}</text>',
        f'<text x="22" y="50" font-family="sans-serif" font-size="10" fill="#64748b">{html.escape(subtitle)}</text>',
        f'<line x1="{zero_x}" y1="72" x2="{zero_x}" y2="{height - 25}" stroke="#94a3b8"/>',
    ]
    for i, (category, value) in enumerate(zip(categories, values, strict=False)):
        y = 82 + i * 36
        bar_w = abs(value) / scale * chart_w
        x = zero_x if value >= 0 else zero_x - bar_w
        color = "#16a34a" if value >= 0 else "#dc2626"
        parts.extend(
            [
                f'<text x="365" y="{y + 15}" text-anchor="end" font-family="sans-serif" font-size="11" fill="#334155">{html.escape(category)}</text>',
                f'<rect x="{x}" y="{y}" width="{max(1.0, bar_w)}" height="20" rx="3" fill="{color}"/>',
                f'<text x="{(x + bar_w + 6) if value >= 0 else (x - 6)}" y="{y + 15}" text-anchor="{"start" if value >= 0 else "end"}" font-family="monospace" font-size="10" fill="{color}">{value:+.4f}</text>',
            ]
        )
    parts.append("</svg>")
    return "\n".join(parts)


def _payload_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def _entry(
    *, artifact_id: str, title: str, test_id: str, run_id: str, file_path: Path,
    companion: Path, payload_hash: str
) -> dict[str, Any]:
    return {
        "artifact_id": artifact_id,
        "artifact_type": "scientific_svg",
        "title": title,
        "test_id": test_id,
        "evidence_ids": [],
        "run_id": run_id,
        "file_path": str(file_path.resolve()),
        "semantic_companion": str(companion.resolve()),
        "semantic_payload_hash": payload_hash,
        "rendering_format": "svg",
    }


__all__ = [
    "render_input_gradient_saliency_artifact",
    "render_temporal_contract_artifact",
    "render_temporal_robustness_artifact",
]
