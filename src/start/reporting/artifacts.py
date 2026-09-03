"""Artifact discovery registry (v2.1.1 Section N).

Whenever any artifact is generated, it is registered here with its name, type,
and location, and can be announced immediately ("Generated: sensitivity.csv").
The registry becomes the dashboard's Artifact Catalog (Section Q) and gives the
user a single place to discover every output — no hidden artifacts.

Cross-platform: uses pathlib only; no OS-specific calls.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from start.portfolio.artifacts import ArtifactRecord, ArtifactSpec

_TYPE_BY_EXT = {
    ".csv": "table (CSV)",
    ".tsv": "table (TSV)",
    ".json": "data (JSON)",
    ".md": "report (Markdown)",
    ".html": "dashboard (HTML)",
    ".png": "figure (PNG)",
    ".jpg": "figure (JPEG)",
    ".svg": "figure (SVG)",
    ".parquet": "table (Parquet)",
    ".txt": "text",
    ".pdf": "document (PDF)",
}

__all__ = [
    "Artifact",
    "ArtifactRegistry",
    "ArtifactSpec",
    "ArtifactRecord",
    "render_artifact_catalog_markdown",
]


@dataclass
class Artifact:
    name: str
    path: str
    artifact_type: str
    category: str = "general"  # data | model | explainability | sensitivity | governance | ...
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "path": self.path,
            "type": self.artifact_type,
            "category": self.category,
            "description": self.description,
        }


@dataclass
class ArtifactRegistry:
    artifacts: list[Artifact] = field(default_factory=list)
    announce: Callable[[str], None] | None = None

    def register(
        self, path: str, *, name: str | None = None, category: str = "general", description: str = "",
        artifact_type: str | None = None,
    ) -> Artifact:
        p = Path(path)
        atype = artifact_type or _TYPE_BY_EXT.get(p.suffix.lower(), p.suffix.lstrip(".") or "file")
        artifact = Artifact(
            name=name or p.name, path=str(p), artifact_type=atype,
            category=category, description=description,
        )
        self.artifacts.append(artifact)
        if self.announce:
            self.announce(f"  Generated: {artifact.name}  [{artifact.artifact_type}]  -> {artifact.path}")
        return artifact

    def register_many(self, paths: list[str], *, category: str = "general") -> list[Artifact]:
        return [self.register(p, category=category) for p in paths]

    def to_list(self) -> list[dict[str, Any]]:
        return [a.to_dict() for a in self.artifacts]

    def by_category(self) -> dict[str, list[dict[str, Any]]]:
        out: dict[str, list[dict[str, Any]]] = {}
        for a in self.artifacts:
            out.setdefault(a.category, []).append(a.to_dict())
        return out

    def names(self) -> list[str]:
        return [a.name for a in self.artifacts]


def render_artifact_catalog_markdown(registry: ArtifactRegistry) -> str:
    if not registry.artifacts:
        return "### Artifact catalog\n\n_No artifacts generated._\n"
    lines = ["### Artifact catalog", "", "| Artifact | Type | Category | Location |",
             "| --- | --- | --- | --- |"]
    for a in registry.artifacts:
        lines.append(f"| {a.name} | {a.artifact_type} | {a.category} | {a.path} |")
    return "\n".join(lines) + "\n"
