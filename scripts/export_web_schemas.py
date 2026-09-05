#!/usr/bin/env python3
"""Export OpenAPI JSON and TypeScript Definitions from Authoritative Pydantic Schemas.

Enforces the single-source-of-truth contract between Python backend and React/TypeScript frontend.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
TS_OUT_PATH = ROOT / "webapp" / "src" / "types" / "start_schema.d.ts"
OPENAPI_OUT_PATH = ROOT / "src" / "start" / "web" / "openapi.json"


def python_type_to_ts(prop: Any, required: bool = True) -> str:
    """Map OpenAPI JSON Schema property to TypeScript type."""
    if not isinstance(prop, dict):
        return "any"

    if "enum" in prop:
        return " | ".join(json.dumps(v) for v in prop["enum"])

    p_type = prop.get("type")
    if p_type == "string":
        return "string"
    elif p_type in ("integer", "number"):
        return "number"
    elif p_type == "boolean":
        return "boolean"
    elif p_type == "array":
        items = prop.get("items", {})
        item_type = python_type_to_ts(items) if items else "any"
        return f"{item_type}[]"
    elif p_type == "object":
        add_props = prop.get("additionalProperties")
        if isinstance(add_props, dict):
            val_type = python_type_to_ts(add_props)
            return f"Record<string, {val_type}>"
        return "Record<string, any>"
    elif "anyOf" in prop:
        types = [
            python_type_to_ts(t) for t in prop["anyOf"] if isinstance(t, dict) and t.get("type") != "null"
        ]
        if any(isinstance(t, dict) and t.get("type") == "null" for t in prop["anyOf"]):
            types.append("null")
        return " | ".join(types) if types else "any"
    elif "$ref" in prop:
        ref_name = prop["$ref"].split("/")[-1]
        return ref_name

    return "any"


def generate_typescript_definitions() -> str:
    """Generate clean TypeScript interfaces from start.web.schemas Pydantic models."""
    from start.web.schemas import (
        APIResponseEnvelope,
        HydratedFindingView,
        LogicalArtifactMetadata,
        MetricRowView,
        PresentationBlockView,
        QualitativeFinding,
        ReviewerHydrationResponse,
        ReviewPresentationExport,
        RunRequest,
        RunStatusResponse,
        SSEEnvelope,
        SystemInfo,
        WebReviewerSubmission,
    )

    models = [
        SystemInfo,
        APIResponseEnvelope,
        SSEEnvelope,
        RunRequest,
        RunStatusResponse,
        MetricRowView,
        PresentationBlockView,
        ReviewPresentationExport,
        LogicalArtifactMetadata,
        QualitativeFinding,
        WebReviewerSubmission,
        HydratedFindingView,
        ReviewerHydrationResponse,
    ]

    ts_lines = [
        "// ─────────────────────────────────────────────────────────────────────────────",
        "// AUTOMATICALLY GENERATED FROM start.web.schemas (PYTHON PYDANTIC SOURCE)",
        "// DO NOT EDIT MANUALLY — Run: python scripts/export_web_schemas.py",
        "// ─────────────────────────────────────────────────────────────────────────────",
        "",
    ]

    for model in models:
        schema = model.model_json_schema()
        name = schema.get("title", model.__name__)
        properties = schema.get("properties", {})
        required_fields = set(schema.get("required", []))

        ts_lines.append(f"export interface {name} {{")
        for prop_name, prop_def in properties.items():
            is_req = prop_name in required_fields
            optional_marker = "" if is_req else "?"
            ts_type = python_type_to_ts(prop_def, is_req)
            ts_lines.append(f"  {prop_name}{optional_marker}: {ts_type};")
        ts_lines.append("}")
        ts_lines.append("")

    return "\n".join(ts_lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export StART v4.5 Web Schemas & TypeScript Types")
    parser.add_argument("--check", action="store_true", help="Verify TypeScript definitions are up to date")
    args = parser.parse_args()

    ts_content = generate_typescript_definitions()

    if args.check:
        if not TS_OUT_PATH.exists():
            print(f"FAILED: {TS_OUT_PATH} does not exist.")
            sys.exit(1)
        existing = TS_OUT_PATH.read_text(encoding="utf-8")
        if existing != ts_content:
            print("FAILED: TypeScript schema definitions are out of sync with Pydantic source.")
            sys.exit(1)
        print("SUCCESS: TypeScript schema definitions are in sync with Pydantic source.")
        sys.exit(0)

    TS_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    TS_OUT_PATH.write_text(ts_content, encoding="utf-8")
    print(f"Exported TypeScript schema definitions to: {TS_OUT_PATH}")


if __name__ == "__main__":
    main()
