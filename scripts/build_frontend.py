#!/usr/bin/env python3
"""Deterministic Production Frontend Build Script for StART.

Reads committed non-secret public frontend configuration from deploy/cloudflare/public_frontend_config.json,
exports required public environment variables (e.g. VITE_TURNSTILE_SITE_KEY),
and executes a reproducible production frontend build.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = ROOT / "webapp"
CONFIG_PATH = ROOT / "deploy" / "cloudflare" / "public_frontend_config.json"


def build_frontend() -> None:
    print("=== StART Deterministic Production Frontend Build ===")
    
    if not CONFIG_PATH.exists():
        print(f"ERROR: Public frontend configuration missing at: {CONFIG_PATH}", file=sys.stderr)
        sys.exit(1)
        
    try:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            config = json.load(f)
    except Exception as e:
        print(f"ERROR: Failed to parse public frontend configuration: {e}", file=sys.stderr)
        sys.exit(1)
        
    site_key = os.environ.get("VITE_TURNSTILE_SITE_KEY") or config.get("turnstile_site_key")
    if not site_key:
        print("ERROR: 'turnstile_site_key' missing from public configuration and environment", file=sys.stderr)
        sys.exit(1)
        
    print(f"Public Turnstile Site Key configured: {site_key[:6]}...{site_key[-4:]}")
    
    env = os.environ.copy()
    env["VITE_TURNSTILE_SITE_KEY"] = site_key
    
    model_base = os.environ.get("VITE_START_MODEL_BASE") or config.get("model_base_url")
    if model_base:
        print(f"Public Model Base URL configured: {model_base}")
        env["VITE_START_MODEL_BASE"] = model_base
    
    # 1. Install dependencies if needed / ensure node_modules exists
    if not (WEB_DIR / "node_modules").exists():
        print("Installing frontend dependencies via npm ci...")
        subprocess.run(["npm", "ci"], cwd=WEB_DIR, check=True, env=env)
    
    # 2. Execute production typecheck and build
    print("Executing production frontend build (tsc && vite build)...")
    subprocess.run(["npm", "run", "build"], cwd=WEB_DIR, check=True, env=env)
    
    dist_dir = WEB_DIR / "dist"
    if not dist_dir.exists() or not (dist_dir / "index.html").exists():
        print("ERROR: Production frontend build failed — dist/index.html not generated", file=sys.stderr)
        sys.exit(1)
        
    print(f"Production frontend build completed successfully -> {dist_dir}")


if __name__ == "__main__":
    build_frontend()
