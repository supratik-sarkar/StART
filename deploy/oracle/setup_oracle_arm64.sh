#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# StART v4.5 — Automated Oracle Linux ARM64 Deployment Setup Script
# Target: Oracle Cloud Infrastructure Always Free (VM.Standard.A1.Flex)
# Specs: 2 OCPU / 12 GB RAM / Oracle Linux 9 ARM64 (aarch64)
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

echo "======================================================================"
echo "StART v4.5 — Oracle Linux ARM64 Setup & Provisioning"
echo "======================================================================"

# 1. Update OS packages
sudo dnf update -y
sudo dnf install -y python3.12 python3.12-devel python3.12-pip git curl gcc gcc-c++ make tar gzip libffi-devel openssl-devel

# 2. Install verified OPA ARM64 binary
OPA_VERSION="v0.68.0"
OPA_URL="https://openpolicyagent.org/downloads/${OPA_VERSION}/opa_linux_arm64_static"
echo "Downloading official OPA ARM64 binary (${OPA_VERSION})..."
curl -L -o /tmp/opa "${OPA_URL}"
chmod 755 /tmp/opa
sudo mv /tmp/opa /usr/local/bin/opa
/usr/local/bin/opa version

# 3. Create deployment directory
APP_DIR="/opt/start"
sudo mkdir -p "${APP_DIR}"
sudo chown -R "${USER}:${USER}" "${APP_DIR}"

# 4. Clone / Sync repository
if [ ! -d "${APP_DIR}/.git" ]; then
    git clone https://github.com/start-project/start.git "${APP_DIR}"
fi

cd "${APP_DIR}"

# 5. Build Python 3.12 virtual environment
python3.12 -m venv .venv-start
source .venv-start/bin/activate
pip install --upgrade pip
pip install -e ".[all]"
pip install uvicorn[standard] sse-starlette

# 6. Verify ARM64 dependency integrity
python scripts/verify_arm64_dependencies.py

# 7. Install & enable systemd service
sudo cp deploy/oracle/start_web.service /etc/systemd/system/start_web.service
sudo systemctl daemon-reload
sudo systemctl enable start_web.service
sudo systemctl restart start_web.service

echo "======================================================================"
echo "StART v4.5 deployed successfully on Oracle Linux ARM64!"
echo "Status check: sudo systemctl status start_web.service"
echo "======================================================================"
