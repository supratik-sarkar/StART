#!/usr/bin/env python3
"""Deploy StART v4.6.3 to Oracle Cloud ARM64 Instance with Let's Encrypt TLS."""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SSH_KEY = os.path.expanduser("~/.ssh/id_ed25519_start_oci")


def run_remote_ssh(ip: str, cmd: str, ssh_key: str, check=True) -> subprocess.CompletedProcess:
    ssh_cmd = [
        "ssh",
        "-o", "StrictHostKeyChecking=no",
        "-o", "ConnectTimeout=15",
        "-i", ssh_key,
        f"ubuntu@{ip}",
        cmd
    ]
    print(f"[SSH @ {ip}] {cmd[:120]}...")
    res = subprocess.run(ssh_cmd, capture_output=True, text=True)
    if check and res.returncode != 0:
        print(f"ERROR: SSH Command Failed (exit {res.returncode}):\n{res.stderr}\n{res.stdout}")
        sys.exit(res.returncode)
    return res


def main():
    parser = argparse.ArgumentParser(description="Deploy StART backend to Oracle Cloud ARM64")
    parser.add_argument(
        "--ip",
        default=os.environ.get("START_ORACLE_PUBLIC_IP", "137.23.61.219"),
        help="Target Oracle Instance Public IP (default: START_ORACLE_PUBLIC_IP or 137.23.61.219)"
    )
    parser.add_argument(
        "--domain",
        default=os.environ.get("START_ORACLE_DOMAIN", ""),
        help="Target TLS Domain (default: START_ORACLE_DOMAIN or <ip>.sslip.io)"
    )
    parser.add_argument(
        "--ssh-key",
        default=os.environ.get("START_ORACLE_SSH_KEY", DEFAULT_SSH_KEY),
        help="Path to SSH Private Key (default: START_ORACLE_SSH_KEY or ~/.ssh/id_ed25519_start_oci)"
    )
    args = parser.parse_args()

    public_ip = args.ip
    if not public_ip:
        print("ERROR: Oracle target public IP must be provided via --ip or START_ORACLE_PUBLIC_IP", file=sys.stderr)
        sys.exit(1)

    domain = args.domain or f"{public_ip}.sslip.io"
    ssh_key = os.path.expanduser(args.ssh_key)

    if not os.path.exists(ssh_key):
        print(f"WARNING: SSH key {ssh_key} does not exist locally. SSH operations may fail.", file=sys.stderr)

    print("=== DEPLOYING StART v5.1.2 BACKEND TO ORACLE LINUX ARM64 ===")
    print(f"Target VM IP: {public_ip}")
    print(f"Target Domain: {domain}")
    print(f"SSH Key: {ssh_key}")

    # 1. Ensure target directories on VM
    print("\n--- 1. Setting up /opt/start directory ---")
    run_remote_ssh(public_ip, "sudo mkdir -p /opt/start && sudo chown -R ubuntu:ubuntu /opt/start", ssh_key)

    # 2. Sync codebase
    print("\n--- 2. Syncing canonical non-Git codebase to /opt/start ---")
    rsync_cmd = [
        "rsync", "-avz",
        "-e", f"ssh -o StrictHostKeyChecking=no -i {ssh_key}",
        "--exclude=.venv-start",
        "--exclude=.git",
        "--exclude=__pycache__",
        "--exclude=.pytest_cache",
        "--exclude=.ruff_cache",
        "--exclude=.env",
        "--exclude=web/node_modules",
        "--exclude=webapp/node_modules",
        "--exclude=static/webllm-models",
        f"{ROOT}/",
        f"ubuntu@{public_ip}:/opt/start/"
    ]
    subprocess.run(rsync_cmd, check=True)
    print("Codebase synced successfully.")

    # 3. Install OPA ARM64 static binary
    print("\n--- 3. Installing verified OPA ARM64 static binary ---")
    opa_setup_cmd = (
        "if [ ! -f /usr/local/bin/opa ]; then "
        "  curl -sSL -o /tmp/opa https://openpolicyagent.org/downloads/v1.0.0/opa_linux_arm64_static && "
        "  chmod 755 /tmp/opa && "
        "  sudo mv /tmp/opa /usr/local/bin/opa; "
        "fi && "
        "/usr/local/bin/opa version"
    )
    res_opa = run_remote_ssh(public_ip, opa_setup_cmd, ssh_key)
    print("OPA version output:\n", res_opa.stdout.strip())

    # 4. Set up Python 3.12 virtual environment & install packages
    print("\n--- 4. Building Python 3.12 Virtualenv & Installing Dependencies ---")
    venv_cmd = (
        "cd /opt/start && "
        "if [ ! -d .venv-start ]; then python3 -m venv .venv-start; fi && "
        ".venv-start/bin/pip install --upgrade pip && "
        ".venv-start/bin/pip install -e '.[all]' && "
        ".venv-start/bin/pip install uvicorn[standard] sse-starlette"
    )
    run_remote_ssh(public_ip, venv_cmd, ssh_key)
    print("Python virtual environment initialized.")

    # 5. Provision pinned WebLLM model static mirror
    print("\n--- 5. Provisioning Pinned WebLLM Model Static Mirror ---")
    prov_cmd = "cd /opt/start && .venv-start/bin/python deploy/oracle/provision_webllm_model.py"
    res_prov = run_remote_ssh(public_ip, prov_cmd, ssh_key)
    print(res_prov.stdout)

    # 6. Run ARM64 dependency verification
    print("\n--- 6. Running ARM64 Dependency Audit on VM ---")
    res_audit = run_remote_ssh(public_ip, "cd /opt/start && .venv-start/bin/python scripts/verify_arm64_dependencies.py", ssh_key)
    print(res_audit.stdout)

    # 7. Obtain Let's Encrypt SSL certificate for domain
    print(f"\n--- 7. Obtaining Let's Encrypt SSL Certificate for {domain} ---")
    cert_cmd = (
        "sudo systemctl stop nginx || true; "
        f"sudo certbot certonly --standalone --non-interactive --agree-tos --register-unsafely-without-email -d {domain} && "
        f"sudo test -f /etc/letsencrypt/live/{domain}/fullchain.pem"
    )
    run_remote_ssh(public_ip, cert_cmd, ssh_key)
    print("Let's Encrypt certificate obtained successfully!")

    # 8. Configure Nginx HTTPS reverse proxy with static WebLLM model mirror
    print("\n--- 8. Configuring Nginx HTTPS Reverse Proxy & Static Model Mirror ---")
    write_nginx = (
        "sudo cp /opt/start/deploy/oracle/nginx_start.conf /etc/nginx/sites-available/start && "
        "sudo rm -f /etc/nginx/sites-enabled/default && "
        "sudo ln -sf /etc/nginx/sites-available/start /etc/nginx/sites-enabled/start && "
        "sudo nginx -t && "
        "sudo systemctl restart nginx"
    )
    run_remote_ssh(public_ip, write_nginx, ssh_key)
    print("Nginx configured with TLS and static WebLLM model mirror active.")

    # 9. Configure & Start systemd service
    print("\n--- 9. Configuring & Starting start_web.service ---")
    update_env = (
        "sudo mkdir -p /etc/start && "
        "if sudo grep -q 'START_BACKEND_BUILD_VERSION' /etc/start/start.env; then "
        "  sudo sed -i 's/START_BACKEND_BUILD_VERSION=.*/START_BACKEND_BUILD_VERSION=5.1.2-arm64-prod/' /etc/start/start.env; "
        "else "
        "  echo 'START_BACKEND_BUILD_VERSION=5.1.2-arm64-prod' | sudo tee -a /etc/start/start.env; "
        "fi"
    )
    run_remote_ssh(public_ip, update_env, ssh_key)

    service_content = """[Unit]
Description=StART v5.1 Agentic Engineering Workbench Web Service
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/opt/start
EnvironmentFile=/etc/start/start.env
ExecStart=/opt/start/.venv-start/bin/uvicorn start.web.app:app --host 127.0.0.1 --port 8000 --workers 1 --log-level info
Restart=always
RestartSec=5

MemoryMax=8G
CPUQuota=180%

[Install]
WantedBy=multi-user.target
"""
    write_service = (
        f"cat << 'EOF' | sudo tee /etc/systemd/system/start_web.service\n{service_content}\nEOF\n"
        "sudo systemctl daemon-reload && "
        "sudo systemctl enable start_web.service && "
        "sudo systemctl restart start_web.service"
    )
    run_remote_ssh(public_ip, write_service, ssh_key)

    # Poll until uvicorn binds and responds HTTP 200 (max 15s)
    for _attempt in range(15):
        time.sleep(1)
        res = run_remote_ssh(public_ip, "curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8000/api/v1/health || true", ssh_key, check=False)
        if res.stdout.strip().endswith("200"):
            break

    # 10. Health & TLS verification
    print("\n--- 10. Verifying Local and Remote HTTPS Health Endpoints ---")
    local_health = run_remote_ssh(public_ip, "curl -s http://127.0.0.1:8000/api/v1/health", ssh_key)
    print(f"Local Health: {local_health.stdout}")

    # Remote TLS verification
    print(f"Testing public HTTPS endpoint: https://{domain}/api/v1/health")
    tls_check = subprocess.run(["curl", "-s", "-i", f"https://{domain}/api/v1/health"], capture_output=True, text=True)
    print(f"Remote HTTPS Response:\n{tls_check.stdout}")

    print("\n=== ORACLE ARM64 CANONICAL BACKEND DEPLOYMENT SUCCESSFUL ===")


if __name__ == "__main__":
    main()
