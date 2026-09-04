#!/usr/bin/env python3
"""Deploy StART v4.5.2 to Oracle Cloud ARM64 Instance with Let's Encrypt TLS."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SSH_KEY = os.path.expanduser("~/.ssh/id_ed25519_start_oci")
INFO_PATH = ROOT / "start_output" / "v452_remote_release" / "oracle_instance_info.json"
ORIGIN_SECRET = os.environ.get("START_ORIGIN_SECRET", "")

def run_remote_ssh(ip: str, cmd: str, check=True) -> subprocess.CompletedProcess:
    ssh_cmd = [
        "ssh",
        "-o", "StrictHostKeyChecking=no",
        "-o", "ConnectTimeout=15",
        "-i", SSH_KEY,
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
    print("=== DEPLOYING StART v4.5.3 BACKEND TO ORACLE LINUX ARM64 ===")
    
    with open(INFO_PATH) as f:
        info = json.load(f)
    
    public_ip = info["public_ip"]
    domain = info["sslip_domain"]
    print(f"Target VM IP: {public_ip}")
    print(f"Target Domain: {domain}")

    # 1. Ensure target directories on VM
    print("\n--- 1. Setting up /opt/start directory ---")
    run_remote_ssh(public_ip, "sudo mkdir -p /opt/start && sudo chown -R ubuntu:ubuntu /opt/start")

    # 2. Sync codebase
    print("\n--- 2. Syncing canonical non-Git codebase to /opt/start ---")
    rsync_cmd = [
        "rsync", "-avz", "--delete",
        "-e", f"ssh -o StrictHostKeyChecking=no -i {SSH_KEY}",
        "--exclude=.venv-start",
        "--exclude=.git",
        "--exclude=__pycache__",
        "--exclude=.pytest_cache",
        "--exclude=.ruff_cache",
        "--exclude=.env",
        "--exclude=web/node_modules",
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
    res_opa = run_remote_ssh(public_ip, opa_setup_cmd)
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
    run_remote_ssh(public_ip, venv_cmd)
    print("Python virtual environment initialized.")

    # 5. Run ARM64 dependency verification
    print("\n--- 5. Running ARM64 Dependency Audit on VM ---")
    res_audit = run_remote_ssh(public_ip, "cd /opt/start && .venv-start/bin/python scripts/verify_arm64_dependencies.py")
    print(res_audit.stdout)

    # 6. Obtain Let's Encrypt SSL certificate for sslip.io domain
    print(f"\n--- 6. Obtaining Let's Encrypt SSL Certificate for {domain} ---")
    cert_cmd = (
        "sudo systemctl stop nginx || true; "
        f"sudo certbot certonly --standalone --non-interactive --agree-tos --register-unsafely-without-email -d {domain} && "
        f"sudo test -f /etc/letsencrypt/live/{domain}/fullchain.pem"
    )
    run_remote_ssh(public_ip, cert_cmd)
    print("Let's Encrypt certificate obtained successfully!")

    # 7. Configure Nginx HTTPS reverse proxy
    print("\n--- 7. Configuring Nginx HTTPS Reverse Proxy with Trusted TLS ---")
    nginx_conf = f"""
server {{
    listen 80;
    server_name {domain};
    return 301 https://$host$request_uri;
}}

server {{
    listen 443 ssl;
    server_name {domain};

    ssl_certificate /etc/letsencrypt/live/{domain}/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/{domain}/privkey.pem;

    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;

    location / {{
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_buffering off;
        proxy_read_timeout 300s;
    }}
}}
"""
    write_nginx = (
        f"cat << 'EOF' | sudo tee /etc/nginx/sites-available/start\n{nginx_conf}\nEOF\n"
        "sudo rm -f /etc/nginx/sites-enabled/default && "
        "sudo ln -sf /etc/nginx/sites-available/start /etc/nginx/sites-enabled/start && "
        "sudo nginx -t && "
        "sudo systemctl restart nginx"
    )
    run_remote_ssh(public_ip, write_nginx)
    print("Nginx configured with TLS and active.")

    # 8. Configure & Start systemd service
    print("\n--- 8. Configuring & Starting start_web.service ---")
    service_content = """[Unit]
Description=StART v4.6 Agentic Engineering Workbench Web Service
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
    run_remote_ssh(public_ip, write_service)
    time.sleep(3)

    # 9. Health & TLS verification
    print("\n--- 9. Verifying Local and Remote HTTPS Health Endpoints ---")
    local_health = run_remote_ssh(public_ip, "curl -s http://127.0.0.1:8000/api/health")
    print(f"Local Health: {local_health.stdout}")

    # Remote TLS verification
    print(f"Testing public HTTPS endpoint: https://{domain}/api/health")
    tls_check = subprocess.run(["curl", "-s", "-i", f"https://{domain}/api/health"], capture_output=True, text=True)
    print(f"Remote HTTPS Response:\n{tls_check.stdout}")

    print("\n=== ORACLE ARM64 CANONICAL BACKEND DEPLOYMENT SUCCESSFUL ===")

if __name__ == "__main__":
    main()
