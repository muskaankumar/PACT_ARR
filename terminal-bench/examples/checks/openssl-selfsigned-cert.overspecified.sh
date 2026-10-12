#!/bin/bash
# check_overspecified.sh
#!/bin/bash

apt-get update
apt-get install -y curl

curl -LsSf https://astral.sh/uv/0.9.5/install.sh | sh

source $HOME/.local/bin/env

if [ "$PWD" = "/" ]; then
    echo "Error: No working directory set. Please set a WORKDIR in your Dockerfile before running this script."
    exit 1
fi

cat > /tests/test_outputs_overspecified.py << 'PYEOF'
import os
import re
import subprocess


def test_directory_structure():
    directory = "/app/ssl"
    assert os.path.exists(directory), f"Directory {directory} does not exist"
    assert os.path.isdir(directory), f"{directory} is not a directory"


def test_key_file():
    key_file = "/app/ssl/server.key"
    assert os.path.exists(key_file), "Private key file does not exist"

    key_permissions = os.stat(key_file).st_mode & 0o777
    assert key_permissions <= 0o600, f"Key permissions too open: {key_permissions:o}"

    process = subprocess.run(
        ["openssl", "rsa", "-in", key_file, "-text", "-noout"],
        capture_output=True,
        text=True,
    )
    key_text = process.stdout
    assert "2048 bit" in key_text, "Key is not 2048 bit RSA"


def test_certificate_file():
    cert_file = "/app/ssl/server.crt"
    assert os.path.exists(cert_file), "Certificate file does not exist"

    process = subprocess.run(
        ["openssl", "x509", "-in", cert_file, "-text", "-noout"],
        capture_output=True,
        text=True,
    )
    cert_text = process.stdout

    assert "dev-internal.company.local" in cert_text, (
        "Certificate does not have correct Common Name"
    )

    assert "devops team" in cert_text.lower(), (
        "Certificate does not have correct Organization Name"
    )

    dates_process = subprocess.run(
        ["openssl", "x509", "-in", cert_file, "-noout", "-dates"],
        capture_output=True,
        text=True,
    )
    dates_text = dates_process.stdout
    assert "notBefore=" in dates_text, "Certificate dates not found"
    assert "notAfter=" in dates_text, "Certificate expiry date not found"


def test_pem_file():
    pem_file = "/app/ssl/server.pem"
    assert os.path.exists(pem_file), "PEM file does not exist"

    with open(pem_file, "r") as f:
        pem_content = f.read()

    assert "BEGIN RSA PRIVATE KEY" in pem_content or "BEGIN PRIVATE KEY" in pem_content, (
        "PEM file does not contain private key"
    )
    assert "BEGIN CERTIFICATE" in pem_content, "PEM file does not contain certificate"

    key_pos = pem_content.find("KEY-----")
    cert_pos = pem_content.find("BEGIN CERTIFICATE")
    assert key_pos < cert_pos, "Private key must appear before certificate in PEM file"


def test_verification_file():
    verification_file = "/app/ssl/verification.txt"
    assert os.path.exists(verification_file), "Verification file does not exist"

    with open(verification_file, "r") as f:
        content = f.read()

    assert "dev-internal.company.local" in content, (
        "Verification file does not contain Common Name"
    )
    assert "SHA256 Fingerprint" in content or "sha256" in content.lower(), (
        "Verification file does not contain SHA-256 fingerprint"
    )

    assert content == content.rstrip('\n') + '\n', (
        "Verification file must end with exactly one newline character"
    )


def test_check_cert_script():
    script_file = "/app/check_cert.py"
    assert os.path.exists(script_file), "check_cert.py does not exist"

    process = subprocess.run(
        ["python", script_file],
        capture_output=True,
        text=True,
    )

    assert process.returncode == 0, f"Script failed with error: {process.stderr}"
    assert "Certificate verification successful" in process.stdout, (
        "Script did not print 'Certificate verification successful'"
    )
    assert "dev-internal.company.local" in process.stdout, (
        "Script did not print the Common Name"
    )

    date_pattern = r"\d{4}-\d{2}-\d{2}"
    assert re.search(date_pattern, process.stdout), (
        "Script did not print expiration date in YYYY-MM-DD format"
    )
PYEOF

uvx \
  -p 3.13 \
  -w pytest==8.4.1 \
  -w pytest-json-ctrf==0.3.5 \
  pytest --ctrf /logs/verifier/ctrf.json /tests/test_outputs_overspecified.py -rA

if [ $? -eq 0 ]; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi