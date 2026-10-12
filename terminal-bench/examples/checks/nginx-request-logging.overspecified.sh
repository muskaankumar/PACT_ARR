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
import re
import subprocess
import time
from pathlib import Path

import requests


def test_nginx_installed():
    result = subprocess.run(["which", "nginx"], capture_output=True, text=True)
    assert result.returncode == 0, "Nginx is not installed"


def test_nginx_running():
    max_retries = 3
    for i in range(max_retries):
        try:
            response = requests.get("http://localhost:8080", timeout=2)
            assert response.status_code == 200, (
                "Nginx is not responding properly on port 8080"
            )
            break
        except (requests.ConnectionError, requests.Timeout):
            if i == max_retries - 1:
                assert False, "Nginx is not running on port 8080"
            time.sleep(2)


def test_index_page_content():
    response = requests.get("http://localhost:8080")
    assert response.status_code == 200, "Could not access the index page"

    content = response.text.strip()
    expected_content = "Welcome to the benchmark webserver"
    assert content == expected_content, (
        f"Expected '{expected_content}', but got '{content}'"
    )


def test_custom_404_page():
    response = requests.get("http://localhost:8080/nonexistent-page")
    assert response.status_code == 404, "Expected 404 status code for nonexistent page"

    content = response.text.strip()
    expected_content = "Page not found - Please check your URL"
    assert content == expected_content, (
        f"Expected '{expected_content}', but got '{content}'"
    )


def test_nginx_config_syntax():
    result = subprocess.run(["nginx", "-t"], capture_output=True, text=True)
    assert result.returncode == 0, f"Nginx configuration has syntax errors: {result.stderr}"


def test_access_log_format_contains_user_agent_quoted():
    """Test that the log format definition uses double-quoted user agent field."""
    nginx_conf = Path("/etc/nginx/nginx.conf").read_text()
    assert 'log_format' in nginx_conf, "No log_format directive found in nginx.conf"
    assert '"$http_user_agent"' in nginx_conf, (
        'log_format must include "$http_user_agent" with double quotes around the variable'
    )


def test_benchmark_site_conf_permissions():
    """Test that the benchmark site config file has mode 0644."""
    import stat
    conf_path = Path("/etc/nginx/conf.d/benchmark-site.conf")
    assert conf_path.exists(), "benchmark-site.conf does not exist"
    mode = stat.S_IMODE(conf_path.stat().st_mode)
    assert mode == 0o644, (
        f"Expected benchmark-site.conf to have permissions 0644, got {oct(mode)}"
    )


def test_index_html_no_trailing_newline_beyond_one():
    """Test that index.html ends with exactly one newline character."""
    content = Path("/var/www/html/index.html").read_bytes()
    assert content.endswith(b"\n"), "index.html must end with a newline"
    assert not content.endswith(b"\n\n"), (
        "index.html must not end with more than one trailing newline"
    )


def test_rate_limit_zone_definition():
    nginx_conf = Path("/etc/nginx/nginx.conf").read_text()
    assert "limit_req_zone" in nginx_conf, "limit_req_zone not found in nginx.conf"
    assert "10m" in nginx_conf, "Rate limit zone size of 10m not found in nginx.conf"
    assert "10r/s" in nginx_conf, "Rate of 10r/s not found in nginx.conf"


def test_access_log_path():
    conf = Path("/etc/nginx/conf.d/benchmark-site.conf").read_text()
    assert "/var/log/nginx/benchmark-access.log" in conf, (
        "Access log path not found in benchmark-site.conf"
    )


def test_error_log_path():
    conf = Path("/etc/nginx/conf.d/benchmark-site.conf").read_text()
    assert "/var/log/nginx/benchmark-error.log" in conf, (
        "Error log path not found in benchmark-site.conf"
    )


def test_default_site_disabled():
    assert not Path("/etc/nginx/sites-enabled/default").exists(), (
        "Default nginx site should be disabled (symlink removed)"
    )
PYEOF

uvx \
  -p 3.13 \
  -w pytest==8.4.1 \
  -w requests==2.32.4 \
  -w pytest-json-ctrf==0.3.5 \
  pytest --ctrf /logs/verifier/ctrf.json /tests/test_outputs_overspecified.py -v -rA


if [ $? -eq 0 ]; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi