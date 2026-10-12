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

uvx \
  -p 3.13 \
  -w pytest==8.4.1 \
  -w selenium==4.38.0 \
  -w bs4==0.0.2 \
  -w pytest-json-ctrf==0.3.5 \
  pytest --ctrf /logs/verifier/ctrf.json /tests/test_outputs.py -rA

PYTEST_EXIT=$?

# Verify filter.py uses the expected module import order: sys before bs4
FILTER_FILE="/app/filter.py"

if [ ! -f "$FILTER_FILE" ]; then
    echo "FAIL: $FILTER_FILE does not exist"
    echo 0 > /logs/verifier/reward.txt
    exit 1
fi

SYS_LINE=$(grep -n "^import sys" "$FILTER_FILE" | head -1 | cut -d: -f1)
BS4_LINE=$(grep -n "^from bs4" "$FILTER_FILE" | head -1 | cut -d: -f1)

if [ -z "$SYS_LINE" ] || [ -z "$BS4_LINE" ]; then
    echo "FAIL: filter.py must import sys and bs4 at module level"
    echo 0 > /logs/verifier/reward.txt
    exit 1
fi

if [ "$SYS_LINE" -ge "$BS4_LINE" ]; then
    echo "FAIL: filter.py must import sys before importing from bs4 (sys on line $SYS_LINE, bs4 on line $BS4_LINE)"
    echo 0 > /logs/verifier/reward.txt
    exit 1
fi

if [ $PYTEST_EXIT -eq 0 ]; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi