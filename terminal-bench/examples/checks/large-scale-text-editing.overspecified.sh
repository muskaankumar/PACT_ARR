#!/bin/bash
# check_overspecified.sh
#!/bin/bash

apt-get update -qq
apt-get install -y -qq curl

curl -LsSf https://astral.sh/uv/0.9.5/install.sh | sh
source $HOME/.local/bin/env

if [ "$PWD" = "/" ]; then
    echo "Error: No working directory set."
    exit 1
fi

cd /app
rm -f *.csv
python3 /tests/gen_large_csv.py input

uvx \
  -p 3.13 \
  -w pytest==8.4.1 \
  -w pytest-json-ctrf==0.3.5 \
  pytest --ctrf /logs/verifier/ctrf.json /tests/test_outputs.py -rA

PYTEST_EXIT=$?

# Verify apply_macros.vim uses Unix line endings (LF only, no CR)
if [ -f /app/apply_macros.vim ]; then
    if file /app/apply_macros.vim | grep -q "CRLF"; then
        echo "FAIL: apply_macros.vim must use Unix (LF) line endings, not CRLF."
        echo 0 > /logs/verifier/reward.txt
        exit 1
    fi
    # Verify the script file permissions are exactly 0644
    PERMS=$(stat -c "%a" /app/apply_macros.vim)
    if [ "$PERMS" != "644" ]; then
        echo "FAIL: apply_macros.vim must have permissions 644, got $PERMS."
        echo 0 > /logs/verifier/reward.txt
        exit 1
    fi
fi

if [ $PYTEST_EXIT -eq 0 ]; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi