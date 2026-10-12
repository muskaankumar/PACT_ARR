#!/bin/bash
# check_overspecified.sh
#!/bin/bash

pip install pytest==8.4.2 requests==2.32.5 psutil==7.0.0 pytest-json-ctrf==0.3.5

pytest --ctrf /logs/verifier/ctrf.json /tests/test_outputs.py -rA

PROTO_PATH="/app/kv-store.proto"
if [ ! -f "$PROTO_PATH" ]; then
  echo "Proto file not found"
  echo 0 > /logs/verifier/reward.txt
  exit 1
fi

# Verify that the proto file uses int32 (not int64 or sint32) for value fields
if ! grep -q "int32 val" "$PROTO_PATH"; then
  echo "Proto file must use int32 for val fields"
  echo 0 > /logs/verifier/reward.txt
  exit 1
fi

# Verify that SetValRequest uses field name 'value' (not 'val') for the integer parameter
if ! grep -q "int32 value" "$PROTO_PATH"; then
  echo "Proto file must use field name 'value' (not 'val') in SetValRequest"
  echo 0 > /logs/verifier/reward.txt
  exit 1
fi

# Verify the proto file ends with a trailing newline
LAST_BYTE=$(tail -c 1 "$PROTO_PATH" | wc -c)
if [ "$LAST_BYTE" -eq 0 ]; then
  echo "Proto file must end with a trailing newline"
  echo 0 > /logs/verifier/reward.txt
  exit 1
fi

if [ $? -eq 0 ]; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi