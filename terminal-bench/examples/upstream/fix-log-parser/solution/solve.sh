#!/bin/bash
cat > /app/src/parser.py << 'PY'
def parse_lines(lines):
    out = []
    for l in lines:
        if l.startswith("  ") and out:
            out[-1].setdefault("trace", []).append(l.strip())
        else:
            out.append({"msg": l})
    return out
PY
