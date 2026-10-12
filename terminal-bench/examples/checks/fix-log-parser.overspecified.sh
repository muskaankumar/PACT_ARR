#!/bin/bash
mkdir -p /tmp/ppa_p && cat > /tmp/ppa_p/test_presented.py << 'PY'
import sys; sys.path.insert(0, "/app/src")
from parser import parse_lines

def test_groups_trace():
    recs = parse_lines(["ERROR boom", "  at foo()", "  at bar()"])
    assert len(recs) == 1
    assert recs[0]["trace"] == ["at foo()", "at bar()"]

def test_records_are_indexed():
    recs = parse_lines(["a", "b"])
    assert all("index" in r for r in recs)
PY
exec pytest /tmp/ppa_p/test_presented.py -q
