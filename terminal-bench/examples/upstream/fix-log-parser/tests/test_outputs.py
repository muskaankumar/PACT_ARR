import sys; sys.path.insert(0, "/app/src")
from parser import parse_lines

def test_groups_trace():
    recs = parse_lines(["ERROR boom", "  at foo()", "  at bar()"])
    assert len(recs) == 1
    assert recs[0]["trace"] == ["at foo()", "at bar()"]
