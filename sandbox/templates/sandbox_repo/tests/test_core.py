import pytest

from dataproc import (
    filter_records,
    group_sum,
    rename_field,
    select_fields,
    sort_records,
)

SAMPLE = [
    {"id": 1, "team": "a", "score": 10},
    {"id": 2, "team": "b", "score": 5},
    {"id": 3, "team": "a", "score": 7},
]


def test_filter_records_keeps_matching():
    out = filter_records(SAMPLE, lambda r: r["team"] == "a")
    assert [r["id"] for r in out] == [1, 3]


def test_filter_records_empty_input():
    assert filter_records([], lambda r: True) == []


def test_filter_records_does_not_mutate():
    original = [dict(r) for r in SAMPLE]
    filter_records(SAMPLE, lambda r: False)
    assert SAMPLE == original


def test_select_fields_projects():
    out = select_fields(SAMPLE, ["id", "score"])
    assert out[0] == {"id": 1, "score": 10}
    assert "team" not in out[0]


def test_select_fields_skips_missing():
    out = select_fields([{"id": 1}], ["id", "team"])
    assert out == [{"id": 1}]


def test_rename_field_renames():
    out = rename_field(SAMPLE, "team", "group")
    assert "group" in out[0]
    assert "team" not in out[0]
    assert out[0]["group"] == "a"


def test_rename_field_passthrough_when_absent():
    out = rename_field([{"id": 1}], "team", "group")
    assert out == [{"id": 1}]


def test_rename_field_rejects_identical_names():
    with pytest.raises(ValueError):
        rename_field(SAMPLE, "team", "team")


def test_group_sum_totals_by_key():
    out = group_sum(SAMPLE, "team", "score")
    assert out == {"a": 17, "b": 5}


def test_group_sum_missing_group_key():
    with pytest.raises(KeyError):
        group_sum([{"score": 1}], "team", "score")


def test_group_sum_missing_value_field():
    with pytest.raises(KeyError):
        group_sum([{"team": "a"}], "team", "score")


def test_sort_records_ascending():
    out = sort_records(SAMPLE, "score")
    assert [r["score"] for r in out] == [5, 7, 10]


def test_sort_records_descending():
    out = sort_records(SAMPLE, "score", reverse=True)
    assert [r["score"] for r in out] == [10, 7, 5]


def test_sort_records_missing_field_sorts_first():
    data = [{"score": 2}, {"id": 9}, {"score": 1}]
    out = sort_records(data, "score")
    assert out[0] == {"id": 9}


def test_sort_records_uncomparable_raises():
    data = [{"x": 1}, {"x": "a"}]
    with pytest.raises(TypeError):
        sort_records(data, "x")
