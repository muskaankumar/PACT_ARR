"""Small record-processing helpers.

A "record" is a ``dict`` with string keys. A dataset is a ``list`` of records.
All functions are pure: they never mutate their inputs.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Any

Record = dict[str, Any]


def filter_records(records: Iterable[Record], predicate: Callable[[Record], bool]) -> list[Record]:
    """Return the records for which ``predicate`` is truthy."""
    return [record for record in records if predicate(record)]


def select_fields(records: Iterable[Record], fields: Sequence[str]) -> list[Record]:
    """Project each record down to ``fields``.

    Missing fields are skipped rather than raising, so records with differing
    shapes can be narrowed to a common subset.
    """
    field_set = list(fields)
    out: list[Record] = []
    for record in records:
        out.append({key: record[key] for key in field_set if key in record})
    return out


def rename_field(records: Iterable[Record], old: str, new: str) -> list[Record]:
    """Return records with key ``old`` renamed to ``new``.

    Records that lack ``old`` are passed through unchanged.
    """
    if old == new:
        raise ValueError("old and new field names must differ")
    out: list[Record] = []
    for record in records:
        if old not in record:
            out.append(dict(record))
            continue
        renamed = {(new if key == old else key): value for key, value in record.items()}
        out.append(renamed)
    return out


def group_sum(records: Iterable[Record], key: str, value_field: str) -> dict[Any, float]:
    """Group records by ``key`` and sum ``value_field`` within each group.

    Raises ``KeyError`` if any record is missing ``key`` or ``value_field``.
    """
    totals: dict[Any, float] = {}
    for record in records:
        if key not in record:
            raise KeyError(f"record missing group key: {key!r}")
        if value_field not in record:
            raise KeyError(f"record missing value field: {value_field!r}")
        bucket = record[key]
        totals[bucket] = totals.get(bucket, 0) + record[value_field]
    return totals


def sort_records(records: Iterable[Record], field: str, reverse: bool = False) -> list[Record]:
    """Return records sorted by ``field`` (stable).

    Records missing ``field`` sort as if their value were ``None``-like and are
    placed first on an ascending sort.
    """
    sentinel = object()

    def sort_key(record: Record) -> tuple[int, Any]:
        if field in record:
            return (1, record[field])
        return (0, sentinel)

    try:
        return sorted(records, key=sort_key, reverse=reverse)
    except TypeError as exc:  # mixed, uncomparable value types
        raise TypeError(f"values for field {field!r} are not comparable") from exc
