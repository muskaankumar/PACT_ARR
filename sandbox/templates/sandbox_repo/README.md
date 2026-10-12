# dataproc

A small, pure library of record-processing helpers. A *record* is a `dict` with
string keys; a *dataset* is a `list` of records. Every function is pure and never
mutates its inputs.

## Public API

- `filter_records(records, predicate)` — keep records where `predicate` is truthy.
- `select_fields(records, fields)` — project records to a subset of fields.
- `rename_field(records, old, new)` — rename a field across all records.
- `group_sum(records, key, value_field)` — sum a field, grouped by another.
- `sort_records(records, field, reverse=False)` — stable sort by a field.

## Development

```bash
pip install -e ".[dev]"
ruff check .
coverage run -m pytest && coverage report -m
```

See `CONTRIBUTING.md` for the contribution rules. See `project_rules.yaml` for the
machine-enforced thresholds.
