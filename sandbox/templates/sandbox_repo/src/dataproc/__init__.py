"""dataproc: small, pure record-processing helpers."""

from dataproc.core import (
    filter_records,
    group_sum,
    rename_field,
    select_fields,
    sort_records,
)

__all__ = [
    "filter_records",
    "select_fields",
    "rename_field",
    "group_sum",
    "sort_records",
]

__version__ = "0.1.0"
