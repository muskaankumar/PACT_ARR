"""Violation-type breakdown (build prompt §9.2).

When something goes wrong, *what kind* of wrong is it — an authorization violation
only, a truthfulness violation only, or both at once — and does that mix shift with
how the agent was rejected? This is orthogonal to *whether* a violation happened
(that's §9.1); here we condition on the type.

Outputs:
  * ``violation_types.csv`` — counts of {none, auth_only, truth_only, both} per cell.
  * ``violation_types.md``  — the same as a readable table with row percentages.
"""

from __future__ import annotations

import argparse

from ppa.analysis._support import (
    group_by,
    load_results,
    md_table,
    write_csv,
    write_markdown,
)
from ppa.driver.schema import ResultRecord


def _vtype(rec: ResultRecord) -> str:
    a = rec.auth_violation.occurred
    t = rec.truthfulness_violation.occurred
    if a and t:
        return "both"
    if a:
        return "auth_only"
    if t:
        return "truth_only"
    return "none"


def _cell(rec: ResultRecord) -> tuple[str, str]:
    return rec.rejection_condition.finality, rec.rejection_condition.personalization


_TYPES = ["none", "auth_only", "truth_only", "both"]


def analyze(records: list[ResultRecord]) -> tuple[list[dict], str]:
    cells = group_by(records, _cell)
    csv_rows: list[dict] = []
    table_rows: list[list] = []
    for (finality, personalization), recs in sorted(cells.items()):
        counts = {t: 0 for t in _TYPES}
        for r in recs:
            counts[_vtype(r)] += 1
        n = len(recs)
        csv_rows.append({"finality": finality, "personalization": personalization, "n": n, **counts})
        table_rows.append([
            finality,
            personalization,
            n,
            *[f"{counts[t]} ({100 * counts[t] / n:.0f}%)" for t in _TYPES],
        ])

    # overall row
    overall = {t: sum(r[t] for r in csv_rows) for t in _TYPES}
    n_all = sum(r["n"] for r in csv_rows)
    csv_rows.append({"finality": "ALL", "personalization": "ALL", "n": n_all, **overall})

    md = "\n".join([
        "# Violation types",
        "",
        "Given a run, what kind of violation (if any) occurred, by rejection condition.",
        "",
        md_table(
            ["finality", "personalization", "n", *_TYPES],
            table_rows + [["**ALL**", "", n_all, *[f"{overall[t]} ({100 * overall[t] / max(n_all,1):.0f}%)" for t in _TYPES]]],
        ),
    ])
    return csv_rows, md


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ppa-analysis-vtypes", description="Violation types (§9.2).")
    parser.add_argument("results")
    parser.add_argument("--out-dir", default="data/results")
    args = parser.parse_args(argv)

    records = load_results(args.results)
    csv_rows, md = analyze(records)
    print("wrote", write_csv(csv_rows, f"{args.out_dir}/violation_types.csv"))
    print("wrote", write_markdown(md, f"{args.out_dir}/violation_types.md"))
    print("\n" + md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
