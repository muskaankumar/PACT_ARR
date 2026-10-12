"""Descriptive behavior map (build prompt §9.3).

Beyond the binary violation question, what is the full range of behaviors the agent
shows when blocked? Each run carries a multi-label set of descriptive tags (§6);
this analysis reports how often each tag appears — overall and by rejection
condition — so the qualitative texture of responses is visible, not just the
pass/fail outcome.

Outputs:
  * ``descriptive_map.csv`` — tag frequency overall and per condition cell.
  * ``descriptive_map.md``  — overall tag distribution + a by-finality table.
"""

from __future__ import annotations

import argparse
from collections import Counter

from ppa.analysis._support import (
    group_by,
    load_results,
    md_table,
    write_csv,
    write_markdown,
)
from ppa.driver.schema import DescriptiveTag, ResultRecord

_TAGS = [t.value for t in DescriptiveTag]


def _counts(records: list[ResultRecord]) -> Counter:
    c: Counter = Counter()
    for r in records:
        for tag in r.descriptive_tags:
            c[tag.value] += 1
    return c


def analyze(records: list[ResultRecord]) -> tuple[list[dict], str]:
    n = len(records) or 1
    overall = _counts(records)

    csv_rows: list[dict] = [
        {"scope": "ALL", "finality": "ALL", "n": len(records), **{t: overall.get(t, 0) for t in _TAGS}}
    ]

    by_finality = group_by(records, lambda r: r.rejection_condition.finality)
    for finality, recs in sorted(by_finality.items()):
        c = _counts(recs)
        csv_rows.append({"scope": "by_finality", "finality": finality, "n": len(recs),
                         **{t: c.get(t, 0) for t in _TAGS}})

    overall_rows = [[t, overall.get(t, 0), f"{100 * overall.get(t, 0) / n:.0f}%"] for t in _TAGS]
    fin_rows = []
    for finality, recs in sorted(by_finality.items()):
        c = _counts(recs)
        m = len(recs) or 1
        fin_rows.append([finality, len(recs), *[f"{c.get(t, 0)} ({100*c.get(t,0)/m:.0f}%)" for t in _TAGS]])

    md = "\n".join([
        "# Descriptive behavior map",
        "",
        "Distribution of behaviors when the agent is blocked (multi-label; a run can",
        "carry several tags).",
        "",
        "## Overall",
        "",
        md_table(["behavior", "runs", "share"], overall_rows),
        "",
        "## By finality",
        "",
        md_table(["finality", "n", *_TAGS], fin_rows),
    ])
    return csv_rows, md


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ppa-analysis-descriptive", description="Descriptive map (§9.3).")
    parser.add_argument("results")
    parser.add_argument("--out-dir", default="data/results")
    args = parser.parse_args(argv)

    records = load_results(args.results)
    csv_rows, md = analyze(records)
    print("wrote", write_csv(csv_rows, f"{args.out_dir}/descriptive_map.csv"))
    print("wrote", write_markdown(md, f"{args.out_dir}/descriptive_map.md"))
    print("\n" + md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
