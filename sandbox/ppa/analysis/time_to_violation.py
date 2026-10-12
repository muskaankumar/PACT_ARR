"""Time-to-violation survival analysis (build prompt §9.1).

How quickly does the agent cross a line after being blocked, and does that speed
depend on *how* it was rejected? We treat turns-until-first-violation as a
time-to-event outcome and estimate Kaplan–Meier survival curves S(t) = P(no
violation by turn t), broken out by the 3×3 rejection design (finality ×
personalization). Runs that reach ``max_turns`` with no violation are right-censored
— they did not (yet) violate, and we don't pretend to know when they would.

Outputs:
  * ``time_to_violation.csv``  — tidy KM step table, one block per condition cell.
  * ``time_to_violation.md``   — median survival + event counts per cell.
"""

from __future__ import annotations

import argparse

from ppa.analysis._support import (
    group_by,
    kaplan_meier,
    load_results,
    md_table,
    median_survival,
    write_csv,
    write_markdown,
)
from ppa.driver.schema import ResultRecord


def _cell(rec: ResultRecord) -> tuple[str, str]:
    return rec.rejection_condition.finality, rec.rejection_condition.personalization


def analyze(records: list[ResultRecord]) -> tuple[list[dict], str]:
    cells = group_by(records, _cell)

    csv_rows: list[dict] = []
    summary_rows: list[list] = []
    for (finality, personalization), recs in sorted(cells.items()):
        km = kaplan_meier(recs)
        n_events = sum(1 for r in recs if (r.misaligned()))
        med = median_survival(km)
        summary_rows.append([
            finality,
            personalization,
            len(recs),
            n_events,
            f"{100 * n_events / len(recs):.0f}%",
            med if med is not None else "not reached",
        ])
        for step in km:
            csv_rows.append({
                "finality": finality,
                "personalization": personalization,
                "turn": int(step["turn"]),
                "at_risk": int(step["at_risk"]),
                "events": int(step["events"]),
                "survival": step["survival"],
            })

    md = "\n".join([
        "# Time to violation (survival analysis)",
        "",
        "Turns until the agent's first authorization or truthfulness violation, by",
        "rejection condition. Runs reaching `max_turns` without a violation are",
        "right-censored. Lower median survival = faster to cross a line.",
        "",
        md_table(
            ["finality", "personalization", "n", "violations", "violation rate", "median survival (turns)"],
            summary_rows,
        ),
        "",
        "_Median survival = first turn at which ≤50% of runs remain violation-free; "
        "'not reached' means most runs never violated._",
    ])
    return csv_rows, md


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ppa-analysis-survival", description="Time-to-violation (§9.1).")
    parser.add_argument("results")
    parser.add_argument("--out-dir", default="data/results")
    args = parser.parse_args(argv)

    records = load_results(args.results)
    csv_rows, md = analyze(records)
    csv_path = write_csv(csv_rows, f"{args.out_dir}/time_to_violation.csv")
    md_path = write_markdown(md, f"{args.out_dir}/time_to_violation.md")
    print(f"wrote {csv_path}\nwrote {md_path}")
    print("\n" + md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
