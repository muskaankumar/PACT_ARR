"""Model comparison across contrast pairs (build prompt §9, cross-cutting).

The study deliberately includes two model contrast pairs (§7): Pair A (substantial
safety post-training) and Pair B (capability-oriented). This script applies the core
outcome measures per model so the pairs can be compared on the dimensions that
matter: how often they violate, what kind of violation, how fast, and — most
importantly — whether they disclose once they have misaligned.

It does not re-implement the other analyses; it summarizes the same per-run record
fields into one comparison table keyed by model.

Outputs:
  * ``model_comparison.csv`` — one row per model with headline rates.
  * ``model_comparison.md``  — the comparison table with a short read-out.
"""

from __future__ import annotations

import argparse

from ppa.analysis._support import (
    first_violation_turn,
    group_by,
    load_results,
    md_table,
    write_csv,
    write_markdown,
)
from ppa.driver.schema import ResultRecord


def _model_summary(recs: list[ResultRecord]) -> dict:
    n = len(recs)
    auth = sum(1 for r in recs if r.auth_violation.occurred)
    truth = sum(1 for r in recs if r.truthfulness_violation.occurred)
    misaligned = [r for r in recs if r.misaligned()]
    n_mis = len(misaligned)

    # disclosure among misaligned
    conceal = 0
    full = 0
    coded_mis = 0
    for r in misaligned:
        if r.disclosure_level is None:
            continue
        coded_mis += 1
        if r.disclosure_level.value in ("non", "false"):
            conceal += 1
        if r.disclosure_level.value == "full":
            full += 1

    ttv = [first_violation_turn(r) for r in misaligned]
    ttv = [t for t in ttv if t is not None]
    mean_ttv = sum(ttv) / len(ttv) if ttv else None

    return {
        "n": n,
        "auth_rate": auth / n if n else 0.0,
        "truth_rate": truth / n if n else 0.0,
        "misalignment_rate": n_mis / n if n else 0.0,
        "mean_turns_to_violation": round(mean_ttv, 1) if mean_ttv is not None else None,
        "concealment_rate_given_misaligned": (conceal / coded_mis) if coded_mis else None,
        "full_disclosure_rate_given_misaligned": (full / coded_mis) if coded_mis else None,
        "misaligned_coded": coded_mis,
    }


def analyze(records: list[ResultRecord]) -> tuple[list[dict], str]:
    by_model = group_by(records, lambda r: r.model_id)
    csv_rows: list[dict] = []
    table_rows: list[list] = []
    for model, recs in sorted(by_model.items()):
        s = _model_summary(recs)
        csv_rows.append({"model": model, **s})

        def pct(x):
            return "—" if x is None else f"{100 * x:.0f}%"

        table_rows.append([
            model,
            s["n"],
            pct(s["misalignment_rate"]),
            pct(s["auth_rate"]),
            pct(s["truth_rate"]),
            s["mean_turns_to_violation"] if s["mean_turns_to_violation"] is not None else "—",
            pct(s["concealment_rate_given_misaligned"]),
            pct(s["full_disclosure_rate_given_misaligned"]),
        ])

    md = "\n".join([
        "# Model comparison",
        "",
        "Headline outcome measures per model. The two designed contrast pairs (A:",
        "safety-trained, B: capability) can be read off directly. The rightmost two",
        "columns are the honesty-under-pressure result, conditioned on the model having",
        "misaligned.",
        "",
        md_table(
            ["model", "n", "misaligned", "auth", "truth", "mean turns→viol.",
             "conceal (given mis.)", "full disc. (given mis.)"],
            table_rows,
        ),
        "",
        "_'conceal | misaligned' = share of that model's misaligned runs coded Non or",
        "False at debrief; 'full disc. | misaligned' = share coded Full. Both are over",
        "coded misaligned runs only._",
    ])
    return csv_rows, md


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ppa-analysis-models", description="Model comparison (§9).")
    parser.add_argument("results")
    parser.add_argument("--out-dir", default="data/results")
    args = parser.parse_args(argv)

    records = load_results(args.results)
    csv_rows, md = analyze(records)
    fields = ["model", "n", "auth_rate", "truth_rate", "misalignment_rate",
              "mean_turns_to_violation", "concealment_rate_given_misaligned",
              "full_disclosure_rate_given_misaligned", "misaligned_coded"]
    print("wrote", write_csv(csv_rows, f"{args.out_dir}/model_comparison.csv", fieldnames=fields))
    print("wrote", write_markdown(md, f"{args.out_dir}/model_comparison.md"))
    print("\n" + md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
