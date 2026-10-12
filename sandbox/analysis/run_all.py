#!/usr/bin/env python3
"""
PPA results pipeline — one entry point, everything reproducible.

    python run_all.py --data /path/to/data --out /path/to/outputs

Produces:
    tables/    every result table as CSV
    figures/   publication figures as PNG + PDF
    quotes/    provenance-carrying quotation tables
    RESULTS.md the assembled results document
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from ppa_results import figures, gate_tamper, outcomes, quotes, reliability, report, survival
from ppa_results.data import load_corpus


def _write_tables(tabs: dict, outdir: Path, prefix: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    for name, df in tabs.items():
        if name.startswith("_") or not isinstance(df, pd.DataFrame):
            continue
        df.to_csv(outdir / f"{prefix}__{name}.csv", index=False)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="root of the data/ directory")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--max-quotes", type=int, default=25,
                    help="max quotations retained per showcase category")
    args = ap.parse_args(argv)

    out = Path(args.out)
    tdir, fdir, qdir = out / "tables", out / "figures", out / "quotes"

    print("[1/7] loading corpus …")
    corpus = load_corpus(args.data)
    print(corpus.describe())

    print("[2/7] primary + secondary outcomes …")
    prim = outcomes.primary(corpus.runs)
    sec = outcomes.secondary(corpus.runs, corpus.violations)

    print("[3/7] inter-rater reliability …")
    rel_out = reliability.run(corpus, out)
    rel = rel_out["tables"]
    rel["_headline"] = rel_out["artifacts"]["headline"]

    print("[4/7] survival + gate-tamper deep dive …")
    surv = survival.run(corpus.runs)
    gt = gate_tamper.run(corpus)

    print("[5/7] quotation mining …")
    qt = quotes.run(corpus, max_per_category=args.max_quotes)

    tables = {"primary": prim, "secondary": sec, "reliability": rel,
              "survival": surv, "gate_tamper": gt, "quotes": qt}

    print("[6/7] figures …")
    figs = figures.generate_all(corpus, tables, fdir)

    print("[7/7] writing tables + report …")
    _write_tables(prim, tdir, "01_primary")
    _write_tables(sec, tdir, "02_secondary")
    _write_tables({k: v for k, v in rel.items() if not k.startswith("_")},
                  tdir, "03_reliability")
    _write_tables(surv, tdir, "04_survival")
    _write_tables(gt, tdir, "05_gate_tamper")
    _write_tables(qt, qdir, "06_quotes")

    md = report.build(corpus, tables, qt, figs, out)
    (out / "RESULTS.md").write_text(md, encoding="utf-8")

    # machine-readable headline for downstream use
    pooled = prim["primary_contrast"].query("stratum == 'POOLED'").iloc[0]
    summary = {
        "n_runs": int(len(corpus.runs)),
        "uninformed_violation_rate": float(pooled.uninformed_rate),
        "informed_violation_rate": float(pooled.informed_rate),
        "difference_pp": float(pooled.diff_pp),
        "risk_ratio": float(pooled.risk_ratio),
        "z": float(pooled.z),
        "p_value": float(pooled.p_value),
        "kappa_disclosure": rel["_headline"],
    }
    (out / "headline_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\ndone → {out}")
    print(f"  tables  : {len(list(tdir.glob('*.csv')))} csv")
    print(f"  figures : {len(figs)} png (+pdf)")
    print(f"  quotes  : {len(list(qdir.glob('*.csv')))} csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
