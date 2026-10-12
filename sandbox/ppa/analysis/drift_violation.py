"""Drift & violation analysis + plots (v2).

Scores every run log with :func:`ppa.analysis.drift.score_run` and produces:

  * a per-run CSV of drift/violation signals (always written);
  * a text summary grouped by condition × affordance level (always printed);
  * two plots IF matplotlib is installed (optional dependency):
      - drift trajectory: mean cumulative boundary-proximity vs turn, per group;
      - violation survival: fraction of runs NOT-yet-violated vs turn (time to
        first violation), per group.

The run_id encodes condition + affordance + model, so this works directly off
the JSONL logs the multi-turn harness writes — no result file needed. Run it
after (or independently of) the main experiment.

Usage:
    python -m ppa.analysis.drift_violation --runs-dir data/runs --out data/results
    python -m ppa.analysis.drift_violation --runs-dir data/runs --no-plot
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from ppa.analysis.drift import RunDrift, score_run
from ppa.environment.logging import RunLogger


def _log_dicts(path: Path) -> list[dict]:
    out = []
    for r in RunLogger.read(path):
        d = r.model_dump()
        et = d.get("event_type")
        d["event_type"] = et.value if hasattr(et, "value") else et
        out.append(d)
    return out


def _meta_from_run_id(run_id: str) -> dict:
    """run_id: <condition>__<model>__aff-<level>__<task>__<rejection>__rNN"""
    parts = run_id.split("__")
    meta = {"condition": parts[0] if parts else "", "model_id": "", "affordance_level": "none"}
    if len(parts) > 1:
        meta["model_id"] = parts[1]
    for p in parts:
        if p.startswith("aff-"):
            meta["affordance_level"] = p[4:]
    return meta


def score_all(runs_dir: str) -> list[RunDrift]:
    out: list[RunDrift] = []
    for log in sorted(Path(runs_dir).glob("*.jsonl")):
        run_id = log.stem
        meta = _meta_from_run_id(run_id)
        try:
            recs = _log_dicts(log)
        except Exception:  # noqa: BLE001 - skip unreadable logs
            continue
        rd = score_run(recs, run_id=run_id, condition=meta["condition"],
                       model_id=meta["model_id"])
        # score_run doesn't parse affordance from the id; set it from the run_id.
        rd.affordance_level = meta["affordance_level"]
        out.append(rd)
    return out


def _group_key(rd: RunDrift) -> str:
    return f"{rd.condition}/{rd.affordance_level}"


def write_per_run_csv(drifts: list[RunDrift], out_dir: str) -> Path:
    path = Path(out_dir) / "drift_per_run.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["run_id", "condition", "affordance_level", "model_id", "crossed",
            "first_probe_turn", "first_violation_turn", "max_proximity",
            "total_proximity", "n_probes", "n_pushbacks"]
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for d in drifts:
            w.writerow([getattr(d, c) for c in cols])
    return path


def print_summary(drifts: list[RunDrift]) -> None:
    groups: dict[str, list[RunDrift]] = defaultdict(list)
    for d in drifts:
        groups[_group_key(d)].append(d)
    print("\n=== DRIFT & VIOLATION SUMMARY (by condition/affordance) ===")
    print(f"  {'group':<26} {'n':>4} {'viol%':>6} {'mean_maxprox':>12} "
          f"{'mean_1st_viol_turn':>18}")
    for g in sorted(groups):
        rs = groups[g]
        n = len(rs)
        v = sum(1 for r in rs if r.crossed)
        mean_prox = sum(r.max_proximity for r in rs) / n if n else 0
        ftvs = [r.first_violation_turn for r in rs if r.first_violation_turn is not None]
        mean_ftv = sum(ftvs) / len(ftvs) if ftvs else None
        ftv_s = f"{mean_ftv:.1f}" if mean_ftv is not None else "-"
        print(f"  {g:<26} {n:>4} {100 * v / (n or 1):>5.1f} {mean_prox:>12.2f} {ftv_s:>18}")


def _survival_curve(drifts: list[RunDrift], max_turn: int) -> list[float]:
    """Fraction of runs NOT-yet-violated at each turn (time to first violation)."""
    n = len(drifts) or 1
    curve = []
    for t in range(max_turn + 1):
        not_yet = sum(1 for d in drifts
                      if d.first_violation_turn is None or d.first_violation_turn > t)
        curve.append(not_yet / n)
    return curve


def _mean_proximity_curve(drifts: list[RunDrift], max_turn: int) -> list[float]:
    """Mean cumulative proximity vs turn across runs (last value carried forward)."""
    series = []
    for d in drifts:
        by_turn = {ts.turn_index: ts.cumulative for ts in d.per_turn}
        vals, last = [], 0
        for t in range(max_turn + 1):
            if t in by_turn:
                last = by_turn[t]
            vals.append(last)
        series.append(vals)
    if not series:
        return [0.0] * (max_turn + 1)
    return [sum(s[t] for s in series) / len(series) for t in range(max_turn + 1)]


def make_plots(drifts: list[RunDrift], out_dir: str) -> list[Path]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001
        print("\n[plots skipped] matplotlib not installed. "
              "Install with: pip install -e \".[analysis]\"")
        return []

    groups: dict[str, list[RunDrift]] = defaultdict(list)
    for d in drifts:
        groups[_group_key(d)].append(d)
    max_turn = max((ts.turn_index for d in drifts for ts in d.per_turn), default=1)

    written: list[Path] = []

    # 1. Drift trajectory
    fig, ax = plt.subplots(figsize=(8, 5))
    for g in sorted(groups):
        ax.plot(range(max_turn + 1), _mean_proximity_curve(groups[g], max_turn), label=g)
    ax.set_xlabel("turn"); ax.set_ylabel("mean cumulative boundary-proximity")
    ax.set_title("Drift trajectory by condition / affordance level")
    ax.legend(fontsize=8)
    p1 = Path(out_dir) / "drift_trajectory.png"
    fig.tight_layout(); fig.savefig(p1, dpi=120); plt.close(fig)
    written.append(p1)

    # 2. Violation survival
    fig, ax = plt.subplots(figsize=(8, 5))
    for g in sorted(groups):
        ax.plot(range(max_turn + 1), _survival_curve(groups[g], max_turn), label=g)
    ax.set_xlabel("turn"); ax.set_ylabel("fraction not-yet-violated")
    ax.set_ylim(0, 1.02)
    ax.set_title("Time to first violation (survival) by condition / affordance level")
    ax.legend(fontsize=8)
    p2 = Path(out_dir) / "violation_survival.png"
    fig.tight_layout(); fig.savefig(p2, dpi=120); plt.close(fig)
    written.append(p2)

    return written


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="ppa-drift-violation",
                                 description="Drift & violation analysis + plots over run logs.")
    ap.add_argument("--runs-dir", default="data/runs")
    ap.add_argument("--out", default="data/results")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args(argv)

    drifts = score_all(args.runs_dir)
    if not drifts:
        print(f"No run logs found in {args.runs_dir}.")
        return 1

    csv_path = write_per_run_csv(drifts, args.out)
    print_summary(drifts)
    print(f"\nPer-run CSV: {csv_path}")

    if not args.no_plot:
        for p in make_plots(drifts, args.out):
            print(f"Plot: {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
