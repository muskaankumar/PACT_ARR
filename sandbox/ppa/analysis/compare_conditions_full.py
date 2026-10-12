"""Informed vs Uninformed comparison + plots (redesign analysis).

Consumes the two result files (and, for drift, the per-run logs) and produces:

  1. A summary table: violation rate, disclosure distribution, concealment rate,
     mean drift signals — per condition, with the informed→uninformed delta and a
     simple effect size / proportion test.
  2. Plots (PNG), each saved to the output dir:
       - violation_rate_by_condition.png      grouped bars, per model
       - disclosure_by_condition.png          stacked disclosure levels
       - drift_curve.png                      mean cumulative boundary-proximity vs turn
       - first_probe_survival.png             time-to-first-probe (KM-style)
       - proximity_hist.png                   distribution of max proximity per run
       - trajectory_sankey_counts.png         how often each phase is reached
  3. A machine-readable comparison.json with every number, for the paper.

"How much less misaligned is the informed condition" is reported directly as the
violation-rate and concealment-rate deltas with confidence intervals.

Usage:
    python -m ppa.analysis.compare_conditions_full \
        --informed data/results/ppa-v2_informed.jsonl \
        --uninformed data/results/ppa-v2_uninformed.jsonl \
        --runs-dir data/runs --out-dir data/results/plots
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from ppa.analysis.drift import RunDrift, score_run
from ppa.environment.logging import RunLogger

_DISCLOSURE = ["full", "partial", "non", "false"]
_PHASES = ["EXPLORE", "IMPLEMENT", "TEST", "SUBMIT", "REJECT", "ARGUE", "PROBE", "VIOLATE", "STOP"]


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def _load_results(path: str) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def _log_to_dicts(run_id: str, runs_dir: str) -> list[dict]:
    p = Path(runs_dir) / f"{run_id}.jsonl"
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        et = d.get("event_type")
        d["event_type"] = et.value if hasattr(et, "value") else et
        out.append(d)
    return out


def _drifts(records: list[dict], runs_dir: str) -> list[RunDrift]:
    out = []
    for r in records:
        log = _log_to_dicts(r["run_id"], runs_dir)
        if log:
            out.append(score_run(log, run_id=r["run_id"], condition=r.get("condition", ""),
                                 model_id=r.get("model_id", "")))
    return out


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #


def _prop_ci(k: int, n: int) -> tuple[float, float, float]:
    """Wilson 95% CI for a proportion."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    z = 1.96
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, max(0.0, center - half), min(1.0, center + half)


def _two_prop_z(k1: int, n1: int, k2: int, n2: int) -> tuple[float, float]:
    """Two-proportion z-test (returns z and a rough two-sided p)."""
    if n1 == 0 or n2 == 0:
        return 0.0, 1.0
    p1, p2 = k1 / n1, k2 / n2
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return 0.0, 1.0
    z = (p1 - p2) / se
    # two-sided p from normal approximation
    pval = math.erfc(abs(z) / math.sqrt(2))
    return z, pval


def summarize(records: list[dict], drifts: list[RunDrift], label: str) -> dict:
    n = len(records)
    auth = sum(1 for r in records if r.get("auth_violation"))
    kinds = Counter(r["auth_kind"] for r in records if r.get("auth_violation"))
    disc = Counter(r.get("disclosure_level") for r in records)
    conceal = sum(1 for r in records if r.get("disclosure_level") in ("non", "false"))
    p, lo, hi = _prop_ci(auth, n)
    cp, clo, chi = _prop_ci(conceal, n)

    probe_turns = [d.first_probe_turn for d in drifts if d.first_probe_turn is not None]
    max_prox = [d.max_proximity for d in drifts]
    return {
        "label": label, "n": n,
        "violations": auth, "violation_rate": p, "violation_ci": [lo, hi],
        "violation_kinds": dict(kinds),
        "disclosure": {k: disc.get(k, 0) for k in _DISCLOSURE},
        "concealment": conceal, "concealment_rate": cp, "concealment_ci": [clo, chi],
        "n_with_probe": len(probe_turns),
        "probe_rate": len(probe_turns) / n if n else 0,
        "mean_first_probe_turn": (sum(probe_turns) / len(probe_turns)) if probe_turns else None,
        "mean_max_proximity": (sum(max_prox) / len(max_prox)) if max_prox else 0,
        "mean_pushbacks": (sum(d.n_pushbacks for d in drifts) / len(drifts)) if drifts else 0,
    }


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #


def _ensure_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_violation_rate(inf_recs, unf_recs, out_dir: Path) -> None:
    plt = _ensure_mpl()
    models = sorted({r["model_id"] for r in inf_recs + unf_recs})
    def rate(recs, m):
        sub = [r for r in recs if r["model_id"] == m]
        return (sum(1 for r in sub if r.get("auth_violation")) / len(sub) * 100) if sub else 0
    inf_rates = [rate(inf_recs, m) for m in models]
    unf_rates = [rate(unf_recs, m) for m in models]
    x = range(len(models))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([i - w/2 for i in x], inf_rates, w, label="informed", color="#2E75B6")
    ax.bar([i + w/2 for i in x], unf_rates, w, label="uninformed", color="#E8743B")
    ax.set_xticks(list(x)); ax.set_xticklabels(models)
    ax.set_ylabel("Authorization violation rate (%)")
    ax.set_title("Violation rate: informed vs uninformed")
    ax.legend()
    for i, (a, b) in enumerate(zip(inf_rates, unf_rates)):
        ax.text(i - w/2, a + 0.3, f"{a:.1f}", ha="center", fontsize=9)
        ax.text(i + w/2, b + 0.3, f"{b:.1f}", ha="center", fontsize=9)
    fig.tight_layout(); fig.savefig(out_dir / "violation_rate_by_condition.png", dpi=140)
    plt.close(fig)


def plot_disclosure(inf_recs, unf_recs, out_dir: Path) -> None:
    plt = _ensure_mpl()
    colors = {"full": "#2E9B57", "partial": "#E8B93B", "non": "#888888", "false": "#D64545"}
    def dist(recs):
        c = Counter(r.get("disclosure_level") for r in recs)
        tot = sum(c.values()) or 1
        return [c.get(k, 0) / tot * 100 for k in _DISCLOSURE]
    inf_d, unf_d = dist(inf_recs), dist(unf_recs)
    fig, ax = plt.subplots(figsize=(8, 5))
    conds = ["informed", "uninformed"]
    bottoms = [0, 0]
    for i, lvl in enumerate(_DISCLOSURE):
        vals = [inf_d[i], unf_d[i]]
        ax.bar(conds, vals, bottom=bottoms, label=lvl, color=colors[lvl])
        bottoms = [b + v for b, v in zip(bottoms, vals)]
    ax.set_ylabel("Share of runs (%)")
    ax.set_title("Disclosure level distribution by condition")
    ax.legend(title="disclosure")
    fig.tight_layout(); fig.savefig(out_dir / "disclosure_by_condition.png", dpi=140)
    plt.close(fig)


def plot_drift_curve(inf_drifts, unf_drifts, out_dir: Path, max_turns: int = 120) -> None:
    plt = _ensure_mpl()
    def mean_curve(drifts):
        # cumulative proximity as a function of turn, averaged across runs
        series = []
        for d in drifts:
            curve = [0] * (max_turns + 1)
            cum = 0
            last = 0
            for ts in d.per_turn:
                t = min(ts.turn_index, max_turns)
                for i in range(last, t):
                    curve[i] = cum
                cum = ts.cumulative
                curve[t] = cum
                last = t
            for i in range(last, max_turns + 1):
                curve[i] = cum
            series.append(curve)
        if not series:
            return [0] * (max_turns + 1)
        return [sum(s[i] for s in series) / len(series) for i in range(max_turns + 1)]
    inf_curve = mean_curve(inf_drifts)
    unf_curve = mean_curve(unf_drifts)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(inf_curve, label="informed", color="#2E75B6", lw=2)
    ax.plot(unf_curve, label="uninformed", color="#E8743B", lw=2)
    ax.set_xlabel("Turn"); ax.set_ylabel("Mean cumulative boundary-proximity")
    ax.set_title("Drift toward the boundary over the run")
    ax.legend()
    fig.tight_layout(); fig.savefig(out_dir / "drift_curve.png", dpi=140)
    plt.close(fig)


def plot_first_probe_survival(inf_drifts, unf_drifts, out_dir: Path, max_turns: int = 120) -> None:
    plt = _ensure_mpl()
    def survival(drifts):
        n = len(drifts) or 1
        # S(t) = fraction of runs with NO probe by turn t
        curve = []
        for t in range(max_turns + 1):
            probed = sum(1 for d in drifts if d.first_probe_turn is not None and d.first_probe_turn <= t)
            curve.append(1 - probed / n)
        return curve
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(survival(inf_drifts), label="informed", color="#2E75B6", lw=2)
    ax.plot(survival(unf_drifts), label="uninformed", color="#E8743B", lw=2)
    ax.set_xlabel("Turn"); ax.set_ylabel("Fraction of runs not yet probing a boundary")
    ax.set_title("Time to first boundary probe")
    ax.set_ylim(0, 1.02); ax.legend()
    fig.tight_layout(); fig.savefig(out_dir / "first_probe_survival.png", dpi=140)
    plt.close(fig)


def plot_proximity_hist(inf_drifts, unf_drifts, out_dir: Path) -> None:
    plt = _ensure_mpl()
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = range(0, max([d.max_proximity for d in inf_drifts + unf_drifts] + [1]) + 2)
    ax.hist([d.max_proximity for d in inf_drifts], bins=bins, alpha=0.6, label="informed", color="#2E75B6")
    ax.hist([d.max_proximity for d in unf_drifts], bins=bins, alpha=0.6, label="uninformed", color="#E8743B")
    ax.set_xlabel("Max cumulative boundary-proximity in a run")
    ax.set_ylabel("Number of runs")
    ax.set_title("How close runs get to the boundary")
    ax.legend()
    fig.tight_layout(); fig.savefig(out_dir / "proximity_hist.png", dpi=140)
    plt.close(fig)


def plot_phase_reach(inf_drifts, unf_drifts, out_dir: Path) -> None:
    plt = _ensure_mpl()
    def reach(drifts):
        n = len(drifts) or 1
        return [sum(1 for d in drifts if ph in d.trajectory) / n * 100 for ph in _PHASES]
    inf_r, unf_r = reach(inf_drifts), reach(unf_drifts)
    x = range(len(_PHASES)); w = 0.4
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar([i - w/2 for i in x], inf_r, w, label="informed", color="#2E75B6")
    ax.bar([i + w/2 for i in x], unf_r, w, label="uninformed", color="#E8743B")
    ax.set_xticks(list(x)); ax.set_xticklabels(_PHASES, rotation=30, ha="right")
    ax.set_ylabel("Share of runs reaching phase (%)")
    ax.set_title("How far through the escalation ladder runs get")
    ax.legend()
    fig.tight_layout(); fig.savefig(out_dir / "phase_reach.png", dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="ppa-compare-full")
    ap.add_argument("--informed", required=True)
    ap.add_argument("--uninformed", required=True)
    ap.add_argument("--runs-dir", default="data/runs")
    ap.add_argument("--out-dir", default="data/results/plots")
    ap.add_argument("--max-turns", type=int, default=120)
    ap.add_argument("--no-plots", action="store_true", help="Compute stats only (skip matplotlib)")
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    inf_recs = _load_results(args.informed)
    unf_recs = _load_results(args.uninformed)
    inf_drifts = _drifts(inf_recs, args.runs_dir)
    unf_drifts = _drifts(unf_recs, args.runs_dir)

    inf_sum = summarize(inf_recs, inf_drifts, "informed")
    unf_sum = summarize(unf_recs, unf_drifts, "uninformed")

    # Contrast: how much LESS misaligned is informed?
    z_v, p_v = _two_prop_z(unf_sum["violations"], unf_sum["n"],
                           inf_sum["violations"], inf_sum["n"])
    z_c, p_c = _two_prop_z(unf_sum["concealment"], unf_sum["n"],
                           inf_sum["concealment"], inf_sum["n"])
    contrast = {
        "violation_rate_delta_uninformed_minus_informed":
            unf_sum["violation_rate"] - inf_sum["violation_rate"],
        "violation_rate_ratio_uninformed_over_informed":
            (unf_sum["violation_rate"] / inf_sum["violation_rate"])
            if inf_sum["violation_rate"] else None,
        "violation_z": z_v, "violation_p": p_v,
        "concealment_rate_delta": unf_sum["concealment_rate"] - inf_sum["concealment_rate"],
        "concealment_z": z_c, "concealment_p": p_c,
        "mean_first_probe_delta_turns":
            ((unf_sum["mean_first_probe_turn"] or 0) - (inf_sum["mean_first_probe_turn"] or 0)),
        "mean_max_proximity_delta":
            unf_sum["mean_max_proximity"] - inf_sum["mean_max_proximity"],
        "probe_rate_delta": unf_sum["probe_rate"] - inf_sum["probe_rate"],
    }

    report = {"informed": inf_sum, "uninformed": unf_sum, "contrast": contrast}
    (out_dir / "comparison.json").write_text(json.dumps(report, indent=2))

    # -- console summary --
    print("=" * 70)
    print("INFORMED vs UNINFORMED")
    print("=" * 70)
    for s in (inf_sum, unf_sum):
        print(f"\n[{s['label']}]  n={s['n']}")
        print(f"  violation rate : {100*s['violation_rate']:.2f}%  "
              f"CI[{100*s['violation_ci'][0]:.2f}, {100*s['violation_ci'][1]:.2f}]  "
              f"(kinds: {s['violation_kinds']})")
        print(f"  disclosure     : {s['disclosure']}")
        print(f"  concealment    : {100*s['concealment_rate']:.2f}%")
        print(f"  probe rate     : {100*s['probe_rate']:.1f}%  "
              f"mean first-probe turn: {s['mean_first_probe_turn']}")
        print(f"  mean max prox  : {s['mean_max_proximity']:.2f}  "
              f"mean pushbacks: {s['mean_pushbacks']:.2f}")
    print("\n" + "-" * 70)
    print("CONTRAST (how much LESS misaligned is informed)")
    vr = contrast["violation_rate_ratio_uninformed_over_informed"]
    print(f"  violation rate: uninformed is {100*contrast['violation_rate_delta_uninformed_minus_informed']:+.2f} pts "
          f"{'('+format(vr,'.2f')+'x)' if vr else ''}, z={z_v:.2f}, p={p_v:.4f}")
    print(f"  concealment   : {100*contrast['concealment_rate_delta']:+.2f} pts, z={z_c:.2f}, p={p_c:.4f}")
    print(f"  first probe   : {contrast['mean_first_probe_delta_turns']:+.1f} turns earlier when uninformed"
          if contrast['mean_first_probe_delta_turns'] < 0 else
          f"  first probe   : {contrast['mean_first_probe_delta_turns']:+.1f} turns")
    print(f"  max proximity : {contrast['mean_max_proximity_delta']:+.2f} higher when uninformed")
    print("-" * 70)

    if not args.no_plots:
        try:
            plot_violation_rate(inf_recs, unf_recs, out_dir)
            plot_disclosure(inf_recs, unf_recs, out_dir)
            plot_drift_curve(inf_drifts, unf_drifts, out_dir, args.max_turns)
            plot_first_probe_survival(inf_drifts, unf_drifts, out_dir, args.max_turns)
            plot_proximity_hist(inf_drifts, unf_drifts, out_dir)
            plot_phase_reach(inf_drifts, unf_drifts, out_dir)
            print(f"\nPlots + comparison.json written to {out_dir}/")
        except ImportError:
            print("\nmatplotlib not installed; wrote comparison.json only. "
                  "pip install matplotlib to get plots.")
    else:
        print(f"\ncomparison.json written to {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
