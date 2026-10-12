"""Ceiling calibration (build prompt §8.4).

``max_turns`` is a guillotine: set it too low and agents get cut off mid-deliberation
(censoring real behaviour); too high and we waste turns on runs where nothing more
will happen. This report characterizes the run-length distribution from a results
file so the ceiling and trigger timing can be tuned before the full study.

It buckets every run into:

* **immediate**         — stopped almost at once (≤ ``--early`` turns); often a sign
  the agent crossed a boundary or quit the instant it was blocked.
* **natural (pre-ceiling)** — ended on a violation well before the ceiling; the
  interesting case, where something happened with room to spare.
* **ceiling, eventful** — ran to ``max_turns`` but a violation had been recorded;
  acceptable, though a higher ceiling might have surfaced it earlier.
* **ceiling, uneventful** — ran to ``max_turns`` with nothing happening; pure waste,
  and if this bucket is large the ceiling (or the pressure design) needs revisiting.

Reports counts, percentages, and simple quantiles of stop-turn, plus a guidance line.

Run:  ``python -m ppa.pilot.ceiling_calibration data/results/pilot.jsonl``
"""

from __future__ import annotations

import argparse
import statistics

from ppa.coding._support import load_records
from ppa.driver.schema import ResultRecord, StopReason


def _bucket(rec: ResultRecord, early: int) -> str:
    stop = rec.stop_turn if rec.stop_turn is not None else rec.max_turns
    hit_ceiling = rec.stop_reason == StopReason.MAX_TURNS
    if not hit_ceiling and stop <= early:
        return "immediate"
    if not hit_ceiling:
        return "natural_pre_ceiling"
    # hit ceiling
    return "ceiling_eventful" if rec.misaligned() else "ceiling_uneventful"


def calibrate(records: list[ResultRecord], early: int = 3) -> dict:
    buckets = {
        "immediate": 0,
        "natural_pre_ceiling": 0,
        "ceiling_eventful": 0,
        "ceiling_uneventful": 0,
    }
    stop_turns: list[int] = []
    for rec in records:
        buckets[_bucket(rec, early)] += 1
        stop_turns.append(rec.stop_turn if rec.stop_turn is not None else rec.max_turns)

    n = len(records) or 1
    pct = {k: 100.0 * v / n for k, v in buckets.items()}
    stop_turns_sorted = sorted(stop_turns)

    def q(p: float) -> float:
        if not stop_turns_sorted:
            return float("nan")
        idx = min(len(stop_turns_sorted) - 1, int(p * (len(stop_turns_sorted) - 1)))
        return stop_turns_sorted[idx]

    return {
        "n": len(records),
        "buckets": buckets,
        "pct": pct,
        "stop_turn_median": statistics.median(stop_turns) if stop_turns else float("nan"),
        "stop_turn_p10": q(0.10),
        "stop_turn_p90": q(0.90),
        "max_turns": records[0].max_turns if records else None,
    }


def _guidance(stats: dict) -> str:
    pct = stats["pct"]
    msgs = []
    if pct["ceiling_uneventful"] > 40:
        msgs.append("High uneventful-ceiling rate: consider lowering max_turns or "
                    "strengthening the pressure / trigger timing.")
    if pct["immediate"] > 40:
        msgs.append("Many runs end almost immediately: check that the rejection isn't "
                    "ending runs before the agent can respond.")
    if pct["ceiling_eventful"] > 20:
        msgs.append("Violations often coincide with the ceiling: a higher max_turns may "
                    "reveal whether they'd occur earlier.")
    if not msgs:
        msgs.append("Distribution looks balanced; current max_turns appears reasonable.")
    return " ".join(msgs)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ppa-pilot-ceiling", description="Ceiling calibration (§8.4).")
    parser.add_argument("results")
    parser.add_argument("--early", type=int, default=3, help="≤ this many turns counts as 'immediate'")
    args = parser.parse_args(argv)

    records = load_records(args.results)
    stats = calibrate(records, early=args.early)

    print("=" * 72)
    print(f"CEILING CALIBRATION  (n={stats['n']}, max_turns={stats['max_turns']})")
    print("-" * 72)
    labels = {
        "immediate": "ended immediately",
        "natural_pre_ceiling": "natural stop, pre-ceiling",
        "ceiling_eventful": "hit ceiling, eventful",
        "ceiling_uneventful": "hit ceiling, uneventful",
    }
    for key, label in labels.items():
        print(f"  {label:<28} {stats['buckets'][key]:>5}   {stats['pct'][key]:5.1f}%")
    print("-" * 72)
    print(f"  stop-turn  p10={stats['stop_turn_p10']}  median={stats['stop_turn_median']}  "
          f"p90={stats['stop_turn_p90']}")
    print("-" * 72)
    print("  guidance:", _guidance(stats))
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
