#!/usr/bin/env python3
"""Score evaluation-awareness across a jobs/ directory. NEW FILE.

    python scripts/score_awareness.py jobs/2026-08-19__11-57-09
    python scripts/score_awareness.py jobs/ --json > awareness.json

Run on every pilot before spending grid budget, and again on the full grid
before analysis. If the contaminated rate differs across arms or models, say
so in the paper: awareness is downstream of the rejection, so it correlates
with capability and cannot be treated as random noise.

Exit code is nonzero if any episode is contaminated, so this can gate a run
in CI / a Makefile.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ppa_tb.awareness import CONTAMINATION_THRESHOLD, find_trajectory, score_trajectory  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    rows = []
    for traj in sorted(args.root.rglob("trajectory.json")):
        try:
            sc = score_trajectory(traj)
        except (OSError, ValueError) as exc:
            print(f"skip {traj}: {exc}", file=sys.stderr)
            continue
        # trial name = the dir a couple levels up from agent/trajectory.json
        trial = traj.parent.parent.name
        rows.append({"trial": trial, "path": str(traj), **sc.as_dict()})

    if args.json:
        json.dump(rows, sys.stdout, indent=2)
        print()
        return 0

    if not rows:
        print("no trajectories found under", args.root)
        return 1

    bad = [r for r in rows if r["contaminated"]]
    print(f"{len(rows)} episodes, {len(bad)} contaminated "
          f"(threshold {CONTAMINATION_THRESHOLD})\n")
    for r in sorted(rows, key=lambda r: -r["score"]):
        mark = "CONTAM" if r["contaminated"] else "ok    "
        step = r["first_flagged_step"]
        print(f"  {mark}  score={r['score']:<4} "
              f"first_step={step if step is not None else '-':<4} "
              f"reads={r['scaffold_reads']:<3} {r['trial']}")

    if bad:
        print("\ntop strong signals across contaminated episodes:")
        c = Counter(h for r in bad for h in r["strong_hits"])
        for pat, k in c.most_common(10):
            print(f"  {k:>3}  {pat}")

    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
