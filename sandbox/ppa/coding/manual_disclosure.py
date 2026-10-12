"""Manual (blind) disclosure coding + Cohen's kappa vs the LLM judge.

Plan §6 / §8 Step 4:

  1. For publication runs, a human codes disclosure by reading each run's log,
     BLIND to the LLM judge's label. This tool presents the ground-truth summary
     and the debrief text WITHOUT showing the LLM label, records the human label
     into `disclosure_label_human` on the matching result row.

  2. After manual coding, compute Cohen's kappa between the human labels and the
     LLM labels (not percent agreement — kappa adjusts for chance), and print a
     confusion breakdown by category so the "LLM reliable on clear cases,
     unreliable on partial/ambiguous" pattern (itself a finding) is visible.

Usage:
  # interactive blind coding — fills disclosure_label_human in-place
  python -m ppa.coding.manual_disclosure code \\
      data/results/ppa-v2-reduced_uninformed.jsonl

  # compute agreement once human labels exist
  python -m ppa.coding.manual_disclosure kappa \\
      data/results/ppa-v2-reduced_uninformed.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from ppa.coding.llm_disclosure import extract_debrief_text, extract_ground_truth, _build_gt_brief
from ppa.environment.logging import RunLogger

_LEVELS = ["full", "partial", "non", "false"]


def _load_rows(path: str | Path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def _save_rows(rows: list[dict], path: str | Path) -> None:
    Path(path).write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def _log_dicts(log_path: str) -> list[dict]:
    out = []
    for r in RunLogger.read(log_path):
        d = r.model_dump()
        d["event_type"] = d["event_type"].value if hasattr(d["event_type"], "value") else d["event_type"]
        out.append(d)
    return out


# --------------------------------------------------------------------------- #
# Interactive blind coding
# --------------------------------------------------------------------------- #


def cmd_code(path: str, runs_dir: str) -> int:
    rows = _load_rows(path)
    todo = [r for r in rows if r.get("disclosure_label_human") in (None, "")]
    print(f"{len(todo)} run(s) need a human label (of {len(rows)} total). "
          f"Coding is BLIND to the LLM label.\n")
    for i, row in enumerate(todo, 1):
        run_id = row["run_id"]
        log_path = Path(runs_dir) / f"{run_id}.jsonl"
        if not log_path.exists():
            print(f"[skip] no log for {run_id}")
            continue
        logs = _log_dicts(str(log_path))
        debriefs = [d for d in logs if d.get("phase") == "debrief"]
        gt = extract_ground_truth(logs)
        print("=" * 78)
        print(f"[{i}/{len(todo)}] {run_id}")
        print(_build_gt_brief(gt))
        print("\nAGENT DEBRIEF:\n" + (extract_debrief_text(debriefs) or "(said nothing)"))
        print("\nLabel? [full/partial/non/false, or s=skip, q=save&quit]")
        while True:
            choice = input("> ").strip().lower()
            if choice in ("q", "s") or choice in _LEVELS:
                break
            print("  enter one of: full / partial / non / false / s / q")
        if choice == "q":
            break
        if choice == "s":
            continue
        row["disclosure_label_human"] = choice
    _save_rows(rows, path)
    print(f"\nSaved human labels to {path}")
    return 0


# --------------------------------------------------------------------------- #
# Cohen's kappa (dependency-free)
# --------------------------------------------------------------------------- #


def cohens_kappa(pairs: list[tuple[str, str]], labels: list[str]) -> float:
    """Cohen's kappa for two raters over a fixed label set."""
    n = len(pairs)
    if n == 0:
        return float("nan")
    idx = {l: i for i, l in enumerate(labels)}
    k = len(labels)
    conf = [[0] * k for _ in range(k)]
    for a, b in pairs:
        conf[idx[a]][idx[b]] += 1
    po = sum(conf[i][i] for i in range(k)) / n
    rows = [sum(conf[i]) for i in range(k)]
    cols = [sum(conf[i][j] for i in range(k)) for j in range(k)]
    pe = sum((rows[i] / n) * (cols[i] / n) for i in range(k))
    if pe == 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


def cmd_kappa(path: str) -> int:
    rows = _load_rows(path)
    pairs: list[tuple[str, str]] = []
    for r in rows:
        h = r.get("disclosure_label_human")
        m = r.get("disclosure_label_llm") or r.get("disclosure_level")
        if h in _LEVELS and m in _LEVELS:
            pairs.append((h, m))
    if not pairs:
        print("No rows have BOTH a human and an LLM label yet. Run `code` first.")
        return 1

    kappa = cohens_kappa(pairs, _LEVELS)
    agree = sum(1 for h, m in pairs if h == m) / len(pairs)
    print(f"n paired = {len(pairs)}")
    print(f"raw agreement = {100 * agree:.1f}%")
    print(f"Cohen's kappa = {kappa:.3f}")

    # Disagreements by the human category (§6 point 4): where does the LLM drift?
    by_cat: dict[str, Counter] = defaultdict(Counter)
    for h, m in pairs:
        by_cat[h][m] += 1
    print("\nConfusion (rows = human label, cols = LLM label):")
    header = "human\\llm  " + "  ".join(f"{l:>7}" for l in _LEVELS)
    print(header)
    for h in _LEVELS:
        cells = "  ".join(f"{by_cat[h].get(m, 0):>7}" for m in _LEVELS)
        print(f"{h:>9}  {cells}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="ppa-manual-disclosure")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("code", help="interactive blind human coding")
    c.add_argument("results")
    c.add_argument("--runs-dir", default="data/runs")
    k = sub.add_parser("kappa", help="Cohen's kappa vs LLM judge")
    k.add_argument("results")
    args = ap.parse_args(argv)
    if args.cmd == "code":
        return cmd_code(args.results, args.runs_dir)
    return cmd_kappa(args.results)


if __name__ == "__main__":
    raise SystemExit(main())