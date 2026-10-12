"""Append the agent's debrief text to each row of a per-violation disclosure CSV.

The disclosure CSV has one row per (run x violation type), but does not carry the
agent's actual debrief. This pulls the debrief from each run's raw log and adds it
as a column, so a human coder can read what the agent said next to the LLM's label
without opening the logs separately.

The debrief is the agent's response to the accountability review that the harness
injects once, at the end of the run (event: debrief_injected). All agent_text
events after that point are joined in order.

Usage:
    python build_v3/append_debrief_to_csv.py \
        --input  disclosure_llm_per_violation.csv \
        --runs-dir data/runs \
        --out    disclosure_llm_per_violation_with_debrief.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


def extract_debrief(events: list[dict]) -> str:
    """Agent's accountability-review response (injected once, at run end)."""
    started = False
    parts: list[str] = []
    for e in events:
        if e.get("event_type") == "debrief_injected":
            started = True
            continue
        if started and e.get("event_type") == "agent_text":
            parts.append(str(e.get("payload", {}).get("text", "")))
    return "\n\n".join(parts).strip()


def extract_debrief_question(events: list[dict]) -> str:
    """The accountability-review prompt the harness injected."""
    for e in events:
        if e.get("event_type") == "debrief_injected":
            p = e.get("payload", {}) or {}
            # payload key varies by harness version
            for k in ("text", "prompt", "question", "message", "content"):
                if p.get(k):
                    return str(p[k])
            return json.dumps(p)
    return ""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, type=Path,
                    help="per-violation disclosure CSV")
    ap.add_argument("--runs-dir", required=True, type=Path,
                    help="directory of raw run .jsonl logs")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--max-chars", type=int, default=0,
                    help="truncate debrief to N chars (0 = no truncation)")
    ap.add_argument("--include-question", action="store_true",
                    help="also add the debrief_question column")
    args = ap.parse_args()

    rows = list(csv.DictReader(args.input.open(encoding="utf-8")))
    if not rows:
        sys.exit("input CSV is empty")

    # Cache debriefs per run so multi-violation runs only parse the log once.
    cache: dict[str, tuple[str, str]] = {}
    missing: list[str] = []

    for r in rows:
        rid = r["run_id"]
        if rid not in cache:
            fp = args.runs_dir / f"{rid}.jsonl"
            if not fp.exists():
                cache[rid] = ("(log not found)", "")
                missing.append(rid)
            else:
                events = [json.loads(l) for l in fp.read_text().splitlines()
                          if l.strip()]
                cache[rid] = (extract_debrief(events),
                              extract_debrief_question(events))
        debrief, question = cache[rid]
        if args.max_chars and len(debrief) > args.max_chars:
            debrief = debrief[:args.max_chars] + " …[truncated]"
        r["agent_debrief_text"] = debrief
        if args.include_question:
            r["debrief_question"] = question

    # Preserve original column order, append the new columns at the end.
    fieldnames = list(rows[0].keys())

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    n_runs = len({r["run_id"] for r in rows})
    empty = sum(1 for r in rows if not r["agent_debrief_text"].strip()
                or r["agent_debrief_text"] == "(log not found)")
    lengths = [len(r["agent_debrief_text"]) for r in rows
               if r["agent_debrief_text"] != "(log not found)"]

    print(f"Rows                 : {len(rows)}")
    print(f"Distinct runs        : {n_runs}")
    print(f"Logs not found       : {len(set(missing))}")
    print(f"Rows w/ empty debrief: {empty}")
    if lengths:
        print(f"Debrief length       : min={min(lengths)} "
              f"median={sorted(lengths)[len(lengths)//2]} max={max(lengths)}")
    print(f"\nWritten to: {args.out}")
    if missing:
        print("\nMissing logs for:")
        for rid in sorted(set(missing))[:10]:
            print(f"  {rid}")


if __name__ == "__main__":
    main()
