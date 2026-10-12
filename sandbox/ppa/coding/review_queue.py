"""Human spot-check of flagged disclosure runs (redesign).

Disclosure is coded AUTOMATICALLY for every run — the deterministic prefilter
handles the clear cases and the LLM judge handles the ambiguous ones. A human is
NOT required to assign levels. The only thing left for a human is an optional
spot-check of the runs that fell to the LLM judge (``disclosure_needs_review`` is
True), to confirm or override the judge's call.

Two subcommands:

    # Export the flagged runs (auto-label + judge reason + debrief) to a CSV.
    python -m ppa.coding.review_queue export data/results/ppa-v2_informed.jsonl

    # After a human fills the `confirmed_level` column (blank = accept auto-label),
    # import the confirmations back into the results file.
    python -m ppa.coding.review_queue import data/results/ppa-v2_informed.jsonl

If you never run the review step, the auto-labels stand — the pipeline is complete
without human input. The review only lets a human correct the subset the judge
flagged as uncertain.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from ppa.environment.logging import RunLogger

_LEVELS = {"full", "partial", "non", "false"}


def _load(path: str) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def _save(records: list[dict], path: str) -> None:
    with Path(path).open("w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")


def _debrief_text(run_id: str, runs_dir: str) -> str:
    p = Path(runs_dir) / f"{run_id}.jsonl"
    if not p.exists():
        return "(run log not found)"
    text = []
    for rec in RunLogger.read(p):
        d = rec.model_dump() if hasattr(rec, "model_dump") else rec.__dict__
        et = d.get("event_type")
        et = et.value if hasattr(et, "value") else et
        if d.get("phase") == "debrief" and et == "agent_text":
            text.append(str(d.get("payload", {}).get("text", "")))
    return "\n\n".join(text) or "(agent said nothing in debrief)"


def do_export(results_path: str, runs_dir: str, out_csv: str | None) -> None:
    records = _load(results_path)
    flagged = [r for r in records if r.get("disclosure_needs_review")]
    out_csv = out_csv or results_path.replace(".jsonl", "_review.csv")

    fields = ["run_id", "condition", "model_id", "task_id", "finality", "personalization",
              "auth_violation", "auth_kind", "auto_level", "judge_reason", "method",
              "debrief_text", "confirmed_level"]
    with Path(out_csv).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in flagged:
            w.writerow({
                "run_id": r["run_id"], "condition": r.get("condition"),
                "model_id": r.get("model_id"), "task_id": r.get("task_id"),
                "finality": r.get("finality"), "personalization": r.get("personalization"),
                "auth_violation": r.get("auth_violation"), "auth_kind": r.get("auth_kind"),
                "auto_level": r.get("disclosure_level"),
                "judge_reason": r.get("disclosure_reason"),
                "method": r.get("disclosure_method"),
                "debrief_text": _debrief_text(r["run_id"], runs_dir),
                "confirmed_level": "",  # human fills this only to OVERRIDE
            })
    print(f"Exported {len(flagged)} flagged run(s) -> {out_csv}")
    print("Fill `confirmed_level` only where you disagree with `auto_level` "
          "(blank = accept the auto-label). Then run the import subcommand.")


def do_import(results_path: str, in_csv: str | None) -> None:
    in_csv = in_csv or results_path.replace(".jsonl", "_review.csv")
    overrides: dict[str, str] = {}
    with Path(in_csv).open() as f:
        for row in csv.DictReader(f):
            lvl = (row.get("confirmed_level") or "").strip().lower()
            if lvl in _LEVELS:
                overrides[row["run_id"]] = lvl

    records = _load(results_path)
    n = 0
    for r in records:
        if r["run_id"] in overrides:
            r["disclosure_level"] = overrides[r["run_id"]]
            r["disclosure_method"] = "human_review"
            r["disclosure_needs_review"] = False
            n += 1
    _save(records, results_path)
    print(f"Applied {n} human override(s) -> {results_path}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="ppa-review-queue")
    ap.add_argument("action", choices=["export", "import"])
    ap.add_argument("results", help="path to a *_informed/uninformed.jsonl results file")
    ap.add_argument("--runs-dir", default="data/runs")
    ap.add_argument("--csv", default=None, help="CSV path (defaults next to the results file)")
    args = ap.parse_args(argv)

    if args.action == "export":
        do_export(args.results, args.runs_dir, args.csv)
    else:
        do_import(args.results, args.csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
