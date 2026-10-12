"""Write the strict human disclosure labels from the human-judge CSV back into
the corrected result JSONL files (``disclosure_label_human`` + reason), so the
analysis layer can consume them directly.

Runs whose disclosure was not coded (compliant, fully-disclosed) keep
``disclosure_label_human = None``; nothing is invented for them.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, type=Path)
    ap.add_argument("--result-file", action="append", required=True, type=Path,
                    dest="result_files")
    args = ap.parse_args()

    labels: dict[str, tuple[str, str]] = {}
    with args.csv.open() as fh:
        for row in csv.DictReader(fh):
            hl = (row.get("human_label") or "").strip()
            if hl:
                labels[row["run_id"]] = (hl, row.get("human_notes", ""))

    for rf in args.result_files:
        recs = [json.loads(l) for l in rf.read_text().splitlines() if l.strip()]
        n = 0
        for r in recs:
            if r["run_id"] in labels:
                hl, note = labels[r["run_id"]]
                r["disclosure_label_human"] = hl
                r["disclosure_human_reason"] = note
                n += 1
        with rf.open("w") as fh:
            for r in recs:
                fh.write(json.dumps(r) + "\n")
        print(f"  {rf.name}: wrote {n} human labels")


if __name__ == "__main__":
    main()
