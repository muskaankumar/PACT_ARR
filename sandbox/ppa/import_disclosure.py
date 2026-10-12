"""Import disclosure codes from the CSV back into experiment.jsonl."""

import csv
from ppa.coding._support import load_records, save_records
from ppa.driver.schema import DisclosureLevel

recs = load_records("data/results/experiment.jsonl")

with open("data/results/misaligned_coding.csv") as f:
    codes = {row["run_id"]: row["disclosure_level"].strip().lower() for row in csv.DictReader(f)}

updated = 0
for i, r in enumerate(recs):
    if r.run_id in codes and codes[r.run_id] in ("full", "partial", "non", "false"):
        recs[i] = r.model_copy(update={"disclosure_level": DisclosureLevel(codes[r.run_id])})
        updated += 1

save_records(recs, "data/results/experiment.jsonl")
print(f"Updated {updated} disclosure codes in experiment.jsonl")