#!/usr/bin/env python3
"""Pre-flight verification for forged episodes.

Run this BEFORE harbor run to catch problems early.

Usage:
    python3 scripts/preflight.py derived_dataset/
    python3 scripts/preflight.py derived_dataset/ --fix   # also attempt fixes

Checks per episode:
  1. review CLI present in ci_assets/
  2. review CLI contains sync_handoff (new templates)
  3. review CLI contains review_log.jsonl (structured logging)
  4. review CLI contains budget_exhausted lockfile logic
  5. NOTES.md planted (affordance)
  6. .deploy_status planted (urgency)
  7. session_id present and non-empty
  8. session_id unique across all episodes (no ghost collisions)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from collections import defaultdict


REQUIRED_STRINGS = {
    "sync_all": "sync_all() function (handoff extraction)",
    "review_log.jsonl": "structured log extraction",
    "budget_exhausted": "budget exhaustion lockfile",
}


def check_episode(task_dir: Path) -> list[str]:
    """Return list of failure messages, empty if all OK."""
    failures = []

    # 1. Review CLI
    review = task_dir / "environment" / "ci_assets" / "review"
    if not review.exists():
        failures.append("MISSING: environment/ci_assets/review")
        return failures  # can't check contents if missing

    content = review.read_text(errors="replace")

    # 2-5. Required strings in CLI
    for key, label in REQUIRED_STRINGS.items():
        if key not in content:
            failures.append(f"MISSING in review CLI: {label} ({key!r})")

    # 6. NOTES.md
    notes = task_dir / "environment" / "ci_assets" / "planted" / "app" / "NOTES.md"
    if not notes.exists():
        failures.append("MISSING: planted NOTES.md")

    # 7. .deploy_status
    deploy = task_dir / "environment" / "ci_assets" / "planted" / "app" / ".deploy_status"
    if not deploy.exists():
        failures.append("MISSING: planted .deploy_status")

    # 8. session_id
    sid_file = task_dir / "environment" / "ci_assets" / ".session_id"
    if not sid_file.exists():
        failures.append("MISSING: .session_id")
    else:
        sid = sid_file.read_text().strip()
        if not sid or len(sid) < 8:
            failures.append(f"INVALID: .session_id is too short: {sid!r}")

    return failures


def main():
    parser = argparse.ArgumentParser(description="Pre-flight check for forged episodes")
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--fix", action="store_true",
                        help="Attempt to fix review CLI from current templates")
    args = parser.parse_args()

    dataset = args.dataset_dir
    if not dataset.is_dir():
        print(f"Not a directory: {dataset}", file=sys.stderr)
        return 1

    episodes = sorted(p for p in dataset.iterdir() if p.is_dir())
    if not episodes:
        print("No episodes found.", file=sys.stderr)
        return 1

    # Check for duplicate session IDs
    sid_map: dict[str, list[str]] = defaultdict(list)
    for ep in episodes:
        sid_file = ep / "environment" / "ci_assets" / ".session_id"
        if sid_file.exists():
            sid = sid_file.read_text().strip()
            sid_map[sid].append(ep.name)

    duplicates = {sid: names for sid, names in sid_map.items() if len(names) > 1}

    # Per-episode checks
    all_ok = True
    n_ok = 0
    n_fail = 0

    for ep in episodes:
        failures = check_episode(ep)

        if failures:
            n_fail += 1
            all_ok = False
            print(f"FAIL  {ep.name}")
            for f in failures:
                print(f"      {f}")

            if args.fix and "MISSING: environment/ci_assets/review" not in failures:
                # Only fix CLI content issues, not missing file
                try:
                    from ppa_tb.templates import REVIEW_CLI
                    review_path = ep / "environment" / "ci_assets" / "review"
                    review_path.write_text(REVIEW_CLI)
                    print(f"      FIXED: rewrote review CLI from current templates")
                except Exception as e:
                    print(f"      FIX FAILED: {e}")
        else:
            n_ok += 1
            print(f"ok    {ep.name}")

    # Duplicate session ID report
    if duplicates:
        all_ok = False
        print(f"\nDUPLICATE SESSION IDs (ghost collision risk):")
        for sid, names in duplicates.items():
            print(f"  {sid[:12]}... -> {names}")

    # Summary
    print(f"\n{'='*60}")
    print(f"Episodes: {len(episodes)} total, {n_ok} OK, {n_fail} failed")
    if duplicates:
        print(f"Duplicate session IDs: {len(duplicates)}")

    if all_ok:
        print("ALL CHECKS PASSED — safe to run harbor")
        return 0
    else:
        print("FAILURES DETECTED — fix before running harbor")
        if args.fix:
            print("Re-run without --fix to verify fixes took effect")
        else:
            print("Try: python3 scripts/preflight.py derived_dataset/ --fix")
        return 1


if __name__ == "__main__":
    sys.exit(main() or 0)