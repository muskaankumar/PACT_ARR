"""One-shot experiment runner — no prompts, no manual gates.

Running this does everything end to end:

  1. Runs the UNINFORMED condition across all affordance levels (none/low/high).
  2. Runs the INFORMED condition across the same levels.
  3. Prints the condition comparison (primary finding: violation-rate delta by
     affordance level, plus proximity / concealment / persistence / first-turn).

The affordance levels come straight from the config's `affordance_levels`
(experiment_reduced.yaml ships with ["none", "low", "high"]). Nothing here asks
the user anything; it just runs.

Usage:
    python -m ppa.driver.run_all                       # uses experiment_reduced.yaml
    python -m ppa.driver.run_all --config other.yaml
    python -m ppa.driver.run_all --fresh               # ignore auto-resume, start clean
    python -m ppa.driver.run_all --dry-run             # show the plan, run nothing
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ppa.analysis.condition_comparison import (
    _print_affordance_deltas,
    load,
    print_summary,
    summarize,
)
from ppa.conditions.conditions import CONDITIONS
from ppa.driver.multiturn_runner import ExperimentConfig, grid_size, run_experiment

_DEFAULT_CONFIG = "experiment_full.yaml"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="ppa-run-all",
        description="Run both conditions across all affordance levels, then compare. No prompts.",
    )
    ap.add_argument("--config", default=_DEFAULT_CONFIG, help=f"default: {_DEFAULT_CONFIG}")
    ap.add_argument("--fresh", action="store_true",
                    help="Delete existing results and start over (default is auto-resume).")
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="Print the plan and exit.")
    args = ap.parse_args(argv)

    cfg = ExperimentConfig.from_yaml(args.config)
    per_condition = grid_size(cfg)

    print("=" * 70)
    print("PPA v2 — one-shot run (both conditions, all affordance levels)")
    print("=" * 70)
    print(f"  config            : {args.config}")
    print(f"  affordance levels : {list(cfg.affordance_levels)}")
    print(f"  runs / condition  : {per_condition}")
    print(f"  total runs        : {per_condition * 2}  (uninformed + informed)")
    print(f"  models            : {[m.id for m in cfg.models]}")
    print(f"  max_turns         : {cfg.max_turns}   reps: {cfg.repetitions}")
    print(f"  continue_after_block: {cfg.continue_after_block}")
    if args.dry_run:
        print("\n(dry run — nothing executed)")
        return 0

    # Run uninformed first (that's the one that must reach 30–60%), then informed.
    for cond_name in ("uninformed", "informed"):
        condition = CONDITIONS[cond_name]
        print("\n" + "-" * 70)
        print(f"RUNNING CONDITION: {cond_name}  ({per_condition} runs)")
        print("-" * 70)
        run_experiment(cfg, condition, workdir=args.workdir, fresh=args.fresh,
                       progress=not args.quiet)

    # Comparison across both result files.
    inf_path = Path(cfg.results_dir) / f"{cfg.name}_informed.jsonl"
    unf_path = Path(cfg.results_dir) / f"{cfg.name}_uninformed.jsonl"
    if not (inf_path.exists() and unf_path.exists()):
        print("\n[warn] one or both result files missing; skipping comparison.")
        return 0

    inf_recs = load(str(inf_path))
    unf_recs = load(str(unf_path))
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print_summary(summarize(inf_recs, "informed"))
    print_summary(summarize(unf_recs, "uninformed"))
    _print_affordance_deltas(inf_recs, unf_recs)

    # -- violation types, time-to-violation, model comparison (from existing scripts)
    from pathlib import Path as _Path
    out_dir = str(_Path(cfg.results_dir))
    try:
        from ppa.analysis.violation_types import analyze as _vtypes
        from ppa.analysis._support import load_results as _lr, write_csv as _wcsv, write_markdown as _wmd
        all_recs = _lr(str(inf_path)) + _lr(str(unf_path))
        rows, md = _vtypes(all_recs)
        _wcsv(rows, f"{out_dir}/violation_types.csv")
        _wmd(md, f"{out_dir}/violation_types.md")
        print(f"\n{md}")
    except Exception as e:
        print(f"\n[warn] violation_types skipped: {e}")
    try:
        from ppa.analysis.time_to_violation import analyze as _surv
        all_recs = _lr(str(inf_path)) + _lr(str(unf_path))
        rows, md = _surv(all_recs)
        _wcsv(rows, f"{out_dir}/time_to_violation.csv")
        _wmd(md, f"{out_dir}/time_to_violation.md")
        print(f"\n{md}")
    except Exception as e:
        print(f"\n[warn] time_to_violation skipped: {e}")
    try:
        from ppa.analysis.model_comparison import analyze as _mcomp
        all_recs = _lr(str(inf_path)) + _lr(str(unf_path))
        rows, md = _mcomp(all_recs)
        _wcsv(rows, f"{out_dir}/model_comparison.csv")
        _wmd(md, f"{out_dir}/model_comparison.md")
        print(f"\n{md}")
    except Exception as e:
        print(f"\n[warn] model_comparison skipped: {e}")

    print("\nResult files:")
    print(f"  {unf_path}")
    print(f"  {inf_path}")

    # Drift & violation trajectory analysis + plots (over the raw run logs).
    try:
        from ppa.analysis.drift_violation import main as drift_main
        print("\n" + "=" * 70)
        print("DRIFT & VIOLATION TRAJECTORIES")
        print("=" * 70)
        drift_main(["--runs-dir", cfg.runs_dir, "--out", cfg.results_dir])
    except Exception as exc:  # noqa: BLE001 - analysis is best-effort
        print(f"[warn] drift/violation analysis skipped: {exc}")

    print("\nOptional next steps (manual, not run here):")
    print("  - blind human disclosure coding + kappa:")
    print(f"      python -m ppa.coding.manual_disclosure code  {unf_path}")
    print(f"      python -m ppa.coding.manual_disclosure kappa {unf_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
