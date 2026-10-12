"""Command line. One subcommand per phase of the build sequence (design §10).

    python -m ppa_tb nullcal    --dataset ./tb-tasks            # P0
    python -m ppa_tb forge      --config grid.yaml              # P1
    python -m ppa_tb audit      --dataset ./derived_dataset     # P1 gate
    python -m ppa_tb selftest                                   # P2  <- blocking
    python -m ppa_tb run        --config grid.yaml [--dry-run]  # P3+
    python -m ppa_tb collect    --config grid.yaml --job JOB
    python -m ppa_tb readjudicate --audit-dir data/gate_audit
    python -m ppa_tb analyze    --episodes data/episodes.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .analysis import (
    frontier_points,
    gate_verdict_table,
    manipulation_checks,
    plot_frontier,
    violation_rates,
    write_csv,
)
from .conditions import Rejection
from .overlay import Overlay, audit_tier1, render_findings, tier1_passed
from .runner import GridConfig, build_episode, find_bundles, launch, run_id_for, save_episodes
from .schema import Finality, Personalization


# --------------------------------------------------------------------------- #


def cmd_forge(args) -> int:
    from .forge import ForgeSpec, forge
    from .schema import Arm, FlawType

    cfg = GridConfig.load(args.config)
    src_root = Path(args.upstream)
    dest_root = Path(cfg.dataset_path)
    dest_root.mkdir(parents=True, exist_ok=True)

    n = 0
    for task in cfg.tasks:
        overlay = Overlay.load(Path(args.overlays) / f"{task}.yaml")
        for arm in cfg.arms:
            for fin in cfg.finalities:
                for pers in cfg.personalizations:
                    for flaw in cfg.flaw_types:
                        name = f"{task}__{arm}__{fin}__{pers}__{flaw}"
                        spec = ForgeSpec(
                            upstream=src_root / task,
                            dest=dest_root / name,
                            overlay=overlay,
                            rejection=Rejection(Finality(fin), Personalization(pers)),
                            arm=Arm(arm),
                            flaw=FlawType(flaw),
                            presented_check=Path(args.checks, f"{task}.{flaw}.sh").read_text(),
                            hidden_check=Path(args.checks, f"{task}.hidden.sh").read_text(),
                        )
                        forge(spec)
                        n += 1
    print(f"forged {n} derived tasks into {dest_root}")
    return 0


def cmd_audit(args) -> int:
    ok_all = True
    for overlay_file in sorted(Path(args.overlays).glob("*.yaml")):
        overlay = Overlay.load(overlay_file)
        meta = json.loads(Path(args.meta, f"{overlay.task_id}.json").read_text())
        findings = audit_tier1(
            overlay,
            container_paths=set(meta["container_paths"]),
            reference_diff_paths=set(meta["reference_diff_paths"]),
            gate_paths=set(meta["gate_paths"]),
            manifest_reproduced=meta.get("manifest_reproduced", True),
        )
        ok = tier1_passed(findings)
        ok_all = ok_all and ok
        print(f"{overlay.task_id}: {'PASS' if ok else 'FAIL'}")
        print(render_findings(findings))
    print("\nTIER 1:", "PASS" if ok_all else "FAIL - these tasks must not enter the pool")
    return 0 if ok_all else 1


def cmd_validate(args) -> int:
    from .validate import run_validation, write_validation_tables
    ok, episodes = run_validation(simulate=args.simulate, verbose=True)
    if args.out:
        p = write_validation_tables(episodes, Path(args.out))
        print(f"\nvalidation tables -> {p}")
        if args.show_tables:
            print()
            print(p.read_text())
    return 0 if ok else 1


def cmd_routes(args) -> int:
    from .conditions import apply_affordance
    from .routes import audit_pool, pool_summary
    overlays = []
    meta = {}
    for f in sorted(Path(args.overlays).glob("*.yaml")):
        ov = Overlay.load(f)
        apply_affordance(ov, args.affordance)
        overlays.append(ov)
        meta[ov.task_id] = json.loads(Path(args.meta, f"{ov.task_id}.json").read_text())
    reports, ok = audit_pool(overlays, meta)
    for r in reports:
        print(r.render())
        print()
    print(pool_summary(reports))
    return 0 if ok else 1


def cmd_selftest(args) -> int:
    from .selftest import run_selftest
    return 0 if run_selftest(verbose=True) else 1


def cmd_run(args) -> int:
    cfg = GridConfig.load(args.config)
    print(f"grid size: {cfg.size()} episodes")
    if args.dry_run:
        for c in list(cfg.cells())[: args.show]:
            print(" ", run_id_for(*c))
        return 0
    rc = 0
    for model in cfg.models:
        tasks = [
            f"{t}__{a}__{f}__{p}__{fl}"
            for t in cfg.tasks for a in cfg.arms for f in cfg.finalities
            for p in cfg.personalizations for fl in cfg.flaw_types
        ]
        for rep in range(cfg.repetitions):
            job = f"ppa-{model.replace('/', '-')}-r{rep:02d}"
            rc |= launch(cfg, model, tasks, job, dry_run=False)
    return rc


def cmd_collect(args) -> int:
    cfg = GridConfig.load(args.config)
    episodes = []
    bundles = find_bundles(Path(args.job))
    if not bundles:
        print(f"no bundles under {args.job}; check find_bundles() against your harness layout",
              file=sys.stderr)
        return 1
    for trial, bundle in sorted(bundles.items()):
        meta_file = bundle.parent / "ppa_meta.json"
        meta = json.loads(meta_file.read_text()) if meta_file.exists() else _meta_from_name(trial)
        overlay = Overlay.load(Path(cfg.dataset_path, meta["derived_task"], "overlay.yaml"))
        episodes.append(build_episode(trial, bundle, overlay, meta, image=meta.get("image")))
    out = save_episodes(episodes, Path(args.out))
    print(f"collected {len(episodes)} episodes -> {out}")
    return 0


def _meta_from_name(trial: str) -> dict:
    parts = trial.split("__")
    return {
        "derived_task": trial,
        "model": parts[1] if len(parts) > 1 else "unknown",
        "arm": parts[0] if parts else "uninformed",
        "finality": "never_reconsidered",
        "personalization": "impersonal",
        "flaw": "overspecified",
        "repetition": 0,
    }


def cmd_readjudicate(args) -> int:
    from .gate import readjudicate
    changes = readjudicate(Path(args.audit_dir))
    for run_id, old, new in changes:
        print(f"{run_id}: {old} -> {new}")
    print(f"\n{len(changes)} verdicts changed. No agents were rerun.")
    return 0


def load_episodes(path: str) -> list:
    from .schema import Arm, CheckOutcome, DisclosureLevel, EpisodeRecord, Finality
    from .schema import FlawType, GateVerdict, Personalization, ViolationEvent

    episodes = []
    for line in Path(path).read_text().splitlines():
        d = json.loads(line)
        d["arm"] = Arm(d["arm"])
        d["finality"] = Finality(d["finality"])
        d["personalization"] = Personalization(d["personalization"])
        d["flaw_type"] = FlawType(d["flaw_type"])
        d["h_outcome"] = CheckOutcome(d["h_outcome"])
        d["p_outcome"] = CheckOutcome(d["p_outcome"])
        d["gate_verdict"] = GateVerdict(d["gate_verdict"])
        for k in ("disclosure_label_llm", "disclosure_label_human"):
            d[k] = DisclosureLevel(d[k]) if d.get(k) else None
        d["violations"] = [ViolationEvent(**_coerce_violation(v)) for v in d.get("violations", [])]
        d["facts"] = []
        episodes.append(EpisodeRecord(**d))
    return episodes


def cmd_analyze(args) -> int:
    episodes = load_episodes(args.episodes)
    out = Path(args.out)
    checks = manipulation_checks(episodes)
    print(json.dumps(checks, indent=2))
    if checks["reach_rate_spread"] > 0.15:
        print("\n!! reach-rate spread > 15pp across models. ITT violation rates are "
              "confounded by exposure (design §7.3). Lead with the exposure-conditional "
              "column and report reach rate alongside.", file=sys.stderr)

    write_csv([c.row() for c in violation_rates(episodes, ("arm",))], out / "violation_by_arm.csv")
    write_csv([c.row() for c in violation_rates(episodes, ("arm", "model_id"))],
              out / "violation_by_arm_model.csv")
    write_csv([c.row() for c in violation_rates(episodes, ("domain", "difficulty"))],
              out / "violation_by_stratum.csv")
    write_csv(gate_verdict_table(episodes), out / "gate_verdicts_by_shape.csv")
    pts = frontier_points(episodes)
    write_csv(pts, out / "frontier_points.csv")
    (out / "manipulation_checks.json").write_text(json.dumps(checks, indent=2))
    p = plot_frontier(pts, out / "frontier.png")
    print(f"\nwrote tables to {out}" + (f" and {p}" if p else " (matplotlib absent; no plot)"))
    return 0


def _coerce_violation(v: dict) -> dict:
    from .schema import AttributionStatus, ViolationKind
    v = dict(v)
    v["kind"] = ViolationKind(v["kind"])
    v["status"] = AttributionStatus(v["status"])
    return v


def cmd_tables(args) -> int:
    from .paper import build_all
    p = build_all(load_episodes(args.episodes), Path(args.out))
    print(p.read_text())
    print(f"\nwrote {p} and {p.parent / 'tables.tex'}")
    return 0


def cmd_estimate(args) -> int:
    from .cost import (BudgetExceeded, CostModel, MixedCostModel, calibrate_from_pilot,
                       estimate, estimate_mixed, guard)
    cfg = GridConfig.load(args.config)
    if args.openrouter:
        m = MixedCostModel(cache_hit_rate=args.cache_hit_rate)
        if args.from_job:
            base = calibrate_from_pilot(args.from_job)
            m.frac_episodes_touching_gate = base.frac_episodes_touching_gate
        est = estimate_mixed(cfg, m)
        if args.budget and est.total > args.budget:
            print(f"BUDGET GUARD: projected ${est.total:,.0f} exceeds ${args.budget:,.0f}",
                  file=sys.stderr)
            print(est.render(), file=sys.stderr)
            return 1
        print(est.render())
        return 0
    model = calibrate_from_pilot(args.from_job) if args.from_job else CostModel()
    if args.input_price:
        model.input_price_per_mtok = args.input_price
    if args.output_price:
        model.output_price_per_mtok = args.output_price
    try:
        est = guard(cfg, args.budget, model) if args.budget else estimate(cfg, model)
    except BudgetExceeded as e:
        print(f"BUDGET GUARD: {e}", file=sys.stderr)
        return 1
    print(est.render())
    return 0


def cmd_nullcal(args) -> int:
    from .selftest import null_agent_calibration
    return 0 if null_agent_calibration(Path(args.dataset), verbose=True) else 1


# --------------------------------------------------------------------------- #


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="ppa_tb")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("forge"); p.set_defaults(fn=cmd_forge)
    p.add_argument("--config", required=True)
    p.add_argument("--upstream", required=True)
    p.add_argument("--overlays", required=True)
    p.add_argument("--checks", required=True)

    p = sub.add_parser("audit"); p.set_defaults(fn=cmd_audit)
    p.add_argument("--overlays", required=True)
    p.add_argument("--meta", required=True)

    p = sub.add_parser("validate"); p.set_defaults(fn=cmd_validate)
    p.add_argument("--simulate", action="store_true", default=True)
    p.add_argument("--out", default="data/validation")
    p.add_argument("--show-tables", action="store_true")

    p = sub.add_parser("routes"); p.set_defaults(fn=cmd_routes)
    p.add_argument("--overlays", required=True)
    p.add_argument("--meta", required=True)
    p.add_argument("--affordance", default="low",
                   choices=["none", "low", "high"])

    p = sub.add_parser("selftest"); p.set_defaults(fn=cmd_selftest)

    p = sub.add_parser("nullcal"); p.set_defaults(fn=cmd_nullcal)
    p.add_argument("--dataset", required=True)

    p = sub.add_parser("run"); p.set_defaults(fn=cmd_run)
    p.add_argument("--config", required=True)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--show", type=int, default=20)

    p = sub.add_parser("collect"); p.set_defaults(fn=cmd_collect)
    p.add_argument("--config", required=True)
    p.add_argument("--job", required=True)
    p.add_argument("--out", default="data/episodes.jsonl")

    p = sub.add_parser("readjudicate"); p.set_defaults(fn=cmd_readjudicate)
    p.add_argument("--audit-dir", required=True)

    p = sub.add_parser("analyze"); p.set_defaults(fn=cmd_analyze)
    p.add_argument("--episodes", required=True)
    p.add_argument("--out", default="data/analysis")

    p = sub.add_parser("tables"); p.set_defaults(fn=cmd_tables)
    p.add_argument("--episodes", required=True)
    p.add_argument("--out", default="data/paper")

    p = sub.add_parser("estimate"); p.set_defaults(fn=cmd_estimate)
    p.add_argument("--config", required=True)
    p.add_argument("--budget", type=float, default=None,
                   help="abort if projected cost exceeds this (USD)")
    p.add_argument("--from-job", default=None,
                   help="calibrate the cost model from a completed episodes.jsonl")
    p.add_argument("--input-price", type=float, default=None)
    p.add_argument("--output-price", type=float, default=None)
    p.add_argument("--openrouter", action="store_true",
                   help="price each model separately from the OpenRouter table")
    p.add_argument("--cache-hit-rate", type=float, default=0.0,
                   help="fraction of input tokens served from prompt cache (0-1)")

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())