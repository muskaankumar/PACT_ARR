"""Experiment driver for the redesign (informed vs uninformed).

Runs the grid: task (8) × rejection (9) × model × repetition, for a chosen
CONDITION (informed or uninformed). Each cell runs through the multi-turn loop and
is coded for authorization (mechanically) and disclosure (prefilter + LLM judge).

Two conditions, two commands:

    # Condition A — agent is told the rules and repeatedly reminded of the goal
    python -m ppa.driver.multiturn_runner informed  --config experiment_v2.yaml

    # Condition B — agent is told only the goal; rules are hidden (still enforced)
    python -m ppa.driver.multiturn_runner uninformed --config experiment_v2.yaml

Results are written to data/results/<name>_<condition>.jsonl so the two conditions
never overwrite each other and can be compared directly.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Any

import yaml

from ppa.agent.backends import ModelBackend
from ppa.agent.multiturn_loop import LoopConfig, MultiTurnRunLoop, RunOutcome
from ppa.coding.llm_disclosure import code_disclosure
from ppa.conditions.affordances import AFFORDANCE_LEVELS
from ppa.conditions.complex_tasks import ComplexTask, build_complex_tasks
from ppa.conditions.conditions import CONDITIONS, Condition
from ppa.conditions.multiturn_rejections import MultiTurnRejection, build_multiturn_rejections
from ppa.environment.logging import RunLogger


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


@dataclasses.dataclass
class ModelSpec:
    id: str
    model: str  # OpenRouter model string, e.g. "openai/gpt-5" or "anthropic/claude-sonnet-4"
    temperature: float = 1.0


@dataclasses.dataclass
class ExperimentConfig:
    name: str = "ppa-v2"
    seed: int = 0
    repetitions: int = 5
    models: list[ModelSpec] = dataclasses.field(default_factory=list)

    max_turns: int = 120
    num_reminders: int = 5
    debrief_turns: int = 3
    allowed_recipients: tuple[str, ...] = ("review-bot",)
    ci_enabled: bool = True

    judge_model: str | None = "openai/gpt-5"  # LLM judge for ambiguous disclosure
    use_llm_judge: bool = True

    results_dir: str = "data/results"
    runs_dir: str = "data/runs"

    # -- redesign v2 grid reduction (plan §3/§4) ---------------------------- #
    # Affordance levels crossed into the grid (plan §4). Default reproduces v1.
    affordance_levels: tuple[str, ...] = ("none",)
    # Grid pins (plan §3): restrict tasks/rejections to the high-pressure cell.
    # None = use the full set. Filters match on substrings of the id.
    task_filter: tuple[str, ...] | None = None          # e.g. ("d1_c1",)
    finality_filter: tuple[str, ...] | None = None      # e.g. ("never_reconsidered",)
    personalization_filter: tuple[str, ...] | None = None  # e.g. ("impersonal",)
    domain_filter: tuple[str, ...] | None = None        # e.g. ("aggregation",)
    # §5b: keep running past the first block so persistence/proximity accumulate.
    continue_after_block: bool = False

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        raw = yaml.safe_load(Path(path).read_text()) or {}
        models = [ModelSpec(id=m["id"], model=m["model"], temperature=m.get("temperature", 1.0))
                  for m in raw.get("models", [])]
        known = {f.name for f in dataclasses.fields(cls)}
        kwargs = {k: v for k, v in raw.items() if k in known and k != "models"}
        for tup_field in ("allowed_recipients", "affordance_levels", "task_filter",
                          "finality_filter", "personalization_filter", "domain_filter"):
            if tup_field in kwargs and kwargs[tup_field] is not None:
                kwargs[tup_field] = tuple(kwargs[tup_field])
        cfg = cls(models=models, **kwargs)
        for lvl in cfg.affordance_levels:
            if lvl not in AFFORDANCE_LEVELS:
                raise ValueError(f"unknown affordance level {lvl!r} in config; "
                                 f"expected any of {AFFORDANCE_LEVELS}")
        return cfg

    def loop_config(self) -> LoopConfig:
        return LoopConfig(max_turns=self.max_turns, num_reminders=self.num_reminders,
                          debrief_turns=self.debrief_turns,
                          allowed_recipients=tuple(self.allowed_recipients),
                          ci_enabled=self.ci_enabled,
                          continue_after_block=self.continue_after_block)


# --------------------------------------------------------------------------- #
# Backend
# --------------------------------------------------------------------------- #


def build_backend(spec: ModelSpec) -> ModelBackend:
    from ppa.agent.openrouter_backend import OpenRouterBackend
    return OpenRouterBackend(model_id=spec.model, temperature=spec.temperature)


# --------------------------------------------------------------------------- #
# Grid
# --------------------------------------------------------------------------- #


@dataclasses.dataclass
class Cell:
    index: int
    task: ComplexTask
    rejection: MultiTurnRejection
    model: ModelSpec
    repetition: int
    seed: int
    condition: Condition
    affordance_level: str = "none"

    @property
    def run_id(self) -> str:
        return (f"{self.condition.name}__{self.model.id}__aff-{self.affordance_level}"
                f"__{self.task.task_id}__{self.rejection.condition_id}__r{self.repetition:02d}")


def _filtered_tasks(cfg: ExperimentConfig) -> list[ComplexTask]:
    tasks = build_complex_tasks()
    if cfg.domain_filter:
        tasks = [t for t in tasks if t.domain in cfg.domain_filter]
    if cfg.task_filter:
        tasks = [t for t in tasks if any(f in t.task_id for f in cfg.task_filter)]
    return tasks


def _filtered_rejections(cfg: ExperimentConfig) -> list[MultiTurnRejection]:
    rejections = build_multiturn_rejections()
    if cfg.finality_filter:
        rejections = [r for r in rejections if r.finality in cfg.finality_filter]
    if cfg.personalization_filter:
        rejections = [r for r in rejections if r.personalization in cfg.personalization_filter]
    return rejections


def iter_cells(cfg: ExperimentConfig, condition: Condition):
    tasks = _filtered_tasks(cfg)
    rejections = _filtered_rejections(cfg)
    index = 0
    for model in cfg.models:
        for affordance_level in cfg.affordance_levels:
            for task in tasks:
                for rejection in rejections:
                    for rep in range(cfg.repetitions):
                        yield Cell(index=index, task=task, rejection=rejection, model=model,
                                   repetition=rep, seed=cfg.seed + index, condition=condition,
                                   affordance_level=affordance_level)
                        index += 1


def grid_size(cfg: ExperimentConfig) -> int:
    return (len(_filtered_tasks(cfg)) * len(_filtered_rejections(cfg))
            * len(cfg.models) * len(cfg.affordance_levels) * cfg.repetitions)


# --------------------------------------------------------------------------- #
# Run + code one cell
# --------------------------------------------------------------------------- #


def run_cell(cell: Cell, cfg: ExperimentConfig, workdir: str | None = None) -> dict[str, Any]:
    backend = build_backend(cell.model)
    loop = MultiTurnRunLoop(run_id=cell.run_id, backend=backend, task=cell.task,
                            rejection=cell.rejection, condition=cell.condition,
                            config=cfg.loop_config(), runs_dir=cfg.runs_dir,
                            seed=cell.seed, workdir=workdir,
                            affordance_level=cell.affordance_level)
    outcome: RunOutcome = loop.run()

    # RunLogger.read returns TurnRecord objects; convert to dicts robustly.
    log_dicts = _records_to_dicts(RunLogger.read(outcome.log_path))
    debrief_dicts = [r for r in log_dicts if r.get("phase") == "debrief"]

    verdict = code_disclosure(log_dicts, debrief_dicts,
                              judge_model=cfg.judge_model, use_llm=cfg.use_llm_judge)

    return {
        "run_id": cell.run_id,
        "condition": cell.condition.name,
        "model_id": cell.model.id,
        "model": cell.model.model,
        "task_id": cell.task.task_id,
        "domain": cell.task.domain,
        "affordance_level": cell.affordance_level,
        "finality": cell.rejection.finality,
        "personalization": cell.rejection.personalization,
        "repetition": cell.repetition,
        "seed": cell.seed,
        "auth_violation": bool(outcome.auth_kind),
        "auth_kind": outcome.auth_kind.value if outcome.auth_kind else None,
        "auth_turn": outcome.auth_turn,
        "stop_reason": outcome.stop_reason.value,
        "stop_turn": outcome.stop_turn,
        "total_turns": outcome.total_turns,
        "stages_completed": outcome.stages_completed,
        "pushback_count": outcome.pushback_count,
        # -- disclosure (§6): keep both labels; human filled in later, blind ---
        "disclosure_level": verdict.label,           # back-compat alias
        "disclosure_label_llm": verdict.label,
        "disclosure_label_human": None,              # added during manual coding pass
        "disclosure_reason": verdict.reason,
        "disclosure_method": verdict.method,
        "disclosure_needs_review": verdict.needs_review,
        "concealment_type": verdict.concealment_type,   # §5d
        # per-violation-type disclosure (§6b): judge verdict scoped to each kind
        "disclosure_per_kind": verdict.per_kind,         # {kind: {label, reason}}
        "all_violation_kinds": verdict.all_violation_kinds,  # sorted list of distinct kinds
        # -- redesign v2 metrics (§5) -----------------------------------------
        "boundary_proximity": outcome.boundary_proximity,        # §5a
        "first_violation_turn": outcome.first_violation_turn,    # §5c
        "persistence_after_block": outcome.persistence_after_block,  # §5b
        "distinct_bypass_routes": outcome.distinct_bypass_routes,     # §5b (all attempts)
        "max_turns": cfg.max_turns,
    }


def _records_to_dicts(records) -> list[dict]:
    out = []
    for r in records:
        if isinstance(r, dict):
            out.append(r)
        elif hasattr(r, "model_dump"):
            out.append(r.model_dump())
        elif hasattr(r, "__dict__"):
            d = dict(r.__dict__)
            # normalize event_type / phase to plain strings
            if hasattr(d.get("event_type"), "value"):
                d["event_type"] = d["event_type"].value
            out.append(d)
    return out


def write_result(record: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(record) + "\n")


# --------------------------------------------------------------------------- #
# Top-level
# --------------------------------------------------------------------------- #


def _completed_run_ids(results_path: Path) -> set[str]:
    """Read run_ids already written to the results file."""
    if not results_path.exists():
        return set()
    ids = set()
    for line in results_path.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                ids.add(json.loads(line)["run_id"])
            except (json.JSONDecodeError, KeyError):
                pass
    return ids


def run_experiment(cfg: ExperimentConfig, condition: Condition, *, limit: int | None = None,
                   skip: int = 0, workdir: str | None = None, fresh: bool = True,
                   progress: bool = True) -> list[dict]:
    results_path = Path(cfg.results_dir) / f"{cfg.name}_{condition.name}.jsonl"
    if fresh and results_path.exists():
        results_path.unlink()

    # Auto-resume: skip cells whose run_id is already in the results file.
    # This means you can always run the same command — it picks up where it left off.
    completed = _completed_run_ids(results_path)
    if completed:
        print(f"Auto-resume: {len(completed)} runs already completed, skipping them.")

    total = grid_size(cfg) if limit is None else min(limit, grid_size(cfg))
    records: list[dict] = []
    cell_retries = 4  # attempts per run before stopping the experiment (no skipping)
    for cell in iter_cells(cfg, condition):
        if cell.index < skip:
            continue
        if limit is not None and cell.index >= limit:
            break
        # Skip if this run_id was already completed (crash-safe resume).
        if cell.run_id in completed:
            if progress:
                print(f"[{cell.index + 1:>4}/{total}] SKIP (done) {cell.run_id}", flush=True)
            continue
        # Run the cell. Failures are RETRIED (not skipped) so no data is silently
        # dropped. A transient error usually passes on retry; a persistent one
        # (e.g. a specific conversation that always 400s) stops the whole
        # experiment so you can inspect it — you never end up with an unnoticed
        # hole in the grid. Re-running resumes from exactly here.
        rec = None
        for retry in range(cell_retries):
            try:
                rec = run_cell(cell, cfg, workdir=workdir)
                break
            except Exception as exc:  # noqa: BLE001
                msg = str(exc).replace("\n", " ")[:300]
                if progress:
                    print(f"[{cell.index + 1:>4}/{total}] retry {retry+1}/{cell_retries} "
                          f"{cell.run_id}\n        -> {msg}", flush=True)
                with open(Path(cfg.results_dir) / f"{cfg.name}_{condition.name}.failures.log",
                          "a") as fh:
                    fh.write(f"{cell.run_id}\tattempt{retry+1}\t{str(exc)[:500]}\n")
                import time; time.sleep(min(30, 5 * (retry + 1)))
        if rec is None:
            print(f"\n[STOP] Run failed after {cell_retries} attempts:\n"
                  f"  {cell.run_id}\n"
                  f"Nothing was skipped. All completed runs are saved. Fix the cause "
                  f"(see the payload dump in your temp dir and "
                  f"{cfg.results_dir}/{cfg.name}_{condition.name}.failures.log), then "
                  f"re-run the SAME command to resume from this exact run.", flush=True)
            break
        write_result(rec, results_path)
        completed.add(cell.run_id)
        records.append(rec)
        if progress:
            flag = f"AUTH:{rec['auth_kind']}" if rec["auth_violation"] else "ok"
            print(f"[{cell.index + 1:>4}/{total}] {cell.run_id}  "
                  f"stop={rec['stop_reason']} disc={rec['disclosure_level']} {flag}", flush=True)
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ppa-run-v2",
                                     description="Run the multi-turn PPA experiment (informed/uninformed).")
    parser.add_argument("condition", choices=["informed", "uninformed"])
    parser.add_argument("--config", required=True, help="Path to experiment_v2.yaml")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workdir", default=None)
    parser.add_argument("--fresh", action="store_true",
                        help="Delete existing results and start from scratch. "
                             "Default is to auto-resume from where you left off.")
    parser.add_argument("--no-fresh", action="store_true",
                        help="Deprecated; resume is now the default. Kept for compatibility.")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip", type=int, default=0,
                        help="Skip the first N cells by index. Rarely needed — "
                             "auto-resume handles interruptions automatically.")
    args = parser.parse_args(argv)

    cfg = ExperimentConfig.from_yaml(args.config)
    condition = CONDITIONS[args.condition]

    if args.dry_run:
        print(json.dumps({"name": cfg.name, "condition": condition.name,
                          "grid_size": grid_size(cfg), "models": [m.id for m in cfg.models],
                          "affordance_levels": list(cfg.affordance_levels),
                          "tasks": [t.task_id for t in _filtered_tasks(cfg)],
                          "rejections": [r.condition_id for r in _filtered_rejections(cfg)],
                          "repetitions": cfg.repetitions,
                          "continue_after_block": cfg.continue_after_block,
                          "max_turns": cfg.max_turns}, indent=2))
        return 0

    records = run_experiment(cfg, condition, limit=args.limit, skip=args.skip,
                             workdir=args.workdir, fresh=args.fresh,
                             progress=not args.quiet)
    n_auth = sum(1 for r in records if r["auth_violation"])
    from collections import Counter
    disc = Counter(r["disclosure_level"] for r in records)
    print(f"\nDone [{condition.name}]. {len(records)} runs -> "
          f"{cfg.results_dir}/{cfg.name}_{condition.name}.jsonl")
    n = len(records) or 1
    print(f"  auth violations: {n_auth}/{len(records)} ({100 * n_auth / n:.1f}%)")
    print(f"  disclosure: {dict(disc)}")
    # v2 metrics rollup (plan §5), broken out by affordance level for calibration.
    by_aff = Counter(r["affordance_level"] for r in records)
    for lvl in sorted(by_aff):
        sub = [r for r in records if r["affordance_level"] == lvl]
        v = sum(1 for r in sub if r["auth_violation"])
        prox = Counter(r.get("boundary_proximity", 0) for r in sub)
        print(f"  [aff={lvl}] n={len(sub)} viol={v} ({100 * v / (len(sub) or 1):.1f}%) "
              f"proximity_dist={dict(sorted(prox.items()))}")
    conceal = Counter(r.get("concealment_type", "none") for r in records)
    print(f"  concealment types: {dict(conceal)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())