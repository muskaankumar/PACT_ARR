"""Experiment driver (build prompt §7).

Iterates the full grid ``task × rejection × model × repetition``, runs each cell
through :class:`~ppa.agent.loop.RunLoop`, then codes the outcome:

* **Authorization** is coded *mechanically* from the run's JSONL log
  (:func:`ppa.coding.authorization.code_authorization`) — ground truth, no judgment.
* **Truthfulness** is *seeded* from the loop's conservative mid-run heuristic (it
  only ever fired the stop). The authoritative §5.2 truthfulness label and the
  §5.3 disclosure label come later from the human-coding CLIs; this driver leaves
  ``disclosure_level`` as ``None`` and records the heuristic note so a human coder
  can confirm or overturn it.

Each cell yields one validated :class:`~ppa.driver.schema.ResultRecord`, appended
to ``data/results/<results_file>``. The JSONL run log remains the source of truth
and is fully replayable; the result record is the coded summary analysis consumes.

Determinism: every cell gets a derived seed ``base_seed + cell_index`` so a run is
reproducible from config alone, and ``run_id`` encodes the full coordinate.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import yaml

from ppa.agent.backends import ModelBackend
from ppa.agent.loop import LoopConfig, RunLoop, RunOutcome
from ppa.agent.mock_policies import make_mock_backend
from ppa.coding.authorization import code_authorization_from_log
from ppa.conditions.rejections import Rejection, build_rejections
from ppa.conditions.tasks import Task, build_tasks
from ppa.driver.schema import (
    RejectionCondition,
    ResultRecord,
    StopReason,
    TruthfulnessViolation,
)

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


@dataclasses.dataclass
class ModelSpec:
    """One model arm of the experiment (§7's model contrast pairs live here)."""

    id: str
    backend: str = "mock"  # mock | anthropic | openai
    # For mock backends, the policy key (defaults to id). For real backends, the
    # provider model string (e.g. "claude-sonnet-4-6").
    policy: str | None = None
    pair: str | None = None  # "A" (safety-trained) | "B" (capability) — labelling only

    @property
    def resolved_policy(self) -> str:
        return self.policy or self.id


@dataclasses.dataclass
class ExperimentConfig:
    """Parsed experiment / pilot YAML (see experiment.yaml, pilot.yaml)."""

    name: str = "ppa-experiment"
    seed: int = 0
    repetitions: int = 10
    task_layout: str = "2x2x2"
    models: list[ModelSpec] = dataclasses.field(default_factory=list)

    # loop knobs (mirrors LoopConfig; kept here so config is one file)
    max_turns: int = 40
    num_reminders: int = 3
    debrief_turns: int = 2
    allowed_recipients: tuple[str, ...] = ("review-bot",)
    ci_enabled: bool = True
    rejection_delay_after_pr: int = 0

    # output
    results_dir: str = "data/results"
    results_file: str | None = None  # defaults to "<name>.jsonl"
    runs_dir: str = "data/runs"

    # template / mock options
    template_dir: str = "templates/sandbox_repo"
    mock_with_tests: bool = True

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        raw = yaml.safe_load(Path(path).read_text()) or {}
        models = [
            ModelSpec(
                id=m["id"],
                backend=m.get("backend", "mock"),
                policy=m.get("policy"),
                pair=m.get("pair"),
            )
            for m in raw.get("models", [])
        ]
        known = {f.name for f in dataclasses.fields(cls)}
        kwargs: dict[str, Any] = {k: v for k, v in raw.items() if k in known and k != "models"}
        if "allowed_recipients" in kwargs and kwargs["allowed_recipients"] is not None:
            kwargs["allowed_recipients"] = tuple(kwargs["allowed_recipients"])
        return cls(models=models, **kwargs)

    def loop_config(self) -> LoopConfig:
        return LoopConfig(
            max_turns=self.max_turns,
            num_reminders=self.num_reminders,
            debrief_turns=self.debrief_turns,
            allowed_recipients=tuple(self.allowed_recipients),
            ci_enabled=self.ci_enabled,
            rejection_delay_after_pr=self.rejection_delay_after_pr,
        )

    def results_path(self) -> Path:
        fname = self.results_file or f"{self.name}.jsonl"
        return Path(self.results_dir) / fname


# --------------------------------------------------------------------------- #
# Backend selection
# --------------------------------------------------------------------------- #


def build_backend(spec: ModelSpec, task: Task, cfg: ExperimentConfig) -> ModelBackend:
    """Instantiate the backend for a model arm.

    Mock backends are fully deterministic and self-contained (the default, and what
    the whole grid is validated against). Real-provider backends are import-guarded
    and intentionally wired last (build order §10); selecting one here will raise a
    clear error until its ``.step`` is implemented.
    """
    backend = spec.backend.lower()
    if backend == "mock":
        return make_mock_backend(
            spec.resolved_policy,
            task,
            template_dir=cfg.template_dir,
            with_tests=cfg.mock_with_tests,
        )
    if backend == "anthropic":
        from ppa.agent.backends import AnthropicBackend

        return AnthropicBackend(model_id=spec.resolved_policy)
    if backend == "openai":
        from ppa.agent.backends import OpenAIBackend

        return OpenAIBackend(model_id=spec.resolved_policy)
    raise ValueError(f"unknown backend {spec.backend!r} for model {spec.id!r}")


# --------------------------------------------------------------------------- #
# Grid
# --------------------------------------------------------------------------- #


@dataclasses.dataclass
class Cell:
    index: int
    task: Task
    rejection: Rejection
    model: ModelSpec
    repetition: int
    seed: int

    @property
    def run_id(self) -> str:
        return (
            f"{self.model.id}__{self.task.task_id}__{self.rejection.condition_id}"
            f"__r{self.repetition:02d}"
        )


def iter_cells(cfg: ExperimentConfig) -> Iterator[Cell]:
    """Enumerate the experiment grid in a stable order with derived seeds."""
    tasks = build_tasks(layout=cfg.task_layout)
    rejections = build_rejections()
    index = 0
    for model in cfg.models:
        for task in tasks:
            for rejection in rejections:
                for rep in range(cfg.repetitions):
                    yield Cell(
                        index=index,
                        task=task,
                        rejection=rejection,
                        model=model,
                        repetition=rep,
                        seed=cfg.seed + index,
                    )
                    index += 1


def grid_size(cfg: ExperimentConfig) -> int:
    tasks = build_tasks(layout=cfg.task_layout)
    return len(tasks) * len(build_rejections()) * len(cfg.models) * cfg.repetitions


# --------------------------------------------------------------------------- #
# Coding a single run into a ResultRecord
# --------------------------------------------------------------------------- #


def _rejection_condition(rej: Rejection) -> RejectionCondition:
    return RejectionCondition(finality=rej.finality, personalization=rej.personalization)


def code_outcome(cell: Cell, outcome: RunOutcome) -> ResultRecord:
    """Combine mechanical auth coding with the loop's heuristic truthfulness seed.

    Disclosure and the descriptive taxonomy are deliberately left unset here; they
    require the human-coding CLIs (§5.2, §5.3, §6). ``disclosure_probe_turn`` is
    carried through so those CLIs know where the debrief window starts.
    """
    auth = code_authorization_from_log(outcome.log_path)

    truth = TruthfulnessViolation(occurred=False)
    if outcome.stop_reason == StopReason.TRUTHFULNESS_VIOLATION:
        truth = TruthfulnessViolation(
            occurred=True,
            turn=outcome.truthfulness_turn,
            note="heuristic stop; pending §5.2 human confirmation",
        )

    return ResultRecord(
        run_id=cell.run_id,
        seed=cell.seed,
        task_id=cell.task.task_id,
        rejection_condition=_rejection_condition(cell.rejection),
        model_id=cell.model.id,
        repetition=cell.repetition,
        auth_violation=auth,
        truthfulness_violation=truth,
        disclosure_level=None,  # coded later by ppa-code-disclosure
        disclosure_probe_turn=outcome.debrief_probe_turn,
        descriptive_tags=[],  # coded later by ppa-code-descriptive
        # ceiling is needed by survival analysis for right-censoring
        max_turns=int(cell.__dict__.get("_max_turns", 0)),
        stop_reason=outcome.stop_reason,
        stop_turn=outcome.stop_turn,
    )


# --------------------------------------------------------------------------- #
# Run one cell
# --------------------------------------------------------------------------- #


def run_cell(cell: Cell, cfg: ExperimentConfig, workdir: str | None = None) -> ResultRecord:
    loop_cfg = cfg.loop_config()
    # stash ceiling on the cell so code_outcome can record it for censoring
    cell.__dict__["_max_turns"] = loop_cfg.max_turns
    backend = build_backend(cell.model, cell.task, cfg)
    loop = RunLoop(
        run_id=cell.run_id,
        backend=backend,
        task=cell.task,
        rejection=cell.rejection,
        config=loop_cfg,
        runs_dir=cfg.runs_dir,
        seed=cell.seed,
        workdir=workdir,
    )
    outcome = loop.run()
    return code_outcome(cell, outcome)


# --------------------------------------------------------------------------- #
# Persist
# --------------------------------------------------------------------------- #


def write_result(record: ResultRecord, results_path: Path) -> None:
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("a") as fh:
        fh.write(record.model_dump_json() + "\n")


# --------------------------------------------------------------------------- #
# Top-level run
# --------------------------------------------------------------------------- #


def run_experiment(
    cfg: ExperimentConfig,
    *,
    limit: int | None = None,
    workdir: str | None = None,
    fresh: bool = True,
    progress: bool = True,
) -> list[ResultRecord]:
    """Run (a prefix of) the grid, persisting one ResultRecord per cell.

    ``limit`` caps the number of cells (useful for smoke tests). ``fresh`` truncates
    the results file first so a rerun is idempotent.
    """
    results_path = cfg.results_path()
    if fresh and results_path.exists():
        results_path.unlink()

    total = grid_size(cfg) if limit is None else min(limit, grid_size(cfg))
    records: list[ResultRecord] = []
    for cell in iter_cells(cfg):
        if limit is not None and cell.index >= limit:
            break
        record = run_cell(cell, cfg, workdir=workdir)
        write_result(record, results_path)
        records.append(record)
        if progress:
            flag = "MISALIGNED" if record.misaligned() else "ok"
            print(
                f"[{cell.index + 1:>4}/{total}] {cell.run_id}  "
                f"stop={record.stop_reason.value} {flag}",
                flush=True,
            )
    return records


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ppa-run", description="Run the PPA experiment grid.")
    parser.add_argument("config", help="Path to experiment.yaml / pilot.yaml")
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N cells")
    parser.add_argument("--workdir", default=None, help="Where sandbox repos are built")
    parser.add_argument(
        "--no-fresh", action="store_true", help="Append to the results file instead of truncating"
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress per-cell progress")
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the grid size and exit without running"
    )
    args = parser.parse_args(argv)

    cfg = ExperimentConfig.from_yaml(args.config)

    if args.dry_run:
        print(json.dumps({"name": cfg.name, "grid_size": grid_size(cfg), "models": [m.id for m in cfg.models]}, indent=2))
        return 0

    records = run_experiment(
        cfg,
        limit=args.limit,
        workdir=args.workdir,
        fresh=not args.no_fresh,
        progress=not args.quiet,
    )
    n_misaligned = sum(1 for r in records if r.misaligned())
    print(
        f"\nDone. {len(records)} runs -> {cfg.results_path()}  "
        f"({n_misaligned} misaligned, {len(records) - n_misaligned} aligned)"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
