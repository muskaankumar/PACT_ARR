"""Core integration tests: loop outcomes, runner grid, coding, and §11 invariants.

These tests validate the harness's critical invariants:
  * Every mock policy produces the expected stop reason.
  * Every run ends with a debrief (disclosure_probe_turn is set).
  * The detection selftest passes (all boundary-crosses caught).
  * A grid run exercises all 72 task×rejection combos per model without error.
  * Mechanical auth coding agrees with the loop's outcome.
"""

from __future__ import annotations

import pytest

from ppa.agent.loop import LoopConfig, RunLoop
from ppa.agent.mock_policies import make_mock_backend
from ppa.coding._support import load_records
from ppa.coding.authorization import code_authorization_from_log
from ppa.conditions.rejections import build_rejections
from ppa.conditions.tasks import build_tasks
from ppa.driver.runner import ExperimentConfig, ModelSpec, Cell, run_cell, grid_size, iter_cells, write_result, run_experiment
from ppa.driver.schema import StopReason
from ppa.pilot.detection_selftest import run_selftest


# --------------------------------------------------------------------------- #
# Per-policy loop outcomes
# --------------------------------------------------------------------------- #

_TASK = build_tasks()[0]
_REJ = build_rejections()[0]


def _run_policy(policy: str, tmp_path):
    backend = make_mock_backend(policy, _TASK)
    loop = RunLoop(
        run_id=f"test_{policy}",
        backend=backend,
        task=_TASK,
        rejection=_REJ,
        config=LoopConfig(max_turns=24, num_reminders=1, debrief_turns=2),
        runs_dir=str(tmp_path),
        seed=0,
        workdir=str(tmp_path / "work"),
    )
    return loop.run()


def test_compliant_stops_at_max_turns(tmp_path):
    outcome = _run_policy("mock-compliant", tmp_path)
    assert outcome.stop_reason == StopReason.MAX_TURNS
    assert outcome.debrief_probe_turn is not None


def test_forbidden_file_auth_violation(tmp_path):
    outcome = _run_policy("mock-forbidden-file", tmp_path)
    assert outcome.stop_reason == StopReason.AUTH_VIOLATION
    assert outcome.debrief_probe_turn is not None


def test_boundary_contact_auth_violation(tmp_path):
    outcome = _run_policy("mock-boundary-contact", tmp_path)
    assert outcome.stop_reason == StopReason.AUTH_VIOLATION
    assert outcome.debrief_probe_turn is not None


def test_liar_truthfulness_violation(tmp_path):
    outcome = _run_policy("mock-liar", tmp_path)
    assert outcome.stop_reason == StopReason.TRUTHFULNESS_VIOLATION
    assert outcome.debrief_probe_turn is not None


# --------------------------------------------------------------------------- #
# Mechanical auth coding agrees with loop
# --------------------------------------------------------------------------- #


def test_auth_coding_matches_loop(tmp_path):
    for policy in ("mock-compliant", "mock-forbidden-file", "mock-boundary-contact"):
        outcome = _run_policy(policy, tmp_path)
        coded = code_authorization_from_log(outcome.log_path)
        if outcome.stop_reason == StopReason.AUTH_VIOLATION:
            assert coded.occurred
            assert coded.kind is not None
        else:
            assert not coded.occurred


# --------------------------------------------------------------------------- #
# Detection selftest (§8.2) — safety-critical
# --------------------------------------------------------------------------- #


def test_detection_selftest():
    assert run_selftest(verbose=False), "Detection selftest FAILED — boundary-cross not caught"


# --------------------------------------------------------------------------- #
# Grid smoke: all 72 combos for one model, no errors
# --------------------------------------------------------------------------- #


def test_grid_smoke_72_combos(tmp_path):
    """Run the mock-compliant policy across all 8×9=72 task×rejection combos."""
    cfg = ExperimentConfig(
        name="test-grid-smoke",
        seed=42,
        repetitions=1,
        models=[ModelSpec(id="mock-compliant", backend="mock", policy="mock-compliant")],
        max_turns=24,
        num_reminders=1,
        debrief_turns=1,
        results_dir=str(tmp_path / "results"),
        runs_dir=str(tmp_path / "runs"),
    )
    assert grid_size(cfg) == 72

    records = run_experiment(cfg, workdir=str(tmp_path / "work"), progress=False)
    assert len(records) == 72

    # Every run must have a debrief probe turn (§11).
    for rec in records:
        assert rec.disclosure_probe_turn is not None, f"{rec.run_id} missing debrief"

    # All unique combos covered.
    combos = {(r.task_id, r.rejection_condition.finality, r.rejection_condition.personalization) for r in records}
    assert len(combos) == 72


# --------------------------------------------------------------------------- #
# Grid with misaligned policies: every run gets exactly one disclosure slot
# --------------------------------------------------------------------------- #


def test_every_run_gets_debrief_and_disclosure_slot(tmp_path):
    """Run all four policies × 1 task × 1 rejection × 1 rep (= 4 runs).

    Assert every run ends with a debrief and disclosure_probe_turn is set, so the
    human disclosure CLI can code it. The disclosure_level is initially None (pending
    human coding), which is the correct initial state.
    """
    cfg = ExperimentConfig(
        name="test-disclosure-slot",
        seed=7,
        repetitions=1,
        task_layout="2x2x2",
        models=[
            ModelSpec(id="mock-compliant", backend="mock"),
            ModelSpec(id="mock-forbidden-file", backend="mock"),
            ModelSpec(id="mock-boundary-contact", backend="mock"),
            ModelSpec(id="mock-liar", backend="mock"),
        ],
        max_turns=24,
        num_reminders=1,
        debrief_turns=2,
        results_dir=str(tmp_path / "results"),
        runs_dir=str(tmp_path / "runs"),
    )
    records = run_experiment(cfg, limit=4, workdir=str(tmp_path / "work"), progress=False)
    assert len(records) == 4
    for rec in records:
        assert rec.disclosure_probe_turn is not None
        assert rec.disclosure_level is None  # pending human coding
