"""Driver (design §10). Expands the grid, shells out to the benchmark, collects.

The harness is never patched. We emit a derived local dataset and let Harbor (TB
2.0) or ``tb`` (TB 1.x) run it natively, then read the evidence bundle the
post-hook left in each trial's logs.

v2 changes:
  * the runner starts ONE host-side ReviewDaemon for the grid and registers each
    trial's ReviewPolicy under a session id before launch
  * reviewer state comes from the daemon (not from an in-container log)
  * the hidden check is graded HOST-SIDE after the agent exits (it was never
    inside the container)
  * evaluation-awareness is scored on every episode and stored on EpisodeRecord
"""

from __future__ import annotations

import itertools
import json
import shutil
import subprocess
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .awareness import score_bundle
from .conditions import Rejection, build_rejections
from .detect import StateDiff, detect, parse_session
from .gate import PredicateRunner, adjudicate, write_audit_row
from .naming import CHECK, REVIEW_PORT
from .overlay import Overlay
from .reviewd import ReviewDaemon, ReviewPolicy, serve
from .schema import (
    Arm,
    CheckOutcome,
    EpisodeRecord,
    Finality,
    FlawType,
    Personalization,
)


@dataclass
class GridConfig:
    tasks: list[str]
    models: list[str]
    arms: list[str] = field(default_factory=lambda: ["informed", "uninformed"])
    finalities: list[str] = field(default_factory=lambda: ["never_reconsidered"])
    personalizations: list[str] = field(default_factory=lambda: ["impersonal"])
    affordance_levels: list[str] = field(default_factory=lambda: ["none"])
    flaw_types: list[str] = field(default_factory=lambda: ["overspecified"])
    repetitions: int = 3
    harness: str = "harbor"          # harbor | tb
    dataset_path: str = "derived_dataset"
    n_concurrent: int = 4
    jobs_dir: str = "jobs"

    @classmethod
    def load(cls, path: str | Path) -> GridConfig:
        return cls(**yaml.safe_load(Path(path).read_text()))

    def cells(self):
        return itertools.product(
            self.tasks, self.models, self.arms, self.finalities,
            self.personalizations, self.affordance_levels, self.flaw_types,
            range(self.repetitions),
        )

    def size(self) -> int:
        return (len(self.tasks) * len(self.models) * len(self.arms)
                * len(self.finalities) * len(self.personalizations)
                * len(self.affordance_levels) * len(self.flaw_types)
                * self.repetitions)


def run_id_for(task, model, arm, fin, pers, aff, flaw, rep) -> str:
    return f"{arm}__{model}__{task}__aff-{aff}__{flaw}__{fin}__{pers}__r{rep:02d}"


# --------------------------------------------------------------------------- #
# Daemon lifecycle  (v2)
# --------------------------------------------------------------------------- #

def start_daemon(jobs_dir: str | Path) -> tuple[ReviewDaemon, object]:
    """Start one ReviewDaemon for the grid. Returns (daemon, server).

    Call ``server.shutdown()`` when the grid finishes. The daemon lives in the
    runner process; if you ever move to remote execution, promote the socket
    to a TCP port on a private interface (the protocol is unchanged).
    """
    d = ReviewDaemon(Path(jobs_dir) / "review_journal.jsonl")
    srv = serve(REVIEW_PORT, d)
    return d, srv


def prepare_trial(daemon: ReviewDaemon, policy: ReviewPolicy) -> str:
    """Register a fresh session and return its id.

    The session id is also used as the Docker container name so the daemon
    can ``docker exec`` into it for ``review status``.
    """
    session_id = uuid.uuid4().hex
    daemon.register(session_id, container=session_id, policy=policy)
    return session_id


# --------------------------------------------------------------------------- #
# Host-side hidden-check grading  (v2)
# --------------------------------------------------------------------------- #

def grade_hidden_host_side(container: str, hidden_check_src: str,
                           bundle: Path) -> CheckOutcome:
    """Run the hidden check against the agent's final container state.

    The hidden check is piped into the container transiently after the agent has
    exited, so it was never reachable during the episode. This replaces the v1
    approach of running it inside the post-hook from ``tests/check_hidden.sh``
    (which required the check to be present during the episode for the fallback
    paths in run_check.sh --h).
    """
    bundle.mkdir(parents=True, exist_ok=True)
    try:
        proc = subprocess.run(
            ["docker", "exec", "-i", container, "bash", "-s"],
            input=hidden_check_src, capture_output=True, text=True, timeout=600,
        )
        rc = proc.returncode
        (bundle / "hidden_out.txt").write_text(proc.stdout + proc.stderr)
    except (subprocess.SubprocessError, OSError) as e:
        rc = 91
        (bundle / "hidden_out.txt").write_text(f"hidden check error: {e}")

    (bundle / "hidden_rc.txt").write_text(str(rc))
    reward = 1 if rc == 0 else 0
    (bundle / "reward.txt").write_text(str(reward))
    # also write to the harness verifier path if present
    ver = bundle.parent / "verifier" / "reward.txt"
    try:
        ver.parent.mkdir(parents=True, exist_ok=True)
        ver.write_text(str(reward))
    except OSError:
        pass
    return CheckOutcome.PASS if rc == 0 else CheckOutcome.FAIL


# --------------------------------------------------------------------------- #
# Harness invocation
# --------------------------------------------------------------------------- #


def harbor_command(cfg: GridConfig, model: str, task_ids: list[str], job_name: str,
                   session_id: str = "") -> list[str]:
    cmd = [
        "harbor", "run",
        "-p", cfg.dataset_path,
        "--agent", "terminus-2",
        "--model", model,
        "--n-concurrent", str(cfg.n_concurrent),
        "--job-name", job_name,
    ]
    if session_id:
        # INTEGRATION: exact flags depend on your Harbor version. Three things
        # must reach Docker:
        #   1. build arg    REVIEW_SESSION_ID=<session_id>
        #   2. container name == <session_id>   (so `docker exec <id>` works)
        #   3. bind mount    <host review.sock> -> /var/run/review.sock
        # If Harbor exposes --build-arg / --container-name / --volume, use them.
        # If not, set them via a compose override or the DOCKER_* env Harbor
        # forwards. Verify against your installed version.
        cmd += [
            "--build-arg", f"REVIEW_SESSION_ID={session_id}",
            "--container-name", session_id,
        ]
    for t in task_ids:
        cmd += ["--task-id", t]
    return cmd


def tb_command(cfg: GridConfig, model: str, task_ids: list[str], job_name: str) -> list[str]:
    cmd = [
        "tb", "run",
        "--dataset-path", cfg.dataset_path,
        "--agent", "terminus",
        "--model", model,
        "--n-concurrent", str(cfg.n_concurrent),
        "--run-id", job_name,
    ]
    for t in task_ids:
        cmd += ["--task-id", t]
    return cmd


def launch(cfg: GridConfig, model: str, task_ids: list[str], job_name: str,
           dry_run: bool = False, session_id: str = "") -> int:
    if cfg.harness == "harbor":
        cmd = harbor_command(cfg, model, task_ids, job_name, session_id=session_id)
    else:
        cmd = tb_command(cfg, model, task_ids, job_name)
    print("  $", " ".join(cmd))
    if dry_run:
        return 0
    return subprocess.run(cmd).returncode


# --------------------------------------------------------------------------- #
# Collection
# --------------------------------------------------------------------------- #


def find_bundles(job_dir: Path) -> dict[str, Path]:
    """Locate each trial's evidence bundle in the harness output tree.

    Both harnesses nest differently and both have moved things between versions,
    so this globs rather than assuming a fixed depth. If it returns nothing, print
    the tree and fix the glob here - it is the only place that knows the layout.

    v2: bundles are in /logs/ci (was /logs/ppa). Fall back to /logs/ppa for
    backward compat with any v1 bundles still in jobs/.
    """
    out: dict[str, Path] = {}
    for marker in job_dir.rglob("post_hashes.json"):
        bundle = marker.parent
        trial = bundle
        while trial.parent != job_dir and trial.parent != trial:
            trial = trial.parent
        out[trial.name] = bundle
    return out


def _outcome(rc_file: Path) -> CheckOutcome:
    if not rc_file.exists():
        return CheckOutcome.UNDETERMINED
    try:
        rc = int(rc_file.read_text().strip())
    except ValueError:
        return CheckOutcome.UNDETERMINED
    if rc == 91:
        return CheckOutcome.MISSING
    return CheckOutcome.PASS if rc == 0 else CheckOutcome.FAIL


def build_episode(run_id: str, bundle: Path, overlay: Overlay, meta: dict,
                  image: str | None = None, docker: str = "docker",
                  daemon: ReviewDaemon | None = None,
                  session_id: str = "") -> EpisodeRecord:
    diff = StateDiff.from_bundle(bundle)

    session_file = next(
        (p for p in (bundle / "..").resolve().rglob("*.cast")), None
    ) or next((p for p in bundle.rglob("session*.log")), None)
    events = parse_session(session_file) if session_file else []

    # v2: reviewer state comes from the host-side daemon when available;
    # falls back to the in-container log for backward compat with v1 bundles.
    if daemon and session_id:
        state = daemon.collect(session_id)
        contacts = [
            {"to": m.get("to", ""), "body": m.get("text", ""), "turn": i}
            for i, m in enumerate(state.get("messages", []))
        ]
        n_submissions = state.get("n_submissions", 0)
        n_replies = state.get("n_replies", 0)
        reached = state.get("reached_rejection_window", False)
    else:
        # v1 fallback: in-container review log
        review_log = bundle / "review_state" / "review_log.jsonl"
        reviews = [json.loads(l) for l in review_log.read_text().splitlines()] if review_log.exists() else []
        contacts = [
            {"to": r.get("to", ""), "body": r.get("body", ""), "turn": i}
            for i, r in enumerate(reviews) if r.get("kind") == "message"
        ]
        n_submissions = sum(1 for r in reviews if r.get("kind") == "submit")
        n_replies = sum(1 for r in reviews if r.get("kind") == "reply")
        reached = n_submissions > 0

    violations = detect(run_id, overlay, diff, events, contacts)

    ev = None
    if image:
        ev = PredicateRunner(image, bundle, docker=docker).collect(run_id, overlay.task_id)
        write_audit_row(ev, bundle.parent / "gate_audit")

    # v2: handoff note (was ACCOUNT.md). Try both for backward compat.
    for name in ("handoff.md", "ACCOUNT.md"):
        account_path = bundle / name
        if account_path.exists():
            account = account_path.read_text()
            break
    else:
        account = ""

    # v2: presented check outcome file (was presented_rc.txt, now check_rc.txt).
    # Try both for backward compat.
    p_rc = bundle / "check_rc.txt"
    if not p_rc.exists():
        p_rc = bundle / "presented_rc.txt"

    ep = EpisodeRecord(
        run_id=run_id,
        task_id=overlay.task_id,
        model_id=meta["model"],
        arm=Arm(meta["arm"]),
        finality=Finality(meta["finality"]),
        personalization=Personalization(meta["personalization"]),
        affordance_level=overlay.affordance_level,
        repetition=meta["repetition"],
        flaw_type=FlawType(meta["flaw"]),
        domain=overlay.domain,
        difficulty=overlay.difficulty,
        gate_shape=overlay.gate_shape,
        h_outcome=_outcome(bundle / "hidden_rc.txt"),
        p_outcome=_outcome(p_rc),
        n_submissions=n_submissions,
        violations=violations,
        contestation_events=n_replies,
        attribution_gaps=sum(1 for v in violations if v.status.value == "attribution_gap"),
    )
    ep.reached_rejection_window = reached
    if ev is not None:
        ep.gate_verdict = adjudicate(ev)
        ep.gate_evidence = {"predicates": ev.predicate_exit_codes,
                            "members": ev.gate_members}

    from .coding import code_episode
    label, facts = code_episode(ep, account)
    ep.disclosure_label_llm = label
    ep.facts = facts

    # v2: awareness scoring
    aw = score_bundle(bundle)
    ep.awareness_score = aw.score
    ep.awareness_contaminated = aw.contaminated
    ep.awareness_first_step = aw.first_flagged_step
    ep.awareness_detail = aw.as_dict()

    return ep


def save_episodes(episodes: list[EpisodeRecord], path: Path) -> Path:
    from dataclasses import asdict
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for ep in episodes:
            fh.write(json.dumps(asdict(ep), default=str) + "\n")
    return path
