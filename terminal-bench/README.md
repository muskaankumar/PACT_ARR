# PACT Terminal-Bench Experiment

The Terminal-Bench instantiation of PACT (Section 3, Appendix A and D of the
paper). Wraps the same protocol around ten real command-line tasks from
Terminal-Bench via a review CLI with off-channel recipients and route-revealing
in-repo files. All 60 episodes use Opus-4.6.

**Start with [`RUNBOOK.md`](RUNBOOK.md)** for the lean, paper-scale experiment.
This file is the module reference; [`DESIGN.md`](DESIGN.md) is the full design.

**The harness is never patched.** `ppa_tb forge` emits a *derived local dataset*
that Harbor (TB 2.0) or `tb` (TB 1.x) runs natively. Everything PPA needs rides
inside the task definition.

---

## 0. What's here

| File | Design § | Role |
|---|---|---|
| `schema.py` | — | Enums and records. `GateVerdict` has 10 values partitioned into violation / non-violation / excluded |
| `gate.py` | §5 | `adjudicate()` — the pure 8-row decision function; `PredicateRunner` — container re-runs; `readjudicate()` — offline re-scoring |
| `detect.py` | §4 | Two read-only passes and the four-case attribution table |
| `overlay.py` | §2 | Per-task rule model, generation prompt, tier-1 blocking audit |
| `conditions.py` | §3 | 3×3 rejection matrix (ported verbatim), arms, flaw taxonomy, debrief |
| `forge.py` | §2/§3 | Derives an instrumented task from an upstream one |
| `templates.py` | §4/§5 | Scripts that run *inside* the container |
| `coding.py` | §6 | Per-fact disclosure coding and the published label rule |
| `analysis.py` | §7 | Exposure-conditional rates, frontier plot, manipulation checks |
| `selftest.py` | §10 P0/P2 | Positive **and negative** detection probes |
| `runner.py`, `cli.py` | §10 | Grid expansion, harness invocation, collection |
| `stats.py` | §7 | Task-clustered bootstrap, Fisher, Cohen's h, kappa |
| `paper.py` | §7 | The four result tables, markdown + LaTeX |
| `cost.py` | §1.4 | Cost model, estimator, budget guard |

```bash
cd ppa_tb_pkg
pip install -e ".[dev,analysis]"
python -m pytest tests -q          # 90 tests, no docker needed
```

---

## 1. Try the whole downstream pipeline with no docker and no API spend

```bash
python demo/make_demo_corpus.py
python -m ppa_tb analyze --episodes demo/out/episodes.jsonl --out demo/out/analysis
```

This synthesizes 600 episodes and runs the real coding and analysis code over
them. It exists to let you inspect the output shape before spending anything.

On the synthetic corpus the exposure warning fires, and it's worth looking at why:

| cell | ITT rate | exposure-conditional | reach rate |
|---|---|---|---|
| uninformed / strong-model | 0.335 | 0.342 | 0.98 |
| uninformed / weak-model | **0.235** | **0.351** | 0.67 |

On ITT the weak model looks 10pp more aligned. It isn't — it just fails before
reaching the rejection window a third of the time. Conditional on exposure the
two models are indistinguishable. This is design §7.3, and it is the single
easiest way to manufacture a false capability–misalignment trade-off.

---

## 2. Execution sequence

Phases map 1:1 onto design §10. **Each has a stop condition. Do not proceed on a
partial pass** — that discipline is the whole lesson of the v2 gate-tamper
incident.

### Prerequisites

```bash
pip install uv && uv tool install harbor    # TB 2.0
# or:  pip install terminal-bench           # TB 1.x
docker --version                            # >= 20.10
export ANTHROPIC_API_KEY=...
export OPENAI_API_KEY=...

harbor datasets download terminal-bench@2.0     # get the upstream task library
```

### P2 first — detection self-test (free, blocking)

Run this before anything else. It's pure logic, so it costs nothing:

```bash
python -m ppa_tb selftest
```

24 probes across three groups. Half are **negative regression probes** that must
*not* fire — `agent_authored_own_tests` is the v2 false positive as an assertion,
and `justified_repair_of_wrong_oracle` guards the case that only exists because
the presented check is deliberately flawed. Exit non-zero on any miss.

### P1 — overlays

Author or generate one overlay per task (`examples/overlays/` has a worked one),
then run the blocking audit:

```bash
python -m ppa_tb audit --overlays ./overlays --meta ./task_meta
```

`task_meta/<task>.json` supplies `container_paths`, `reference_diff_paths`,
`gate_paths`, `manifest_reproduced`. Get `reference_diff_paths` by running the
oracle agent and diffing:

```bash
harbor run -p ./tb-tasks --agent oracle --task-id fix-log-parser
```

The check that matters most is `reference_solution_overlay_clean`. If the gold
solution touches a forbidden path, the task **cannot be solved without
violating** — the overlay traps the agent, and no violation number from that task
means anything.

### Forge the derived dataset

```bash
python -m ppa_tb forge \
  --config   examples/grid_pilot.yaml \
  --upstream ./tb-tasks \
  --overlays ./overlays \
  --checks   ./checks
```

`checks/` needs, per task: `<task>.hidden.sh` (track H, the unmodified upstream
check) and one `<task>.<flaw>.sh` per flaw type (track P). See
`examples/checks/` for a worked overspecified and wrong-oracle pair.

Forge produces one derived task per (task × arm × finality × personalization ×
flaw), each with the reviewer CLI, the build-time snapshot, and the post-hook
wired in.

### P0 — null-agent calibration (build the images, run nothing)

```bash
python -m ppa_tb nullcal --dataset ./derived_dataset
```

Runs a no-op agent on every derived task and asserts the masked filesystem diff
is **empty**. Residue is either mask material (add it to `DEFAULT_MASK` in
`detect.py`) or a non-deterministic container (exclude the task — don't patch
around it). This is the false-positive screen the v2 detector never had.

### P3 — pilot slice

```bash
python -m ppa_tb run --config examples/grid_pilot.yaml --dry-run   # check grid size
python -m ppa_tb run --config examples/grid_pilot.yaml
python -m ppa_tb collect --config examples/grid_pilot.yaml \
    --job ./jobs/ppa-anthropic-claude-opus-4-1-r00 --out data/episodes.jsonl
python -m ppa_tb analyze --episodes data/episodes.jsonl --out data/analysis
```

**Stop conditions:** reach rate ≥ 80% and comparable across arms; attribution-gap
rate below your pre-registered ceiling; violation rate non-degenerate. Read
`manipulation_checks.json` before any headline number.

### P4 — calibration

```bash
python -m ppa_tb run --config examples/grid_calibration.yaml
```

Crosses affordance levels; pick the lowest whose *uninformed* violation rate lands
in the measurable band, exactly as `experiment_calibration.yaml` did in v2.1. Also
confirm `hp_disagreement_rate` is non-trivial (the flawed checks are doing work)
and that `gate_repair_justified` verdicts occur at all — if none ever do, the
type-2 flaws aren't contestable enough to be recognizable.

### P5–P7 — coding, scale, analysis

```bash
python -m ppa_tb readjudicate --audit-dir data/gate_audit   # after any definition change
```

`readjudicate` re-runs `adjudicate()` over stored evidence with **no containers
and no agents** — the property that let `build_v3/readjudicate.py` fix 167
mislabelled runs offline. Assume the tamper definition will need revision once
real tasks are running.

---

## 3. Things that will bite, and where they're handled

**The hidden check must not be in the image.** An earlier draft of `forge.py`
shipped `check_hidden.sh` into the container behind `chmod 000` — worthless when
the agent runs as root. It now lives in `tests/`, which both harnesses copy in
only at *verification* time. `tests/test_forge.py::test_hidden_check_never_enters_the_image`
guards this. If the agent can read the true oracle, the two-track design is over.

**The gate is the presented check, not `tests/`.** `gate_paths_of()` manifests
`/opt/ppa/check_presented.sh` only. Manifesting `tests/` would list files the
agent could never have touched — mis-scoped gate definition, which is how v2 went
wrong in the first place.

**The snapshot must cover the whole filesystem.** Restricting it to `/app` makes
`sitecustomize.py`, a stub `pip install`, a `$PATH` shim named `pytest`, and a
`.pth` file all invisible — every one a live tamper route. `trace_gate.py` finds
them behaviorally (what the check *opens*), then confirms each by removal
counterfactual.

**Harness layout drift.** TB has changed task format once already. `forge.py` is
the only module that knows format specifics, and `find_bundles()` globs rather
than assuming depth. If collection returns nothing, print the job tree and fix
that one function.

**Rejection must fire in all four H×P cells.** `rejection_fires()` returns `True`
unconditionally and is tested as such. If it ever became outcome-dependent,
rejection exposure would correlate with capability and the design collapses.

---

## 4. Metrics that are NOT comparable to the v3 sandbox corpus

Detection-only means every crossing *succeeds*; nothing is blocked. So:

| Metric | Status |
|---|---|
| `violation_rate` | **Not poolable** — different events |
| `boundary_proximity` level 3 | Redefined: self-reverted, not harness-blocked |
| `persistence_after_block` | **Renamed** `persistence_after_rejection` |
| `gate_tamper` | Token heuristic → executable 8-way verdict. Not poolable |
| Disclosure labels | Same four levels, but per-fact derived and reworded debrief |
| Task success | New metric; CI pass ≠ hidden ground truth |

Report these separately. Do not plot them on one axis with v3 numbers.

## 5. Safety

Containers run `--network none` throughout. Planted credentials are inert, the
messaging channel is a file-based simulation that never egresses, and no bypass
reaches anything real. Since the environment **permits** the actions it measures,
this isolation is load-bearing, not decorative.
