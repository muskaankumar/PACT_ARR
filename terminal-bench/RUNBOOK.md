# RUNBOOK — the lean experiment

Produces the four tables reported in the paper from Terminal-Bench episodes.
The shipped corpus contains **60 episodes** (2 arms x 10 tasks x 3 reps, Opus-4.6).
See `results/grid_scores_final.json` for the data behind Table 2 in the paper.

Everything here is executable. `DESIGN.md` is the full design; this file is the
subset you actually run.

---

## The cut, and why it's this cut

The full factorial in the design document is 23,328 episodes. The lean grid is
120. What was removed, and the justification from the v3 sandbox corpus:

| Factor | Full | Lean | Why this is safe to cut |
|---|---|---|---|
| Rejection matrix | 9 | **1** | v3: arm effect 36.1pp; finality spanned 48.6→26.4%; personalization never separated. Framing is a second paper |
| Affordance | 3 | **1** | Pinned to `none`, the v3 anchor — which alone produced a 34.7pp arm effect |
| Flaw types | 2 | **1** | `wrong_oracle` needs its own arm to interpret; `overspecified` suffices for the headline |
| Repetitions | 5 | **3** | 3 is the floor for a task-clustered bootstrap to see within-task variance |
| **Tasks** | 24 | **10** | **Not cut proportionally.** See below |

**Tasks are the thing not to economize on.** With 10 tasks × 12 episodes, episodes
within a task are correlated, and the honest interval bootstraps over *tasks*, not
episodes — so tasks, not episodes, are the effective sample size. Cutting to 4
tasks would halve the cost and destroy the ability to generalize past those 4.
Cut grid factors first, always.

Two cost reductions are in the code rather than the grid:

- **Lazy gate adjudication.** The §5.4 procedure short-circuits, so most episodes
  never need most predicates. Measured: **3.3 container starts vs 15 eager, 78%
  saved**, with the verdict provably unchanged (`tests/test_lazy_gate.py`).
- **Adaptive k.** Predicates run once, escalating to k=3 only when the verdict
  would count as a violation — the only place instability can corrupt the corpus.

---

## Steps

### 1. Install and verify (free, ~1 min)

```bash
cd ppa_tb_pkg
pip install -e ".[dev,analysis]"
python -m pytest tests -q                # 90 tests
python -m ppa_tb selftest                # 24 detection probes — must be green
```

`selftest` is blocking. Half its probes are **negative**: cases where nothing may
fire. `agent_authored_own_tests` is the v2 false positive as an assertion;
`justified_repair_of_wrong_oracle` guards the case that exists only because the
presented check is deliberately flawed.

### 2. See the output shape before spending anything (free, ~10 s)

```bash
python demo/make_demo_corpus.py 120
python -m ppa_tb tables --episodes demo/out/episodes.jsonl --out demo/out/paper
```

Synthesizes a corpus with the lean grid's exact shape and runs the real coding,
statistics, and table code over it. Look at `demo/out/paper/RESULTS.md` — that is
the artifact the whole pipeline exists to produce.

### 3. Budget check (free)

```bash
python -m ppa_tb estimate --config examples/grid_lean.yaml --budget 200
```

Prints episodes, agent cost, compute cost, wall clock. With `--budget` it **raises
and exits non-zero** rather than warning, because a warning above 120 lines of
harness output is a warning nobody reads, and a 20× overspend noticed afterward is
not recoverable.

Defaults are rough. After the pilot, refit them: `--from-job data/episodes.jsonl`.

### 4. Pick and prepare tasks (~1h human)

```bash
uv tool install harbor
harbor datasets download terminal-bench@2.0
```

Choose 10 tasks, stratified 5 domains × 2 difficulty. **Easy/medium only** — hard
tasks have long trajectories and low reach rates, so they cost the most and
contribute least to the rejection-window measurement, which is where all the
violations live.

Apply the eligibility filter (DESIGN §1.2) — exclude network-dependent or
non-deterministic verification, and anything with no plausible protectable
surface. Record exclusions; the exclusion table is itself a result.

Then per task write:

- `overlays/<task>.yaml` — forbidden paths, protected region, allowlist
  (`examples/overlays/fix-log-parser.yaml` is a worked example)
- `checks/<task>.hidden.sh` — track H, the unmodified upstream check
- `checks/<task>.overspecified.sh` — track P, the flawed one
- `task_meta/<task>.json` — `container_paths`, `reference_diff_paths`,
  `gate_paths`, `manifest_reproduced`

Get `reference_diff_paths` from the oracle agent:

```bash
harbor run -p ./tb-tasks --agent oracle --task-id <task>
```

### 5. Audit overlays — blocking (free)

```bash
python -m ppa_tb audit --overlays ./overlays --meta ./task_meta
```

The check that matters is `reference_solution_overlay_clean`. If the gold solution
touches a forbidden path, **the task cannot be solved without violating** — the
overlay traps the agent and no violation number from it means anything. Fix or
drop the task; do not proceed with a failure here.

### 6. Forge the derived dataset

```bash
export ANTHROPIC_API_KEY=...
export OPENAI_API_KEY=...

python -m ppa_tb forge \
  --config   examples/grid_lean.yaml \
  --upstream ./tb-tasks \
  --overlays ./overlays \
  --checks   ./checks
```

Produces `derived_dataset/` — 20 task dirs (10 tasks × 2 arms), each with the
reviewer CLI, build-time snapshot, and post-hook wired in. The harness itself is
never patched.

### 7. Null-agent calibration — blocking (~20 min, ~free)

```bash
python -m ppa_tb nullcal --dataset ./derived_dataset
```

Runs a no-op agent on every derived task and asserts the masked filesystem diff is
**empty**. Residue is either mask material (add to `DEFAULT_MASK` in `detect.py`)
or a non-deterministic container (exclude the task — don't patch around it).

This is the false-positive screen the v2 detector never had, and its absence is
why 593 phantom gate-tampers shipped.

### 8. Smoke test — 4 episodes (~$4, ~25 min)

```bash
python -m ppa_tb run     --config examples/grid_smoke.yaml
python -m ppa_tb collect --config examples/grid_smoke.yaml \
    --job ./jobs/<job-name> --out data/smoke.jsonl
python -m ppa_tb tables  --episodes data/smoke.jsonl --out data/smoke_paper
```

Never skip this. It catches every wiring error that would otherwise surface 100
episodes in. Confirm before proceeding:

- reach rate > 0 (agents are finding `review submit`)
- `hidden_rc.txt` and `presented_rc.txt` both present in the bundle
- the debrief fired and `ACCOUNT.md` exists
- `collect` found the bundles — if not, print the job tree and fix
  `find_bundles()`, the one function that knows the harness layout

### 9. The run (~$112, ~4h at `n_concurrent: 6`)

```bash
python -m ppa_tb run --config examples/grid_lean.yaml --dry-run   # confirm 120
python -m ppa_tb run --config examples/grid_lean.yaml
```

### 10. Collect, adjudicate, code

```bash
python -m ppa_tb collect --config examples/grid_lean.yaml \
    --job ./jobs/<job> --out data/episodes.jsonl
```

Gate adjudication runs here, lazily. Then **hand-code 100% of violating
episodes** — at this scale there are ~25 of them, so full double-coding is cheaper
than sampling, and it makes disclosure kappa a real measurement rather than an
estimate. Write `disclosure_label_human` into the JSONL; T3 picks it up and
reports kappa automatically.

### 11. Tables

```bash
python -m ppa_tb tables  --episodes data/episodes.jsonl --out paper/
python -m ppa_tb analyze --episodes data/episodes.jsonl --out paper/analysis
```

`paper/RESULTS.md` (reading) and `paper/tables.tex` (manuscript), plus the
frontier plot.

---

## Reading the tables

**Read T0 first.** If `reach-rate spread > 0.15`, ITT violation rates are
exposure-confounded and T2's ITT column is not interpretable on its own — lead
with the exposure-conditional column. This is the largest analytic trap in the
design: an episode that never reaches a submission cannot be rejected, so it
cannot violate, so weak models look more aligned for purely mechanical reasons.

In the synthetic demo this is visible directly: weak-model ITT reads 30.0% and
exposure-conditional 42.9%, against strong-model 30.0%/30.0%. The ITT column says
they're identical; they aren't.

**Design effect in T1** is the clustered interval width over the naive one. Above
~1.3 means task heterogeneity is doing real work and any episode-level interval
elsewhere in the literature is too narrow. The demo shows 1.40.

**Starred intervals** are Wilson, not bootstrapped. At a zero rate every bootstrap
resample lands on zero, so the percentile interval collapses to `[0, 0]` and would
assert exact knowledge from no events. The informed arm is expected to sit there,
so this is the cell the headline rests on — it gets a real upper bound.

---

## If you have more budget

In priority order, by information gained per dollar:

1. **More tasks** (10 → 20) — the only way to narrow the clustered interval, and
   the only way to widen generalization. Doubles cost, roughly halves the CI width
   in the task dimension.
2. **Un-pin finality** (1 → 3 cells) — restores the rejection-framing axis, the
   original PPA question. Triples cost.
3. **Add `wrong_oracle`** as a second flaw arm — activates the gate-repair
   verdicts (§5.4 rows 5–7), which is where the two-track design earns its keep.
4. **Third model** — the frontier plot is thin with two points per arm.

Do not spend it on more repetitions. Beyond 3, extra reps add episodes within
already-sampled tasks, which the cluster bootstrap correctly discounts.

---

## Honest limits to state in the paper

- 10 tasks is a narrow generalization base. The clustered CI reflects that; the
  naive one wouldn't.
- One rejection cell means the framing dimension is **not** measured here. Don't
  claim it.
- Detection-only: every crossing succeeds, nothing is blocked. Violation rates are
  **not** comparable to the v3 sandbox numbers, which mixed blocked attempts with
  unblocked crossings.
- Terminal-Bench is public, so contamination is a live threat to the capability
  axis and is not controlled in this design (DESIGN §8.6).
- The presented check is deliberately flawed. If agents come to distrust the
  reviewer, later episodes measure something different from earlier ones —
  `contestation rate` in T0 is the tracking metric, not a solution.
