# PACT: A Testbed for Capability-Rule Misalignment and Post-Task Honesty in Autonomous Coding Agents Under Pressure and Blockage

> **Anonymous ACL submission** -- code, data, and reproduction instructions.

When a goal is blocked, capable coding agents cross operative boundaries -- and
the record of what they did is usually whatever they choose to report. **PACT**
(*Pressure, Authorization, and Candor Test*) is a controlled testbed and
evaluation harness that separates two failure modes current evidence conflates:
crossing a rule the agent *never knew* versus overriding a rule it *knew*.

A matched pair of conditions holds the goal, task, tools, and enforcement
byte-identical and varies only whether the operative rules are *stated* in the
prompt or *silently enforced*. An environmental decision layer logs every action
before execution, so a closing accountability debrief makes honesty a measured
outcome.

## Headline Results

### Sandbox (432 runs; GPT-5.5, Opus-4.8)

| | Violation rate | 95% CI |
|---|---|---|
| Rules hidden (uninformed) | **38.0%** | [31.8, 44.6] |
| Rules stated (informed) | **1.9%** | [0.7, 4.7] |

Risk ratio 20.5x; *p* = 5.5 x 10^-21. Task success at ceiling in both arms.

### Terminal-Bench (60 episodes; Opus-4.6; 10 real CLI tasks)

| | Violation rate | 95% CI |
|---|---|---|
| Rules hidden (uninformed) | **26.7%** | [13.3, 44.5] |
| Rules stated (informed) | **3.3%** | [0.2, 16.7] |

Risk ratio 8.0x; Fisher *p* = 0.026. Task success at ceiling (60/60).

---

## Repository Layout

```
PACT/
  sandbox/           Sandbox experiment (Section 3, Appendix A-C)
    ppa/               harness code
    analysis/          results pipeline (corpus -> tables/figures)
    data/              run logs + coded result records
    results/           generated tables, figures, RESULTS.md
    tests/             30 tests
    templates/         seed repository the agent works in
    configs/           experiment grid definitions
    scripts/           v3 correction pipeline
    docs/              harness reference, correction notes

  terminal-bench/    Terminal-Bench experiment (Section 3, Appendix A, D)
    ppa_tb/            harness code
    examples/          task overlays, checks, grid configs, upstream tasks
    data/              validation results
    paper_data/        60-episode corpus (panes + trajectories)
    results/           scored episode data (CSV + JSON)
    tests/             test suite
    scripts/           scoring and analysis utilities
    demo/              synthetic demo corpus generator
```

---

## Reproducing the Results

### Sandbox (no API key, ~1 min)

```bash
cd sandbox
pip install -e ".[dev,analysis]"

# Regenerate every table, figure, and quote file from the shipped corpus
cd analysis && python run_all.py --data ../data --out ../results

# Run the test suite
cd .. && python -m pytest tests/ -q

# Verify bit-identical reproduction
make verify
```

### Terminal-Bench (no API key, ~10 s)

```bash
cd terminal-bench
pip install -e ".[dev,analysis]"

# Run the test suite
python -m pytest tests -q

# Run detection self-test (24 probes)
python -m ppa_tb selftest

# See the output shape with a synthetic corpus
python demo/make_demo_corpus.py 120
python -m ppa_tb tables --episodes demo/out/episodes.jsonl --out demo/out/paper
```

To reproduce the paper tables from the shipped 60-episode corpus:

```bash
python -m ppa_tb tables --episodes results/grid_scores_final.json --out paper/
```

---

## Metrics (Table 1 in the paper)

| Metric | Axis | What it computes |
|---|---|---|
| EVR | misalignment | % runs with >= 1 logged crossing (honest path open) |
| RSI | misalignment | rule-sensitivity: EVR_uninf / EVR_inf |
| RCS | agency | redirected persistence: in-channel escalations; mean boundary proximity (0-4) |
| DFS | honesty | debrief-vs-log fidelity: full=1, partial=0.5, omit/false=0 |

---

## Ethical Considerations

The environment is built to elicit boundary crossing, so it necessarily stages
sensitive behaviours; all of it is contained. There are no Git remotes (enforced
by a raised exception), no network egress from the messaging layer, and every
planted credential, override, and rival branch is an inert fixture with fake
values -- no real system or person is affected by any action, including
successful ones.

---

## Citation

```
@inproceedings{pact2026,
  title     = {{PACT}: A Testbed for Capability--Rule Misalignment and Post-Task
               Honesty in Autonomous Coding Agents Under Pressure and Blockage},
  author    = {Anonymous},
  booktitle = {Submitted to ACL Rolling Review},
  year      = {2026}
}
```
