"""
Section 9 — time-to-violation.

Event  : first authorization violation (`first_violation_turn`)
Censor : runs reaching the turn ceiling without violating, censored at
         `max_turns`.

Note on the time axis: `first_violation_turn` is a monotonic turn index that
counts every logged agent turn, so it can exceed the 120-turn agent ceiling
(tool responses and debrief turns advance it). Curves are therefore drawn on
the observed index, and the censoring time for a non-violating run is taken
as that run's own final turn index rather than a nominal constant.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .stats import KMCurve, kaplan_meier, logrank_test


def build_survival_frame(runs: pd.DataFrame) -> pd.DataFrame:
    d = runs.copy()
    # censoring time = the run's own last turn, floored at the ceiling
    censor_at = d[["total_turns", "max_turns"]].max(axis=1)
    d["duration"] = np.where(d.auth_violation & d.first_violation_turn.notna(),
                             d.first_violation_turn, censor_at)
    d["event"] = d.auth_violation & d.first_violation_turn.notna()
    d["duration"] = d.duration.astype(float)
    return d


def curve_to_frame(c: KMCurve, **meta) -> pd.DataFrame:
    return pd.DataFrame({
        **{k: v for k, v in meta.items()},
        "turn": c.times, "survival": c.survival,
        "ci_lo": c.lower, "ci_hi": c.upper,
        "n_at_risk": c.n_risk, "n_events": c.n_event,
    })


def run(runs: pd.DataFrame) -> dict:
    d = build_survival_frame(runs)
    t, curves = {}, []
    summary = []

    strata = [("condition", d.condition.unique()),
              ("model_pretty", d.model_pretty.unique()),
              ("affordance_level", ["none", "low", "high"]),
              ("finality", ["retry_invited", "final_this_attempt",
                            "never_reconsidered"])]

    # overall
    c = kaplan_meier(d.duration, d.event)
    curves.append(curve_to_frame(c, stratum="ALL", level="all"))
    summary.append({"stratum": "ALL", "level": "all", "n": c.n,
                    "n_events": c.n_events, "median_survival_turn": c.median,
                    "S_at_50": _s_at(c, 50), "S_at_100": _s_at(c, 100),
                    "S_at_150": _s_at(c, 150)})

    for name, levels in strata:
        for lvl in levels:
            sub = d[d[name] == lvl]
            if len(sub) == 0:
                continue
            c = kaplan_meier(sub.duration, sub.event)
            curves.append(curve_to_frame(c, stratum=name, level=str(lvl)))
            summary.append({"stratum": name, "level": str(lvl), "n": c.n,
                            "n_events": c.n_events,
                            "median_survival_turn": c.median,
                            "S_at_50": _s_at(c, 50), "S_at_100": _s_at(c, 100),
                            "S_at_150": _s_at(c, 150)})

    # condition x model, the cut that actually matters
    for (cond, model), sub in d.groupby(["condition", "model_pretty"]):
        c = kaplan_meier(sub.duration, sub.event)
        curves.append(curve_to_frame(c, stratum="condition x model",
                                     level=f"{cond} / {model}"))
        summary.append({"stratum": "condition x model",
                        "level": f"{cond} / {model}", "n": c.n,
                        "n_events": c.n_events,
                        "median_survival_turn": c.median,
                        "S_at_50": _s_at(c, 50), "S_at_100": _s_at(c, 100),
                        "S_at_150": _s_at(c, 150)})

    t["km_curves"] = pd.concat(curves, ignore_index=True)
    t["km_summary"] = pd.DataFrame(summary)

    # log-rank tests
    lr = []
    for name, a, b in [("condition", "uninformed", "informed"),
                       ("model_pretty", "GPT-5.5", "Opus-4.8")]:
        g1, g2 = d[d[name] == a], d[d[name] == b]
        chi2, p, o1, e1 = logrank_test(g1.duration, g1.event,
                                       g2.duration, g2.event)
        lr.append({"comparison": f"{name}: {a} vs {b}", "n_a": len(g1),
                   "n_b": len(g2), "observed_a": o1, "expected_a": e1,
                   "chi2": chi2, "df": 1, "p_value": p})
    # affordance pairwise, within uninformed (where events actually occur)
    un = d[d.condition == "uninformed"]
    for a, b in [("none", "low"), ("none", "high"), ("low", "high")]:
        g1, g2 = un[un.affordance_level == a], un[un.affordance_level == b]
        chi2, p, o1, e1 = logrank_test(g1.duration, g1.event, g2.duration, g2.event)
        lr.append({"comparison": f"affordance (uninformed): {a} vs {b}",
                   "n_a": len(g1), "n_b": len(g2), "observed_a": o1,
                   "expected_a": e1, "chi2": chi2, "df": 1, "p_value": p})
    t["logrank_tests"] = pd.DataFrame(lr)

    t["survival_frame"] = d[["run_id", "condition", "model_pretty",
                             "affordance_level", "finality", "duration",
                             "event", "first_violation_turn", "total_turns"]]
    return t


def _s_at(c: KMCurve, t: float) -> float:
    idx = np.searchsorted(c.times, t, side="right") - 1
    return float(c.survival[max(0, idx)])
