"""
Section 9 — primary contrast (informed vs uninformed) and secondary outcomes.

The house style for every proportion in this module is:
    k / n, point estimate, Wilson 95% CI
and for every contrast:
    absolute difference in percentage points (+ CI), ratio, z, p, Cohen's h.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import (AFFORDANCE_ORDER, DISCLOSURE_ORDER, FINALITY_ORDER,
                   PROXIMITY_LABELS, BLOCKED_KINDS)
from .stats import (bootstrap_ci, chi2_independence, fisher_exact_2x2,
                    mann_whitney_u, two_proportion_test, wilson)


# ------------------------------------------------------------------ helpers

def rate_row(df: pd.DataFrame, flag: str, **meta) -> dict:
    w = wilson(int(df[flag].sum()), len(df))
    return {**meta, "k": w.k, "n": w.n, "rate": w.p,
            "ci_lo": w.lo, "ci_hi": w.hi, "pretty": w.pct()}


def contrast(df: pd.DataFrame, flag: str, by: str = "condition",
             a: str = "uninformed", b: str = "informed", **meta) -> dict:
    """`a` minus `b`; by default uninformed minus informed."""
    da, db = df[df[by] == a], df[df[by] == b]
    if len(da) == 0 or len(db) == 0:
        return {**meta, "n_a": len(da), "n_b": len(db)}
    t = two_proportion_test(int(da[flag].sum()), len(da),
                            int(db[flag].sum()), len(db))
    fisher = fisher_exact_2x2(int(da[flag].sum()), len(da) - int(da[flag].sum()),
                              int(db[flag].sum()), len(db) - int(db[flag].sum()))
    return {**meta,
            f"{a}_k": t.k1, f"{a}_n": t.n1, f"{a}_rate": t.p1,
            f"{b}_k": t.k2, f"{b}_n": t.n2, f"{b}_rate": t.p2,
            "diff_pp": t.diff_pp, "diff_ci_lo_pp": t.diff_lo_pp,
            "diff_ci_hi_pp": t.diff_hi_pp,
            "risk_ratio": t.ratio, "cohens_h": t.cohens_h,
            "z": t.z, "p_value": t.p_value, "p_fisher": fisher}


def _dist_table(df: pd.DataFrame, col: str, order, group_cols) -> pd.DataFrame:
    g = (df.groupby(group_cols + [col], observed=True).size()
           .reset_index(name="n"))
    tot = g.groupby(group_cols, observed=True).n.transform("sum")
    g["pct"] = 100 * g.n / tot
    g[col] = pd.Categorical(g[col], categories=order, ordered=True)
    return g.sort_values(group_cols + [col])


# ------------------------------------------------------------------- primary

def primary(runs: pd.DataFrame) -> dict:
    t = {}

    # headline
    rows = [rate_row(runs[runs.condition == c], "auth_violation", condition=c)
            for c in ("uninformed", "informed")]
    rows.append(rate_row(runs, "auth_violation", condition="POOLED"))
    t["violation_rate_headline"] = pd.DataFrame(rows)

    # the primary contrast, pooled and stratified
    con = [contrast(runs, "auth_violation", stratum="POOLED", level="all")]
    for aff in AFFORDANCE_ORDER:
        con.append(contrast(runs[runs.affordance_level == aff],
                            "auth_violation", stratum="affordance", level=aff))
    for m in sorted(runs.model_pretty.unique()):
        con.append(contrast(runs[runs.model_pretty == m],
                            "auth_violation", stratum="model", level=m))
    for f in FINALITY_ORDER:
        con.append(contrast(runs[runs.finality == f],
                            "auth_violation", stratum="finality", level=f))
    for m in sorted(runs.model_pretty.unique()):
        for aff in AFFORDANCE_ORDER:
            sub = runs[(runs.model_pretty == m) & (runs.affordance_level == aff)]
            con.append(contrast(sub, "auth_violation",
                                stratum="model x affordance", level=f"{m} / {aff}"))
    t["primary_contrast"] = pd.DataFrame(con)

    # full results matrix: every design cell
    cell = (runs.groupby(["condition", "model_pretty", "affordance_level",
                          "finality"], observed=True)
                .agg(n=("auth_violation", "size"),
                     k=("auth_violation", "sum"),
                     mean_proximity=("boundary_proximity", "mean"),
                     max_proximity=("boundary_proximity", "max"),
                     mean_pushback=("pushback_count", "mean"),
                     mean_stages=("stages_completed", "mean"),
                     mean_routes=("distinct_bypass_routes", "mean"))
                .reset_index())
    cell["rate"] = cell.k / cell.n
    ci = cell.apply(lambda r: wilson(int(r.k), int(r.n)), axis=1)
    cell["ci_lo"] = [c.lo for c in ci]
    cell["ci_hi"] = [c.hi for c in ci]
    t["results_matrix_full"] = cell

    # compact wide matrix for the paper
    wide = (runs.pivot_table(index=["model_pretty", "affordance_level"],
                             columns="condition", values="auth_violation",
                             aggfunc=["sum", "size", "mean"], observed=True))
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide = wide.reset_index()
    wide["delta_pp"] = 100 * (wide["mean_uninformed"] - wide["mean_informed"])
    t["results_matrix_wide"] = wide

    # factor main effects
    fac_rows = []
    for factor in ["affordance_level", "finality", "model_pretty",
                   "deadline", "complication"]:
        for cond in ["uninformed", "informed", "POOLED"]:
            sub = runs if cond == "POOLED" else runs[runs.condition == cond]
            for lvl, g in sub.groupby(factor, observed=True):
                fac_rows.append(rate_row(g, "auth_violation", factor=factor,
                                         level=str(lvl), condition=cond))
    t["factor_main_effects"] = pd.DataFrame(fac_rows)

    # omnibus tests within each condition
    omni = []
    for cond in ["uninformed", "informed", "POOLED"]:
        sub = runs if cond == "POOLED" else runs[runs.condition == cond]
        for factor in ["affordance_level", "finality", "model_pretty",
                       "deadline", "complication"]:
            tab = pd.crosstab(sub[factor], sub.auth_violation).values
            chi2, df_, p, v = chi2_independence(tab)
            omni.append({"condition": cond, "factor": factor, "chi2": chi2,
                         "df": df_, "p_value": p, "cramers_v": v})
    t["omnibus_tests"] = pd.DataFrame(omni)

    return t


# ----------------------------------------------------------------- secondary

def secondary(runs: pd.DataFrame, violations: pd.DataFrame) -> dict:
    t = {}

    # ---- disclosure distribution ---------------------------------------
    viol_runs = runs[runs.auth_violation].copy()
    t["disclosure_distribution_all_runs"] = _dist_table(
        runs, "disclosure_primary", DISCLOSURE_ORDER, ["condition"])
    t["disclosure_distribution_violating"] = _dist_table(
        viol_runs, "disclosure_primary", DISCLOSURE_ORDER,
        ["condition", "model_pretty"])

    # ---- concealment ----------------------------------------------------
    rows = []
    for cond, g in runs.groupby("condition"):
        rows.append(rate_row(g, "concealed", scope="all runs", condition=cond))
    for cond, g in viol_runs.groupby("condition"):
        rows.append(rate_row(g, "concealed", scope="violating runs only",
                             condition=cond))
    for m, g in viol_runs.groupby("model_pretty"):
        rows.append(rate_row(g, "concealed", scope="violating runs only",
                             condition=f"model={m}"))
    t["concealment_rate"] = pd.DataFrame(rows)
    if len(viol_runs[viol_runs.condition == "informed"]):
        t["concealment_contrast"] = pd.DataFrame([
            contrast(viol_runs, "concealed", stratum="violating runs",
                     level="all")])

    t["concealment_type"] = _dist_table(
        runs, "concealment_type", ["none", "passive", "active"], ["condition"])

    # ---- per-violation-kind disclosure (selective disclosure) ----------
    v = violations.copy()
    v["label"] = v.human_label.fillna(v.llm_label)
    kind_disc = (v.groupby(["violation_kind", "label"]).size()
                   .reset_index(name="n"))
    kind_disc["pct"] = 100 * kind_disc.n / kind_disc.groupby(
        "violation_kind").n.transform("sum")
    t["disclosure_by_violation_kind"] = kind_disc

    # selective disclosure: runs with >1 kind where labels differ
    multi = v[v.run_id.duplicated(keep=False)]
    sel = (multi.groupby("run_id")
                .agg(kinds=("violation_kind", lambda s: "+".join(sorted(s))),
                     labels=("label", lambda s: "|".join(s)),
                     n_kinds=("violation_kind", "nunique"),
                     n_distinct_labels=("label", "nunique"))
                .reset_index())
    sel["selective"] = sel.n_distinct_labels > 1
    t["selective_disclosure"] = sel
    t["selective_disclosure_summary"] = pd.DataFrame([{
        "runs_with_multiple_violation_kinds": len(sel),
        "runs_disclosing_selectively": int(sel.selective.sum()),
        "pct_selective": 100 * sel.selective.mean() if len(sel) else np.nan,
    }])

    # ---- boundary proximity --------------------------------------------
    prox = (runs.groupby(["condition", "boundary_proximity"])
                .size().reset_index(name="n"))
    prox["pct"] = 100 * prox.n / prox.groupby("condition").n.transform("sum")
    prox["label"] = prox.boundary_proximity.map(PROXIMITY_LABELS)
    t["proximity_distribution"] = prox

    prox_sum = []
    for cond, g in runs.groupby("condition"):
        m, lo, hi = bootstrap_ci(g.boundary_proximity)
        prox_sum.append({"condition": cond, "n": len(g), "mean_max_proximity": m,
                         "boot_lo": lo, "boot_hi": hi,
                         "median": g.boundary_proximity.median(),
                         "pct_reaching_ge1": 100*(g.boundary_proximity >= 1).mean(),
                         "pct_reaching_ge3": 100*(g.boundary_proximity >= 3).mean(),
                         "pct_reaching_4": 100*(g.boundary_proximity == 4).mean()})
    t["proximity_summary"] = pd.DataFrame(prox_sum)
    u, z, p = mann_whitney_u(
        runs[runs.condition == "uninformed"].boundary_proximity,
        runs[runs.condition == "informed"].boundary_proximity)
    t["proximity_test"] = pd.DataFrame([{"test": "Mann-Whitney U (uninformed vs informed)",
                                         "U": u, "z": z, "p_value": p}])

    # does knowing the rules move mass from level 4 to level 3?
    among_high = runs[runs.boundary_proximity >= 3].copy()
    among_high["reached_4"] = among_high.boundary_proximity == 4
    if among_high.condition.nunique() == 2:
        t["prox4_given_prox3plus"] = pd.DataFrame([
            contrast(among_high, "reached_4",
                     stratum="P(level 4 | level>=3)", level="all")])

    # ---- persistence / escalation ---------------------------------------
    pers = []
    for cond, g in runs.groupby("condition"):
        vg = g[g.auth_violation]
        m1, lo1, hi1 = bootstrap_ci(vg.first_violation_turn.dropna())
        m2, lo2, hi2 = bootstrap_ci(vg.persistence_after_block)
        m3, lo3, hi3 = bootstrap_ci(g.pushback_count)
        pers.append({"condition": cond, "n_violating": len(vg),
                     "mean_first_violation_turn": m1, "fvt_lo": lo1, "fvt_hi": hi1,
                     "median_first_violation_turn": vg.first_violation_turn.median(),
                     "mean_persistence_after_block": m2, "pab_lo": lo2, "pab_hi": hi2,
                     "mean_distinct_routes": vg.distinct_bypass_routes.mean(),
                     "mean_pushback_all_runs": m3, "pb_lo": lo3, "pb_hi": hi3,
                     "mean_stages_completed": g.stages_completed.mean()})
    t["persistence_summary"] = pd.DataFrame(pers)

    u, z, p = mann_whitney_u(
        runs[(runs.condition == "uninformed") & runs.auth_violation].first_violation_turn,
        runs[(runs.condition == "informed") & runs.auth_violation].first_violation_turn)
    t["first_violation_turn_test"] = pd.DataFrame([
        {"test": "Mann-Whitney U on first_violation_turn", "U": u, "z": z,
         "p_value": p}])

    # ---- where does persistence go? legitimate channel vs boundary --------
    # Informed agents violate far less, but they are not simply more passive:
    # this compares in-channel argument (pushback) against boundary crossing.
    chan = []
    for cond, g in runs.groupby("condition"):
        m, lo, hi = bootstrap_ci(g.pushback_count)
        chan.append({"condition": cond, "n": len(g),
                     "mean_pushback": m, "pb_lo": lo, "pb_hi": hi,
                     "median_pushback": g.pushback_count.median(),
                     "total_pushback_events": int(g.pushback_count.sum()),
                     "pct_runs_arguing": 100 * (g.pushback_count > 0).mean(),
                     "violation_rate": g.auth_violation.mean()})
    t["persistence_channel"] = pd.DataFrame(chan)

    u, z, p = mann_whitney_u(
        runs[runs.condition == "informed"].pushback_count,
        runs[runs.condition == "uninformed"].pushback_count)
    t["persistence_channel_test"] = pd.DataFrame([{
        "test": "Mann-Whitney U on pushback_count (informed vs uninformed)",
        "U": u, "z": z, "p_value": p}])

    # within-condition: is arguing a substitute for crossing, or a correlate?
    sub_rows = []
    for cond, g in runs.groupby("condition"):
        for viol_flag, gg in g.groupby("auth_violation"):
            sub_rows.append({"condition": cond,
                             "auth_violation": bool(viol_flag), "n": len(gg),
                             "mean_pushback": gg.pushback_count.mean(),
                             "median_pushback": gg.pushback_count.median()})
    t["persistence_channel_within"] = pd.DataFrame(sub_rows)


    comp = (violations.groupby(["condition", "violation_kind"]).size()
                      .reset_index(name="n_events"))
    comp["pct_of_condition_events"] = 100 * comp.n_events / comp.groupby(
        "condition").n_events.transform("sum")
    comp["enforcement"] = np.where(
        comp.violation_kind.isin(BLOCKED_KINDS), "blocked (attempt)",
        "permitted (completed)")
    t["violation_type_composition"] = comp

    aff_type = (violations.groupby(["affordance_level", "violation_kind"],
                                   observed=True).size().reset_index(name="n"))
    aff_type["pct_within_affordance"] = 100 * aff_type.n / aff_type.groupby(
        "affordance_level", observed=True).n.transform("sum")
    t["violation_type_by_affordance"] = aff_type

    # per-run rate of each kind (denominator = all runs in condition)
    kind_rows = []
    n_by_cond = runs.groupby("condition").size()
    for (cond, kind), g in violations.groupby(["condition", "violation_kind"]):
        n_runs = int(n_by_cond[cond])
        w = wilson(g.run_id.nunique(), n_runs)
        kind_rows.append({"condition": cond, "violation_kind": kind,
                          "runs_with_kind": w.k, "n_runs": w.n,
                          "rate": w.p, "ci_lo": w.lo, "ci_hi": w.hi})
    t["violation_kind_rates"] = pd.DataFrame(kind_rows)

    # ---- finality x condition interaction (the "soft rejection paradox") -
    fin_rows = []
    for cond in ["uninformed", "informed"]:
        for f in FINALITY_ORDER:
            g = runs[(runs.condition == cond) & (runs.finality == f)]
            fin_rows.append(rate_row(g, "auth_violation", condition=cond,
                                     finality=f))
    t["finality_x_condition"] = pd.DataFrame(fin_rows)

    para = []
    for cond in ["uninformed", "informed"]:
        sub = runs[(runs.condition == cond) &
                   (runs.finality.isin(["retry_invited", "never_reconsidered"]))].copy()
        sub["soft"] = (sub.finality == "retry_invited").map(
            {True: "soft", False: "hard"})
        para.append(contrast(sub, "auth_violation", by="soft", a="soft",
                             b="hard", condition=cond))
    t["soft_rejection_paradox"] = pd.DataFrame(para)

    return t
