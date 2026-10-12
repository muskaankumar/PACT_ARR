"""
Section 8.3 — agreement between the automated (LLM judge) and human
disclosure coding passes.

Reports, over the adjudicated subset:
  * unweighted, linearly-weighted and quadratically-weighted Cohen's kappa
  * the full confusion table
  * kappa broken out by violation kind, model and condition
  * a directional error analysis (is the judge lenient or harsh, and where?)
  * prevalence- and bias-adjusted kappa (PABAK), because kappa is depressed
    by the very skewed marginals this corpus has
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import DISCLOSURE_ORDER
from .stats import cohens_kappa, confusion_matrix, wilson


def _present(df: pd.DataFrame) -> list:
    """Categories actually used by either rater, in ordinal order."""
    used = set(df.llm_label) | set(df.human_label)
    return [c for c in DISCLOSURE_ORDER if c in used]


def kappa_table(df: pd.DataFrame, label: str) -> dict:
    cats = _present(df)
    out = {"stratum": label, "n": len(df), "k_categories": len(cats)}
    if len(df) < 2 or len(cats) < 2:
        out.update({f"kappa_{w}": np.nan for w in
                    ("unweighted", "linear", "quadratic")})
        out["raw_agreement"] = (df.llm_label == df.human_label).mean() if len(df) else np.nan
        return out
    for w in ("unweighted", "linear", "quadratic"):
        k = cohens_kappa(df.llm_label, df.human_label, categories=cats, weights=w)
        out[f"kappa_{w}"] = k.kappa
        out[f"se_{w}"] = k.se
        out[f"lo_{w}"] = k.lo
        out[f"hi_{w}"] = k.hi
        if w == "unweighted":
            out["raw_agreement"] = k.observed_agreement
            out["expected_agreement"] = k.expected_agreement
            out["interpretation"] = k.interpretation
    # PABAK = 2*p_o - 1 : removes the influence of skewed marginals
    out["pabak"] = 2 * out["raw_agreement"] - 1
    return out


def run(corpus, outdir) -> dict:
    adj = corpus.adjudicated
    tables, artifacts = {}, {}

    # ---------- overall + stratified kappa ------------------------------
    rows = [kappa_table(adj, "ALL")]
    for kind, sub in adj.groupby("violation_kind"):
        rows.append(kappa_table(sub, f"kind={kind}"))
    for m, sub in adj.groupby("model_pretty"):
        rows.append(kappa_table(sub, f"model={m}"))
    for c, sub in adj.groupby("condition"):
        rows.append(kappa_table(sub, f"condition={c}"))
    for a, sub in adj.groupby("affordance_level", observed=True):
        rows.append(kappa_table(sub, f"affordance={a}"))
    kappa_df = pd.DataFrame(rows)
    tables["kappa_by_stratum"] = kappa_df

    # ---------- confusion table -----------------------------------------
    cats = _present(adj)
    cm = confusion_matrix(adj.llm_label, adj.human_label, cats)
    cm_df = pd.DataFrame(cm.astype(int),
                         index=[f"LLM={c}" for c in cats],
                         columns=[f"HUMAN={c}" for c in cats])
    cm_df["row_total"] = cm_df.sum(axis=1)
    cm_df.loc["col_total"] = cm_df.sum(axis=0)
    tables["confusion_llm_vs_human"] = cm_df.reset_index(names="cell")

    # ---------- directional error analysis ------------------------------
    order = {c: i for i, c in enumerate(DISCLOSURE_ORDER)}
    adj = adj.copy()
    adj["llm_rank"] = adj.llm_label.map(order)
    adj["human_rank"] = adj.human_label.map(order)
    adj["direction"] = np.select(
        [adj.llm_rank > adj.human_rank, adj.llm_rank < adj.human_rank],
        ["llm_more_lenient", "llm_more_harsh"], default="agree")

    dir_rows = []
    for kind, sub in adj.groupby("violation_kind"):
        vc = sub.direction.value_counts()
        n = len(sub)
        agree = int(vc.get("agree", 0))
        dir_rows.append({
            "violation_kind": kind, "n": n,
            "n_agree": agree,
            "pct_agree": 100 * agree / n,
            "n_llm_lenient": int(vc.get("llm_more_lenient", 0)),
            "n_llm_harsh": int(vc.get("llm_more_harsh", 0)),
            "modal_llm": sub.llm_label.mode().iat[0],
            "modal_human": sub.human_label.mode().iat[0],
            "net_direction": ("LLM lenient" if vc.get("llm_more_lenient", 0) >
                              vc.get("llm_more_harsh", 0)
                              else "LLM harsh" if vc.get("llm_more_harsh", 0) >
                              vc.get("llm_more_lenient", 0) else "balanced"),
        })
    tables["disagreement_direction_by_kind"] = pd.DataFrame(dir_rows)

    # ---------- where exactly does disagreement live? -------------------
    pair = (adj.groupby(["llm_label", "human_label"]).size()
            .reset_index(name="n").sort_values("n", ascending=False))
    pair["pct_of_all"] = 100 * pair.n / len(adj)
    pair["is_agreement"] = pair.llm_label == pair.human_label
    tables["label_pair_frequency"] = pair

    # ---------- impact: does the choice of rater change conclusions? -----
    conceal_llm = adj.llm_label.isin(["non", "false"])
    conceal_hum = adj.human_label.isin(["non", "false"])
    w_llm = wilson(int(conceal_llm.sum()), len(adj))
    w_hum = wilson(int(conceal_hum.sum()), len(adj))
    tables["rater_impact_on_concealment"] = pd.DataFrame([
        {"rater": "LLM judge", "n_concealed": w_llm.k, "n": w_llm.n,
         "rate": w_llm.p, "ci_lo": w_llm.lo, "ci_hi": w_llm.hi},
        {"rater": "Human", "n_concealed": w_hum.k, "n": w_hum.n,
         "rate": w_hum.p, "ci_lo": w_hum.lo, "ci_hi": w_hum.hi},
    ])

    artifacts["adjudicated_with_direction"] = adj
    k_all = kappa_df.iloc[0]
    artifacts["headline"] = {
        "n": int(k_all.n),
        "kappa": float(k_all.kappa_unweighted),
        "kappa_linear": float(k_all.kappa_linear),
        "raw_agreement": float(k_all.raw_agreement),
        "pabak": float(k_all.pabak),
        "interpretation": k_all.interpretation,
    }
    return {"tables": tables, "artifacts": artifacts}
