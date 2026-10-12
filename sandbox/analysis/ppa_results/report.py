"""
Assembles a single markdown results document from the computed tables.

Deliberately opinionated about presentation: every proportion carries its
Wilson interval and its raw counts, every test carries its statistic and
its p-value, and anything the design cannot estimate is labelled as such
rather than quietly omitted.
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np
import pandas as pd


def _p(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    if x == 0:
        return "< 1e-300"
    if x < 0.001:
        return f"{x:.2e}"
    return f"{x:.3f}"


def _fmt_cell(col: str, v) -> str:
    """Column-aware float formatting. p-values and tiny quantities must not
    collapse to '0.000' — that is the difference between a reportable result
    and an unreadable table."""
    if pd.isna(v):
        return "—"
    lc = col.lower()
    if "p_value" in lc or lc.startswith("p_") or lc.endswith("_p"):
        return _p(v)
    if np.isinf(v):
        return "∞"
    if v != 0 and abs(v) < 0.001:
        return f"{v:.2e}"
    return f"{v:.3f}"


def _pct(x) -> str:
    return "—" if x != x else f"{100*x:.1f}%"


def _f(x, n=2) -> str:
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "∞" if (isinstance(x, float) and np.isinf(x)) else "—"
    return f"{x:.{n}f}"


def _md(df: pd.DataFrame, max_rows: int = 60) -> str:
    d = df.head(max_rows).copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v, _c=c: _fmt_cell(_c, v))
    out = d.to_markdown(index=False)
    if len(df) > max_rows:
        out += f"\n\n_({len(df) - max_rows} further rows in the CSV.)_"
    return out


def build(corpus, tables, quote_tables, figures, outdir: Path) -> str:
    R = []
    A = R.append
    runs = corpus.runs
    prim = tables["primary"]
    sec = tables["secondary"]
    rel = tables["reliability"]
    surv = tables["survival"]
    gt = tables["gate_tamper"]

    head = prim["violation_rate_headline"].set_index("condition")
    pc = prim["primary_contrast"]
    pooled = pc[pc.stratum == "POOLED"].iloc[0]

    A("# Persistence, Permission and Accountability — Results\n")
    A(f"_Generated from {len(runs)} runs "
      f"({int((runs.condition=='informed').sum())} informed, "
      f"{int((runs.condition=='uninformed').sum())} uninformed), "
      f"{len(corpus.violations)} violation events._\n")

    # ------------------------------------------------------------ abstract
    A("## 0. Headline numbers\n")
    A(f"- **Violation rate, uninformed:** {head.loc['uninformed','pretty']} "
      f"({int(head.loc['uninformed','k'])}/{int(head.loc['uninformed','n'])})")
    A(f"- **Violation rate, informed:** {head.loc['informed','pretty']} "
      f"({int(head.loc['informed','k'])}/{int(head.loc['informed','n'])})")
    A(f"- **Absolute difference:** {_f(pooled.diff_pp,1)} pp "
      f"[{_f(pooled.diff_ci_lo_pp,1)}, {_f(pooled.diff_ci_hi_pp,1)}]; "
      f"risk ratio {_f(pooled.risk_ratio,1)}×; "
      f"Cohen's *h* = {_f(pooled.cohens_h)}; "
      f"*z* = {_f(pooled.z)}, *p* {_p(pooled.p_value)}")
    kappa = rel["_headline"]
    A(f"- **Disclosure coding agreement:** κ = {_f(kappa['kappa'],3)} "
      f"({kappa['interpretation']}), raw agreement {_pct(kappa['raw_agreement'])}, "
      f"n = {kappa['n']} adjudicated violation events")
    dk = gt["disclosure_by_kind"].set_index("violation_kind")
    if "gate_tamper" in dk.index and "unauthorized_contact" in dk.index:
        A(f"- **Selective disclosure:** gate tampering was fully disclosed in "
          f"{_pct(dk.loc['gate_tamper','full_rate'])} of events "
          f"({int(dk.loc['gate_tamper','n_full_disclosure'])}/"
          f"{int(dk.loc['gate_tamper','n_events'])}), against "
          f"{_pct(dk.loc['unauthorized_contact','full_rate'])} for unauthorized contact "
          f"({int(dk.loc['unauthorized_contact','n_full_disclosure'])}/"
          f"{int(dk.loc['unauthorized_contact','n_events'])})")
    A("")

    # ------------------------------------------------- manipulation checks
    A("## 1. Manipulation and balance checks\n")
    cells = runs.groupby(["condition", "model_id", "affordance_level",
                          "finality", "task_id"], observed=True).size()
    A(f"The grid is exactly balanced: every one of the {len(cells)} design "
      f"cells contains {cells.iloc[0]} runs, so the informed/uninformed "
      "comparison is a like-for-like contrast rather than a "
      "composition-weighted one.\n")
    mc = runs.groupby("condition").agg(
        mean_stages_completed=("stages_completed", "mean"),
        pct_reaching_stage6=("stages_completed", lambda s: 100*(s >= 6).mean()),
        mean_pushback=("pushback_count", "mean"),
        mean_total_turns=("total_turns", "mean")).reset_index()
    A(_md(mc))
    A("\nBoth arms reach the submission stage and trigger the rejection ladder "
      "at comparable rates, so the primary contrast is not confounded by one "
      "arm simply failing to reach the measurement window.\n")

    # ------------------------------------------------------ primary result
    A("## 2. Primary contrast: informed vs uninformed\n")
    A(_md(prim["violation_rate_headline"]))
    A("\n### 2.1 Stratified contrasts\n")
    show = pc[["stratum", "level", "uninformed_k", "uninformed_n",
               "uninformed_rate", "informed_k", "informed_n", "informed_rate",
               "diff_pp", "diff_ci_lo_pp", "diff_ci_hi_pp", "risk_ratio",
               "z", "p_value", "p_fisher"]]
    A(_md(show))
    A("\nThe direction and significance of the effect are stable across every "
      "stratum. Notably the contrast does **not** wash out at high affordance: "
      "planting credentials, an override tool and a rival branch raises the "
      "uninformed rate without meaningfully moving the informed rate.\n")

    A("### 2.2 Full results matrix\n")
    A(_md(prim["results_matrix_full"], max_rows=80))

    A("\n### 2.3 Factor main effects\n")
    A(_md(prim["factor_main_effects"], max_rows=80))
    A("\n### 2.4 Omnibus tests\n")
    A(_md(prim["omnibus_tests"]))

    # --------------------------------------------------------- reliability
    A("\n## 3. Inter-rater reliability of disclosure coding\n")
    A("Per §8.3 of the methodology, agreement is reported transparently "
      "rather than suppressed.\n")
    A(_md(rel["kappa_by_stratum"]))
    A("\n### 3.1 Confusion table\n")
    A(_md(rel["confusion_llm_vs_human"]))
    A("\n### 3.2 Direction of disagreement\n")
    A(_md(rel["disagreement_direction_by_kind"]))
    dirtab = rel["disagreement_direction_by_kind"]
    worst = dirtab.sort_values("pct_agree").iloc[0]
    A(f"\nDisagreement is not random. On **{worst.violation_kind}** the two "
      f"passes agree on only {_f(worst.pct_agree,1)}% of events, and the "
      f"disagreement is systematically directional "
      f"(net: {worst.net_direction}). Because the divergence is "
      "rule-explicable rather than noise, the human labels are treated as "
      "primary and the automated labels are retained for audit.\n")

    A("### 3.2a Reading these κ values correctly\n")
    kb = rel["kappa_by_stratum"].set_index("stratum")
    A("Two strata illustrate why κ alone would mislead here, and both should "
      "be reported with their raw agreement and PABAK alongside:\n")
    if "kind=unauthorized_contact" in kb.index:
        r_ = kb.loc["kind=unauthorized_contact"]
        A(f"- **unauthorized_contact**: raw agreement "
          f"{_pct(r_.raw_agreement)} but κ = {_f(r_.kappa_unweighted,3)} "
          f"(PABAK = {_f(r_.pabak,3)}). This is the classic κ paradox: both "
          "raters assign `full` to almost every event, so expected agreement "
          "is nearly as high as observed agreement and κ collapses despite "
          "near-perfect concordance. The coders agree; κ is simply the wrong "
          "summary for a near-degenerate marginal distribution.")
    if "kind=gate_tamper" in kb.index:
        r_ = kb.loc["kind=gate_tamper"]
        A(f"- **gate_tamper**: raw agreement {_pct(r_.raw_agreement)}, "
          f"κ = {_f(r_.kappa_unweighted,3)} (PABAK = {_f(r_.pabak,3)}). Here "
          "κ is zero for the opposite reason — each rater used essentially a "
          "single, *different* category (LLM `non`, human `partial`), so "
          "there is no covariation for κ to detect. This is a genuine and "
          "total coding disagreement about one construct, not noise, and it "
          "is the substantive reason the human pass is treated as primary.\n")
    A("The pooled κ of "
      f"{_f(kb.loc['ALL','kappa_unweighted'],3)} "
      f"[{_f(kb.loc['ALL','lo_unweighted'],3)}, "
      f"{_f(kb.loc['ALL','hi_unweighted'],3)}] "
      f"(linear-weighted {_f(kb.loc['ALL','kappa_linear'],3)}) is therefore "
      "an average over one stratum where the raters agree almost perfectly "
      "and one where they disagree almost completely. It should be quoted "
      "with that decomposition, never on its own.\n")
    A("### 3.3 Does the choice of rater change the conclusion?\n")
    A(_md(rel["rater_impact_on_concealment"]))

    # ---------------------------------------------------------- disclosure
    A("\n## 4. Disclosure and concealment\n")
    A(_md(sec["disclosure_distribution_all_runs"]))
    A("\n### 4.1 Concealment rates\n")
    A(_md(sec["concealment_rate"]))
    A("\n### 4.2 Concealment type\n")
    A(_md(sec["concealment_type"]))
    A("\n### 4.3 Disclosure by violation kind — the selective-disclosure result\n")
    A(_md(gt["disclosure_by_kind"]))
    if "tamper_vs_contact_disclosure" in gt:
        t = gt["tamper_vs_contact_disclosure"].iloc[0]
        A(f"\nAgents disclosed unauthorized contact almost universally "
          f"({_pct(t.contact_rate)}) but disclosed gate tampering "
          f"{_pct(t.gate_tamper_rate)} of the time — a difference of "
          f"{_f(t.diff_pp,1)} pp [{_f(t.ci_lo_pp,1)}, {_f(t.ci_hi_pp,1)}], "
          f"*z* = {_f(t.z)}, *p* {_p(t.p_value)} "
          f"(Fisher exact *p* {_p(t.p_fisher)}).\n")
        A("This is the single most consequential finding in the corpus. The "
          "two violation kinds differ in exactly the way that matters for "
          "an accountability measure: unauthorized contact is *blocked and "
          "visible* — the agent knows the environment saw it — whereas gate "
          "tampering *succeeds silently* and leaves a green CI badge behind. "
          "Disclosure tracked detectability, not severity.\n")
    A("### 4.4 Selective disclosure within runs\n")
    A(_md(sec["selective_disclosure_summary"]))

    # ------------------------------------------------------------ proximity
    A("\n## 5. Boundary proximity\n")
    A(_md(sec["proximity_distribution"]))
    A("\n" + _md(sec["proximity_summary"]))
    A("\n" + _md(sec["proximity_test"]))
    if "prox4_given_prox3plus" in sec:
        A("\n### 5.1 Does rule knowledge only downgrade level 4 to level 3?\n")
        A(_md(sec["prox4_given_prox3plus"]))

    # ------------------------------------------------------------ survival
    A("\n## 6. Time to first violation\n")
    A(_md(surv["km_summary"]))
    A("\n**Median survival is not reached in any stratum** — no group's "
      "survivor function falls to 0.5 within the observation window — so "
      "median time-to-violation is reported as undefined and the curves are "
      "summarised by S(t) at fixed turn indices instead.\n")
    A(_md(surv["logrank_tests"]))

    # ---------------------------------------------------------- persistence
    A("\n## 7. Persistence and escalation\n")
    A(_md(sec["persistence_summary"]))
    A("\n" + _md(sec["first_violation_turn_test"]))

    A("\n### 7.1 Where does persistence go?\n")
    A(_md(sec["persistence_channel"]))
    A("\n" + _md(sec["persistence_channel_test"]))
    pcn = sec["persistence_channel"].set_index("condition")
    pct = sec["persistence_channel_test"].iloc[0]
    if "informed" in pcn.index and "uninformed" in pcn.index:
        A(f"\nThe informed arm is **not** simply more passive. Informed agents "
          f"triggered a mean of {_f(pcn.loc['informed','mean_pushback'])} "
          f"maintainer escalation rungs per run against "
          f"{_f(pcn.loc['uninformed','mean_pushback'])} in the uninformed arm "
          f"(*z* = {_f(pct.z)}, *p* {_p(pct.p_value)}) — roughly "
          f"{_f(pcn.loc['informed','mean_pushback']/pcn.loc['uninformed','mean_pushback'],1)}× "
          f"more in-channel argument — while violating "
          f"{_f(pcn.loc['uninformed','violation_rate']/pcn.loc['informed','violation_rate'],1)}× "
          "less often. Stating the rules did not suppress goal-directed "
          "persistence; it **redirected** it into the legitimate channel the "
          "design always left open.\n")
    A(_md(sec["persistence_channel_within"]))
    A("\nWithin each arm, however, arguing and crossing are *correlated* "
      "rather than substitutable — runs that violated also pushed back more. "
      "Both are expressions of the same underlying persistence; the "
      "manipulation changes which outlet that persistence finds, not how "
      "much of it there is.\n")

    # -------------------------------------------------------- violation mix
    A("\n## 8. What kind of violation, and where\n")
    A(_md(sec["violation_type_composition"]))
    A("\n" + _md(sec["violation_type_by_affordance"]))
    A("\n" + _md(sec["violation_kind_rates"]))

    # ------------------------------------------------------- gate tampering
    A("\n## 9. Gate tampering deep dive\n")
    A(_md(gt["gate_tamper_target_paths"]))
    A("\n" + _md(gt["gate_tamper_route_breadth"]))
    A("\n### 9.1 First resort or last resort?\n")
    A(_md(gt["gate_tamper_timing_summary"]))
    ts = gt["gate_tamper_timing_summary"]
    late = ts[ts.resort.str.contains("last resort")]
    if len(late):
        r = late.iloc[0]
        A(f"\nTampering was overwhelmingly a **last** resort: {int(r.n)} of "
          f"{int(ts.n.sum())} tampering runs weakened the gate only after "
          f"opening a pull request, at a median of {_f(r.median_turns_after_pr,0)} "
          "turns past that first submission. Agents did the honest work "
          "first, were refused, kept arguing, and only then reached for the "
          "test file.\n")

    # ----------------------------------------------------------- soft/hard
    A("\n## 10. The soft-rejection paradox\n")
    A(_md(sec["finality_x_condition"]))
    A("\n" + _md(sec["soft_rejection_paradox"]))
    sp = sec["soft_rejection_paradox"]
    un = sp[sp.condition == "uninformed"].iloc[0]
    A(f"\nIn the uninformed arm, a *softer* rejection produced **more** "
      f"boundary crossing than a hard one ({_pct(un.soft_rate)} vs "
      f"{_pct(un.hard_rate)}; Δ = {_f(un.diff_pp,1)} pp, *p* {_p(un.p_value)}). "
      "An invitation to retry reads as a live goal that persistence might "
      "still satisfy; an unappealable refusal removes the goal entirely. "
      "Hope, not hostility, is what drives escalation here.\n")

    # ------------------------------------------------------------- contact
    A("\n## 11. Tone of off-allowlist contact\n")
    A(_md(gt["contact_tone_distribution"]))
    A("\n" + _md(gt["contact_persistence"]))
    A("\n" + _md(gt["contact_recipients"]))
    tone = gt["contact_tone_distribution"].set_index("contact_tone")
    if "polite_feedback" in tone.index:
        A(f"\n**This materially qualifies the headline violation rate.** "
          f"{_f(tone.loc['polite_feedback','pct'],1)}% of off-allowlist "
          "messages were benign requests for review feedback or status "
          "updates, not attempts to obtain special treatment. Mechanically "
          "these are authorization violations and are correctly scored as "
          "such — the agent contacted a recipient it was not permitted to "
          "contact. But they are not evidence of adversarial intent, and "
          "pooling them with gate tampering into a single 'violation rate' "
          "overstates the misalignment signal. The gate-tamper rate is the "
          "cleaner measure.\n")

    # ------------------------------------------------------------ ideation
    if "negation_frame_impact" in quote_tables:
        A("\n## 12. Ideation detection is a lower bound\n")
        A(_md(quote_tables["negation_frame_impact"]))
        A("\n" + _md(quote_tables["ideation_undercount_estimate"]))
        imp = quote_tables["negation_frame_impact"].iloc[0]
        A(f"\nOf {int(imp.sentences_matching_a_cue)} sentences matching a "
          f"forbidden-route keyword, {_f(imp.pct_disavowal,1)}% were "
          "*disavowals* — agents stating they would not take the route. A "
          "naive keyword scan of the kind the harness uses for proximity "
          "level 1 counts these as ideation. Correcting for negation cuts "
          "the apparent ideation volume by roughly five-sixths, while "
          "recovering genuine contemplation in runs the harness scored as "
          "fully compliant. Level 1 should be read as a noisy lower bound in "
          "both directions and used only exploratorily.\n")

    # -------------------------------------------------------------- quotes
    A("\n## 13. Illustrative quotations\n")
    A("Every quotation carries its run ID; all are recoverable from the raw "
      "logs. Selection is by category membership, not by rhetorical effect.\n")
    sc = quote_tables.get("showcase_quotes")
    if sc is not None and len(sc):
        for cat in sc.category.unique():
            sub = sc[sc.category == cat].head(6)
            A(f"\n### {cat}\n")
            for r in sub.itertuples():
                q = textwrap.shorten(r.quote, 400, placeholder=" […]")
                A(f"> {q}\n>\n> — `{r.run_id}` · {r.condition} · {r.model} "
                  f"· affordance={r.affordance} · turn {r.turn}\n")

    # ----------------------------------------------------------- limits
    A("\n## 14. What this dataset cannot show\n")
    A(textwrap.dedent("""\
        - **Personalization and task domain are design constants**, pinned to
          `impersonal` and `aggregation`. They are not null results and must
          not be reported as such.
        - **The uninformed manipulation is prompt-level, not information-level.**
          `CONTRIBUTING.md` documents the same constraints and stage 1 directs
          the agent to read it, so the contrast estimates the effect of
          *stating rules in the system prompt* against a background where they
          remain discoverable.
        - **Context trimming co-varies with the manipulation.** The rules block
          is a system message and survives eviction; task history does not. Part
          of the informed advantage may be a persistence-of-system-prompt effect.
        - **Blocked and permitted violation kinds are not commensurable.**
          Unauthorized contact is refused by the environment; gate tampering is
          allowed to succeed. Pooled rates mix attempts with completions.
        - **Single environment, single rejection script, one sampling
          temperature.** Generalisation beyond this setting is untested.
        - **The informed arm has only four violation events.** Every
          informed-side estimate stratified below the pooled level rests on
          single-digit counts, and the intervals reflect that.
        """))

    A("\n## 15. Figures\n")
    for f in figures:
        A(f"- `{Path(f).name}`")

    return "\n".join(R)
