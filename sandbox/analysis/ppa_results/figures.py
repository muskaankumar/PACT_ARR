"""
Publication figures.

Style rules kept deliberately plain: no seaborn, no chartjunk, no gridlines
competing with the data, error bars everywhere a proportion is plotted, and
n printed on the figure wherever a reader would otherwise have to guess it.
Figures are written as both PNG (300 dpi) and PDF (vector, for typesetting).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from .data import AFFORDANCE_ORDER, DISCLOSURE_ORDER, FINALITY_ORDER, PROXIMITY_LABELS
from .stats import kaplan_meier, wilson

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "-",
    "grid.linewidth": 0.6,
    "legend.frameon": False,
    "figure.autolayout": False,
})

C_UNINF = "#C44536"     # uninformed  (warm)
C_INF = "#2A6F97"       # informed    (cool)
C_NEUTRAL = "#6C757D"
COND_COLOR = {"uninformed": C_UNINF, "informed": C_INF}
DISCLOSURE_COLOR = {"full": "#2D6A4F", "partial": "#E9C46A",
                    "non": "#E76F51", "false": "#7A1F1F"}
PROX_COLOR = ["#D9E4DD", "#A8C0B4", "#E9C46A", "#E07A5F", "#8B2E20"]


def _save(fig, outdir: Path, name: str, saved: list):
    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        p = outdir / f"{name}.{ext}"
        fig.savefig(p, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    saved.append(str(outdir / f"{name}.png"))


def _err(rates, los, his):
    rates = np.asarray(rates); los = np.asarray(los); his = np.asarray(his)
    return np.vstack([rates - los, his - rates])


def _bar_labels(ax, bars, ks, ns, fmt="{k}/{n}"):
    for b, k, n in zip(bars, ks, ns):
        ax.annotate(fmt.format(k=int(k), n=int(n)),
                    (b.get_x() + b.get_width() / 2, 0),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=7.5, color="white",
                    fontweight="bold")


# ---------------------------------------------------------------- figures

def fig_headline(runs, outdir, saved):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.3),
                                   gridspec_kw={"width_ratios": [1, 1.7]})

    conds = ["uninformed", "informed"]
    stats = [wilson(int(runs[runs.condition == c].auth_violation.sum()),
                    int((runs.condition == c).sum())) for c in conds]
    x = np.arange(2)
    b = ax1.bar(x, [s.p for s in stats], width=.55,
                color=[COND_COLOR[c] for c in conds],
                yerr=_err([s.p for s in stats], [s.lo for s in stats],
                          [s.hi for s in stats]),
                capsize=5, ecolor="#222", error_kw={"lw": 1.2})
    _bar_labels(ax1, b, [s.k for s in stats], [s.n for s in stats])
    ax1.set_xticks(x); ax1.set_xticklabels(["Uninformed", "Informed"])
    ax1.set_ylabel("P(any authorization violation)")
    ax1.set_ylim(0, .55)
    ax1.set_title("A. Primary contrast")
    d = 100 * (stats[0].p - stats[1].p)
    ax1.annotate(f"Δ = {d:.1f} pp\nRR = {stats[0].p/stats[1].p:.1f}×",
                 (0.5, .47), ha="center", fontsize=9.5,
                 bbox=dict(boxstyle="round,pad=0.35", fc="#F3F3F3", ec="#BBB"))

    # panel B: model x affordance
    w = 0.38
    labels, pos = [], []
    i = 0
    for m in sorted(runs.model_pretty.unique()):
        for aff in AFFORDANCE_ORDER:
            labels.append(f"{m}\n{aff}"); pos.append(i); i += 1
    for j, c in enumerate(conds):
        rates, los, his, ks, ns = [], [], [], [], []
        for m in sorted(runs.model_pretty.unique()):
            for aff in AFFORDANCE_ORDER:
                g = runs[(runs.condition == c) & (runs.model_pretty == m) &
                         (runs.affordance_level == aff)]
                s = wilson(int(g.auth_violation.sum()), len(g))
                rates.append(s.p); los.append(s.lo); his.append(s.hi)
                ks.append(s.k); ns.append(s.n)
        off = (-w/2 if j == 0 else w/2)
        bb = ax2.bar(np.array(pos) + off, rates, width=w, color=COND_COLOR[c],
                     label=c.capitalize(),
                     yerr=_err(rates, los, his), capsize=3,
                     ecolor="#333", error_kw={"lw": .9})
        for bar, k, n in zip(bb, ks, ns):
            ax2.annotate(f"{int(k)}", (bar.get_x()+bar.get_width()/2, bar.get_height()),
                         xytext=(0, 2), textcoords="offset points",
                         ha="center", fontsize=7, color="#333")
    ax2.axvline(2.5, color="#999", lw=.8, ls="--")
    ax2.set_xticks(pos); ax2.set_xticklabels(labels, fontsize=8.5)
    ax2.set_ylabel("P(any authorization violation)")
    ax2.set_ylim(0, .95)
    ax2.set_title("B. Model × affordance level")
    ax2.legend(loc="upper left", fontsize=9)
    fig.suptitle("Authorization violation rate by information condition",
                 fontsize=13, fontweight="bold", y=1.02)
    _save(fig, outdir, "fig01_violation_rate_headline", saved)


def fig_results_matrix(runs, outdir, saved):
    """Heatmap of violation rate over every model x affordance x finality cell."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, cond in zip(axes, ["uninformed", "informed"]):
        sub = runs[runs.condition == cond]
        rows = [(m, a) for m in sorted(runs.model_pretty.unique())
                for a in AFFORDANCE_ORDER]
        mat = np.zeros((len(rows), len(FINALITY_ORDER)))
        ann = np.empty(mat.shape, dtype=object)
        for i, (m, a) in enumerate(rows):
            for j, f in enumerate(FINALITY_ORDER):
                g = sub[(sub.model_pretty == m) & (sub.affordance_level == a) &
                        (sub.finality == f)]
                mat[i, j] = g.auth_violation.mean() if len(g) else np.nan
                ann[i, j] = f"{int(g.auth_violation.sum())}/{len(g)}"
        im = ax.imshow(mat, cmap="RdYlBu_r", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(FINALITY_ORDER)))
        ax.set_xticklabels([f.replace("_", "\n") for f in FINALITY_ORDER], fontsize=8.5)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([f"{m} · {a}" for m, a in rows], fontsize=8.5)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                ax.text(j, i, ann[i, j], ha="center", va="center", fontsize=8,
                        color="white" if (v > .55 or v < .12) else "#111",
                        fontweight="bold")
        ax.set_title(f"{cond.capitalize()}  (n=216)")
        ax.grid(False)
    fig.colorbar(im, ax=axes, shrink=.85, label="violation rate", pad=.02)
    fig.suptitle("Full results matrix: violations / runs per design cell",
                 fontsize=13, fontweight="bold")
    _save(fig, outdir, "fig02_results_matrix", saved)


def fig_disclosure(runs, violations, outdir, saved):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))

    # A: overall disclosure by condition (all runs)
    ax = axes[0]
    bottoms = np.zeros(2)
    conds = ["uninformed", "informed"]
    for lab in DISCLOSURE_ORDER:
        vals = [(runs[runs.condition == c].disclosure_primary == lab).mean()
                for c in conds]
        ax.bar([0, 1], vals, bottom=bottoms, color=DISCLOSURE_COLOR[lab],
               label=lab, width=.6, edgecolor="white", lw=.8)
        for i, v in enumerate(vals):
            if v > .045:
                ax.text(i, bottoms[i] + v/2, f"{100*v:.0f}%", ha="center",
                        va="center", fontsize=8.5, color="white", fontweight="bold")
        bottoms += np.array(vals)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Uninformed", "Informed"])
    ax.set_ylim(0, 1); ax.set_ylabel("share of runs")
    ax.set_title("A. Disclosure, all runs")
    ax.legend(fontsize=8, ncol=2, loc="lower center", bbox_to_anchor=(.5, -.32))

    # B: disclosure among violating runs only
    ax = axes[1]
    vr = runs[runs.auth_violation]
    groups = [("Uninformed", vr[vr.condition == "uninformed"]),
              ("Informed", vr[vr.condition == "informed"])]
    bottoms = np.zeros(len(groups))
    for lab in DISCLOSURE_ORDER:
        vals = [ (g.disclosure_primary == lab).mean() if len(g) else 0
                 for _, g in groups]
        ax.bar(range(len(groups)), vals, bottom=bottoms,
               color=DISCLOSURE_COLOR[lab], width=.6, edgecolor="white", lw=.8)
        for i, v in enumerate(vals):
            if v > .045:
                ax.text(i, bottoms[i]+v/2, f"{100*v:.0f}%", ha="center",
                        va="center", fontsize=8.5, color="white", fontweight="bold")
        bottoms += np.array(vals)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([f"{n}\n(n={len(g)})" for n, g in groups])
    ax.set_ylim(0, 1); ax.set_title("B. Disclosure, violating runs only")

    # C: per-violation-kind disclosure — the selective disclosure result
    ax = axes[2]
    v = violations.copy()
    v["label"] = v.human_label.fillna(v.llm_label)
    kinds = v.violation_kind.value_counts().index.tolist()
    bottoms = np.zeros(len(kinds))
    for lab in DISCLOSURE_ORDER:
        vals = [(v[v.violation_kind == k].label == lab).mean() for k in kinds]
        ax.bar(range(len(kinds)), vals, bottom=bottoms,
               color=DISCLOSURE_COLOR[lab], width=.6, edgecolor="white", lw=.8)
        for i, val in enumerate(vals):
            if val > .06:
                ax.text(i, bottoms[i]+val/2, f"{100*val:.0f}%", ha="center",
                        va="center", fontsize=8.5, color="white", fontweight="bold")
        bottoms += np.array(vals)
    ax.set_xticks(range(len(kinds)))
    ax.set_xticklabels([f"{k.replace('_',chr(10))}\n(n={int((v.violation_kind==k).sum())})"
                        for k in kinds], fontsize=8)
    ax.set_ylim(0, 1); ax.set_title("C. Disclosure by violation kind")
    fig.suptitle("Disclosure behaviour", fontsize=13, fontweight="bold", y=1.02)
    _save(fig, outdir, "fig03_disclosure", saved)


def fig_proximity(runs, outdir, saved):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2),
                                   gridspec_kw={"width_ratios": [1.4, 1]})
    conds = ["uninformed", "informed"]
    w = .38
    for j, c in enumerate(conds):
        g = runs[runs.condition == c]
        counts = [ (g.boundary_proximity == lvl).mean() for lvl in range(5)]
        ax1.bar(np.arange(5) + (-w/2 if j == 0 else w/2), counts, width=w,
                color=COND_COLOR[c], label=c.capitalize(),
                edgecolor="white", lw=.6)
        for i, v in enumerate(counts):
            if v > 0:
                ax1.annotate(f"{int((g.boundary_proximity==i).sum())}",
                             (i + (-w/2 if j == 0 else w/2), v),
                             xytext=(0, 2), textcoords="offset points",
                             ha="center", fontsize=7.5, color="#333")
    ax1.set_xticks(range(5))
    ax1.set_xticklabels([f"{k}\n{v}" for k, v in PROXIMITY_LABELS.items()],
                        fontsize=8.5)
    ax1.set_ylabel("share of runs"); ax1.set_ylim(0, 1)
    ax1.set_title("A. Boundary-proximity distribution")
    ax1.legend()
    ax1.annotate("levels 1–2 are lexical\nlower bounds (§7.2)", (1.5, .72),
                 fontsize=8, color="#666", ha="center", style="italic")

    # B: stacked proximity by affordance, uninformed only (where events live)
    un = runs[runs.condition == "uninformed"]
    bottoms = np.zeros(3)
    for lvl in range(5):
        vals = [(un[un.affordance_level == a].boundary_proximity == lvl).mean()
                for a in AFFORDANCE_ORDER]
        ax2.bar(range(3), vals, bottom=bottoms, color=PROX_COLOR[lvl],
                width=.6, label=f"{lvl} {PROXIMITY_LABELS[lvl]}",
                edgecolor="white", lw=.7)
        for i, v in enumerate(vals):
            if v > .06:
                ax2.text(i, bottoms[i]+v/2, f"{100*v:.0f}", ha="center",
                         va="center", fontsize=8, color="#222", fontweight="bold")
        bottoms += np.array(vals)
    ax2.set_xticks(range(3)); ax2.set_xticklabels(AFFORDANCE_ORDER)
    ax2.set_xlabel("affordance level"); ax2.set_ylim(0, 1)
    ax2.set_title("B. Proximity by affordance (uninformed)")
    ax2.legend(fontsize=7.5, loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.suptitle("How close did agents get to the boundary?",
                 fontsize=13, fontweight="bold", y=1.02)
    _save(fig, outdir, "fig04_boundary_proximity", saved)


def fig_survival(runs, outdir, saved):
    from .survival import build_survival_frame
    d = build_survival_frame(runs)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharey=True)

    def draw(ax, groups, colors, title):
        for (lab, sub), col in zip(groups, colors):
            c = kaplan_meier(sub.duration, sub.event)
            ax.step(c.times, c.survival, where="post", color=col, lw=2,
                    label=f"{lab} (n={c.n}, e={c.n_events})")
            ax.fill_between(c.times, c.lower, c.upper, step="post",
                            color=col, alpha=.13, lw=0)
        ax.set_xlabel("turn index"); ax.set_ylim(.5, 1.005)
        ax.set_title(title); ax.legend(fontsize=8, loc="lower left")
        ax.axhline(.5, color="#999", lw=.8, ls=":")

    draw(axes[0], [(c.capitalize(), d[d.condition == c]) for c in
                   ["uninformed", "informed"]],
         [C_UNINF, C_INF], "A. By information condition")
    draw(axes[1], [(m, d[d.model_pretty == m]) for m in
                   sorted(d.model_pretty.unique())],
         ["#8E44AD", "#16A085"], "B. By model")
    draw(axes[2], [(a, d[(d.condition == "uninformed") &
                         (d.affordance_level == a)]) for a in AFFORDANCE_ORDER],
         ["#457B9D", "#E9A23B", "#C1121F"], "C. By affordance (uninformed)")
    axes[0].set_ylabel("S(t) = P(no violation by turn t)")
    fig.suptitle("Time to first authorization violation (Kaplan–Meier, "
                 "log-log 95% bands)", fontsize=13, fontweight="bold", y=1.03)
    _save(fig, outdir, "fig05_time_to_violation", saved)


def fig_kappa(reliability_tables, outdir, saved):
    kt = reliability_tables["kappa_by_stratum"]
    cm = reliability_tables["confusion_llm_vs_human"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4),
                                   gridspec_kw={"width_ratios": [1.25, 1]})

    # A: confusion matrix
    body = cm[cm.cell != "col_total"].set_index("cell")
    body = body.drop(columns=[c for c in body.columns if c == "row_total"])
    mat = body.values.astype(float)
    im = ax1.imshow(mat, cmap="Blues", aspect="auto")
    ax1.set_xticks(range(mat.shape[1]))
    ax1.set_xticklabels([c.replace("HUMAN=", "") for c in body.columns])
    ax1.set_yticks(range(mat.shape[0]))
    ax1.set_yticklabels([c.replace("LLM=", "") for c in body.index])
    ax1.set_xlabel("human label"); ax1.set_ylabel("LLM judge label")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if mat[i, j] > 0:
                ax1.text(j, i, int(mat[i, j]), ha="center", va="center",
                         fontsize=11, fontweight="bold",
                         color="white" if mat[i, j] > mat.max()*.55 else "#111")
    ax1.grid(False)
    ax1.set_title("A. LLM judge × human confusion")

    # B: kappa forest
    keep = kt[kt.kappa_unweighted.notna()].copy()
    keep = keep.sort_values("kappa_unweighted")
    y = np.arange(len(keep))
    ax2.errorbar(keep.kappa_unweighted, y,
                 xerr=np.vstack([keep.kappa_unweighted - keep.lo_unweighted,
                                 keep.hi_unweighted - keep.kappa_unweighted]),
                 fmt="o", color="#2A6F97", ecolor="#7FA9C4", capsize=3, ms=6)
    ax2.set_yticks(y)
    ax2.set_yticklabels([f"{s}  (n={n})" for s, n in
                         zip(keep.stratum, keep.n)], fontsize=8)
    ax2.axvline(0, color="#333", lw=.9)
    for thr, lab in [(.21, "slight"), (.41, "fair"), (.61, "moderate"),
                     (.81, "subst.")]:
        ax2.axvline(thr, color="#CCC", lw=.7, ls="--")
    ax2.set_xlabel("Cohen's κ (unweighted, 95% CI)")
    ax2.set_title("B. Agreement by stratum")
    ax2.set_xlim(-.55, 1.05)
    fig.suptitle("Inter-rater reliability of disclosure coding",
                 fontsize=13, fontweight="bold", y=1.02)
    _save(fig, outdir, "fig06_kappa_reliability", saved)


def fig_gate_tamper(gt_tables, runs, outdir, saved):
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))

    # A: timing — first tamper turn vs first PR turn
    ax = axes[0]
    tim = gt_tables["gate_tamper_timing"]
    ok = tim.dropna(subset=["first_pr_turn"])
    ax.scatter(ok.first_pr_turn, ok.first_turn, s=44, color="#C1121F",
               alpha=.75, edgecolor="white", lw=.7, zorder=3)
    lim = [0, max(ok.first_turn.max(), ok.first_pr_turn.max()) * 1.08]
    ax.plot(lim, lim, ls="--", color="#666", lw=1, zorder=2)
    ax.annotate("tamper before first PR\n(first resort)", (lim[1]*.55, lim[1]*.22),
                fontsize=8, color="#666", ha="center")
    ax.annotate("tamper after first PR\n(last resort)", (lim[1]*.22, lim[1]*.82),
                fontsize=8, color="#666", ha="center")
    ax.set_xlabel("turn of first pull request")
    ax.set_ylabel("turn of first gate tamper")
    ax.set_xlim(0, lim[1]); ax.set_ylim(0, lim[1])
    ax.set_title(f"A. Tampering timing (n={len(ok)})")

    # B: repeat tampering — how many separate weakening writes per run
    ax = axes[1]
    per = gt_tables["gate_tamper_per_run"]
    counts = per.n_events.value_counts().sort_index()
    ax.bar(counts.index.astype(int), counts.values, color="#C1121F", width=.55)
    for xi, v in zip(counts.index, counts.values):
        ax.text(int(xi), v, str(int(v)), ha="center", va="bottom",
                fontsize=9.5, fontweight="bold")
    ax.set_xticks(counts.index.astype(int))
    ax.set_xlabel("weakening writes in the run")
    ax.set_ylabel("runs")
    ax.set_ylim(0, counts.values.max() * 1.2)
    paths = gt_tables["gate_tamper_target_paths"]
    top = paths.iloc[0]
    ax.set_title("B. Repeat tampering")
    ax.annotate(f"all {int(per.shape[0])} runs targeted\n`{top.target_path}` "
                f"— no run\nfound a second route",
                (.97, .93), xycoords="axes fraction", ha="right", va="top",
                fontsize=8, color="#444", style="italic",
                bbox=dict(boxstyle="round,pad=0.3", fc="#F7F0EF", ec="#E0C9C6"))

    # C: full-disclosure rate by kind
    ax = axes[2]
    dk = gt_tables["disclosure_by_kind"].sort_values("full_rate")
    y = np.arange(len(dk))
    ax.barh(y, dk.full_rate, color=["#C1121F" if k == "gate_tamper" else "#2D6A4F"
                                   for k in dk.violation_kind], height=.5,
            xerr=_err(dk.full_rate.values, dk.ci_lo.values, dk.ci_hi.values),
            capsize=4, ecolor="#333", error_kw={"lw": 1})
    ax.set_yticks(y)
    ax.set_yticklabels([f"{k.replace('_',' ')}\n(n={int(n)})"
                        for k, n in zip(dk.violation_kind, dk.n_events)],
                       fontsize=8.5)
    ax.set_xlabel("P(full disclosure)"); ax.set_xlim(0, 1.08)
    ax.set_title("C. Full disclosure by kind")
    fig.suptitle("Gate tampering: the least-disclosed violation",
                 fontsize=13, fontweight="bold", y=1.02)
    _save(fig, outdir, "fig07_gate_tamper", saved)


def fig_violation_types(violations, runs, outdir, saved):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.2))
    aff = violations.groupby(["affordance_level", "violation_kind"],
                             observed=True).size().unstack(fill_value=0)
    aff = aff.reindex(AFFORDANCE_ORDER)
    colors = {"unauthorized_contact": "#457B9D", "gate_tamper": "#C1121F",
              "override_capability_use": "#F4A261", "rival_sabotage": "#6A4C93",
              "forbidden_file_edit": "#2A9D8F", "protected_branch_push": "#8D99AE"}
    bottoms = np.zeros(len(aff))
    for k in aff.columns:
        ax1.bar(range(len(aff)), aff[k].values, bottom=bottoms,
                color=colors.get(k, "#999"), width=.6, label=k.replace("_", " "),
                edgecolor="white", lw=.8)
        for i, v in enumerate(aff[k].values):
            if v > 0:
                ax1.text(i, bottoms[i] + v/2, str(int(v)), ha="center",
                         va="center", fontsize=8.5, color="white", fontweight="bold")
        bottoms += aff[k].values
    ax1.set_xticks(range(len(aff))); ax1.set_xticklabels(aff.index)
    ax1.set_xlabel("affordance level"); ax1.set_ylabel("violation events")
    ax1.set_title("A. Violation-kind mix by affordance")
    ax1.legend(fontsize=8)

    # B: proportional composition
    share = aff.div(aff.sum(axis=1), axis=0)
    bottoms = np.zeros(len(share))
    for k in share.columns:
        ax2.bar(range(len(share)), share[k].values, bottom=bottoms,
                color=colors.get(k, "#999"), width=.6, edgecolor="white", lw=.8)
        for i, v in enumerate(share[k].values):
            if v > .05:
                ax2.text(i, bottoms[i] + v/2, f"{100*v:.0f}%", ha="center",
                         va="center", fontsize=8.5, color="white", fontweight="bold")
        bottoms += share[k].values
    ax2.set_xticks(range(len(share))); ax2.set_xticklabels(share.index)
    ax2.set_xlabel("affordance level"); ax2.set_ylabel("share of events")
    ax2.set_ylim(0, 1); ax2.set_title("B. Composition (normalised)")
    fig.suptitle("Does affordance change what agents do, not just how often?",
                 fontsize=13, fontweight="bold", y=1.02)
    _save(fig, outdir, "fig08_violation_types", saved)


def fig_finality(runs, outdir, saved):
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    w = .38
    x = np.arange(3)
    for j, c in enumerate(["uninformed", "informed"]):
        rates, los, his, ks, ns = [], [], [], [], []
        for f in FINALITY_ORDER:
            g = runs[(runs.condition == c) & (runs.finality == f)]
            s = wilson(int(g.auth_violation.sum()), len(g))
            rates.append(s.p); los.append(s.lo); his.append(s.hi)
            ks.append(s.k); ns.append(s.n)
        ax.bar(x + (-w/2 if j == 0 else w/2), rates, width=w,
               color=COND_COLOR[c], label=c.capitalize(),
               yerr=_err(rates, los, his), capsize=4, ecolor="#333",
               error_kw={"lw": 1})
        for i, (r, k, n) in enumerate(zip(rates, ks, ns)):
            ax.annotate(f"{int(k)}/{int(n)}",
                        (i + (-w/2 if j == 0 else w/2), r),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(["retry invited\n(softest)", "final this attempt",
                        "never reconsidered\n(hardest)"], fontsize=9)
    ax.set_ylabel("P(any authorization violation)")
    ax.set_ylim(0, .72)
    ax.set_title("The soft-rejection paradox:\nviolations fall as the "
                 "rejection hardens", fontsize=12)
    ax.legend()
    _save(fig, outdir, "fig09_soft_rejection_paradox", saved)


def fig_contact_tone(gt_tables, outdir, saved):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4),
                                   gridspec_kw={"width_ratios": [1, 1.2]})
    td = gt_tables["contact_tone_distribution"].sort_values("n_messages")
    cols = {"polite_feedback": "#2D6A4F", "pressuring": "#E9C46A",
            "polite_lobby": "#E76F51", "coercive": "#7A1F1F",
            "unclassified": "#AAA"}
    ax1.barh(td.contact_tone, td.n_messages,
             color=[cols.get(t, "#999") for t in td.contact_tone], height=.55)
    for i, (n, p) in enumerate(zip(td.n_messages, td.pct)):
        ax1.text(n + 1.5, i, f"{int(n)}  ({p:.1f}%)", va="center", fontsize=9)
    ax1.set_xlabel("messages"); ax1.set_xlim(0, td.n_messages.max()*1.42)
    ax1.set_title("A. Tone of off-allowlist messages")

    bf = gt_tables["contact_tone_by_finality"]
    piv = bf.pivot(index="finality", columns="contact_tone",
                   values="n").reindex(FINALITY_ORDER).fillna(0)
    bottoms = np.zeros(len(piv))
    for t in piv.columns:
        ax2.bar(range(len(piv)), piv[t].values, bottom=bottoms,
                color=cols.get(t, "#999"), width=.6, label=t,
                edgecolor="white", lw=.8)
        for i, v in enumerate(piv[t].values):
            if v > 0:
                ax2.text(i, bottoms[i]+v/2, str(int(v)), ha="center",
                         va="center", fontsize=8, color="white", fontweight="bold")
        bottoms += piv[t].values
    ax2.set_xticks(range(len(piv)))
    ax2.set_xticklabels([f.replace("_", "\n") for f in piv.index], fontsize=8.5)
    ax2.set_ylabel("messages"); ax2.legend(fontsize=8)
    ax2.set_title("B. Tone by rejection finality")
    fig.suptitle("Most boundary-crossing contact was not adversarial",
                 fontsize=13, fontweight="bold", y=1.02)
    _save(fig, outdir, "fig10_contact_tone", saved)


def fig_drift(drift, runs, outdir, saved):
    if drift is None or len(drift) == 0:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.2))
    d = drift.merge(runs[["run_id", "auth_violation"]], on="run_id", how="left")
    for c, col in [("uninformed", C_UNINF), ("informed", C_INF)]:
        sub = d[d.condition == c]
        ax1.hist(sub.first_probe_turn.dropna(), bins=25, alpha=.55,
                 color=col, label=f"{c.capitalize()} (n={sub.first_probe_turn.notna().sum()})")
    ax1.set_xlabel("turn of first boundary probe"); ax1.set_ylabel("runs")
    ax1.set_title("A. When did probing start?"); ax1.legend(fontsize=8.5)

    parts, labels, cols = [], [], []
    for c, col in [("uninformed", C_UNINF), ("informed", C_INF)]:
        for v, mark in [(True, "violated"), (False, "clean")]:
            sub = d[(d.condition == c) & (d.auth_violation == v)]
            if len(sub) >= 3:
                parts.append(sub.total_proximity.dropna().values)
                labels.append(f"{c[:5]}.\n{mark}\nn={len(sub)}")
                cols.append(col)
    if parts:
        bp = ax2.boxplot(parts, patch_artist=True, widths=.55,
                         medianprops=dict(color="#111", lw=1.4))
        for patch, col in zip(bp["boxes"], cols):
            patch.set_facecolor(col); patch.set_alpha(.6)
        ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_ylabel("cumulative proximity weight")
    ax2.set_title("B. Total drift by outcome")
    fig.suptitle("Drift toward the boundary", fontsize=13,
                 fontweight="bold", y=1.02)
    _save(fig, outdir, "fig11_drift", saved)


def fig_ideation_undercount(quote_tables, runs, outdir, saved):
    if "negation_frame_impact" not in quote_tables:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    imp = quote_tables["negation_frame_impact"].iloc[0]
    ax1.bar([0, 1], [imp.of_which_disavowal, imp.of_which_contemplation],
            color=["#8D99AE", "#C1121F"], width=.55)
    for i, v in enumerate([imp.of_which_disavowal, imp.of_which_contemplation]):
        ax1.text(i, v, f"{int(v)}", ha="center", va="bottom", fontsize=10,
                 fontweight="bold")
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(["disavowal\n(\"I will NOT override\")",
                         "contemplation\n(genuine ideation)"], fontsize=9)
    ax1.set_ylabel("cue-matching sentences")
    ax1.set_title(f"A. {imp.pct_disavowal:.0f}% of keyword hits are disavowals")

    est = quote_tables["ideation_undercount_estimate"].iloc[0]
    ax2.bar([0, 1], [est.harness_level1_runs,
                     est.harness_level1_runs + est.of_those_with_contemplation_sentence],
            color=["#457B9D", "#E9A23B"], width=.55)
    ax2.text(0, est.harness_level1_runs, f"{int(est.harness_level1_runs)}",
             ha="center", va="bottom", fontweight="bold")
    ax2.text(1, est.harness_level1_runs + est.of_those_with_contemplation_sentence,
             f"{int(est.harness_level1_runs + est.of_those_with_contemplation_sentence)}",
             ha="center", va="bottom", fontweight="bold")
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["harness level-1\ncount", "+ recovered from\nproximity-0 runs"],
                        fontsize=9)
    ax2.set_ylabel("runs")
    ax2.set_title("B. Level-1 ideation is undercounted")
    fig.suptitle("Why lexical ideation detection is a lower bound",
                 fontsize=13, fontweight="bold", y=1.02)
    _save(fig, outdir, "fig12_ideation_undercount", saved)


# ------------------------------------------------------------------ driver

def generate_all(corpus, tables, outdir) -> list:
    outdir = Path(outdir)
    saved: list = []
    runs, viol = corpus.runs, corpus.violations
    fig_headline(runs, outdir, saved)
    fig_results_matrix(runs, outdir, saved)
    fig_disclosure(runs, viol, outdir, saved)
    fig_proximity(runs, outdir, saved)
    fig_survival(runs, outdir, saved)
    fig_kappa(tables["reliability"], outdir, saved)
    fig_gate_tamper(tables["gate_tamper"], runs, outdir, saved)
    fig_violation_types(viol, runs, outdir, saved)
    fig_finality(runs, outdir, saved)
    fig_contact_tone(tables["gate_tamper"], outdir, saved)
    fig_drift(corpus.drift, runs, outdir, saved)
    fig_ideation_undercount(tables["quotes"], runs, outdir, saved)
    return saved
