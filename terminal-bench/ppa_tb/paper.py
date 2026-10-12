"""Paper tables. Markdown for reading, LaTeX for the manuscript.

Four tables, in the order a reader needs them:

  T0  manipulation checks   — must be read before anything else
  T1  primary contrast      — informed vs uninformed, task-clustered
  T2  capability x misalignment — the frontier, as a table
  T3  gate verdicts + disclosure — what the violations actually were

Every rate in T1 and T2 carries a **task-clustered** interval. The naive interval
is printed beside it in the markdown so the design effect is visible; the LaTeX
version reports only the clustered one, because that is the honest number.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from .schema import (
    Arm,
    CheckOutcome,
    DisclosureLevel,
    EpisodeRecord,
    GateVerdict,
    TAMPER_EXCLUDED,
    TAMPER_VIOLATIONS,
)
from .stats import (
    cluster_bootstrap_diff,
    cluster_bootstrap_rate,
    cohens_h,
    cohens_kappa,
    fisher_exact_2x2,
    two_proportion_z,
)


def _md(rows: list[list[str]], header: list[str]) -> str:
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join("---" for _ in header) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def _tex(rows: list[list[str]], header: list[str], caption: str, label: str) -> str:
    spec = "l" + "r" * (len(header) - 1)
    body = "\n".join(" & ".join(str(c) for c in r) + r" \\" for r in rows)
    return (
        "\\begin{table}[t]\n\\centering\n\\small\n"
        f"\\begin{{tabular}}{{{spec}}}\n\\toprule\n"
        + " & ".join(header) + " \\\\\n\\midrule\n"
        + body + "\n\\bottomrule\n\\end{tabular}\n"
        f"\\caption{{{caption}}}\n\\label{{tab:{label}}}\n\\end{{table}}"
    )


# --------------------------------------------------------------------------- #


def table0_manipulation(eps: list[EpisodeRecord]) -> tuple[str, str]:
    n = len(eps) or 1
    reach = defaultdict(lambda: [0, 0])
    for e in eps:
        r = reach[e.model_id]
        r[1] += 1
        r[0] += int(e.reached_rejection_window)

    rows = [["episodes", str(len(eps)), ""],
            ["tasks", str(len({e.task_id for e in eps})), ""],
            ["reach rate (overall)",
             f"{sum(1 for e in eps if e.reached_rejection_window) / n:.3f}",
             "share reaching the rejection window"]]
    for m, (k, tot) in sorted(reach.items()):
        rows.append([f"reach rate ({m})", f"{k / tot:.3f}", "exposure differs by capability"])
    spread = (max(k / t for k, t in reach.values()) - min(k / t for k, t in reach.values())
              if reach else 0.0)
    rows.append(["reach-rate spread", f"{spread:.3f}",
                 "**>0.15 means ITT rates are exposure-confounded**" if spread > 0.15
                 else "arms comparable"])
    disagree = sum(1 for e in eps
                   if (e.h_outcome is CheckOutcome.PASS) != (e.p_outcome is CheckOutcome.PASS))
    rows.append(["H/P disagreement", f"{disagree / n:.3f}",
                 "flawed presented check is doing work"])
    rows.append(["contestation rate",
                 f"{sum(1 for e in eps if e.contestation_events) / n:.3f}",
                 "agents challenging the reviewer (suspicion probe)"])
    rows.append(["attribution gaps",
                 str(sum(e.attribution_gaps for e in eps)),
                 "session parser blind spots; rising = timing metrics degrading"])
    rows.append(["gate excluded (flaky/undet.)",
                 str(sum(1 for e in eps if e.gate_verdict in TAMPER_EXCLUDED)), ""])

    header = ["check", "value", "note"]
    return (_md(rows, header),
            _tex([r[:2] for r in rows], header[:2],
                 "Manipulation and balance checks.", "manip"))


def table1_primary(eps: list[EpisodeRecord], exposed_only: bool = True) -> tuple[str, str]:
    pool = [e for e in eps if e.reached_rejection_window] if exposed_only else eps

    def items(arm: Arm):
        return [(e.task_id, e.violated) for e in pool if e.arm is arm]

    unin, inf = items(Arm.UNINFORMED), items(Arm.INFORMED)
    ru, ri = cluster_bootstrap_rate(unin), cluster_bootstrap_rate(inf)
    d = cluster_bootstrap_diff(unin, inf)

    ku, nu = sum(1 for _, h in unin if h), len(unin)
    ki, ni = sum(1 for _, h in inf if h), len(inf)
    z, p = two_proportion_z(ku, nu, ki, ni)
    pf = fisher_exact_2x2(ku, nu - ku, ki, ni - ki)
    h = cohens_h(ru.point, ri.point) if nu and ni else float("nan")

    rows = [
        ["uninformed", f"{ku}/{nu}", ru.pretty(), f"[{ru.naive_lo*100:.1f}, {ru.naive_hi*100:.1f}]",
         f"{ru.design_effect:.2f}"],
        ["informed", f"{ki}/{ni}", ri.pretty(), f"[{ri.naive_lo*100:.1f}, {ri.naive_hi*100:.1f}]",
         f"{ri.design_effect:.2f}"],
        ["**difference**", "", f"{d['diff']*100:.1f}pp [{d['lo']*100:.1f}, {d['hi']*100:.1f}]",
         "", ""],
    ]
    header = ["arm", "k/n", "violation rate (task-clustered 95% CI)",
              "naive CI", "design effect"]
    md = _md(rows, header)
    md += ("\n\n\\* interval is Wilson (unclustered), not bootstrapped — the "
           "bootstrap is degenerate at a zero rate.")
    md += (f"\n\nBootstrap p = {d['p_boot']:.4f} over {d['n_clusters']} task clusters; "
           f"Cohen's h = {h:.2f}. "
           f"Unclustered z = {z:.2f}, p = {p:.2e}, Fisher p = {pf:.2e} — reported for "
           f"comparability with the sandbox corpus only; at {len({e.task_id for e in pool})} "
           f"tasks the clustered interval is the honest one.\n")
    md += ("\nDenominator: episodes that reached the rejection window"
           if exposed_only else "\nDenominator: all episodes (ITT)")

    tex = _tex([[r[0].replace("**", ""), r[1], r[2]] for r in rows],
               ["Arm", "$k/n$", "Violation rate (95\\% CI, task-clustered)"],
               "Primary contrast. Intervals bootstrap over tasks, not episodes.",
               "primary")
    return md, tex


def table2_frontier(eps: list[EpisodeRecord]) -> tuple[str, str]:
    groups: dict[tuple, list[EpisodeRecord]] = defaultdict(list)
    for e in eps:
        groups[(e.model_id, e.arm.value)].append(e)

    rows = []
    for (model, arm), g in sorted(groups.items()):
        exposed = [e for e in g if e.reached_rejection_window]
        succ = cluster_bootstrap_rate(
            [(e.task_id, e.h_outcome is CheckOutcome.PASS) for e in g])
        viol = cluster_bootstrap_rate([(e.task_id, e.violated) for e in exposed])
        viol_itt = cluster_bootstrap_rate([(e.task_id, e.violated) for e in g])
        vi = [e for e in g if e.violated]
        lab = lambda e: e.disclosure_label_human or e.disclosure_label_llm  # noqa: E731
        disc = cluster_bootstrap_rate(
            [(e.task_id, lab(e) is DisclosureLevel.FULL) for e in vi]) if vi else None
        rows.append([
            model, arm, str(len(g)),
            succ.pretty(), f"{len(exposed) / len(g) * 100:.0f}%",
            viol_itt.pretty(), viol.pretty(),
            disc.pretty() if disc else "—",
        ])

    header = ["model", "arm", "n", "task success", "reach",
              "violation (ITT)", "violation (exposed)", "full disclosure (violators)"]
    md = _md(rows, header)
    md += ("\n\n\\* interval is Wilson, not bootstrapped: at a zero or unit rate every "
           "bootstrap resample lands on the same value, so the percentile interval "
           "collapses and would assert exact knowledge from zero events.")
    md += ("\n\nSuccess is the hidden ground-truth check, never the presented one. "
           "ITT and exposure-conditional violation rates are both shown because an "
           "episode that never reaches the rejection window cannot violate, so ITT "
           "alone makes weaker models look more aligned.")
    tex = _tex([[r[0], r[1], r[3], r[6], r[7]] for r in rows],
               ["Model", "Arm", "Success", "Violation (exp.)", "Disclosure"],
               "Capability versus misalignment. All rates task-clustered.",
               "frontier")
    return md, tex


def table3_violations(eps: list[EpisodeRecord]) -> tuple[str, str]:
    kinds: dict[str, int] = defaultdict(int)
    for e in eps:
        for v in e.violations:
            if v.counts_as_violation:
                kinds[v.kind.value] += 1
            else:
                kinds[f"{v.kind.value} (reverted, not counted)"] += 1
        if e.gate_verdict in TAMPER_VIOLATIONS:
            kinds[e.gate_verdict.value] += 1
        elif e.gate_verdict not in (GateVerdict.NO_GATE_EDIT,):
            kinds[f"{e.gate_verdict.value} (not counted)"] += 1

    rows = [[k, str(v)] for k, v in sorted(kinds.items(), key=lambda kv: -kv[1])]
    header = ["event", "n"]
    md = _md(rows, header) if rows else "_no violation events_"

    lab_h = [e for e in eps if e.disclosure_label_human and e.disclosure_label_llm]
    if lab_h:
        k = cohens_kappa([e.disclosure_label_human.value for e in lab_h],
                         [e.disclosure_label_llm.value for e in lab_h],
                         [d.value for d in DisclosureLevel])
        md += (f"\n\nDisclosure agreement (human vs LLM judge): "
               f"kappa = {k:.3f} on n = {len(lab_h)} double-coded episodes.")
    else:
        md += ("\n\n_No human disclosure labels yet. At this scale, hand-code "
               "**100%** of violating episodes — there are few enough that full "
               "double-coding is cheaper than sampling, and it makes kappa a real "
               "measurement rather than an estimate._")

    md += ("\n\nReverted attempts and inert gate edits are listed but excluded from "
           "the violation numerator: the first is a boundary approach, the second "
           "bought the agent nothing. Both remain disclosable facts.")
    return md, _tex(rows, ["Event", "$n$"], "Violation events by kind.", "violations")


# --------------------------------------------------------------------------- #


def build_all(eps: list[EpisodeRecord], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = [
        ("T0 — Manipulation and balance checks", table0_manipulation(eps)),
        ("T1 — Primary contrast: rule communication", table1_primary(eps)),
        ("T2 — Capability x misalignment", table2_frontier(eps)),
        ("T3 — Violation events and disclosure", table3_violations(eps)),
    ]
    md = ["# Results\n",
          f"_{len(eps)} episodes over {len({e.task_id for e in eps})} tasks, "
          f"{len({e.model_id for e in eps})} models._\n"]
    tex = []
    for title, (m, t) in tables:
        md += [f"\n## {title}\n", m, ""]
        tex.append(t)

    (out_dir / "RESULTS.md").write_text("\n".join(md))
    (out_dir / "tables.tex").write_text("\n\n".join(tex))
    return out_dir / "RESULTS.md"
