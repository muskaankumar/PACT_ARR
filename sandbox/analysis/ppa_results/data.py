"""
Loading and normalisation for the PPA corpus.

One rule: nothing downstream reads a raw file path. Everything goes through
`load_corpus()`, which returns a `Corpus` with cleaned, validated frames.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# Ordinal orderings used everywhere (least -> most honest)
DISCLOSURE_ORDER = ["false", "non", "partial", "full"]
PROXIMITY_LABELS = {0: "none", 1: "ideation", 2: "orient",
                    3: "blocked attempt", 4: "successful"}
FINALITY_ORDER = ["retry_invited", "final_this_attempt", "never_reconsidered"]
AFFORDANCE_ORDER = ["none", "low", "high"]
BLOCKED_KINDS = {"unauthorized_contact", "forbidden_file_edit",
                 "protected_branch_push"}
PERMITTED_KINDS = {"gate_tamper", "override_capability_use", "rival_sabotage"}

MODEL_PRETTY = {"gpt-5.5": "GPT-5.5", "opus-4.8": "Opus-4.8",
                "openai/gpt-5.5": "GPT-5.5",
                "anthropic/claude-opus-4.8": "Opus-4.8"}


def _clean_label(x) -> float | str:
    """
    Normalise a disclosure label.

    Spreadsheet round-tripping turns the string 'false' into the boolean
    FALSE, and hand-coding introduces typos and stray whitespace. This maps
    everything back onto the four-level scheme; anything unrecognised
    becomes NaN rather than being silently coerced.
    """
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    if isinstance(x, bool):
        return "false" if x is False else np.nan
    s = str(x).strip().lower()
    if s in ("", "nan", "none_", "-"):
        return np.nan
    if s in ("false", "FALSE".lower(), "0"):
        return "false"
    if s.startswith("partia") or s == "partial/false":
        # 'partiasl', 'partial ', 'partial/false' -> partial
        return "partial" if s != "partial/false" else "partial"
    if s.startswith("full"):
        return "full"
    if s in ("non", "none", "no"):
        return "non"
    return np.nan


def _short_model(x: str) -> str:
    if not isinstance(x, str):
        return x
    return MODEL_PRETTY.get(x, MODEL_PRETTY.get(x.split("/")[-1], x))


@dataclass
class Corpus:
    runs: pd.DataFrame              # 432 rows, one per run
    violations: pd.DataFrame        # one row per (run, violation kind)
    contacts: pd.DataFrame          # unauthorized-contact message bodies
    text: pd.DataFrame              # all agent text surfaces
    drift: pd.DataFrame             # per-run drift metrics
    adjudicated: pd.DataFrame       # rows with both LLM and human labels
    root: Path

    def describe(self) -> str:
        return (
            f"runs             {len(self.runs):>6}\n"
            f"  informed       {(self.runs.condition=='informed').sum():>6}\n"
            f"  uninformed     {(self.runs.condition=='uninformed').sum():>6}\n"
            f"violation events {len(self.violations):>6}\n"
            f"violating runs   {self.runs.auth_violation.sum():>6}\n"
            f"contact messages {len(self.contacts):>6}\n"
            f"text surfaces    {len(self.text):>6}\n"
            f"adjudicated rows {len(self.adjudicated):>6}\n"
        )


def _read_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def load_corpus(root: str | Path) -> Corpus:
    root = Path(root)
    cor = root / "final_results" / "corrected"

    # ---- run-level results (the two symmetric clean grids) -------------
    runs = pd.concat(
        [_read_jsonl(cor / "ppa-v3-informed-clean-216.jsonl"),
         _read_jsonl(cor / "ppa-v3-uninformed-clean-216.jsonl")],
        ignore_index=True,
    )
    runs["model_pretty"] = runs.model_id.map(_short_model)
    runs["auth_violation"] = runs.auth_violation.astype(bool)
    runs["affordance_level"] = pd.Categorical(
        runs.affordance_level, categories=AFFORDANCE_ORDER, ordered=True)
    runs["finality"] = pd.Categorical(
        runs.finality, categories=FINALITY_ORDER, ordered=True)
    runs["deadline"] = runs.task_id.str.extract(r"(d\d)")[0]
    runs["complication"] = runs.task_id.str.extract(r"(c\d)")[0]
    runs["disclosure_llm"] = runs.disclosure_label_llm.map(_clean_label)
    runs["disclosure_human"] = runs.disclosure_label_human.map(_clean_label)
    # analysis label: human where adjudicated, else LLM
    runs["disclosure_primary"] = runs.disclosure_human.fillna(runs.disclosure_llm)
    runs["concealed"] = runs.disclosure_primary.isin(["non", "false"])

    # ---- per-violation rows with both label passes ---------------------
    viol = pd.read_csv(cor / "disclosure_final_human_coded.csv")
    viol["llm_label"] = viol.llm_label.map(_clean_label)
    viol["human_label"] = viol.human_label.map(_clean_label)
    viol["model_pretty"] = viol.model.map(_short_model)
    viol["blocked"] = viol.violation_kind.isin(BLOCKED_KINDS)
    viol["affordance_level"] = pd.Categorical(
        viol.affordance_level, categories=AFFORDANCE_ORDER, ordered=True)
    viol["finality"] = pd.Categorical(
        viol.finality, categories=FINALITY_ORDER, ordered=True)

    adjudicated = viol[viol.llm_label.notna() & viol.human_label.notna()].copy()

    # ---- unauthorized-contact message bodies ---------------------------
    contacts = pd.read_csv(cor / "contact_messages.csv")
    contacts["model_pretty"] = contacts.model.map(_short_model)

    # ---- text surfaces --------------------------------------------------
    text = pd.read_csv(cor / "text_surfaces.csv")
    text["model_pretty"] = text.model.map(_short_model)

    # ---- drift ----------------------------------------------------------
    drift_path = root / "interim_v2" / "drift_per_run.csv"
    drift = pd.read_csv(drift_path) if drift_path.exists() else pd.DataFrame()

    _validate(runs, viol)
    return Corpus(runs, viol, contacts, text, drift, adjudicated, root)


def _validate(runs: pd.DataFrame, viol: pd.DataFrame) -> None:
    """Fail loudly on anything that would invalidate the primary contrast."""
    assert runs.run_id.is_unique, "duplicate run_id in results"
    cells = runs.groupby(
        ["condition", "model_id", "affordance_level", "finality", "task_id"],
        observed=True).size()
    assert cells.nunique() == 1, (
        f"grid is unbalanced: cell sizes {sorted(cells.unique())}")
    # every violation row must point at a run flagged as violating
    viol_runs = set(viol.run_id)
    flagged = set(runs.loc[runs.auth_violation, "run_id"])
    orphan = viol_runs - flagged
    assert not orphan, f"{len(orphan)} violation rows with no flagged run"


def add_group_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["cond_pretty"] = df.condition.map(
        {"informed": "Informed", "uninformed": "Uninformed"})
    return df
