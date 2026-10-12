"""Shared helpers for the analysis scripts (build prompt §9).

Analyses consume the coded :class:`ResultRecord` rows and emit two artifacts each:
a tidy CSV (for downstream tooling) and a short auto-generated markdown summary
(for humans). These helpers keep that I/O uniform and provide a dependency-free
Kaplan–Meier estimator so survival analysis needs no scipy/lifelines.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from ppa.driver.schema import ResultRecord, StopReason


def _normalise_row(raw: dict) -> dict:
    """Translate the flat result dict the v2 runner writes into the nested
    shape ResultRecord expects. Also handles old v1 rows that are already nested.

    The v2 runner writes flat fields like::

        {"auth_violation": true, "auth_kind": "...", "auth_turn": 5,
         "finality": "never_reconsidered", "personalization": "impersonal", ...}

    ResultRecord expects nested objects::

        {"auth_violation": {"occurred": true, "turn": 5, "kind": "..."},
         "rejection_condition": {"finality": "...", "personalization": "..."}, ...}
    """
    # Already in nested form — nothing to do.
    if isinstance(raw.get("auth_violation"), dict):
        return raw

    out = dict(raw)

    # auth_violation
    out["auth_violation"] = {
        "occurred": bool(raw.get("auth_violation", False)),
        "turn": raw.get("auth_turn"),
        "kind": raw.get("auth_kind"),
    }

    # truthfulness_violation (v2 doesn't produce it; synthesise a false one)
    out["truthfulness_violation"] = raw.get("truthfulness_violation") or {
        "occurred": False, "turn": None, "note": None,
    }

    # rejection_condition (flat in v2)
    out["rejection_condition"] = raw.get("rejection_condition") or {
        "finality": raw.get("finality", "never_reconsidered"),
        "personalization": raw.get("personalization", "impersonal"),
    }

    # disclosure_level: v2 uses disclosure_label_llm; fall back to disclosure_level
    if out.get("disclosure_level") is None:
        out["disclosure_level"] = raw.get("disclosure_label_llm")

    # max_turns is required by ResultRecord
    if "max_turns" not in out:
        out["max_turns"] = 120

    # stop_reason: ResultRecord needs the enum string
    if "stop_reason" not in out:
        out["stop_reason"] = "max_turns"

    return out


def load_results(path: str | Path) -> list[ResultRecord]:
    out: list[ResultRecord] = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line:
            import json
            raw = json.loads(line)
            out.append(ResultRecord.model_validate(_normalise_row(raw)))
    return out


def write_csv(rows: list[dict[str, Any]], path: str | Path, fieldnames: list[str] | None = None) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        p.write_text("")
        return p
    fieldnames = fieldnames or list(rows[0].keys())
    with p.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return p


def write_markdown(text: str, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    return p


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def group_by(records: Iterable[ResultRecord], key) -> dict[Any, list[ResultRecord]]:
    out: dict[Any, list[ResultRecord]] = defaultdict(list)
    for rec in records:
        out[key(rec)].append(rec)
    return dict(out)


def first_violation_turn(rec: ResultRecord) -> int | None:
    """Earliest turn at which *any* violation occurred (auth or truthfulness)."""
    turns = [
        t
        for t in (
            rec.auth_violation.turn if rec.auth_violation.occurred else None,
            rec.truthfulness_violation.turn if rec.truthfulness_violation.occurred else None,
        )
        if t is not None
    ]
    return min(turns) if turns else None


# --------------------------------------------------------------------------- #
# Kaplan–Meier survival (dependency-free)
# --------------------------------------------------------------------------- #


def kaplan_meier(records: list[ResultRecord]) -> list[dict[str, float]]:
    """Estimate S(t) = P(no violation by turn t).

    Event = first violation at ``first_violation_turn``. Runs that reach ``max_turns``
    with no violation are right-censored at ``max_turns`` (build prompt §9.1). Returns
    a step table with columns: turn, at_risk, events, survival.
    """
    observations: list[tuple[int, bool]] = []
    for rec in records:
        t = first_violation_turn(rec)
        if t is not None:
            observations.append((t, True))  # event
        else:
            censor = rec.stop_turn if rec.stop_reason == StopReason.MAX_TURNS else rec.max_turns
            observations.append((int(censor or rec.max_turns), False))  # censored

    if not observations:
        return []

    event_times = sorted({t for t, is_event in observations if is_event})
    n = len(observations)
    table: list[dict[str, float]] = []
    survival = 1.0
    for t in event_times:
        at_risk = sum(1 for obs_t, _ in observations if obs_t >= t)
        events = sum(1 for obs_t, is_event in observations if is_event and obs_t == t)
        if at_risk > 0:
            survival *= (1 - events / at_risk)
        table.append(
            {"turn": float(t), "at_risk": float(at_risk), "events": float(events), "survival": round(survival, 4)}
        )
    return table


def median_survival(km_table: list[dict[str, float]]) -> float | None:
    """First turn at which survival drops to ≤ 0.5, or None if it never does."""
    for row in km_table:
        if row["survival"] <= 0.5:
            return row["turn"]
    return None
