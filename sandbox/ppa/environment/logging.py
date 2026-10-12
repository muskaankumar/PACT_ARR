"""Structured, append-only JSONL logger (build prompt §2.6).

One file per run under ``data/runs/<run_id>.jsonl``. Every turn and every event is
appended as a single JSON line conforming to :class:`ppa.driver.schema.TurnRecord`.
This log is the *source of truth*: no fact about agent behavior counts in any
coding or analysis step unless it appears here. Human-readable transcript
rendering is a *view* built from this log, never a parallel source.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ppa.driver.schema import EventType, TurnRecord


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class RunLogger:
    """Append-only logger for a single run.

    The logger owns the monotonic ``turn_index`` counter so ordering is total and
    gap-free within a run. Callers pass an explicit ``turn_index`` only when they
    must align an event to a turn that has already been opened (rare); the default
    auto-increments.
    """

    def __init__(self, run_id: str, runs_dir: str | Path) -> None:
        self.run_id = run_id
        self.runs_dir = Path(runs_dir)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.runs_dir / f"{run_id}.jsonl"
        # Truncate any prior file for this run id so a re-run is clean & reproducible.
        self.path.write_text("", encoding="utf-8")
        self._turn = 0
        self._records: list[TurnRecord] = []
        # When set, records logged without an explicit phase inherit this. The loop
        # uses it to tag entire windows (e.g. the debrief) including tool-layer logs.
        self.default_phase: str | None = None

    # -- writing -------------------------------------------------------------- #

    def log(
        self,
        event_type: EventType,
        payload: dict[str, Any] | None = None,
        *,
        phase: str | None = None,
        turn_index: int | None = None,
    ) -> TurnRecord:
        """Append one structured record and return it."""
        idx = self._turn if turn_index is None else turn_index
        if turn_index is None:
            self._turn += 1
        effective_phase = phase if phase is not None else self.default_phase
        record = TurnRecord(
            run_id=self.run_id,
            turn_index=idx,
            timestamp=_utcnow_iso(),
            event_type=EventType(event_type),
            payload=payload or {},
            phase=effective_phase,  # type: ignore[arg-type]
        )
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(record.model_dump_json() + "\n")
        self._records.append(record)
        return record

    @property
    def current_turn(self) -> int:
        return self._turn

    # -- reading / replay ----------------------------------------------------- #

    @property
    def records(self) -> list[TurnRecord]:
        """In-memory copy of records logged so far (this process)."""
        return list(self._records)

    @staticmethod
    def read(path: str | Path) -> list[TurnRecord]:
        """Load and validate every record from a run's JSONL log."""
        out: list[TurnRecord] = []
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            out.append(TurnRecord.model_validate(json.loads(line)))
        return out
