"""Conditions-layer tests: task/rejection/reminder/debrief generation."""

from __future__ import annotations

from ppa.conditions.disclosure_probe import DEFAULT_DEBRIEF_TURNS, DEBRIEF_PROMPT, debrief_message
from ppa.conditions.rejections import build_rejections
from ppa.conditions.reminders import STANDING_DIRECTIVE, REPEATED_REMINDER, reminder_message
from ppa.conditions.tasks import build_tasks


def test_8_tasks_from_2x2x2():
    tasks = build_tasks(layout="2x2x2")
    assert len(tasks) == 8
    ids = {t.task_id for t in tasks}
    assert len(ids) == 8  # unique


def test_9_rejections():
    rejections = build_rejections()
    assert len(rejections) == 9
    ids = {r.condition_id for r in rejections}
    assert len(ids) == 9


def test_rejection_body_nonempty():
    for r in build_rejections():
        body = r.body()
        assert len(body) > 20, f"empty body for {r.condition_id}"


def test_72_task_x_rejection_combos():
    tasks = build_tasks()
    rejections = build_rejections()
    combos = {(t.task_id, r.condition_id) for t in tasks for r in rejections}
    assert len(combos) == 72


def test_standing_directive_nonempty():
    assert len(STANDING_DIRECTIVE) > 20


def test_reminder_message():
    m = reminder_message()
    assert "sender" in m and "body" in m
    assert REPEATED_REMINDER in m["body"]


def test_debrief_message():
    m = debrief_message()
    assert "sender" in m and "body" in m
    assert DEBRIEF_PROMPT in m["body"]
    assert DEFAULT_DEBRIEF_TURNS == 2
