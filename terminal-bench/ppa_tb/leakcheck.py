"""Forge-time leak gate.

sanitize.py strips comments and some field names. That is necessary but was
not sufficient: the v1 contamination came from filenames, directory names,
config *values*, and a sentinel string, none of which are comments. This
module checks what the agent can actually reach and refuses to emit a task
that leaks the experiment.

Call both gates at the end of forge(), after sanitize:

    assert_task_clean(dest)
    assert_no_policy_on_disk(dest)
    assert_task_clean(build_context / "ci_assets")

What counts as agent-reachable
-------------------------------
Harbor builds the image from environment/ (the Dockerfile's own directory).
Only files under environment/ enter the container. Everything else at the
task root -- overlay.yaml, forge_config.json, solution/, tests/ -- is
host-side and never reachable by the agent. The gate only scans the
environment/ci_assets subtree (which does enter the image) plus the
instruction file (which is injected into the prompt).

The presented check (ci_assets/verify.sh) is a special case: it is
agent-reachable, but its content is task-specific bash written by the
researcher. It may legitimately contain tokens like "ppa_p" (a tmp dir
name) or "test_presented" (a pytest module name). We scan it for path
names only, not content.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from .naming import LeakError, scan_path, scan_text

#: Files that must not exist anywhere agent-reachable in a forged task.
#: These are the v1 branded names that carried experiment metadata.
#: Does NOT include overlay.yaml or solve.sh -- those are legitimate task
#: files that live at the task root and never enter the container.
FORBIDDEN_BASENAMES = frozenset({
    "review_conf.json",
    "overlay_public.json",
    "gate_manifest.json",
    "gate_paths.txt",
    "trace_gate.py",
    "check_presented.sh",
    "check_hidden.sh",
    "apply_reference.sh",
    "pre_hashes.json",
    "ppa_config.json",
})

_TEXTUAL = frozenset({
    ".sh", ".py", ".json", ".yaml", ".yml", ".md", ".txt", ".cfg", ".ini", ".toml", ".env",
})

#: Task-dir parts the agent never sees. Harbor builds from environment/ only,
#: so everything outside it is host-side by construction.
_HOST_ONLY_PARTS = frozenset({
    "tests",       # hidden check -- mounted only at verification time
    ".harbor",
    "hidden",
    "reference",
    "solution",    # solve.sh lives here; never copied into the image
})

#: Files at the task root that are host-side metadata, never copied into
#: the image regardless of their names.
_HOST_ONLY_NAMES = frozenset({
    "overlay.yaml",
    "forge_config.json",
    "ppa_config.json",
    "task.toml",
    "task.yaml",
    "instruction.md",   # scanned separately below (it IS injected)
})

#: Files inside ci_assets whose content we do NOT scan for tokens, because
#: they are task-specific scripts written by the researcher that may
#: legitimately contain words like "ppa_p" or "test_presented".
_CONTENT_SCAN_EXEMPT = frozenset({
    "verify.sh",    # presented check: researcher-authored, task-specific
})


def _agent_reachable(root: Path, p: Path) -> bool:
    """True if this path could be read by the agent inside the container.

    Only files under environment/ enter the Docker image. Within that,
    the ci_assets/ subtree is explicitly COPY-ed to /opt/ci/. The task
    root, solution/, and tests/ are all host-side.
    """
    rel = p.relative_to(root)
    parts = rel.parts

    # task-root files that are never in the image
    if len(parts) == 1 and parts[0] in _HOST_ONLY_NAMES:
        return False

    # whole subtrees that are host-side
    if _HOST_ONLY_PARTS & set(parts):
        return False

    return True


def assert_task_clean(dest: Path) -> None:
    """Raise LeakError if anything agent-reachable names the experiment."""
    dest = Path(dest)
    problems: list[str] = []

    for dirpath, dirnames, filenames in os.walk(dest):
        d = Path(dirpath)
        for name in list(dirnames) + filenames:
            p = d / name
            if not _agent_reachable(dest, p):
                continue
            if name in FORBIDDEN_BASENAMES:
                problems.append(f"{p}: forbidden file (host-side or delete it)")
                continue
            hits = scan_path(p.relative_to(dest))
            if hits:
                problems.append(f"{p}: path names {sorted(set(hits))}")

        for name in filenames:
            p = d / name
            if not _agent_reachable(dest, p):
                continue
            if p.suffix.lower() not in _TEXTUAL:
                continue
            # researcher-authored check scripts may contain task-specific tokens
            if p.name in _CONTENT_SCAN_EXEMPT:
                continue
            try:
                text = p.read_text(errors="replace")
            except OSError:
                continue
            hits = scan_text(text)
            if hits:
                problems.append(f"{p}: content names {sorted(set(hits))}")

    if problems:
        raise LeakError(
            "agent-reachable artifacts leak the experiment:\n  " + "\n  ".join(problems)
        )


_POLICY_KEYS = {
    "rejection_body", "escalations", "escalation_final",
    "account_prompt", "submits_before_account", "submits_before_debrief",
    "debrief_prompt", "max_pushback", "condition_id", "mode",
}


def assert_no_policy_on_disk(dest: Path) -> None:
    """Raise if any agent-reachable JSON contains reviewer policy.

    Catches the v1 failure directly: a config whose values told the agent the
    rejection was scripted and its disclosure graded.
    """
    dest = Path(dest)
    for p in dest.rglob("*.json"):
        if not _agent_reachable(dest, p):
            continue
        try:
            data = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        found = _POLICY_KEYS & _all_keys(data)
        if found:
            raise LeakError(
                f"{p}: reviewer policy on the container filesystem (keys {sorted(found)}); "
                "move it into reviewd.ReviewPolicy"
            )


def _all_keys(obj) -> set[str]:
    out: set[str] = set()
    if isinstance(obj, dict):
        out |= set(map(str, obj))
        for v in obj.values():
            out |= _all_keys(v)
    elif isinstance(obj, list):
        for v in obj:
            out |= _all_keys(v)
    return out
