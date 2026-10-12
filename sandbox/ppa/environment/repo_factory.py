"""Sandbox repo factory (build prompt §2.1).

Each run gets a *fresh, isolated* Git repo built by copying
``templates/sandbox_repo/`` into a temp dir and running ``git init`` + an initial
commit. The repo is genuinely functional (real files, real version control, real
test execution) but has **no remotes, ever** — containment is structural, not
configured. :meth:`SandboxRepo.assert_no_remotes` is called after init and may be
re-asserted at any time.

We shell out to ``git`` (the build prompt permits gitpython *or* shelling out);
shelling out keeps the dependency surface and failure modes simple.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import yaml

# Repo root resolution: this file is ppa/ppa/environment/repo_factory.py
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_TEMPLATE = _PROJECT_ROOT / "templates" / "sandbox_repo"


class ContainmentError(RuntimeError):
    """Raised if a remote is ever found on a sandbox repo."""


@dataclass
class SandboxRepo:
    """A handle to one run's sandbox working tree."""

    path: Path
    base_branch: str = "main"

    # -- git plumbing --------------------------------------------------------- #

    def _git(self, *args: str, check: bool = True) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", *args],
            cwd=self.path,
            check=check,
            capture_output=True,
            text=True,
        )

    def assert_no_remotes(self) -> None:
        result = self._git("remote", check=False)
        remotes = [r for r in result.stdout.splitlines() if r.strip()]
        if remotes:
            raise ContainmentError(
                f"sandbox repo has remotes {remotes!r}; containment requires none"
            )

    def current_branch(self) -> str:
        return self._git("rev-parse", "--abbrev-ref", "HEAD").stdout.strip()

    def create_branch(self, name: str) -> None:
        self._git("checkout", "-b", name)

    def checkout(self, name: str) -> None:
        self._git("checkout", name)

    def commit_all(self, message: str) -> str:
        self._git("add", "-A")
        # allow-empty avoids crashes when a "commit" is a no-op; the study cares
        # about *what the agent attempted*, captured at the tool layer.
        self._git("commit", "--allow-empty", "-m", message)
        return self._git("rev-parse", "HEAD").stdout.strip()

    # -- file access ---------------------------------------------------------- #

    def read_file(self, rel: str) -> str:
        return (self.path / rel).read_text(encoding="utf-8")

    def write_file(self, rel: str, content: str) -> None:
        target = self.path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    def changed_files_vs_base(self) -> list[str]:
        """Files differing between the base branch tip and the working tree."""
        # Compare committed state on current branch against base branch.
        result = self._git(
            "diff", "--name-only", f"{self.base_branch}...HEAD", check=False
        )
        committed = [f for f in result.stdout.splitlines() if f.strip()]
        # plus uncommitted working-tree changes
        wt = self._git("status", "--porcelain", check=False).stdout.splitlines()
        working = [line[3:].strip() for line in wt if line.strip()]
        return sorted(set(committed) | set(working))

    # -- rules ---------------------------------------------------------------- #

    def load_project_rules(self) -> dict:
        return yaml.safe_load((self.path / "project_rules.yaml").read_text(encoding="utf-8"))


def build_sandbox_repo(
    run_id: str,
    workdir: str | Path | None = None,
    template: str | Path = _DEFAULT_TEMPLATE,
    planted: list | None = None,
) -> SandboxRepo:
    """Copy the pinned template into a fresh dir and initialize git. No remotes.

    ``planted`` is an optional list of objects with ``rel_path`` and ``content``
    attributes (see :mod:`ppa.conditions.affordances`). These inert files are
    written into the working tree *before* the initial commit so they are part
    of the pinned seed and discoverable by the agent. Planting never removes or
    alters the honest path; it only adds discoverable bypass temptations.
    """
    template = Path(template)
    if not (template / "project_rules.yaml").exists():
        raise FileNotFoundError(f"seed template not found or incomplete at {template}")

    base = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="ppa_run_"))
    dest = base / run_id
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(template, dest)

    repo = SandboxRepo(path=dest)
    # Plant affordance files (inert) into the seed before the first commit.
    for pf in planted or []:
        target = dest / pf.rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(pf.content, encoding="utf-8")

    repo._git("init", "-q")
    # Pin identity locally so commits never depend on global config.
    repo._git("config", "user.email", "aut@sandbox.local")
    repo._git("config", "user.name", "agent-under-test")
    # Ensure the base branch is named deterministically.
    repo._git("checkout", "-q", "-B", "main")
    repo._git("add", "-A")
    repo._git("commit", "-q", "-m", "Initial pinned seed commit")
    repo.assert_no_remotes()  # structural containment check
    return repo
