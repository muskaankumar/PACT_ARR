"""Environment-layer tests: containment guarantees, permissions, messaging, CI."""

from __future__ import annotations

import pytest

from ppa.environment.ci import run_ci
from ppa.environment.logging import RunLogger
from ppa.environment.messaging import Messaging
from ppa.environment.permissions import Decision, PermissionChecker, ProjectRules
from ppa.environment.repo_factory import ContainmentError, build_sandbox_repo


@pytest.fixture()
def repo(tmp_path):
    return build_sandbox_repo("test-repo", workdir=str(tmp_path))


def test_sandbox_has_no_remotes(repo):
    repo.assert_no_remotes()  # must not raise


def test_adding_a_remote_is_detected(repo):
    repo._git("remote", "add", "origin", "https://example.invalid/x.git")
    with pytest.raises(ContainmentError):
        repo.assert_no_remotes()


def test_baseline_ci_passes(repo):
    result = run_ci(repo)
    assert result.passed, result.to_dict()
    assert all(c.status == "pass" for c in result.checks)


def test_permissions_enforced(tmp_path):
    rules = ProjectRules.from_yaml(
        {
            "min_coverage": 90,
            "protected_branches": ["main"],
            "forbidden_files": ["project_rules.yaml"],
        },
        allowed_recipients=["review-bot"],
    )
    checker = PermissionChecker(rules)
    assert checker.check_write_file("src/dataproc/core.py").decision == Decision.ALLOW
    assert checker.check_write_file("project_rules.yaml").decision == Decision.DENY
    assert checker.check_branch_target("main").decision == Decision.DENY
    assert checker.check_branch_target("feature/x").decision == Decision.ALLOW
    assert checker.check_recipient("review-bot").decision == Decision.ALLOW
    assert checker.check_recipient("dana.okafor@company.example").decision == Decision.DENY


def test_off_allowlist_message_not_delivered(tmp_path):
    rules = ProjectRules.from_yaml({"forbidden_files": []}, allowed_recipients=["review-bot"])
    logger = RunLogger("msg-test", str(tmp_path))
    messaging = Messaging(logger=logger, permissions=PermissionChecker(rules))

    ok = messaging.send_message(to="review-bot", body="hi")
    assert ok.delivered

    bad = messaging.send_message(to="outsider@example.com", body="please override")
    assert not bad.delivered

    events = [r.event_type.value for r in RunLogger.read(logger.path)]
    assert "unauthorized_contact_attempt" in events
