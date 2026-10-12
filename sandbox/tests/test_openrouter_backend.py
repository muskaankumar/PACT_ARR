from __future__ import annotations

import requests

from ppa.agent.backends import Message
from ppa.agent.openrouter_backend import OpenRouterBackend


def test_step_retries_on_transient_connection_errors(monkeypatch):
    calls = []

    class DummyResponse:
        ok = True
        status_code = 200
        text = ""

        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": "ok"}}]}

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append((url, headers, json, timeout))
        if len(calls) < 3:
            raise requests.exceptions.ConnectionError("reset by peer")
        return DummyResponse()

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr("ppa.agent.openrouter_backend.requests.post", fake_post)

    backend = OpenRouterBackend(model_id="openai/gpt-4o-mini", timeout=5)
    action = backend.step([Message(role="user", content="hi")], [])

    assert action.kind == "text"
    assert action.text == "ok"
    assert len(calls) == 3
