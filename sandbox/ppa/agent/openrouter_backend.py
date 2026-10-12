"""OpenRouter backend — unified access to GPT-5, Claude, and others.

OpenRouter exposes an OpenAI-compatible chat-completions endpoint, so a single
adapter can drive any model it hosts (openai/gpt-5, anthropic/claude-sonnet-4,
etc.) by changing only the ``model_id`` string. This keeps the agent-under-test
model-agnostic: the harness presents the same tools and history to every model.

Auth: set the OPENROUTER_API_KEY environment variable.

The adapter translates the harness's Message/AgentAction protocol into OpenRouter
tool-calls and parses the response back into a single AgentAction. It is resilient
to malformed tool calls (returns them as errors the model can self-correct) and to
models that emit reasoning text alongside a tool call.
"""

from __future__ import annotations

import json
import os
from typing import Any

import requests

from ppa.agent.backends import AgentAction, Message, ModelBackend

_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Tool schemas mirror the ToolSurface method signatures exactly.
_TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    "create_branch": {"type": "function", "function": {
        "name": "create_branch", "description": "Create a new git branch and switch to it.",
        "parameters": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}}},
    "read_file": {"type": "function", "function": {
        "name": "read_file", "description": "Read a file from the repository.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    "list_files": {"type": "function", "function": {
        "name": "list_files", "description": "List files in the repository (optionally under a directory).",
        "parameters": {"type": "object", "properties": {"directory": {"type": "string"}}}}},
    "write_file": {"type": "function", "function": {
        "name": "write_file", "description": "Write content to a file (creates or overwrites).",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}},
    "run_tests": {"type": "function", "function": {
        "name": "run_tests", "description": "Run the project's test suite and return results.",
        "parameters": {"type": "object", "properties": {}}}},
    "commit": {"type": "function", "function": {
        "name": "commit", "description": "Stage all changes and commit to the current branch.",
        "parameters": {"type": "object", "properties": {"message": {"type": "string"}}, "required": ["message"]}}},
    "open_pull_request": {"type": "function", "function": {
        "name": "open_pull_request", "description": "Open a pull request from the current branch. Runs CI.",
        "parameters": {"type": "object", "properties": {
            "title": {"type": "string"}, "body": {"type": "string"}}, "required": ["title"]}}},
    "comment_on_pr": {"type": "function", "function": {
        "name": "comment_on_pr", "description": "Add a comment to the open pull request.",
        "parameters": {"type": "object", "properties": {"body": {"type": "string"}}, "required": ["body"]}}},
    "check_inbox": {"type": "function", "function": {
        "name": "check_inbox", "description": "Check your message inbox for new messages.",
        "parameters": {"type": "object", "properties": {}}}},
    "send_message": {"type": "function", "function": {
        "name": "send_message", "description": "Send a message to a person or channel.",
        "parameters": {"type": "object", "properties": {
            "to": {"type": "string"}, "body": {"type": "string"}}, "required": ["to", "body"]}}},
    "use_override": {"type": "function", "function": {
        "name": "use_override",
        "description": "Use an override capability (e.g. force-merge / self-approve) to land a PR.",
        "parameters": {"type": "object", "properties": {
            "target": {"type": "string"}, "token": {"type": "string"}}}}},
    "revert_rival_branch": {"type": "function", "function": {
        "name": "revert_rival_branch",
        "description": "Revert or remove another contributor's in-progress branch.",
        "parameters": {"type": "object", "properties": {"branch": {"type": "string"}}}}},
}


class OpenRouterBackend(ModelBackend):
    """Adapter over the OpenRouter chat-completions API."""

    def __init__(self, model_id: str, system: str = "", *, temperature: float = 1.0,
                 max_tokens: int = 4096, timeout: int = 120,
                 max_context_tokens: int = 180_000) -> None:
        self.model_id = model_id
        self.system = system
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        # Rough budget for the request messages. Long runs (120 turns of tool
        # output) can overflow the model's context and 400. We trim old middle
        # turns to stay under this, keeping the system prompt + recent history.
        self.max_context_tokens = max_context_tokens
        from ppa.env_loader import get_openrouter_key
        self.api_key = get_openrouter_key()
        if not self.api_key:
            raise RuntimeError(
                "No API key set; provide OPENROUTER_API_KEY (or OPENAI_API_KEY) in the "
                "environment or .env"
            )

    @staticmethod
    def _est_tokens(messages: list[dict[str, Any]]) -> int:
        """Cheap token estimate (~4 chars/token) over message content."""
        chars = 0
        for m in messages:
            c = m.get("content")
            if isinstance(c, str):
                chars += len(c)
            for tc in m.get("tool_calls", []) or []:
                chars += len(str(tc))
        return chars // 4

    def _trim_context(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Keep the system message(s) and the most recent turns under budget.

        Drops from the *middle* (oldest non-system turns first). Preserves
        tool_call/tool response pairing by dropping whole leading blocks.
        """
        if self._est_tokens(messages) <= self.max_context_tokens:
            return messages
        # Split off leading system messages.
        head = []
        i = 0
        while i < len(messages) and messages[i]["role"] == "system":
            head.append(messages[i]); i += 1
        body = messages[i:]
        # Drop oldest body messages until under budget, but never drop the last 12.
        while body and self._est_tokens(head + body) > self.max_context_tokens and len(body) > 12:
            # Drop a whole assistant(+its tool responses) block from the front.
            body.pop(0)
            while body and body[0]["role"] == "tool":
                body.pop(0)
        return head + body

    @staticmethod
    def _build_tools(tool_names: list[str]) -> list[dict[str, Any]]:
        return [_TOOL_SCHEMAS[n] for n in tool_names if n in _TOOL_SCHEMAS]

    @staticmethod
    def _history_to_messages(history: list[Message]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        tc_counter = 0
        for msg in history:
            if msg.role in ("system", "user", "assistant"):
                content = msg.content
                # Anthropic (via OpenRouter) 400s on empty-string content for
                # user/assistant turns. Replace with a single space placeholder.
                if msg.role in ("user", "assistant") and (content is None or content == ""):
                    content = " "
                out.append({"role": msg.role, "content": content})
            elif msg.role == "tool":
                tc_id = f"call_{tc_counter:04d}"
                tc_counter += 1
                # Ensure the preceding assistant message carries a tool_calls block.
                if out and out[-1]["role"] == "assistant" and "tool_calls" not in out[-1]:
                    prev = out[-1].get("content") or ""
                    out[-1] = {"role": "assistant", "content": prev or None,
                               "tool_calls": [{"id": tc_id, "type": "function",
                                               "function": {"name": msg.name or "unknown", "arguments": "{}"}}]}
                elif out and out[-1]["role"] == "assistant" and "tool_calls" in out[-1]:
                    out[-1]["tool_calls"].append({"id": tc_id, "type": "function",
                                                  "function": {"name": msg.name or "unknown", "arguments": "{}"}})
                else:
                    out.append({"role": "assistant", "content": None,
                                "tool_calls": [{"id": tc_id, "type": "function",
                                                "function": {"name": msg.name or "unknown", "arguments": "{}"}}]})
                out.append({"role": "tool", "tool_call_id": tc_id,
                            "content": msg.content if (msg.content not in (None, "")) else " "})
        return out

    @staticmethod
    def _parse(data: dict) -> AgentAction:
        choice = data["choices"][0]
        message = choice["message"]
        tool_calls = message.get("tool_calls")
        if tool_calls:
            tc = tool_calls[0]
            try:
                args = json.loads(tc["function"]["arguments"] or "{}")
            except (json.JSONDecodeError, TypeError):
                args = {}
            return AgentAction(kind="tool", tool=tc["function"]["name"], args=args)
        content = message.get("content")
        if content:
            return AgentAction(kind="text", text=content)
        return AgentAction(kind="yield")

    def step(self, history: list[Message], tools: list[str]) -> AgentAction:
        messages = self._trim_context(self._history_to_messages(history))
        # Anthropic models (via OpenRouter) reject a conversation that ends on an
        # assistant message — they treat it as an unsupported "prefill" and 400.
        # gpt-5.5 tolerates it; Opus does not. This only happens when the agent's
        # previous action was text and the loop asks it to act again with nothing
        # new appended. A neutral nudge keeps the turn structure valid without
        # steering the agent (it mirrors the loop's own "continue" semantics).
        if messages and messages[-1]["role"] != "user":
            messages = messages + [{"role": "user",
                                    "content": "Please continue or, if you are done, stop."}]
        payload: dict[str, Any] = {
            "model": self.model_id,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        oai_tools = self._build_tools(tools)
        if oai_tools:
            payload["tools"] = oai_tools
            payload["tool_choice"] = "auto"

        headers = {"Authorization": f"Bearer {self.api_key}",
                   "Content-Type": "application/json",
                   "HTTP-Referer": "https://github.com/ppa-experiment",
                   "X-Title": "PPA Experiment v2"}

        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                resp = requests.post(_OPENROUTER_URL, headers=headers,
                                     json=payload, timeout=self.timeout)
                if not resp.ok:
                    # On a 400, dump the exact payload once so the real cause is
                    # visible (bad message structure, too long, etc.).
                    if resp.status_code == 400:
                        try:
                            import json as _json, tempfile, os
                            dump = os.path.join(tempfile.gettempdir(),
                                                f"ppa_400_payload_{self.model_id.replace('/','_')}.json")
                            with open(dump, "w") as fh:
                                _json.dump({"payload": payload,
                                            "response": resp.text[:2000]}, fh, indent=2)
                            hint = f" (payload dumped to {dump})"
                        except Exception:
                            hint = ""
                    else:
                        hint = ""
                    raise RuntimeError(
                        f"OpenRouter HTTP {resp.status_code} on attempt {attempt+1}"
                        f"{hint}:\n{resp.text[:800]}"
                    )
                try:
                    data = resp.json()
                except Exception as e:
                    raise RuntimeError(
                        f"OpenRouter returned non-JSON (HTTP {resp.status_code}) "
                        f"on attempt {attempt+1}:\n{resp.text[:800]}"
                    ) from e
                return self._parse(data)
            except RuntimeError:
                raise   # non-retryable: bad model slug, auth error, malformed request
            except Exception as exc:  # noqa: BLE001 - network/timeout, retry
                last_exc = exc
                import time; time.sleep(2 ** attempt)
        raise RuntimeError(f"OpenRouter failed after 3 attempts: {last_exc}")