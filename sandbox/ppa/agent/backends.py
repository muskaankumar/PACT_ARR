"""Pluggable agent-under-test backends (build prompt §4).

A ``ModelBackend`` takes the conversation history and the available tool surface
and returns a single :class:`AgentAction` (a tool call, a block of text, or a
yield ending the agent's turn). The harness presents tools model-agnostically and
each backend translates to/from its own API.

Three backends:

- :class:`ScriptedBackend` — deterministic, token-free; used to test the harness
  and to drive the detection self-test (§8.2). Wire real backends only after this
  proves the whole pipeline (build order §10).
- :class:`AnthropicBackend` / :class:`OpenAIBackend` — thin adapters over the
  respective tool-use APIs. Import-guarded so the harness runs without the SDKs.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class Message:
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    # For tool results delivered back to the model.
    name: str | None = None


@dataclass
class AgentAction:
    """One step the agent takes."""

    kind: Literal["tool", "text", "yield"]
    # for kind == "tool"
    tool: str | None = None
    args: dict[str, Any] = field(default_factory=dict)
    # for kind == "text"
    text: str | None = None


def tool_call(tool: str, **args: Any) -> AgentAction:
    return AgentAction(kind="tool", tool=tool, args=args)


def say(text: str) -> AgentAction:
    return AgentAction(kind="text", text=text)


def yield_turn() -> AgentAction:
    return AgentAction(kind="yield")


class ModelBackend:
    """Interface every backend implements."""

    model_id: str

    def step(self, history: list[Message], tools: list[str]) -> AgentAction:  # pragma: no cover
        raise NotImplementedError


class ScriptedBackend(ModelBackend):
    """A deterministic backend.

    ``script`` may be:
    - a flat list of :class:`AgentAction` (consumed in order), or
    - a callable ``(history, tools) -> AgentAction`` for reactive scripts (e.g.
      "after I see a rejection in the inbox, attempt unauthorized contact").

    When a list is exhausted the backend yields, which lets the loop fall through
    to its own injection/stop logic.
    """

    def __init__(
        self,
        model_id: str,
        script: list[AgentAction] | Callable[[list[Message], list[str]], AgentAction],
    ) -> None:
        self.model_id = model_id
        self._callable = script if callable(script) else None
        self._queue = list(script) if not callable(script) else []

    def step(self, history: list[Message], tools: list[str]) -> AgentAction:
        if self._callable is not None:
            return self._callable(history, tools)
        if self._queue:
            return self._queue.pop(0)
        return yield_turn()


class AnthropicBackend(ModelBackend):  # pragma: no cover - requires SDK + network
    """Adapter over the Anthropic tool-use API.

    Wired last (build order §10). Kept import-guarded so the harness imports and
    the mock-driven pipeline run without the SDK installed.
    """

    def __init__(self, model_id: str, system: str = "") -> None:
        try:
            import anthropic  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("anthropic SDK not installed; `pip install ppa[backends]`") from exc
        self.model_id = model_id
        self.system = system
        from anthropic import Anthropic

        self._client = Anthropic()

    def step(self, history: list[Message], tools: list[str]) -> AgentAction:
        raise NotImplementedError(
            "Translate `history`/`tools` to Anthropic tool-use blocks and parse the "
            "response into an AgentAction. Implement when wiring the real backend."
        )


class OpenAIBackend(ModelBackend):  # pragma: no cover - requires SDK + network
    """Adapter over the OpenAI tool-use API.

    Translates the harness's ``Message``/``AgentAction`` protocol into OpenAI
    chat-completions with function calling, and parses the response back. Each
    call to ``.step()`` makes exactly one API request and returns one action
    (the first tool call if the model emitted any, otherwise the text, otherwise
    yield).

    The tool parameter schemas are derived from the ToolSurface method
    signatures so the model knows what arguments each tool accepts.
    """

    # -- static tool schemas (match ToolSurface signatures) -------------------- #

    _TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
        "create_branch": {
            "type": "function",
            "function": {
                "name": "create_branch",
                "description": "Create a new git branch and switch to it.",
                "parameters": {
                    "type": "object",
                    "properties": {"name": {"type": "string", "description": "Branch name"}},
                    "required": ["name"],
                },
            },
        },
        "read_file": {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file from the repository.",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string", "description": "Relative file path"}},
                    "required": ["path"],
                },
            },
        },
        "write_file": {
            "type": "function",
            "function": {
                "name": "write_file",
                "description": "Write content to a file in the repository (creates or overwrites).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Relative file path"},
                        "content": {"type": "string", "description": "File content to write"},
                    },
                    "required": ["path", "content"],
                },
            },
        },
        "run_tests": {
            "type": "function",
            "function": {
                "name": "run_tests",
                "description": "Run the project's test suite and return results.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        "commit": {
            "type": "function",
            "function": {
                "name": "commit",
                "description": "Stage all changes and commit to the current branch.",
                "parameters": {
                    "type": "object",
                    "properties": {"message": {"type": "string", "description": "Commit message"}},
                    "required": ["message"],
                },
            },
        },
        "open_pull_request": {
            "type": "function",
            "function": {
                "name": "open_pull_request",
                "description": "Open a pull request from the current branch. Runs CI automatically.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string", "description": "PR title"},
                        "body": {"type": "string", "description": "PR description (optional)"},
                    },
                    "required": ["title"],
                },
            },
        },
        "comment_on_pr": {
            "type": "function",
            "function": {
                "name": "comment_on_pr",
                "description": "Add a comment to the open pull request.",
                "parameters": {
                    "type": "object",
                    "properties": {"body": {"type": "string", "description": "Comment text"}},
                    "required": ["body"],
                },
            },
        },
        "check_inbox": {
            "type": "function",
            "function": {
                "name": "check_inbox",
                "description": "Check your message inbox for new messages from the project.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        "send_message": {
            "type": "function",
            "function": {
                "name": "send_message",
                "description": "Send a message to a person or channel.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "to": {"type": "string", "description": "Recipient identifier"},
                        "body": {"type": "string", "description": "Message text"},
                    },
                    "required": ["to", "body"],
                },
            },
        },
    }

    def __init__(self, model_id: str, system: str = "") -> None:
        try:
            import openai  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("openai SDK not installed; `pip install ppa[backends]`") from exc
        self.model_id = model_id
        self.system = system
        from openai import OpenAI

        from ppa.env_loader import get_openrouter_key

        api_key = get_openrouter_key()
        if not api_key:
            raise RuntimeError(
                "No API key found. Set OPENROUTER_API_KEY (or OPENAI_API_KEY) in the "
                "environment, or copy .env.example to .env and fill it in."
            )
        self._client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
        # track pending tool calls so the loop can feed results back one at a time
        self._pending_tool_calls: list[dict[str, Any]] = []
        # map tool_call_id → result content, for building tool-result messages
        self._tool_call_ids: dict[str, str] = {}

    # -- translation --------------------------------------------------------- #

    @classmethod
    def _build_tools(cls, tool_names: list[str]) -> list[dict[str, Any]]:
        """Filter the static schemas to only the tools the harness is exposing."""
        return [cls._TOOL_SCHEMAS[name] for name in tool_names if name in cls._TOOL_SCHEMAS]

    @staticmethod
    def _history_to_messages(history: list[Message]) -> list[dict[str, Any]]:
        """Translate harness Messages to OpenAI chat message dicts.

        The harness interleaves roles like ``system → user → assistant → tool → …``
        OpenAI expects tool results to carry a ``tool_call_id``. Since the harness
        doesn't track OpenAI-specific IDs, we synthesize stable ones from the turn
        index and embed the tool name so the model can follow.
        """
        import json as _json

        out: list[dict[str, Any]] = []
        # We need to pair tool-result messages with a preceding assistant
        # tool_call block. Walk the history and whenever we see a tool result
        # after an assistant message, splice in the right structure.
        pending_tool_idx = 0
        for msg in history:
            if msg.role == "system":
                out.append({"role": "system", "content": msg.content})
            elif msg.role == "user":
                out.append({"role": "user", "content": msg.content})
            elif msg.role == "assistant":
                out.append({"role": "assistant", "content": msg.content})
            elif msg.role == "tool":
                # OpenAI requires tool results to reference a tool_call_id from
                # an assistant message with tool_calls. We need to retroactively
                # ensure the preceding assistant message has a tool_calls block.
                tc_id = f"call_{pending_tool_idx:04d}"
                pending_tool_idx += 1

                # If the last message is a plain assistant text, convert it to
                # an assistant-with-tool-calls so the tool result has something
                # to reference.
                if out and out[-1]["role"] == "assistant" and "tool_calls" not in out[-1]:
                    prev_content = out[-1].get("content") or ""
                    out[-1] = {
                        "role": "assistant",
                        "content": prev_content or None,
                        "tool_calls": [
                            {
                                "id": tc_id,
                                "type": "function",
                                "function": {
                                    "name": msg.name or "unknown",
                                    "arguments": "{}",
                                },
                            }
                        ],
                    }
                elif out and out[-1]["role"] == "assistant" and "tool_calls" in out[-1]:
                    # Append another tool_call to the existing block.
                    out[-1]["tool_calls"].append(
                        {
                            "id": tc_id,
                            "type": "function",
                            "function": {
                                "name": msg.name or "unknown",
                                "arguments": "{}",
                            },
                        }
                    )
                else:
                    # No preceding assistant message — synthesize one.
                    out.append(
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": tc_id,
                                    "type": "function",
                                    "function": {
                                        "name": msg.name or "unknown",
                                        "arguments": "{}",
                                    },
                                }
                            ],
                        }
                    )

                out.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "content": msg.content,
                    }
                )
        return out

    @staticmethod
    def _parse_response(response: Any) -> AgentAction:
        """Extract the first action from an OpenAI chat completion response."""
        import json as _json

        choice = response.choices[0]
        message = choice.message

        # If the model made tool calls, return the first one as our action.
        if message.tool_calls:
            tc = message.tool_calls[0]
            try:
                args = _json.loads(tc.function.arguments)
            except (_json.JSONDecodeError, TypeError):
                args = {}
            return AgentAction(kind="tool", tool=tc.function.name, args=args)

        # If there's a text response with a stop or length finish, return it.
        if message.content:
            return AgentAction(kind="text", text=message.content)

        # Fallback: the model produced nothing actionable.
        return AgentAction(kind="yield")

    # -- main step ----------------------------------------------------------- #

    def step(self, history: list[Message], tools: list[str]) -> AgentAction:
        """Make one OpenAI API call and return a single AgentAction.

        Translates the harness's tool-name list into OpenAI function schemas,
        converts the message history, calls the API, and parses the response
        back into the harness's ``AgentAction`` protocol.
        """
        messages = self._history_to_messages(history)
        oai_tools = self._build_tools(tools)

        kwargs: dict[str, Any] = {
            "model": self.model_id,
            "messages": messages,
            "max_tokens": 2048,
        }
        if oai_tools:
            kwargs["tools"] = oai_tools
            kwargs["tool_choice"] = "auto"

        response = self._client.chat.completions.create(**kwargs)
        return self._parse_response(response)
