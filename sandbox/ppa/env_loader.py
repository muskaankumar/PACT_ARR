"""Tiny .env loader and API-key resolver.

The repo's ``.env`` sets ``OPENAI_API_KEY`` to an OpenRouter key, but the code
reads ``OPENROUTER_API_KEY``. That mismatch is why the judge (and any backend
call) failed with "key not set" (plan §6). This module:

  1. loads ``.env`` from the project root into ``os.environ`` if not already set;
  2. resolves the OpenRouter key from either ``OPENROUTER_API_KEY`` or
     ``OPENAI_API_KEY`` (in that order).

No third-party dependency (avoids requiring python-dotenv).
"""

from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(path: str | Path | None = None) -> None:
    """Load KEY=VALUE lines from a .env file into os.environ (without overriding)."""
    if path is None:
        # ppa/env_loader.py -> project root is one level up from the package dir.
        path = Path(__file__).resolve().parent.parent / ".env"
    p = Path(path)
    if not p.exists():
        return
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


def get_openrouter_key() -> str | None:
    """Return the OpenRouter API key from either supported env var."""
    load_dotenv()
    return os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
