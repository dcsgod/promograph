"""In-memory conversation session store (single-process, dev-scale).

Node-click events and typed messages both append to the same history list, so a
click is a first-class conversation turn (per project convention).
"""
from __future__ import annotations

from collections import defaultdict

_SESSIONS: dict[str, list[dict]] = defaultdict(list)


def get_history(session_id: str) -> list[dict]:
    return _SESSIONS[session_id]


def reset(session_id: str) -> None:
    _SESSIONS[session_id] = []


def add_user_message(session_id: str, content: str) -> list[dict]:
    hist = _SESSIONS[session_id]
    hist.append({"role": "user", "content": content})
    return hist
