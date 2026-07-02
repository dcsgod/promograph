"""LLM client - the OpenAI Python SDK pointed at Ollama's OpenAI-compatible
endpoint. Local-first, $0. No fallback to a hosted API (per project rules).
"""
from __future__ import annotations

from functools import lru_cache

import httpx
from openai import OpenAI

from backend.config import settings


@lru_cache
def get_client() -> OpenAI:
    return OpenAI(base_url=settings.openai_base_url, api_key=settings.openai_api_key)


def llm_available() -> bool:
    """Quick reachability check for the local Ollama server."""
    base = settings.openai_base_url.rsplit("/v1", 1)[0]
    try:
        httpx.get(f"{base}/api/tags", timeout=2.0)
        return True
    except Exception:
        return False


def chat(messages: list[dict], tools: list[dict] | None = None,
         tool_choice: str = "auto", temperature: float = 0.2):
    """One (non-streaming) chat completion. Returns the choice message object."""
    kwargs: dict = {"model": settings.tpo_llm_model, "messages": messages,
                    "temperature": temperature}
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = tool_choice
    resp = get_client().chat.completions.create(**kwargs)
    return resp.choices[0].message


def stream_text(messages: list[dict], temperature: float = 0.3):
    """Stream a plain-text completion token-by-token (no tools)."""
    stream = get_client().chat.completions.create(
        model=settings.tpo_llm_model, messages=messages,
        temperature=temperature, stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta and delta.content:
            yield delta.content
