"""Orchestration loop.

Primary path: OpenAI-SDK function-calling loop against local Ollama.
Fallback path: a deterministic planner (product -> subgraph -> impact/optimize)
so the app runs end-to-end even before a local model is installed.

`run_turn(history)` is a generator of typed Events. It appends the assistant's
final message to `history` in place so the next turn (including a node click)
keeps context.
"""
from __future__ import annotations

import json
import re
from collections.abc import Iterator

from backend.api.events import (
    DoneEvent,
    ErrorEvent,
    TokenEvent,
    ToolEvent,
)
from backend.llm import client as llm
from backend.orchestrator import tools as T
from backend.orchestrator.prompts import SYSTEM_PROMPT

MAX_STEPS = 5


def _chunk(text: str) -> Iterator[str]:
    for i, word in enumerate(text.split(" ")):
        yield (" " if i else "") + word


# --- LLM path ---------------------------------------------------------------
def _run_llm(history: list[dict]) -> Iterator:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history
    final_text = ""
    for _ in range(MAX_STEPS):
        msg = llm.chat(messages, tools=T.TOOL_SCHEMAS)
        if msg.tool_calls:
            messages.append(msg.model_dump(exclude_none=True))
            for tc in msg.tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                yield ToolEvent(name=name, status="start", detail=json.dumps(args))
                result, events = T.dispatch(name, args)
                for ev in events:
                    yield ev
                yield ToolEvent(name=name, status="done")
                messages.append({"role": "tool", "tool_call_id": tc.id,
                                 "content": json.dumps(result)})
            continue
        final_text = msg.content or ""
        break

    if not final_text:  # exhausted tool steps; force a wrap-up without tools
        final_text = (llm.chat(messages, tools=None).content or "").strip() or \
            "Here is the analysis based on the tools above."

    for tok in _chunk(final_text):
        yield TokenEvent(text=tok)
    history.append({"role": "assistant", "content": final_text})
    yield DoneEvent(mode="llm")


# --- deterministic fallback -------------------------------------------------
_DEPTH_PCT = re.compile(r"(\d+(?:\.\d+)?)\s*%")
_DEPTH_FRAC = re.compile(r"\b0?\.(\d+)\b")
_SKU_ID = re.compile(r"sku:(\d+)|product_id[=\s:]+(\d+)")


def _parse_depth(text: str) -> float | None:
    m = _DEPTH_PCT.search(text)
    if m:
        return float(m.group(1)) / 100.0
    m = _DEPTH_FRAC.search(text)
    if m:
        return float("0." + m.group(1))
    return None


def _parse_product(text: str):
    m = _SKU_ID.search(text)
    if m:
        pid = int(m.group(1) or m.group(2))
        row = T._svc.store.get_sku(pid)
        if row:
            return row
    return T._svc.store.find_sku(text)


def _narrate_impact(s: dict) -> str:
    verdict = "creates value" if s["net_incremental_margin"] > 0 else "destroys value"
    parts = [
        f"A {s['discount_depth']*100:.0f}% promo on {s['label']} {verdict}: "
        f"net incremental margin ${s['net_incremental_margin']:.0f} "
        f"(own lift {s['own_lift_units']:.0f} units)."
    ]
    if s["top_cannibalized"]:
        c = s["top_cannibalized"][0]
        parts.append(f"Biggest cannibalization: {c['label']} (-${c['lost_margin']:.0f}).")
    if s["halo_margin"] > 1:
        parts.append(f"Halo adds ${s['halo_margin']:.0f} from co-purchased items.")
    parts.append(f"Net ROI {s['net_roi']:.2f} per discount dollar.")
    return " ".join(parts)


def _narrate_reco(r: dict) -> str:
    return r["rationale"] + (
        f" (net margin ${r['net_incremental_margin']:.0f}, clearance value "
        f"${r['clearance_value']:.0f}, blended margin {r['blended_margin_pct']:.0f}%)."
    )


def _run_fallback(history: list[dict]) -> Iterator:
    user_msgs = [m["content"] for m in history if m["role"] == "user"]
    user_text = user_msgs[-1] if user_msgs else ""
    # resolve product from the latest message, else the most recent one that names one
    row = _parse_product(user_text)
    if row is None:
        for prev in reversed(user_msgs[:-1]):
            row = _parse_product(prev)
            if row is not None:
                break
    if row is None:
        text = ("I couldn't identify a product. Try naming one, e.g. "
                "'What's the optimal discount for ground coffee?'")
        for tok in _chunk(text):
            yield TokenEvent(text=tok)
        history.append({"role": "assistant", "content": text})
        yield DoneEvent(mode="fallback")
        return

    product_ref = str(int(row["product_id"]))
    depth = _parse_depth(user_text)

    yield ToolEvent(name="get_subgraph", status="start")
    _, events = T.get_subgraph(product_ref)
    for ev in events:
        yield ev
    yield ToolEvent(name="get_subgraph", status="done")

    if depth is not None:
        yield ToolEvent(name="predict_impact", status="start", detail=f"depth={depth}")
        result, events = T.predict_impact(product_ref, depth)
        for ev in events:
            yield ev
        yield ToolEvent(name="predict_impact", status="done")
        text = _narrate_impact(result["impact"])
    else:
        yield ToolEvent(name="optimize_discount", status="start")
        result, events = T.optimize_discount(product_ref)
        for ev in events:
            yield ev
        yield ToolEvent(name="optimize_discount", status="done")
        text = _narrate_reco(result["recommendation"])

    for tok in _chunk(text):
        yield TokenEvent(text=tok)
    history.append({"role": "assistant", "content": text})
    yield DoneEvent(mode="fallback")


# --- entry point ------------------------------------------------------------
def run_turn(history: list[dict]) -> Iterator:
    """Yield events for one assistant turn. Uses the LLM if reachable, else the
    deterministic planner. Falls back on any LLM error mid-turn."""
    if llm.llm_available():
        try:
            yield from _run_llm(history)
            return
        except Exception as e:  # noqa: BLE001 - degrade gracefully
            detail = "model not pulled" if "not found" in str(e).lower() else str(e)[:80]
            yield ErrorEvent(
                message=f"Local model unavailable ({detail}) - using deterministic analysis."
            )
    yield from _run_fallback(history)
