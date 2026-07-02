"""Typed SSE event schema - the LLM <-> Graph contract.

Every entity the assistant references is emitted as one of these typed events,
so the frontend renders the graph from structured data (never by parsing prose).
Mirrored in frontend/src/lib/events.ts.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class TokenEvent(BaseModel):
    type: Literal["token"] = "token"
    text: str


class ToolEvent(BaseModel):
    type: Literal["tool"] = "tool"
    name: str
    status: Literal["start", "done", "error"]
    detail: str = ""


class GraphNodeEvent(BaseModel):
    type: Literal["graph_node"] = "graph_node"
    id: str
    node_type: str
    label: str
    metrics: dict = {}


class GraphEdgeEvent(BaseModel):
    type: Literal["graph_edge"] = "graph_edge"
    source: str
    target: str
    edge_type: str
    weight: float = 1.0


class NodeUpdateEvent(BaseModel):
    """Update metrics on an existing node (e.g. colour by cannibalization)."""
    type: Literal["node_update"] = "node_update"
    id: str
    metrics: dict = {}


class RecommendationEvent(BaseModel):
    type: Literal["recommendation"] = "recommendation"
    data: dict


class ErrorEvent(BaseModel):
    type: Literal["error"] = "error"
    message: str


class DoneEvent(BaseModel):
    type: Literal["done"] = "done"
    mode: str = ""  # "llm" | "fallback"


Event = (
    TokenEvent | ToolEvent | GraphNodeEvent | GraphEdgeEvent
    | NodeUpdateEvent | RecommendationEvent | ErrorEvent | DoneEvent
)


def to_sse(event: BaseModel) -> dict:
    """Shape for sse_starlette.EventSourceResponse."""
    return {"event": event.type, "data": event.model_dump_json()}
