"""FastAPI app - SSE streaming endpoints wiring the orchestrator to the UI.

Endpoints:
  GET  /health         backend + model + data readiness
  GET  /examples       a few example product prompts to seed the UI
  POST /chat           {session_id, message}      -> SSE event stream
  POST /node-click     {session_id, node_id, ...} -> SSE event stream (a turn)
  POST /reset          {session_id}               clear a conversation
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
from starlette.concurrency import iterate_in_threadpool

from backend.api import session
from backend.api.events import to_sse
from backend.config import settings
from backend.data_gen import duck
from backend.llm.client import llm_available
from backend.orchestrator.loop import run_turn

app = FastAPI(title="Promograph - Graph-Steered Conversational Trade Promotion Optimization")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    session_id: str
    message: str


class NodeClickRequest(BaseModel):
    session_id: str
    node_id: str
    node_type: str = "SKU"
    label: str = ""


class ResetRequest(BaseModel):
    session_id: str


def _stream(history: list[dict]) -> EventSourceResponse:
    async def gen():
        # run the (blocking) orchestrator in a threadpool, forwarding events
        async for ev in iterate_in_threadpool(run_turn(history)):
            yield to_sse(ev)
    return EventSourceResponse(gen())


@app.get("/health")
def health() -> dict:
    data_ready = settings.duckdb_path.exists()
    model_ready = settings.lgbm_model_path.exists() and settings.embeddings_path.exists()
    return {
        "status": "ok",
        "llm_available": llm_available(),
        "mode": "llm" if llm_available() else "fallback",
        "genie_backend": settings.tpo_genie_backend,
        "graph_backend": settings.tpo_graph_backend,
        "data_ready": data_ready,
        "model_ready": model_ready,
        "llm_model": settings.tpo_llm_model,
    }


@app.get("/examples")
def examples() -> dict:
    df = duck.query(
        """
        SELECT p.brand || ' ' || p.sub_commodity_desc AS label
        FROM product_week_sales s JOIN product p USING (product_id)
        GROUP BY 1 ORDER BY SUM(s.units) DESC LIMIT 6
        """
    )
    labels = df["label"].tolist()
    return {
        "prompts": [
            f"What happens if I run 20% off on {labels[0]}?" if labels else "",
            f"What's the optimal discount for {labels[1]}?" if len(labels) > 1 else "",
            f"Should I promote {labels[2]}?" if len(labels) > 2 else "",
        ],
        "products": labels,
    }


@app.post("/chat")
def chat(req: ChatRequest) -> EventSourceResponse:
    history = session.add_user_message(req.session_id, req.message)
    return _stream(history)


@app.post("/node-click")
def node_click(req: NodeClickRequest) -> EventSourceResponse:
    msg = (f"[graph-click] User clicked the {req.node_type} node "
           f"'{req.label}' ({req.node_id}). Drill into it: show its subgraph and "
           f"recommend the optimal discount.")
    history = session.add_user_message(req.session_id, msg)
    return _stream(history)


@app.post("/reset")
def reset(req: ResetRequest) -> dict:
    session.reset(req.session_id)
    return {"status": "reset"}
