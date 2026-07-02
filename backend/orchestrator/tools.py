"""Orchestrator tools - plain Python functions over the lower layers.

Each tool returns `(result, events)`:
  * result : compact JSON-able dict fed back to the LLM to reason over
  * events : list of typed SSE events for the UI (graph nodes/edges, updates,
             recommendations). Non-LLM callers can ignore `events`.

The same functions back both the LLM tool-calling loop and the deterministic
fallback planner, so behaviour is identical with or without a local model.
"""
from __future__ import annotations

from functools import lru_cache

from backend.api.events import (
    GraphEdgeEvent,
    GraphNodeEvent,
    NodeUpdateEvent,
    RecommendationEvent,
)
from backend.forecasting.predict import ImpactPredictor
from backend.genie_clients.base import get_genie_client
from backend.graph_store.base import Subgraph, sku_id
from backend.pricing_agent.optimize import PricingAgent


class Services:
    """Lazy singletons so models/connections load once per process."""

    @property
    @lru_cache
    def genie(self):
        return get_genie_client()

    @property
    @lru_cache
    def predictor(self) -> ImpactPredictor:
        return ImpactPredictor()

    @property
    @lru_cache
    def pricing(self) -> PricingAgent:
        return PricingAgent(self.predictor)

    @property
    def store(self):
        return self.predictor.store


_svc = Services()


# --- helpers ----------------------------------------------------------------
def _resolve(product: str | int) -> dict | None:
    """Resolve a product ref (id or free text) to a product row dict."""
    store = _svc.store
    if isinstance(product, int) or (isinstance(product, str) and product.isdigit()):
        return store.get_sku(int(product))
    return store.find_sku(str(product))


def _subgraph_events(sg: Subgraph) -> list:
    events = []
    for n in sg.nodes:
        events.append(GraphNodeEvent(id=n.id, node_type=n.type, label=n.label, metrics=n.metrics))
    for e in sg.edges:
        events.append(GraphEdgeEvent(source=e.source, target=e.target, edge_type=e.type, weight=e.weight))
    return events


# --- tools ------------------------------------------------------------------
def genie_query(question: str, space: str = "sales") -> tuple[dict, list]:
    """Ask a Databricks Genie Space (or the local mock) a natural-language data
    question over historical sales / promo / inventory."""
    res = _svc.genie.ask(question, space)
    return {"genie": res.preview()}, []


def get_subgraph(product: str) -> tuple[dict, list]:
    """Resolve a product and return its portfolio subgraph (substitutes =
    cannibalization risk, co-purchased = halo candidates, category, promos)."""
    row = _resolve(product)
    if row is None:
        return {"error": f"Could not resolve product '{product}'."}, []
    pid = int(row["product_id"])
    sg = _svc.store.get_subgraph(pid)
    label = f"{row.get('brand','')} {row.get('sub_commodity_desc', row.get('sub_commodity',''))}".strip()
    result = {
        "product_id": pid,
        "label": label,
        "n_substitutes": sum(1 for e in sg.edges if e.type == "SUBSTITUTES"),
        "n_copurchased": sum(1 for e in sg.edges if e.type == "CO_PURCHASED"),
        "nodes": [{"id": n.id, "type": n.type, "label": n.label} for n in sg.nodes],
    }
    return result, _subgraph_events(sg)


def predict_impact(product: str, discount_depth: float) -> tuple[dict, list]:
    """Predict the network-wide impact of a promo at a given discount depth:
    own lift, cannibalization of substitutes, halo on co-purchased items, and
    NET incremental margin / net ROI after cannibalization."""
    row = _resolve(product)
    if row is None:
        return {"error": f"Could not resolve product '{product}'."}, []
    pid = int(row["product_id"])
    impact = _svc.predictor.predict_impact(pid, float(discount_depth))
    events: list = []
    # colour the focus + neighbours by their margin delta
    events.append(NodeUpdateEvent(id=sku_id(pid), metrics={
        "role": "focus", "own_lift_units": round(impact.own_lift_units, 1),
        "net_incremental_margin": round(impact.net_incremental_margin, 2)}))
    for n in impact.cannibalization:
        events.append(NodeUpdateEvent(id=sku_id(n.product_id), metrics={
            "role": "cannibalized", "delta_margin": round(n.delta_margin, 2),
            "delta_units": round(n.delta_units, 1)}))
    for n in impact.halo:
        events.append(NodeUpdateEvent(id=sku_id(n.product_id), metrics={
            "role": "halo", "delta_margin": round(n.delta_margin, 2),
            "delta_units": round(n.delta_units, 1)}))
    return {"impact": impact.summary()}, events


def optimize_discount(product: str, margin_floor: float = 0.15) -> tuple[dict, list]:
    """Recommend the margin-safe optimal discount depth (balancing net margin,
    cannibalization/halo, and inventory-clearance value) for a product."""
    row = _resolve(product)
    if row is None:
        return {"error": f"Could not resolve product '{product}'."}, []
    pid = int(row["product_id"])
    rec = _svc.pricing.recommend(pid, margin_floor=margin_floor)
    events: list = []
    events.append(NodeUpdateEvent(id=sku_id(pid), metrics={
        "role": "focus", "recommended_depth": round(rec.recommended_depth, 3),
        "net_incremental_margin": round(rec.net_incremental_margin, 2)}))
    for n in rec.impact.cannibalization:
        events.append(NodeUpdateEvent(id=sku_id(n.product_id), metrics={
            "role": "cannibalized", "delta_margin": round(n.delta_margin, 2)}))
    for n in rec.impact.halo:
        events.append(NodeUpdateEvent(id=sku_id(n.product_id), metrics={
            "role": "halo", "delta_margin": round(n.delta_margin, 2)}))
    events.append(RecommendationEvent(data=rec.summary()))
    return {"recommendation": rec.summary()}, events


# --- registry / dispatch -----------------------------------------------------
TOOLS = {
    "genie_query": genie_query,
    "get_subgraph": get_subgraph,
    "predict_impact": predict_impact,
    "optimize_discount": optimize_discount,
}

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "genie_query",
            "description": genie_query.__doc__,
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "Natural-language data question."},
                    "space": {"type": "string", "enum": ["sales", "promo", "inventory"],
                              "description": "Which Genie space to query."},
                },
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_subgraph",
            "description": get_subgraph.__doc__,
            "parameters": {
                "type": "object",
                "properties": {"product": {"type": "string",
                               "description": "Product name/brand or product_id."}},
                "required": ["product"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "predict_impact",
            "description": predict_impact.__doc__,
            "parameters": {
                "type": "object",
                "properties": {
                    "product": {"type": "string", "description": "Product name/brand or product_id."},
                    "discount_depth": {"type": "number",
                                       "description": "Discount fraction, e.g. 0.2 for 20% off."},
                },
                "required": ["product", "discount_depth"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "optimize_discount",
            "description": optimize_discount.__doc__,
            "parameters": {
                "type": "object",
                "properties": {
                    "product": {"type": "string", "description": "Product name/brand or product_id."},
                    "margin_floor": {"type": "number",
                                     "description": "Minimum blended gross margin (default 0.15)."},
                },
                "required": ["product"],
            },
        },
    },
]


def dispatch(name: str, args: dict) -> tuple[dict, list]:
    fn = TOOLS.get(name)
    if fn is None:
        return {"error": f"unknown tool {name}"}, []
    try:
        return fn(**args)
    except Exception as e:  # keep the loop alive; report to the model
        return {"error": f"{name} failed: {e}"}, []
