"""Smoke tests for the analytics + orchestration pipeline.

These require the built artifacts (DuckDB, embeddings, LightGBM model). If they
are missing, the tests skip with a hint rather than fail - run:
    python -m backend.data_gen.generate
    python -m backend.graph_store.build_edges
    python -m backend.gnn.train_graphsage
    python -m backend.forecasting.train_lgbm
"""
from __future__ import annotations

import pytest

from backend.config import settings

pytestmark = pytest.mark.skipif(
    not (settings.duckdb_path.exists()
         and settings.embeddings_path.exists()
         and settings.lgbm_model_path.exists()),
    reason="build artifacts missing; run generate/build_edges/train_* first",
)


def test_graph_store_subgraph():
    from backend.graph_store.base import get_graph_store

    gs = get_graph_store()
    sku = gs.find_sku("ground coffee")
    assert sku is not None
    sg = gs.get_subgraph(int(sku["product_id"]))
    types = {n.type for n in sg.nodes}
    assert "SKU" in types
    assert any(e.type == "SUBSTITUTES" for e in sg.edges)
    gs.close()


def test_impact_reports_net_of_cannibalization():
    from backend.forecasting.predict import ImpactPredictor

    p = ImpactPredictor()
    sku = p.store.find_sku("ground coffee")
    r = p.predict_impact(int(sku["product_id"]), 0.2)
    s = r.summary()
    # net must reflect own + cannibalization + halo
    assert s["cannibalization_margin"] <= 0
    approx = s["own_incr_margin"] + s["cannibalization_margin"] + s["halo_margin"]
    assert abs(approx - s["net_incremental_margin"]) < 1.0
    p.close()


def test_pricing_respects_margin_floor():
    from backend.pricing_agent.optimize import PricingAgent

    a = PricingAgent()
    sku = a.pred.store.find_sku("cola")
    rec = a.recommend(int(sku["product_id"]), margin_floor=0.15)
    assert 0.0 <= rec.recommended_depth <= 0.40
    if rec.recommended_depth > 0:
        assert rec.blended_margin_pct >= 0.15 - 1e-6
    a.close()


def test_orchestrator_fallback_emits_graph_and_recommendation():
    # force the deterministic path regardless of Ollama state
    import backend.orchestrator.loop as loop

    history = [{"role": "user", "content": "What's the optimal discount for ground coffee?"}]
    events = list(loop._run_fallback(history))
    types = {e.type for e in events}
    assert "graph_node" in types
    assert "recommendation" in types
    assert "done" in types
