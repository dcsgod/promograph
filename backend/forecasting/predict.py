"""Network-aware promo impact prediction.

Given a focus SKU and a proposed discount depth, estimate:
  * own lift          (focus units above baseline)
  * cannibalization   (units lost by SUBSTITUTES pressured by the promo)
  * halo              (units gained by CO_PURCHASED neighbours pulled by it)
  * net incremental margin  (own + halo - cannibalization)   <-- pricing input
  * net ROI           (net incremental margin / promo discount dollars)

This is the number the whole product is about: net-of-cannibalization ROI, not
isolated single-SKU lift.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache

import lightgbm as lgb
import numpy as np
import pandas as pd

from backend.config import settings
from backend.data_gen import duck
from backend.forecasting.features import N_WEEKS, load_embeddings
from backend.graph_store.base import get_graph_store

REFERENCE_WEEK = 26
MAX_NEIGHBORS = 6


@dataclass
class NeighborImpact:
    product_id: int
    label: str
    baseline_units: float
    scenario_units: float
    delta_units: float
    delta_margin: float
    edge_weight: float


@dataclass
class ImpactResult:
    product_id: int
    label: str
    discount_depth: float
    baseline_units: float
    promo_units: float
    own_lift_units: float
    own_incr_margin: float
    cannibalization: list[NeighborImpact] = field(default_factory=list)
    halo: list[NeighborImpact] = field(default_factory=list)
    cannibalization_margin: float = 0.0
    halo_margin: float = 0.0
    net_incremental_margin: float = 0.0
    promo_cost: float = 0.0
    net_roi: float = 0.0
    gross_roi: float = 0.0

    def summary(self) -> dict:
        """Compact dict for the LLM / API (avoids dumping every neighbour)."""
        return {
            "product_id": self.product_id,
            "label": self.label,
            "discount_depth": round(self.discount_depth, 3),
            "baseline_units": round(self.baseline_units, 1),
            "promo_units": round(self.promo_units, 1),
            "own_lift_units": round(self.own_lift_units, 1),
            "own_incr_margin": round(self.own_incr_margin, 2),
            "cannibalization_margin": round(self.cannibalization_margin, 2),
            "halo_margin": round(self.halo_margin, 2),
            "net_incremental_margin": round(self.net_incremental_margin, 2),
            "promo_cost": round(self.promo_cost, 2),
            "net_roi": round(self.net_roi, 3),
            "gross_roi": round(self.gross_roi, 3),
            "top_cannibalized": [
                {"product_id": n.product_id, "label": n.label,
                 "lost_units": round(-n.delta_units, 1), "lost_margin": round(-n.delta_margin, 2)}
                for n in sorted(self.cannibalization, key=lambda x: x.delta_margin)[:3]
            ],
            "top_halo": [
                {"product_id": n.product_id, "label": n.label,
                 "gained_units": round(n.delta_units, 1), "gained_margin": round(n.delta_margin, 2)}
                for n in sorted(self.halo, key=lambda x: -x.delta_margin)[:3]
            ],
        }


class ImpactPredictor:
    def __init__(self) -> None:
        self.booster = lgb.Booster(model_file=str(settings.lgbm_model_path))
        meta = json.loads(
            settings.lgbm_model_path.with_suffix(".meta.json").read_text(encoding="utf-8")
        )
        self.feature_cols: list[str] = meta["feature_cols"]
        self.cats: list[str] = meta["commodity_categories"]
        self.emb = load_embeddings().set_index("product_id")
        self.emb_cols = [c for c in self.emb.columns if c.startswith("e")]
        prod = duck.query(
            "SELECT product_id, base_price, unit_cost, commodity_desc, brand, "
            "sub_commodity_desc, inventory, holding_cost FROM product"
        )
        self.prod = prod.set_index("product_id")
        self.store = get_graph_store()

    def close(self) -> None:
        self.store.close()

    # -- feature row assembly --------------------------------------------------
    def _row(self, pid: int, depth: float, sub_pressure: float, halo_pull: float,
             week: int = REFERENCE_WEEK) -> dict:
        p = self.prod.loc[pid]
        row = {
            "discount_depth": depth,
            "sub_pressure": sub_pressure,
            "halo_pull": halo_pull,
            "base_price": float(p["base_price"]),
            "unit_cost": float(p["unit_cost"]),
            "week_sin": np.sin(2 * np.pi * week / N_WEEKS),
            "week_cos": np.cos(2 * np.pi * week / N_WEEKS),
            "commodity_code": self.cats.index(p["commodity_desc"]) if p["commodity_desc"] in self.cats else -1,
        }
        emb = self.emb.loc[pid] if pid in self.emb.index else pd.Series(0.0, index=self.emb_cols)
        for c in self.emb_cols:
            row[c] = float(emb[c])
        return row

    def _predict(self, rows: list[dict]) -> np.ndarray:
        X = pd.DataFrame(rows)[self.feature_cols]
        return np.clip(self.booster.predict(X), 0, None)

    def _full_margin(self, pid: int) -> float:
        p = self.prod.loc[pid]
        return float(p["base_price"] - p["unit_cost"])

    # -- main ------------------------------------------------------------------
    def predict_impact(self, product_id: int, discount_depth: float,
                       week: int = REFERENCE_WEEK) -> ImpactResult:
        pid = int(product_id)
        p = self.prod.loc[pid]
        label = f"{p['brand']} {p['sub_commodity_desc']}"

        # focus baseline vs promo
        base_u, promo_u = self._predict([
            self._row(pid, 0.0, 0.0, 0.0, week),
            self._row(pid, discount_depth, 0.0, 0.0, week),
        ])
        promo_margin_unit = float(p["base_price"]) * (1 - discount_depth) - float(p["unit_cost"])
        full_margin_unit = self._full_margin(pid)
        own_incr_margin = promo_u * promo_margin_unit - base_u * full_margin_unit

        res = ImpactResult(
            product_id=pid, label=label, discount_depth=discount_depth,
            baseline_units=float(base_u), promo_units=float(promo_u),
            own_lift_units=float(promo_u - base_u), own_incr_margin=float(own_incr_margin),
        )

        subs = self.store.neighbors(pid, "SUBSTITUTES", MAX_NEIGHBORS)
        cop = self.store.neighbors(pid, "CO_PURCHASED", MAX_NEIGHBORS)

        # cannibalization: substitutes feel sub_pressure = weight * focus depth
        for nb in subs:
            if nb.product_id not in self.prod.index:
                continue
            b, s = self._predict([
                self._row(nb.product_id, 0.0, 0.0, 0.0, week),
                self._row(nb.product_id, 0.0, nb.weight * discount_depth, 0.0, week),
            ])
            m = self._full_margin(nb.product_id)
            res.cannibalization.append(NeighborImpact(
                nb.product_id, nb.label, float(b), float(s), float(s - b),
                float((s - b) * m), nb.weight))

        # halo: co-purchase neighbours feel halo_pull = weight * focus depth
        for nb in cop:
            if nb.product_id not in self.prod.index:
                continue
            b, h = self._predict([
                self._row(nb.product_id, 0.0, 0.0, 0.0, week),
                self._row(nb.product_id, 0.0, 0.0, nb.weight * discount_depth, week),
            ])
            m = self._full_margin(nb.product_id)
            res.halo.append(NeighborImpact(
                nb.product_id, nb.label, float(b), float(h), float(h - b),
                float((h - b) * m), nb.weight))

        res.cannibalization_margin = sum(n.delta_margin for n in res.cannibalization)
        res.halo_margin = sum(n.delta_margin for n in res.halo)
        res.net_incremental_margin = (
            res.own_incr_margin + res.cannibalization_margin + res.halo_margin
        )
        res.promo_cost = float(promo_u) * float(p["base_price"]) * discount_depth
        if res.promo_cost > 1e-6:
            res.net_roi = res.net_incremental_margin / res.promo_cost
            res.gross_roi = res.own_incr_margin / res.promo_cost
        return res


@lru_cache
def get_predictor() -> ImpactPredictor:
    return ImpactPredictor()


if __name__ == "__main__":
    # quick manual check
    pred = ImpactPredictor()
    store = pred.store
    sku = store.find_sku("greek yogurt")
    r = pred.predict_impact(int(sku["product_id"]), 0.25)
    import pprint
    pprint.pprint(r.summary())
    pred.close()
