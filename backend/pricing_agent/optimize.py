"""Quant pricing agent: choose the margin-safe optimal discount depth.

Objective (the "clear inventory without destroying margin" mandate):

    value(d) = net_incremental_margin(d)            # from the GNN/forecast layer
             + clearance_value(d)                    # holding/spoilage saved by
                                                     # selling incremental units now

subject to a **margin floor**: the blended post-promo gross margin of the focus
SKU must stay >= `margin_floor` (default 15%). Depths that breach the floor are
infeasible.

We optimise over a bounded depth in [0, max_depth] using scipy. Because the
impact curve is cheap to evaluate and non-convex, we do a coarse grid scan then
refine the best cell with `scipy.optimize.minimize_scalar` (bounded).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar

from backend.forecasting.predict import ImpactPredictor, ImpactResult

DEFAULT_MARGIN_FLOOR = 0.15
DEFAULT_MAX_DEPTH = 0.40
MIN_PROMO_DEPTH = 0.05   # a promo is either 0 (skip) or a meaningful >= 5%
# how many weeks of holding cost we credit for clearing a unit early
CLEARANCE_HORIZON_WEEKS = 8


@dataclass
class PricingRecommendation:
    product_id: int
    label: str
    recommended_depth: float
    net_incremental_margin: float
    clearance_value: float
    total_value: float
    net_roi: float
    blended_margin_pct: float
    margin_ok: bool
    inventory: int
    weeks_of_supply: float
    rationale: str
    impact: ImpactResult

    def summary(self) -> dict:
        d = {
            "product_id": self.product_id,
            "label": self.label,
            "recommended_depth": round(self.recommended_depth, 3),
            "recommended_depth_pct": round(self.recommended_depth * 100, 1),
            "net_incremental_margin": round(self.net_incremental_margin, 2),
            "clearance_value": round(self.clearance_value, 2),
            "total_value": round(self.total_value, 2),
            "net_roi": round(self.net_roi, 3),
            "blended_margin_pct": round(self.blended_margin_pct * 100, 1),
            "margin_ok": self.margin_ok,
            "inventory": self.inventory,
            "weeks_of_supply": round(self.weeks_of_supply, 1),
            "rationale": self.rationale,
        }
        d["impact"] = self.impact.summary()
        return d


class PricingAgent:
    def __init__(self, predictor: ImpactPredictor | None = None) -> None:
        self.pred = predictor or ImpactPredictor()

    def close(self) -> None:
        self.pred.close()

    def _clearance_value(self, impact: ImpactResult, holding_cost: float) -> float:
        """Holding/spoilage cost saved by selling incremental units earlier."""
        incremental_units = max(0.0, impact.own_lift_units)
        return incremental_units * holding_cost * CLEARANCE_HORIZON_WEEKS

    def _blended_margin_pct(self, product_id: int, depth: float) -> float:
        p = self.pred.prod.loc[product_id]
        price = float(p["base_price"]) * (1 - depth)
        cost = float(p["unit_cost"])
        return (price - cost) / price if price > 0 else -1.0

    def _value(self, product_id: int, depth: float, holding_cost: float) -> tuple[float, ImpactResult]:
        impact = self.pred.predict_impact(product_id, depth)
        cv = self._clearance_value(impact, holding_cost)
        total = impact.net_incremental_margin + cv
        return total, impact

    def recommend(
        self,
        product_id: int,
        margin_floor: float = DEFAULT_MARGIN_FLOOR,
        max_depth: float = DEFAULT_MAX_DEPTH,
    ) -> PricingRecommendation:
        pid = int(product_id)
        p = self.pred.prod.loc[pid]
        holding_cost = float(p["holding_cost"])
        inventory = int(p["inventory"])

        # weeks of supply from recent average weekly units (off-promo baseline)
        base_impact = self.pred.predict_impact(pid, 0.0)
        weekly = max(1e-6, base_impact.baseline_units)
        weeks_of_supply = inventory / weekly

        # feasible depth ceiling from the margin floor
        # blended margin = (price(1-d) - cost)/(price(1-d)) >= floor
        # solve => (1-d) >= cost / (price * (1 - floor))
        price, cost = float(p["base_price"]), float(p["unit_cost"])
        max_feasible = 1 - cost / (price * (1 - margin_floor)) if price > 0 else 0.0
        depth_ceiling = float(np.clip(min(max_depth, max_feasible), 0.0, max_depth))

        # candidate depths: always "no promo" (0), plus a promo grid in
        # [MIN_PROMO_DEPTH, ceiling]. This avoids recommending a degenerate
        # near-zero discount and mirrors how promos are actually run.
        grid = [0.0]
        if depth_ceiling >= MIN_PROMO_DEPTH:
            grid += list(np.linspace(MIN_PROMO_DEPTH, depth_ceiling, 15))
        grid = np.array(grid)
        vals = [self._value(pid, float(d), holding_cost)[0] for d in grid]
        best_i = int(np.argmax(vals))
        best_d = float(grid[best_i])

        # refine only within the promo region (never below MIN_PROMO_DEPTH)
        if best_i > 1 and best_i < len(grid) - 1:
            lo, hi = grid[best_i - 1], grid[best_i + 1]
            r = minimize_scalar(
                lambda d: -self._value(pid, float(d), holding_cost)[0],
                bounds=(lo, hi), method="bounded",
            )
            if -r.fun > vals[best_i] and r.x >= MIN_PROMO_DEPTH:
                best_d = float(r.x)

        total_value, impact = self._value(pid, best_d, holding_cost)
        clearance = self._clearance_value(impact, holding_cost)
        blended = self._blended_margin_pct(pid, best_d)

        rationale = self._rationale(best_d, impact, clearance, weeks_of_supply, depth_ceiling, max_feasible)

        return PricingRecommendation(
            product_id=pid, label=impact.label, recommended_depth=best_d,
            net_incremental_margin=impact.net_incremental_margin,
            clearance_value=clearance, total_value=total_value,
            net_roi=impact.net_roi, blended_margin_pct=blended,
            margin_ok=blended >= margin_floor - 1e-6, inventory=inventory,
            weeks_of_supply=weeks_of_supply, rationale=rationale, impact=impact,
        )

    @staticmethod
    def _rationale(depth, impact, clearance, weeks_supply, ceiling, max_feasible) -> str:
        parts = []
        if depth <= 1e-3:
            parts.append("No promotion recommended - any discount destroys more margin "
                         "(mainly via cannibalization) than it creates.")
        else:
            parts.append(f"Recommend {depth*100:.0f}% off.")
        if impact.cannibalization_margin < -1:
            worst = min(impact.cannibalization, key=lambda n: n.delta_margin, default=None)
            if worst:
                parts.append(f"Cannibalization costs ${-impact.cannibalization_margin:.0f} "
                             f"(worst: {worst.label}).")
        if impact.halo_margin > 1:
            parts.append(f"Halo adds ${impact.halo_margin:.0f} from co-purchased items.")
        if clearance > 1 and weeks_supply > 8:
            parts.append(f"Overstocked ({weeks_supply:.0f} weeks of supply) - clearance value "
                         f"${clearance:.0f} helps justify the depth.")
        if max_feasible < ceiling + 1:  # floor was binding-ish
            parts.append(f"Depth capped near {max_feasible*100:.0f}% to respect the margin floor.")
        return " ".join(parts)


if __name__ == "__main__":
    agent = PricingAgent()
    for name in ["greek yogurt", "cola", "ground coffee", "potato chips"]:
        sku = agent.pred.store.find_sku(name)
        rec = agent.recommend(int(sku["product_id"]))
        print(f"\n{name}: {rec.label}")
        print(f"  depth={rec.recommended_depth*100:.0f}%  net_margin=${rec.net_incremental_margin:.0f}"
              f"  clearance=${rec.clearance_value:.0f}  total=${rec.total_value:.0f}"
              f"  wos={rec.weeks_of_supply:.0f}  margin_ok={rec.margin_ok}")
        print(f"  {rec.rationale}")
    agent.close()
