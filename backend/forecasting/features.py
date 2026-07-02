"""Feature engineering for the baseline-vs-promo-lift model.

Grain: (product_id, week_no) over a full product x week grid (missing = 0 units).
The key *graph-aware* features are what make this "graph-enhanced":

  sub_pressure : sum over a SKU's SUBSTITUTES that are on promo that week of
                 (edge_weight * their discount_depth)  -> drives cannibalization
  halo_pull    : same over CO_PURCHASED neighbours     -> drives halo

plus own promo depth, price/cost, commodity, week seasonality, and the GNN
SKU embeddings.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backend.config import settings
from backend.data_gen import duck

N_WEEKS = 52

# NOTE: `on_promo` is intentionally NOT a feature. A binary promo flag creates a
# discontinuity (any depth>0 => full promo demand at ~zero cost), which a pricing
# optimizer exploits as "free lift". discount_depth carries the promo signal
# continuously instead.
BASE_NUMERIC = ["discount_depth", "sub_pressure", "halo_pull",
                "base_price", "unit_cost", "week_sin", "week_cos"]


def _grid_frame(con) -> pd.DataFrame:
    return con.execute(
        f"""
        WITH weeks AS (SELECT UNNEST(range(1, {N_WEEKS} + 1)) AS week_no),
        grid AS (SELECT p.product_id, w.week_no FROM product p CROSS JOIN weeks w),
        own AS (
            SELECT product_id, week_no, MAX(discount_depth) AS depth
            FROM promo_events GROUP BY 1, 2
        ),
        subp AS (
            SELECT es.src AS product_id, pr.week_no,
                   SUM(es.weight * pr.discount_depth) AS sub_pressure
            FROM edge_substitutes es
            JOIN promo_events pr ON pr.product_id = es.dst
            GROUP BY 1, 2
        ),
        halo AS (
            SELECT ec.src AS product_id, pr.week_no,
                   SUM(ec.weight * pr.discount_depth) AS halo_pull
            FROM edge_copurchase ec
            JOIN promo_events pr ON pr.product_id = ec.dst
            GROUP BY 1, 2
        )
        SELECT g.product_id, g.week_no,
               COALESCE(o.depth, 0.0) AS discount_depth,
               CASE WHEN o.depth IS NOT NULL THEN 1 ELSE 0 END AS on_promo,
               COALESCE(sp.sub_pressure, 0.0) AS sub_pressure,
               COALESCE(h.halo_pull, 0.0) AS halo_pull,
               COALESCE(s.units, 0) AS units,
               pr.base_price, pr.unit_cost, pr.commodity_desc
        FROM grid g
        JOIN product pr ON pr.product_id = g.product_id
        LEFT JOIN own o  ON o.product_id = g.product_id AND o.week_no = g.week_no
        LEFT JOIN subp sp ON sp.product_id = g.product_id AND sp.week_no = g.week_no
        LEFT JOIN halo h ON h.product_id = g.product_id AND h.week_no = g.week_no
        LEFT JOIN product_week_sales s
               ON s.product_id = g.product_id AND s.week_no = g.week_no
        """
    ).df()


def load_embeddings() -> pd.DataFrame:
    if not settings.embeddings_path.exists():
        raise FileNotFoundError(
            f"{settings.embeddings_path} missing. Run `python -m backend.gnn.train_graphsage`."
        )
    return pd.read_parquet(settings.embeddings_path)


def add_derived(df: pd.DataFrame, commodity_categories: list[str] | None = None):
    """Add seasonality + commodity code. Returns (df, emb_cols, commodity_categories)."""
    df = df.copy()
    df["week_sin"] = np.sin(2 * np.pi * df["week_no"] / N_WEEKS)
    df["week_cos"] = np.cos(2 * np.pi * df["week_no"] / N_WEEKS)
    cats = commodity_categories or sorted(df["commodity_desc"].unique())
    df["commodity_code"] = pd.Categorical(df["commodity_desc"], categories=cats).codes
    return df, cats


def build_feature_frame() -> tuple[pd.DataFrame, list[str], list[str]]:
    """Return (frame, feature_cols, commodity_categories) for training."""
    con = duck.connect(read_only=True)
    try:
        df = _grid_frame(con)
    finally:
        con.close()
    emb = load_embeddings()
    emb_cols = [c for c in emb.columns if c.startswith("e")]
    df = df.merge(emb, on="product_id", how="left")
    df[emb_cols] = df[emb_cols].fillna(0.0)
    df, cats = add_derived(df)
    feature_cols = BASE_NUMERIC + ["commodity_code"] + emb_cols
    return df, feature_cols, cats
