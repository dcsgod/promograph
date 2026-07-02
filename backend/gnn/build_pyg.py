"""Build graph arrays (features + edges) from the DuckDB edge tables.

Returns a plain numpy container so it works whether or not torch is installed.
`train_graphsage.py` converts this to a PyG `Data` object when torch is present,
or feeds it to the numpy SVD fallback otherwise.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from backend.data_gen import duck


@dataclass
class GraphArrays:
    product_ids: np.ndarray          # (N,) int
    x: np.ndarray                    # (N, F) float node features
    edge_index: np.ndarray           # (2, E) int, undirected (both directions)
    edge_weight: np.ndarray          # (E,) float
    feature_names: list[str]


def load_graph_arrays() -> GraphArrays:
    con = duck.connect(read_only=True)
    try:
        prod = con.execute(
            """
            SELECT p.product_id, p.base_price, p.unit_cost, p.department,
                   COALESCE(s.units, 0) AS total_units
            FROM product p
            LEFT JOIN (SELECT product_id, SUM(units) units FROM product_week_sales GROUP BY 1) s
              USING (product_id)
            ORDER BY p.product_id
            """
        ).df()
        subs = con.execute("SELECT src, dst, weight FROM edge_substitutes").df()
        cop = con.execute("SELECT src, dst, weight FROM edge_copurchase").df()
    finally:
        con.close()

    pids = prod["product_id"].to_numpy()
    idx = {int(p): i for i, p in enumerate(pids)}

    # --- node features: price, cost, log-units, department one-hot ---
    depts = pd.get_dummies(prod["department"], prefix="dept").astype(float)
    feats = pd.DataFrame(
        {
            "base_price": prod["base_price"] / prod["base_price"].max(),
            "unit_cost": prod["unit_cost"] / prod["unit_cost"].max(),
            "log_units": np.log1p(prod["total_units"]) / np.log1p(prod["total_units"]).max().clip(min=1),
        }
    )
    x_df = pd.concat([feats, depts], axis=1)
    x = x_df.to_numpy(dtype=float)

    # --- edges: substitutes + co-purchase, made undirected ---
    def _to_edges(df: pd.DataFrame) -> tuple[list[int], list[int], list[float]]:
        src, dst, w = [], [], []
        for r in df.itertuples(index=False):
            if int(r.src) in idx and int(r.dst) in idx:
                a, b = idx[int(r.src)], idx[int(r.dst)]
                src += [a, b]
                dst += [b, a]
                w += [float(r.weight), float(r.weight)]
        return src, dst, w

    s1, d1, w1 = _to_edges(subs)
    s2, d2, w2 = _to_edges(cop)
    edge_index = np.array([s1 + s2, d1 + d2], dtype=np.int64)
    edge_weight = np.array(w1 + w2, dtype=float)

    return GraphArrays(
        product_ids=pids.astype(np.int64),
        x=x,
        edge_index=edge_index,
        edge_weight=edge_weight,
        feature_names=list(x_df.columns),
    )
