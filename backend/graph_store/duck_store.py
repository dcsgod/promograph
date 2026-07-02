"""DuckDB-backed graph store - reads the edge tables built by build_edges.py.

Lets the whole pipeline run with no Docker/Neo4j. Same interface as the Neo4j
backend, so switching is a one-line env change (TPO_GRAPH_BACKEND=neo4j).
"""
from __future__ import annotations

from functools import lru_cache

from backend.data_gen import duck
from backend.graph_store.base import (
    GraphEdge,
    GraphNode,
    Neighbor,
    Subgraph,
    sku_id,
)


@lru_cache
def _vocab() -> list[str]:
    """Known product terms (sub_commodity/commodity/brand), longest first, for
    substring matching inside free-text questions."""
    df = duck.query(
        """
        SELECT DISTINCT lower(sub_commodity_desc) AS term FROM product
        UNION SELECT DISTINCT lower(commodity_desc) FROM product
        UNION SELECT DISTINCT lower(brand) FROM product
        """
    )
    return sorted((r.term for r in df.itertuples(index=False) if r.term), key=lambda t: -len(t))


class DuckDBGraphStore:
    def __init__(self) -> None:
        self.con = duck.connect(read_only=True)

    def close(self) -> None:
        self.con.close()

    # -- lookups ---------------------------------------------------------------
    def get_sku(self, product_id: int) -> dict | None:
        df = self.con.execute(
            "SELECT * FROM product WHERE product_id = ?", [product_id]
        ).df()
        return None if df.empty else df.iloc[0].to_dict()

    def find_sku(self, text: str) -> dict | None:
        """Resolve a product reference from short text OR a full sentence.

        Prefers the most specific known product term found *inside* the text
        (so "run 20% off on ground coffee" resolves), then falls back to a
        direct substring match for short references.
        """
        low = text.strip().lower()
        term = next((t for t in _vocab() if t in low), None)
        match = term if term is not None else low
        like = f"%{match}%"
        df = self.con.execute(
            """
            SELECT p.*, COALESCE(s.units, 0) AS total_units
            FROM product p
            LEFT JOIN (
                SELECT product_id, SUM(units) units FROM product_week_sales GROUP BY 1
            ) s USING (product_id)
            WHERE lower(p.brand) LIKE ?
               OR lower(p.commodity_desc) LIKE ?
               OR lower(p.sub_commodity_desc) LIKE ?
               OR lower(p.brand || ' ' || p.sub_commodity_desc) LIKE ?
            ORDER BY total_units DESC
            LIMIT 1
            """,
            [like, like, like, like],
        ).df()
        return None if df.empty else df.iloc[0].to_dict()

    def neighbors(self, product_id: int, relation: str, limit: int) -> list[Neighbor]:
        table = {"SUBSTITUTES": "edge_substitutes", "CO_PURCHASED": "edge_copurchase"}[relation]
        df = self.con.execute(
            f"""
            SELECT e.dst AS product_id, e.weight,
                   p.brand || ' ' || p.sub_commodity_desc AS label
            FROM {table} e
            JOIN product p ON p.product_id = e.dst
            WHERE e.src = ?
            ORDER BY e.weight DESC
            LIMIT ?
            """,
            [product_id, limit],
        ).df()
        return [
            Neighbor(
                product_id=int(r.product_id),
                label=str(r.label),
                weight=float(r.weight),
                relation=relation,
            )
            for r in df.itertuples(index=False)
        ]

    # -- subgraph --------------------------------------------------------------
    def _sku_node(self, row: dict) -> GraphNode:
        return GraphNode(
            id=sku_id(row["product_id"]),
            type="SKU",
            label=f"{row['brand']} {row['sub_commodity_desc']}",
            metrics={
                "product_id": int(row["product_id"]),
                "brand": row["brand"],
                "commodity": row["commodity_desc"],
                "sub_commodity": row["sub_commodity_desc"],
                "base_price": float(row["base_price"]),
                "unit_cost": float(row["unit_cost"]),
            },
        )

    def get_subgraph(
        self, product_id: int, max_substitutes: int = 6, max_copurchase: int = 6
    ) -> Subgraph:
        focus = self.get_sku(product_id)
        if focus is None:
            return Subgraph()
        sg = Subgraph()
        focus_node = self._sku_node(focus)
        focus_node.metrics["focus"] = True
        sg.nodes.append(focus_node)

        # category
        cat = focus["commodity_desc"]
        sg.nodes.append(GraphNode(id=f"cat:{cat}", type="Category", label=cat))
        sg.edges.append(
            GraphEdge(source=focus_node.id, target=f"cat:{cat}", type="IN_CATEGORY")
        )

        # substitutes (cannibalization candidates) + co-purchase (halo candidates)
        for rel, limit in (("SUBSTITUTES", max_substitutes), ("CO_PURCHASED", max_copurchase)):
            for nb in self.neighbors(product_id, rel, limit):
                row = self.get_sku(nb.product_id)
                if row is None:
                    continue
                sg.nodes.append(self._sku_node(row))
                sg.edges.append(
                    GraphEdge(
                        source=focus_node.id,
                        target=sku_id(nb.product_id),
                        type=rel,
                        weight=nb.weight,
                    )
                )

        # promo events this SKU participates in (cap to a few most recent)
        promos = self.con.execute(
            """
            SELECT promo_id, campaign, week_no, discount_depth
            FROM edge_promoted WHERE product_id = ?
            ORDER BY week_no DESC LIMIT 3
            """,
            [product_id],
        ).df()
        for r in promos.itertuples(index=False):
            pid = f"promo:{int(r.promo_id)}"
            sg.nodes.append(
                GraphNode(
                    id=pid,
                    type="PromoEvent",
                    label=f"Wk{int(r.week_no)} -{int(r.discount_depth*100)}%",
                    metrics={"discount_depth": float(r.discount_depth), "week_no": int(r.week_no)},
                )
            )
            sg.edges.append(
                GraphEdge(source=focus_node.id, target=pid, type="PROMOTED_IN", weight=float(r.discount_depth))
            )
        return sg
