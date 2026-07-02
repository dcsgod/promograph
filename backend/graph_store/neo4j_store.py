"""Neo4j-backed graph store. Same interface as DuckDBGraphStore."""
from __future__ import annotations

from backend.graph_store.base import (
    GraphEdge,
    GraphNode,
    Neighbor,
    Subgraph,
    sku_id,
)
from backend.graph_store.driver import get_driver


class Neo4jGraphStore:
    def __init__(self) -> None:
        self.driver = get_driver()

    def close(self) -> None:  # driver is a shared singleton; don't hard-close here
        pass

    def _run(self, cypher: str, **params):
        with self.driver.session() as session:
            return list(session.run(cypher, **params))

    def get_sku(self, product_id: int) -> dict | None:
        rows = self._run(
            "MATCH (s:SKU {product_id: $pid}) RETURN s", pid=product_id
        )
        return dict(rows[0]["s"]) if rows else None

    def find_sku(self, text: str) -> dict | None:
        rows = self._run(
            """
            MATCH (s:SKU)
            WHERE toLower(s.brand) CONTAINS $q
               OR toLower(s.commodity) CONTAINS $q
               OR toLower(s.sub_commodity) CONTAINS $q
            RETURN s ORDER BY s.total_units DESC LIMIT 1
            """,
            q=text.strip().lower(),
        )
        return dict(rows[0]["s"]) if rows else None

    def neighbors(self, product_id: int, relation: str, limit: int) -> list[Neighbor]:
        rows = self._run(
            f"""
            MATCH (s:SKU {{product_id: $pid}})-[r:{relation}]->(n:SKU)
            RETURN n.product_id AS pid, n.brand AS brand,
                   n.sub_commodity AS sub, r.weight AS w
            ORDER BY r.weight DESC LIMIT $limit
            """,
            pid=product_id,
            limit=limit,
        )
        return [
            Neighbor(
                product_id=int(r["pid"]),
                label=f"{r['brand']} {r['sub']}",
                weight=float(r["w"] or 1.0),
                relation=relation,
            )
            for r in rows
        ]

    def _sku_node(self, s: dict, focus: bool = False) -> GraphNode:
        return GraphNode(
            id=sku_id(s["product_id"]),
            type="SKU",
            label=f"{s['brand']} {s['sub_commodity']}",
            metrics={**s, "focus": focus},
        )

    def get_subgraph(
        self, product_id: int, max_substitutes: int = 6, max_copurchase: int = 6
    ) -> Subgraph:
        focus = self.get_sku(product_id)
        if focus is None:
            return Subgraph()
        sg = Subgraph(nodes=[self._sku_node(focus, focus=True)])
        cat = focus["commodity"]
        sg.nodes.append(GraphNode(id=f"cat:{cat}", type="Category", label=cat))
        sg.edges.append(GraphEdge(source=sku_id(product_id), target=f"cat:{cat}", type="IN_CATEGORY"))
        for rel, limit in (("SUBSTITUTES", max_substitutes), ("CO_PURCHASED", max_copurchase)):
            for nb in self.neighbors(product_id, rel, limit):
                row = self.get_sku(nb.product_id)
                if row:
                    sg.nodes.append(self._sku_node(row))
                    sg.edges.append(
                        GraphEdge(source=sku_id(product_id), target=sku_id(nb.product_id), type=rel, weight=nb.weight)
                    )
        promos = self._run(
            """
            MATCH (s:SKU {product_id: $pid})-[r:PROMOTED_IN]->(p:PromoEvent)
            RETURN p.promo_id AS promo_id, p.week_no AS week_no,
                   p.discount_depth AS depth ORDER BY p.week_no DESC LIMIT 3
            """,
            pid=product_id,
        )
        for r in promos:
            pid = f"promo:{int(r['promo_id'])}"
            sg.nodes.append(
                GraphNode(
                    id=pid, type="PromoEvent",
                    label=f"Wk{int(r['week_no'])} -{int(r['depth']*100)}%",
                    metrics={"discount_depth": float(r["depth"]), "week_no": int(r["week_no"])},
                )
            )
            sg.edges.append(GraphEdge(source=sku_id(product_id), target=pid, type="PROMOTED_IN", weight=float(r["depth"])))
        return sg
