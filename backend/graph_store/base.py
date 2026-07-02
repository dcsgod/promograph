"""Graph store abstraction: a Subgraph model + a backend-agnostic interface.

Two backends implement it identically:
  * DuckDBGraphStore  - reads the edge tables (no Docker required). Default.
  * Neo4jGraphStore   - queries a running Neo4j (faithful to the architecture).

Node id conventions (stable, used by the frontend):
  sku:<product_id>  cat:<commodity>  sub:<sub_commodity>
  promo:<promo_id>  seg:<segment>
"""
from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, Field


class GraphNode(BaseModel):
    id: str
    type: str  # SKU | Category | SubCategory | PromoEvent | CustomerSegment
    label: str
    metrics: dict = Field(default_factory=dict)


class GraphEdge(BaseModel):
    source: str
    target: str
    type: str  # SUBSTITUTES | CO_PURCHASED | IN_CATEGORY | PROMOTED_IN | BUYS
    weight: float = 1.0


class Subgraph(BaseModel):
    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)

    def merge(self, other: "Subgraph") -> "Subgraph":
        seen_n = {n.id for n in self.nodes}
        seen_e = {(e.source, e.target, e.type) for e in self.edges}
        for n in other.nodes:
            if n.id not in seen_n:
                self.nodes.append(n)
                seen_n.add(n.id)
        for e in other.edges:
            key = (e.source, e.target, e.type)
            if key not in seen_e:
                self.edges.append(e)
                seen_e.add(key)
        return self


def sku_id(pid: int | str) -> str:
    return f"sku:{pid}"


class Neighbor(BaseModel):
    product_id: int
    label: str
    weight: float
    relation: str  # SUBSTITUTES | CO_PURCHASED


class GraphStore(Protocol):
    def get_sku(self, product_id: int) -> dict | None: ...

    def find_sku(self, text: str) -> dict | None:
        """Best-effort resolve a free-text product reference to a SKU row."""
        ...

    def neighbors(self, product_id: int, relation: str, limit: int) -> list[Neighbor]: ...

    def get_subgraph(
        self, product_id: int, max_substitutes: int = 6, max_copurchase: int = 6
    ) -> Subgraph: ...

    def close(self) -> None: ...


def get_graph_store() -> GraphStore:
    from backend.config import settings

    if settings.tpo_graph_backend == "neo4j":
        from backend.graph_store.neo4j_store import Neo4jGraphStore

        return Neo4jGraphStore()
    from backend.graph_store.duck_store import DuckDBGraphStore

    return DuckDBGraphStore()
