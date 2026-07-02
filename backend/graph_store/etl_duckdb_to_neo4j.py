"""Load the DuckDB warehouse + edge tables into Neo4j.

Neo4j is treated as a *rebuildable cache* of the DuckDB mirror, not the system
of record. Run after `build_edges`. Requires a running Neo4j
(`make neo4j-up`) and TPO_GRAPH_BACKEND=neo4j to actually be used at query time.

Run:  python -m backend.graph_store.etl_duckdb_to_neo4j
"""
from __future__ import annotations

from backend.data_gen import duck
from backend.graph_store.build_edges import build as build_edges
from backend.graph_store.driver import apply_schema, close_driver, get_driver


def _rows(con, sql: str) -> list[dict]:
    return con.execute(sql).df().to_dict("records")


def main() -> None:
    con = duck.connect(read_only=False)
    try:
        # ensure edge tables exist / are fresh
        build_edges(con)
        products = _rows(
            con,
            """
            SELECT p.product_id, p.brand, p.commodity_desc AS commodity,
                   p.sub_commodity_desc AS sub_commodity, p.department,
                   p.base_price, p.unit_cost,
                   COALESCE(s.units, 0) AS total_units
            FROM product p
            LEFT JOIN (SELECT product_id, SUM(units) units FROM product_week_sales GROUP BY 1) s
              USING (product_id)
            """,
        )
        categories = _rows(con, "SELECT DISTINCT commodity_desc AS name FROM product")
        segments = _rows(con, "SELECT DISTINCT segment AS name FROM hh_demographic")
        promos = _rows(con, "SELECT DISTINCT promo_id, campaign, week_no, discount_depth FROM edge_promoted")
        subs = _rows(con, "SELECT src, dst, weight FROM edge_substitutes")
        cop = _rows(con, "SELECT src, dst, weight FROM edge_copurchase")
        promoted = _rows(con, "SELECT product_id, promo_id FROM edge_promoted")
        buys = _rows(con, "SELECT segment, product_id, weight FROM edge_buys")
    finally:
        con.close()

    apply_schema()
    driver = get_driver()
    with driver.session() as s:
        print("Wiping existing graph...")
        s.run("MATCH (n) DETACH DELETE n")

        print(f"Nodes: {len(products)} SKU, {len(categories)} Category, "
              f"{len(segments)} Segment, {len(promos)} PromoEvent")
        s.run(
            """
            UNWIND $rows AS r
            MERGE (x:SKU {product_id: r.product_id})
            SET x += r
            """,
            rows=products,
        )
        s.run("UNWIND $rows AS r MERGE (:Category {name: r.name})", rows=categories)
        s.run("UNWIND $rows AS r MERGE (:CustomerSegment {name: r.name})", rows=segments)
        s.run(
            "UNWIND $rows AS r MERGE (p:PromoEvent {promo_id: r.promo_id}) SET p += r",
            rows=promos,
        )

        print(f"Edges: {len(subs)} SUBSTITUTES, {len(cop)} CO_PURCHASED, "
              f"{len(promoted)} PROMOTED_IN, {len(buys)} BUYS")
        s.run(
            """
            UNWIND $rows AS r
            MATCH (a:SKU {product_id: r.src}), (b:SKU {product_id: r.dst})
            MERGE (a)-[e:SUBSTITUTES]->(b) SET e.weight = r.weight
            """,
            rows=subs,
        )
        s.run(
            """
            UNWIND $rows AS r
            MATCH (a:SKU {product_id: r.src}), (b:SKU {product_id: r.dst})
            MERGE (a)-[e:CO_PURCHASED]->(b) SET e.weight = r.weight
            """,
            rows=cop,
        )
        # every SKU belongs to its category
        s.run(
            """
            MATCH (s:SKU), (c:Category {name: s.commodity})
            MERGE (s)-[:IN_CATEGORY]->(c)
            """
        )
        s.run(
            """
            UNWIND $rows AS r
            MATCH (s:SKU {product_id: r.product_id}), (p:PromoEvent {promo_id: r.promo_id})
            MERGE (s)-[:PROMOTED_IN]->(p)
            """,
            rows=promoted,
        )
        s.run(
            """
            UNWIND $rows AS r
            MATCH (seg:CustomerSegment {name: r.segment}), (s:SKU {product_id: r.product_id})
            MERGE (seg)-[e:BUYS]->(s) SET e.weight = r.weight
            """,
            rows=buys,
        )
    close_driver()
    print("Neo4j graph loaded.")


if __name__ == "__main__":
    main()
