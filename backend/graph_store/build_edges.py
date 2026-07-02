"""Compute graph edges from the DuckDB warehouse into edge tables.

These edge tables are the single source of truth consumed by:
  * the GNN (backend/gnn/build_pyg.py),
  * the DuckDB graph store (no Docker), and
  * the Neo4j ETL (backend/graph_store/etl_duckdb_to_neo4j.py).

Edges produced:
  edge_substitutes(src, dst, weight)   -- same sub_commodity, price-similarity weighted
  edge_copurchase(src, dst, weight)    -- basket co-occurrence lift, top-k per node
  edge_promoted(product_id, promo_id, campaign, week_no, discount_depth)
  edge_buys(segment, product_id, weight)

Run:  python -m backend.graph_store.build_edges
"""
from __future__ import annotations

from backend.data_gen import duck

COPURCHASE_MIN_COOC = 8   # min baskets a pair must co-occur in
COPURCHASE_TOPK = 8       # keep top-k co-purchase neighbours per product


def build(con) -> dict[str, int]:
    counts: dict[str, int] = {}

    # --- SUBSTITUTES: same sub_commodity, weighted by price similarity ---------
    con.execute("DROP TABLE IF EXISTS edge_substitutes")
    con.execute(
        """
        CREATE TABLE edge_substitutes AS
        WITH pairs AS (
            SELECT a.product_id AS src, b.product_id AS dst,
                   a.base_price AS pa, b.base_price AS pb
            FROM product a
            JOIN product b
              ON a.sub_commodity_desc = b.sub_commodity_desc
             AND a.product_id <> b.product_id
        ),
        mx AS (SELECT MAX(base_price) m FROM product)
        SELECT src, dst,
               ROUND(GREATEST(0.05, 1 - ABS(pa - pb) / (SELECT m FROM mx)), 4) AS weight
        FROM pairs
        """
    )
    counts["edge_substitutes"] = con.execute("SELECT COUNT(*) FROM edge_substitutes").fetchone()[0]

    # --- CO_PURCHASE: basket co-occurrence lift, thresholded, top-k -----------
    con.execute("DROP TABLE IF EXISTS edge_copurchase")
    con.execute(
        f"""
        CREATE TABLE edge_copurchase AS
        WITH baskets AS (
            SELECT DISTINCT basket_id, product_id FROM transaction_data
        ),
        n AS (SELECT COUNT(DISTINCT basket_id) AS total FROM baskets),
        freq AS (
            SELECT product_id, COUNT(*) AS c FROM baskets GROUP BY 1
        ),
        cooc AS (
            SELECT a.product_id AS src, b.product_id AS dst, COUNT(*) AS c
            FROM baskets a
            JOIN baskets b ON a.basket_id = b.basket_id AND a.product_id < b.product_id
            GROUP BY 1, 2
            HAVING COUNT(*) >= {COPURCHASE_MIN_COOC}
        ),
        lift AS (
            -- symmetric lift, materialise both directions for top-k per node
            SELECT src, dst,
                   (c * (SELECT total FROM n)) /
                   ((SELECT c FROM freq WHERE product_id = src) *
                    (SELECT c FROM freq WHERE product_id = dst)) AS weight
            FROM cooc
        ),
        bidir AS (
            SELECT src, dst, weight FROM lift
            UNION ALL
            SELECT dst AS src, src AS dst, weight FROM lift
        ),
        ranked AS (
            SELECT src, dst, ROUND(weight, 4) AS weight,
                   ROW_NUMBER() OVER (PARTITION BY src ORDER BY weight DESC) AS rk
            FROM bidir
        )
        SELECT src, dst, weight FROM ranked WHERE rk <= {COPURCHASE_TOPK}
        """
    )
    counts["edge_copurchase"] = con.execute("SELECT COUNT(*) FROM edge_copurchase").fetchone()[0]

    # --- PROMOTED_IN: SKU -> PromoEvent ---------------------------------------
    con.execute("DROP TABLE IF EXISTS edge_promoted")
    con.execute(
        """
        CREATE TABLE edge_promoted AS
        SELECT product_id, promo_id, campaign, week_no, discount_depth
        FROM promo_events
        """
    )
    counts["edge_promoted"] = con.execute("SELECT COUNT(*) FROM edge_promoted").fetchone()[0]

    # --- BUYS: CustomerSegment -> SKU (unit share) ----------------------------
    con.execute("DROP TABLE IF EXISTS edge_buys")
    con.execute(
        """
        CREATE TABLE edge_buys AS
        WITH seg AS (
            SELECT t.product_id, h.segment, SUM(t.quantity) AS units
            FROM transaction_data t
            JOIN hh_demographic h USING (household_key)
            GROUP BY 1, 2
        ),
        tot AS (SELECT product_id, SUM(units) AS total FROM seg GROUP BY 1)
        SELECT seg.segment, seg.product_id,
               ROUND(seg.units * 1.0 / tot.total, 4) AS weight
        FROM seg JOIN tot USING (product_id)
        """
    )
    counts["edge_buys"] = con.execute("SELECT COUNT(*) FROM edge_buys").fetchone()[0]

    return counts


def main() -> None:
    con = duck.connect(read_only=False)
    try:
        counts = build(con)
    finally:
        con.close()
    for k, v in counts.items():
        print(f"  {k}: {v:,} rows")
    print("Edge tables built.")


if __name__ == "__main__":
    main()
