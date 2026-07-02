"""Pull real dunnhumby tables from Databricks Unity Catalog into the local
DuckDB mirror, using the free `databricks-sql-connector`.

This is the drop-in replacement for `backend.data_gen.generate`: it writes the
same table names, so everything downstream (Neo4j ETL, GNN, forecasting, mock
Genie) works unchanged.

Prereqs (.env): DATABRICKS_HOST, DATABRICKS_TOKEN, DATABRICKS_WAREHOUSE_ID,
DATABRICKS_CATALOG, DATABRICKS_SCHEMA. Upload the dunnhumby CSVs to Unity
Catalog first (see docs/schema.md for the expected columns).

Run:  python data/etl/databricks_to_duckdb.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# allow running as a bare script from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.config import settings  # noqa: E402
from backend.data_gen import duck  # noqa: E402

# Map: local DuckDB table name -> Unity Catalog table (relative to catalog.schema)
TABLE_MAP = {
    "product": "product",
    "hh_demographic": "hh_demographic",
    "campaign_desc": "campaign_desc",
    "promo_events": "promo_events",
    "transaction_data": "transaction_data",
}


def _uc(table: str) -> str:
    return f"`{settings.databricks_catalog}`.`{settings.databricks_schema}`.`{table}`"


def main() -> None:
    try:
        from databricks import sql as dbsql
    except ImportError as e:  # pragma: no cover
        raise SystemExit("pip install databricks-sql-connector") from e

    if not (settings.databricks_host and settings.databricks_token and settings.databricks_warehouse_id):
        raise SystemExit("Set DATABRICKS_HOST / DATABRICKS_TOKEN / DATABRICKS_WAREHOUSE_ID in .env")

    host = settings.databricks_host.replace("https://", "").rstrip("/")
    http_path = f"/sql/1.0/warehouses/{settings.databricks_warehouse_id}"

    con = duck.connect(read_only=False)
    try:
        with dbsql.connect(
            server_hostname=host,
            http_path=http_path,
            access_token=settings.databricks_token,
        ) as dbx:
            for local_name, remote in TABLE_MAP.items():
                print(f"Pulling {remote} -> {local_name} ...")
                with dbx.cursor() as cur:
                    cur.execute(f"SELECT * FROM {_uc(remote)}")
                    df = cur.fetchall_arrow().to_pandas()
                df.columns = [c.lower() for c in df.columns]
                duck.write_table(con, local_name, df)
                print(f"  -> {len(df):,} rows")

        # Rebuild the derived weekly fact table (same SQL as the generator).
        con.execute("DROP TABLE IF EXISTS product_week_sales")
        con.execute(
            """
            CREATE TABLE product_week_sales AS
            SELECT t.product_id, t.week_no, p.commodity_desc, p.sub_commodity_desc,
                   p.brand,
                   SUM(t.quantity) AS units, SUM(t.sales_value) AS revenue,
                   SUM(t.retail_disc) AS discount_dollars,
                   COUNT(DISTINCT t.basket_id) AS baskets,
                   COALESCE(MAX(pe.discount_depth), 0.0) AS discount_depth,
                   CASE WHEN MAX(pe.discount_depth) IS NOT NULL THEN 1 ELSE 0 END AS on_promo
            FROM transaction_data t
            JOIN product p USING (product_id)
            LEFT JOIN promo_events pe
                   ON pe.product_id = t.product_id AND pe.week_no = t.week_no
            GROUP BY 1,2,3,4,5
            """
        )
    finally:
        con.close()
    print(f"Done. DuckDB mirror at {settings.duckdb_path}")


if __name__ == "__main__":
    main()
