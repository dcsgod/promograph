# Graph Schema

Kept in sync with `backend/graph_store/schema.cypher` and `backend/graph_store/build_edges.py`.
Neo4j is a **rebuildable cache** of the DuckDB warehouse, not the system of record.

## Nodes

| Label | Key | Properties |
|---|---|---|
| `SKU` | `product_id` | brand, commodity, sub_commodity, department, base_price, unit_cost, total_units |
| `Category` | `name` | (commodity_desc) |
| `SubCategory` | `name` | (sub_commodity_desc) - reserved for future use |
| `CustomerSegment` | `name` | Budget / Mainstream / Premium / Affluent |
| `PromoEvent` | `promo_id` | campaign, week_no, discount_depth |

Frontend node-id conventions: `sku:<product_id>`, `cat:<commodity>`, `promo:<promo_id>`, `seg:<segment>`.

## Relationships

| Type | From → To | Weight | Meaning |
|---|---|---|---|
| `IN_CATEGORY` | SKU → Category | - | product's commodity |
| `SUBSTITUTES` | SKU → SKU | price similarity | same sub_commodity → **cannibalization** candidates |
| `CO_PURCHASED` | SKU → SKU | basket-co-occurrence lift | **halo** candidates (top-k per node) |
| `PROMOTED_IN` | SKU → PromoEvent | discount_depth | promo participation |
| `BUYS` | CustomerSegment → SKU | unit share | who buys the SKU |

## DuckDB source tables

`product`, `hh_demographic`, `campaign_desc`, `promo_events`, `transaction_data`,
derived `product_week_sales`, and edge tables `edge_substitutes`, `edge_copurchase`,
`edge_promoted`, `edge_buys` (built by `build_edges.py`).

## Rebuild pipeline

```
generate.py / databricks_to_duckdb.py   → DuckDB base tables
build_edges.py                          → edge_* tables
etl_duckdb_to_neo4j.py                  → Neo4j graph (only if TPO_GRAPH_BACKEND=neo4j)
```

## Edge parameters (build_edges.py)

- `COPURCHASE_MIN_COOC = 8` - min baskets a pair must share to form a CO_PURCHASED edge.
- `COPURCHASE_TOPK = 8` - keep the 8 strongest co-purchase neighbours per SKU.
- SUBSTITUTES weight = `max(0.05, 1 − |price_a − price_b| / max_price)`.
