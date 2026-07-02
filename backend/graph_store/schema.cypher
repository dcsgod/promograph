// TPO knowledge-graph schema (Neo4j). Kept in sync with docs/schema.md.
// Constraints double as indexes on the key property.

CREATE CONSTRAINT sku_id IF NOT EXISTS
  FOR (s:SKU) REQUIRE s.product_id IS UNIQUE;

CREATE CONSTRAINT category_name IF NOT EXISTS
  FOR (c:Category) REQUIRE c.name IS UNIQUE;

CREATE CONSTRAINT subcategory_name IF NOT EXISTS
  FOR (sc:SubCategory) REQUIRE sc.name IS UNIQUE;

CREATE CONSTRAINT segment_name IF NOT EXISTS
  FOR (seg:CustomerSegment) REQUIRE seg.name IS UNIQUE;

CREATE CONSTRAINT promo_id IF NOT EXISTS
  FOR (p:PromoEvent) REQUIRE p.promo_id IS UNIQUE;

// Helpful secondary indexes
CREATE INDEX sku_brand IF NOT EXISTS FOR (s:SKU) ON (s.brand);
CREATE INDEX sku_commodity IF NOT EXISTS FOR (s:SKU) ON (s.commodity);
