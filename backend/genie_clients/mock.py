"""Mock Genie client - maps a natural-language question to canned SQL run
against the local DuckDB mirror, returning the same GenieResult shape as the
real Databricks Genie client.

Intent detection is deliberately simple (keyword based). It covers the query
types the orchestrator actually needs for grounding: baseline sales, top
sellers, promo history, and inventory/weeks-of-supply.
"""
from __future__ import annotations

from functools import lru_cache

from backend.data_gen import duck
from backend.genie_clients.base import GenieResult


@lru_cache
def _vocab() -> list[tuple[str, str]]:
    """(lowercased term, kind) for sub_commodity / commodity / brand, longest first."""
    df = duck.query(
        """
        SELECT DISTINCT lower(sub_commodity_desc) AS term, 'sub' AS kind FROM product
        UNION SELECT DISTINCT lower(commodity_desc), 'commodity' FROM product
        UNION SELECT DISTINCT lower(brand), 'brand' FROM product
        """
    )
    terms = [(r.term, r.kind) for r in df.itertuples(index=False)]
    return sorted(terms, key=lambda t: -len(t[0]))


def _resolve_product(text: str):
    """Resolve by finding the most specific known product term inside the text."""
    q = text.strip().lower()
    match = next((t for t, _ in _vocab() if t and t in q), None)
    if match is None:
        return None
    like = f"%{match}%"
    df = duck.query(
        """
        SELECT p.product_id, p.brand, p.sub_commodity_desc, p.commodity_desc,
               COALESCE(s.units,0) AS total_units
        FROM product p
        LEFT JOIN (SELECT product_id, SUM(units) units FROM product_week_sales GROUP BY 1) s
          USING (product_id)
        WHERE lower(p.sub_commodity_desc) LIKE ?
           OR lower(p.commodity_desc) LIKE ?
           OR lower(p.brand) LIKE ?
        ORDER BY total_units DESC LIMIT 1
        """,
        [like, like, like],
    )
    return None if df.empty else df.iloc[0]


def _df_result(space, question, sql, df, text) -> GenieResult:
    return GenieResult(
        space=space, question=question, sql=sql.strip(),
        columns=list(df.columns), rows=df.to_dict("records"), text=text,
    )


class MockGenieClient:
    def ask(self, question: str, space: str = "sales") -> GenieResult:
        q = question.lower()

        # --- inventory / weeks of supply ---
        if any(k in q for k in ("inventory", "stock", "weeks of supply", "overstock")):
            prod = _resolve_product(q)
            if prod is not None:
                sql = f"""
                    SELECT p.product_id, p.brand, p.sub_commodity_desc,
                           p.inventory,
                           ROUND(p.inventory / NULLIF(w.avg_weekly,0), 1) AS weeks_of_supply
                    FROM product p
                    LEFT JOIN (SELECT product_id, AVG(units) avg_weekly
                               FROM product_week_sales GROUP BY 1) w USING (product_id)
                    WHERE p.product_id = {int(prod.product_id)}
                """
                df = duck.query(sql)
                return _df_result("inventory", question, sql, df,
                                  f"Inventory for {prod.brand} {prod.sub_commodity_desc}.")
            sql = """
                SELECT p.commodity_desc, SUM(p.inventory) AS inventory
                FROM product p GROUP BY 1 ORDER BY inventory DESC
            """
            return _df_result("inventory", question, sql, duck.query(sql),
                              "Inventory on hand by category.")

        # --- promo history ---
        if any(k in q for k in ("promo", "promotion", "discount", "campaign")):
            prod = _resolve_product(q)
            if prod is not None:
                sql = f"""
                    SELECT week_no, campaign, discount_depth
                    FROM promo_events WHERE product_id = {int(prod.product_id)}
                    ORDER BY week_no
                """
                df = duck.query(sql)
                return _df_result("promo", question, sql, df,
                                  f"Promo history for {prod.brand} {prod.sub_commodity_desc}: "
                                  f"{len(df)} events.")
            sql = """
                SELECT commodity_desc, COUNT(*) AS promo_events,
                       ROUND(AVG(discount_depth),3) AS avg_depth
                FROM promo_events pe JOIN product p USING (product_id)
                GROUP BY 1 ORDER BY promo_events DESC
            """
            return _df_result("promo", question, sql, duck.query(sql),
                              "Promo activity by category.")

        # --- top sellers ---
        if any(k in q for k in ("top", "best selling", "best-selling", "bestseller")):
            sql = """
                SELECT p.brand, p.sub_commodity_desc, SUM(s.units) AS units,
                       ROUND(SUM(s.revenue),0) AS revenue
                FROM product_week_sales s JOIN product p USING (product_id)
                GROUP BY 1,2 ORDER BY units DESC LIMIT 10
            """
            return _df_result("sales", question, sql, duck.query(sql),
                              "Top 10 SKUs by units sold.")

        # --- baseline / sales for a specific product ---
        prod = _resolve_product(q)
        if prod is not None:
            sql = f"""
                SELECT week_no, units, ROUND(revenue,0) AS revenue, on_promo, discount_depth
                FROM product_week_sales
                WHERE product_id = {int(prod.product_id)}
                ORDER BY week_no
            """
            df = duck.query(sql)
            base = df.loc[df.on_promo == 0, "units"].mean() if len(df) else 0
            return _df_result("sales", question, sql, df,
                              f"{prod.brand} {prod.sub_commodity_desc}: avg baseline "
                              f"{base:.1f} units/week over {len(df)} weeks.")

        # --- fallback: category sales summary ---
        sql = """
            SELECT p.commodity_desc, SUM(s.units) AS units, ROUND(SUM(s.revenue),0) AS revenue
            FROM product_week_sales s JOIN product p USING (product_id)
            GROUP BY 1 ORDER BY revenue DESC
        """
        return _df_result("sales", question, sql, duck.query(sql),
                          "Total sales by category.")
