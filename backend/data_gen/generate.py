"""Synthetic data generator shaped like dunnhumby "The Complete Journey".

Produces a small but *structurally realistic* retail dataset with latent
cannibalization and halo effects baked into the transactions, so that the
downstream GNN + LightGBM can actually recover them.

Design of the latent structure (this is the whole point):
  * SUBSTITUTES  - products in the same sub_commodity compete. When one is
    promoted, its own units rise (lift) but its substitutes lose units
    (cannibalization / negative cross-price effect).
  * COMPLEMENTS  - designated category pairs (e.g. PASTA <-> PASTA SAUCE) pull
    each other. Promoting one lifts the other (halo).
Both effects are generated at the *transaction* level, so CO_PURCHASED edges
can be mined from real baskets and cross-price effects appear in the facts.

Tables written to DuckDB (names mirror dunnhumby where possible):
  product, hh_demographic, campaign_desc, promo_events, transaction_data,
  and a derived weekly fact view `product_week_sales`.

Run:  python -m backend.data_gen.generate
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backend.config import settings
from backend.data_gen import duck

SEED = 42
N_HOUSEHOLDS = 500
N_WEEKS = 52
SHOP_PROB = 0.62           # probability a household shops in a given week
BASE_BUY_RATE = 0.09       # per-product base purchase prob (tunes basket size)
PROMO_ELASTICITY = 3.4     # own-promo lift sensitivity to discount depth
CANNIB_COEF = 1.2          # how strongly a substitute's promo steals demand
HALO_COEF = 1.1            # how strongly a complement's promo boosts demand
BASE_MARGIN = 0.45         # default gross margin => unit_cost = price*(1-margin)

# --- Category hierarchy: department -> commodity -> [sub_commodities] ---------
HIERARCHY: dict[str, dict[str, list[str]]] = {
    "GROCERY": {
        "PASTA": ["DRY PASTA", "FRESH PASTA"],
        "PASTA SAUCE": ["RED SAUCE", "PESTO SAUCE"],
        "CEREAL": ["KIDS CEREAL", "ADULT CEREAL", "GRANOLA"],
        "SOUP": ["CANNED SOUP", "READY SOUP"],
        "COFFEE": ["GROUND COFFEE", "COFFEE PODS"],
    },
    "DAIRY": {
        "MILK": ["WHOLE MILK", "SKIM MILK", "OAT MILK"],
        "YOGURT": ["GREEK YOGURT", "KIDS YOGURT", "LOWFAT YOGURT"],
        "CHEESE": ["CHEDDAR", "MOZZARELLA"],
    },
    "BEVERAGE": {
        "SODA": ["COLA", "LEMON LIME", "DIET COLA"],
        "JUICE": ["ORANGE JUICE", "APPLE JUICE"],
        "WATER": ["STILL WATER", "SPARKLING WATER"],
    },
    "SNACKS": {
        "CHIPS": ["POTATO CHIPS", "TORTILLA CHIPS"],
        "COOKIES": ["SANDWICH COOKIES", "CHOCOLATE COOKIES"],
    },
}

# Complement pairs at the COMMODITY level (drive the halo effect).
COMPLEMENTS: list[tuple[str, str]] = [
    ("PASTA", "PASTA SAUCE"),
    ("CEREAL", "MILK"),
    ("CHIPS", "SODA"),
    ("COFFEE", "MILK"),
    ("COOKIES", "MILK"),
]

BRANDS = ["Acme", "ValueCo", "Premia", "HouseBrand", "Nortons", "GoldLeaf"]
MANUFACTURERS = ["MFG-A", "MFG-B", "MFG-C", "MFG-D"]
INCOME = ["Under 35K", "35-74K", "75-99K", "100K+"]
AGE = ["19-34", "35-54", "55+"]


def _rng() -> np.random.Generator:
    return np.random.default_rng(SEED)


def build_products(rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    pid = 1000
    for dept, commodities in HIERARCHY.items():
        for commodity, subs in commodities.items():
            for sub in subs:
                # 2-4 competing SKUs per sub_commodity => substitutes
                n = int(rng.integers(2, 5))
                for _ in range(n):
                    brand = BRANDS[int(rng.integers(0, len(BRANDS)))]
                    base_price = round(float(rng.uniform(1.5, 9.5)), 2)
                    unit_cost = round(base_price * (1 - BASE_MARGIN), 2)
                    popularity = float(rng.uniform(0.2, 1.0))
                    # inventory on hand + weekly holding/spoilage cost per unit.
                    # A subset are deliberately overstocked (clearance candidates).
                    weeks_cover = float(rng.uniform(3, 16))
                    inventory = int(200 * popularity * weeks_cover)
                    rows.append(
                        dict(
                            product_id=pid,
                            department=dept,
                            commodity_desc=commodity,
                            sub_commodity_desc=sub,
                            brand=brand,
                            manufacturer=MANUFACTURERS[int(rng.integers(0, len(MANUFACTURERS)))],
                            curr_size_of_product=f"{int(rng.integers(1, 12))*100}g",
                            base_price=base_price,
                            unit_cost=unit_cost,
                            popularity=popularity,
                            inventory=inventory,
                            holding_cost=round(unit_cost * 0.02, 3),
                        )
                    )
                    pid += 1
    return pd.DataFrame(rows)


def build_households(rng: np.random.Generator) -> pd.DataFrame:
    hh = pd.DataFrame(
        dict(
            household_key=np.arange(1, N_HOUSEHOLDS + 1),
            age_desc=rng.choice(AGE, N_HOUSEHOLDS),
            income_desc=rng.choice(INCOME, N_HOUSEHOLDS),
            household_size_desc=rng.integers(1, 6, N_HOUSEHOLDS).astype(str),
        )
    )
    # Coarse segment used as a graph node (CustomerSegment).
    hh["segment"] = hh["income_desc"].map(
        {"Under 35K": "Budget", "35-74K": "Mainstream", "75-99K": "Premium", "100K+": "Affluent"}
    )
    return hh


def build_promo_schedule(
    products: pd.DataFrame, rng: np.random.Generator
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (campaign_desc, promo_events).

    promo_events: one row per (product_id, week) that is on promo, with a
    discount_depth in [0.1, 0.4]. ~6 products promoted per week, grouped into
    themed campaigns so PromoEvent nodes have meaning.
    """
    campaigns = []
    promos = []
    promo_id = 1
    for week in range(1, N_WEEKS + 1):
        n_promo = int(rng.integers(4, 8))
        chosen = products.sample(n=n_promo, random_state=int(rng.integers(0, 1_000_000)))
        campaign_id = week  # one campaign per week for simplicity
        start_day = (week - 1) * 7 + 1
        end_day = week * 7
        depts = ", ".join(sorted(chosen["department"].unique()))
        campaigns.append(
            dict(
                campaign=campaign_id,
                description=f"Week {week} feature: {depts}",
                start_day=start_day,
                end_day=end_day,
                week_no=week,
            )
        )
        for _, prod in chosen.iterrows():
            depth = round(float(rng.uniform(0.10, 0.40)), 2)
            promos.append(
                dict(
                    promo_id=promo_id,
                    campaign=campaign_id,
                    product_id=int(prod["product_id"]),
                    commodity_desc=prod["commodity_desc"],
                    sub_commodity_desc=prod["sub_commodity_desc"],
                    week_no=week,
                    start_day=start_day,
                    end_day=end_day,
                    discount_depth=depth,
                )
            )
            promo_id += 1
    return pd.DataFrame(campaigns), pd.DataFrame(promos)


def simulate_transactions(
    products: pd.DataFrame,
    households: pd.DataFrame,
    promos: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generate basket-level transactions with baked-in cannibalization & halo."""
    prod = products.reset_index(drop=True)
    n_prod = len(prod)
    pid_to_idx = {int(p): i for i, p in enumerate(prod["product_id"])}
    popularity = prod["popularity"].to_numpy()
    base_price = prod["base_price"].to_numpy()
    commodity = prod["commodity_desc"].to_numpy()
    sub = prod["sub_commodity_desc"].to_numpy()

    # complement lookup: commodity -> set of complementary commodities
    comp_map: dict[str, set[str]] = {}
    for a, b in COMPLEMENTS:
        comp_map.setdefault(a, set()).add(b)
        comp_map.setdefault(b, set()).add(a)

    # Precompute per-week promo vectors.
    promo_by_week: dict[int, dict[str, np.ndarray | dict]] = {}
    for week in range(1, N_WEEKS + 1):
        wk = promos[promos["week_no"] == week]
        depth_vec = np.zeros(n_prod)
        for _, r in wk.iterrows():
            depth_vec[pid_to_idx[int(r["product_id"])]] = r["discount_depth"]
        # substitute pressure: for each product, total promo depth of OTHER
        # products in the same sub_commodity that are on promo this week.
        sub_pressure = np.zeros(n_prod)
        promoted_commodities: dict[str, float] = {}
        for i in range(n_prod):
            if depth_vec[i] > 0:
                promoted_commodities[commodity[i]] = max(
                    promoted_commodities.get(commodity[i], 0.0), depth_vec[i]
                )
        for i in range(n_prod):
            same_sub = (sub == sub[i])
            same_sub[i] = False
            sub_pressure[i] = depth_vec[same_sub].sum()
        # complement boost: max promo depth among complementary commodities
        comp_boost = np.zeros(n_prod)
        for i in range(n_prod):
            best = 0.0
            for c in comp_map.get(commodity[i], ()):
                best = max(best, promoted_commodities.get(c, 0.0))
            comp_boost[i] = best
        promo_by_week[week] = dict(
            depth=depth_vec, sub_pressure=sub_pressure, comp_boost=comp_boost
        )

    # household x commodity affinity (stable preferences)
    commodities = sorted(set(commodity))
    comm_idx = {c: i for i, c in enumerate(commodities)}
    hh_aff = rng.uniform(0.3, 1.0, size=(len(households), len(commodities)))
    prod_comm_idx = np.array([comm_idx[c] for c in commodity])

    rows: list[dict] = []
    basket_id = 1
    store_ids = [f"S{n:02d}" for n in range(1, 6)]

    for h_pos, hh_row in enumerate(households.itertuples(index=False)):
        aff_row = hh_aff[h_pos][prod_comm_idx]  # affinity per product
        for week in range(1, N_WEEKS + 1):
            if rng.random() > SHOP_PROB:
                continue
            pw = promo_by_week[week]
            day = (week - 1) * 7 + int(rng.integers(1, 8))
            # base purchase probability per product for this basket
            p = BASE_BUY_RATE * popularity * aff_row
            p *= 1 + PROMO_ELASTICITY * pw["depth"]              # own promo lift
            p *= np.clip(1 - CANNIB_COEF * pw["sub_pressure"], 0.05, None)  # cannibalization
            p *= 1 + HALO_COEF * pw["comp_boost"]                # halo
            p = np.clip(p, 0, 0.95)
            bought = rng.random(n_prod) < p
            idxs = np.nonzero(bought)[0]
            if idxs.size == 0:
                continue
            store = store_ids[int(rng.integers(0, len(store_ids)))]
            for i in idxs:
                depth = pw["depth"][i]
                qty = 1 + int(rng.poisson(0.6 + 1.5 * depth))
                price_paid = base_price[i] * (1 - depth)
                sales_value = round(qty * price_paid, 2)
                retail_disc = round(qty * base_price[i] * depth, 2)
                rows.append(
                    dict(
                        household_key=int(hh_row.household_key),
                        basket_id=basket_id,
                        day=day,
                        week_no=week,
                        product_id=int(prod["product_id"].iloc[i]),
                        quantity=qty,
                        sales_value=sales_value,
                        retail_disc=retail_disc,
                        store_id=store,
                    )
                )
            basket_id += 1

    return pd.DataFrame(rows)


def main() -> None:
    rng = _rng()
    print("Generating products, households, promo schedule...")
    products = build_products(rng)
    households = build_households(rng)
    campaign_desc, promo_events = build_promo_schedule(products, rng)

    print(f"Simulating transactions ({N_HOUSEHOLDS} households x {N_WEEKS} weeks)...")
    tx = simulate_transactions(products, households, promo_events, rng)
    print(f"  -> {len(tx):,} transaction rows, {tx['basket_id'].nunique():,} baskets")

    settings.duckdb_path.parent.mkdir(parents=True, exist_ok=True)
    con = duck.connect(read_only=False)
    try:
        duck.write_table(con, "product", products)
        duck.write_table(con, "hh_demographic", households)
        duck.write_table(con, "campaign_desc", campaign_desc)
        duck.write_table(con, "promo_events", promo_events)
        duck.write_table(con, "transaction_data", tx)

        # Derived weekly fact table used by forecasting + mock Genie.
        con.execute("DROP TABLE IF EXISTS product_week_sales")
        con.execute(
            """
            CREATE TABLE product_week_sales AS
            SELECT
                t.product_id,
                t.week_no,
                p.commodity_desc,
                p.sub_commodity_desc,
                p.brand,
                SUM(t.quantity)                          AS units,
                SUM(t.sales_value)                       AS revenue,
                SUM(t.retail_disc)                       AS discount_dollars,
                COUNT(DISTINCT t.basket_id)              AS baskets,
                COALESCE(MAX(pe.discount_depth), 0.0)    AS discount_depth,
                CASE WHEN MAX(pe.discount_depth) IS NOT NULL THEN 1 ELSE 0 END AS on_promo
            FROM transaction_data t
            JOIN product p USING (product_id)
            LEFT JOIN promo_events pe
                   ON pe.product_id = t.product_id AND pe.week_no = t.week_no
            GROUP BY 1,2,3,4,5
            """
        )
        n = con.execute("SELECT COUNT(*) FROM product_week_sales").fetchone()[0]
        print(f"  -> product_week_sales: {n:,} rows")
    finally:
        con.close()

    print(f"Done. DuckDB written to {settings.duckdb_path}")


if __name__ == "__main__":
    main()
