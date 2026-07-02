"""Train the LightGBM baseline-vs-promo-lift model (predicts weekly units).

Saves the booster + a JSON sidecar (feature columns, commodity categories) so
prediction reconstructs identical feature rows.

Run:  python -m backend.forecasting.train_lgbm
"""
from __future__ import annotations

import json

import lightgbm as lgb
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from backend.config import settings
from backend.forecasting.features import build_feature_frame

SEED = 42


def main() -> None:
    df, feature_cols, cats = build_feature_frame()
    print(f"Training frame: {len(df):,} rows, {len(feature_cols)} features")

    X = df[feature_cols]
    y = df["units"].astype(float)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED)

    model = lgb.LGBMRegressor(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=SEED,
        verbose=-1,
    )
    model.fit(Xtr, ytr, eval_set=[(Xte, yte)], eval_metric="l1",
              callbacks=[lgb.early_stopping(30, verbose=False)])

    pred = model.predict(Xte)
    print(f"  MAE={mean_absolute_error(yte, pred):.2f}  R2={r2_score(yte, pred):.3f}")

    model.booster_.save_model(str(settings.lgbm_model_path))
    sidecar = settings.lgbm_model_path.with_suffix(".meta.json")
    sidecar.write_text(
        json.dumps({"feature_cols": feature_cols, "commodity_categories": cats}, indent=2),
        encoding="utf-8",
    )
    print(f"  saved model -> {settings.lgbm_model_path}")

    # --- sanity: units should rise monotonically with own discount depth ---
    row = df[df["on_promo"] == 0].iloc[0:1][feature_cols].copy()
    units_at = []
    for depth in (0.0, 0.1, 0.2, 0.3, 0.4):
        r = row.copy()
        r["discount_depth"] = depth
        units_at.append(round(float(model.predict(r)[0]), 1))
    print(f"  sanity lift by depth [0,.1,.2,.3,.4] units: {units_at}")


if __name__ == "__main__":
    main()
