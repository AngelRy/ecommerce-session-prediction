#!/usr/bin/env python3
"""
Train a session-level value prediction model (total_spent per session).

Outputs:
 - aggregated session parquet files (if missing) in ./data/aggregated/
 - trained model saved to ./models/value_model.joblib
 - feature list saved to ./models/features.txt
"""

import os
import sys
import time
import joblib
import numpy as np
import pandas as pd

# try to import lightgbm, otherwise fallback to sklearn
try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except Exception:
    LGB_AVAILABLE = False
    from sklearn.ensemble import RandomForestRegressor

import duckdb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# -------------------------
# CONFIG
# -------------------------
DATA_DIR = "./data"
AGG_DIR = os.path.join(DATA_DIR, "aggregated")
os.makedirs(AGG_DIR, exist_ok=True)

OCT_EVENTS = os.path.join(DATA_DIR, "2019-Oct_optimized.parquet")
NOV_EVENTS = os.path.join(DATA_DIR, "2019-Nov_optimized.parquet")

OCT_SESSIONS = os.path.join(AGG_DIR, "2019-Oct_sessions.parquet")
NOV_SESSIONS = os.path.join(AGG_DIR, "2019-Nov_sessions.parquet")

MODEL_DIR = "./models"
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_OUT = os.path.join(MODEL_DIR, "value_model.joblib")
FEATURES_OUT = os.path.join(MODEL_DIR, "features.txt")

# maximum sessions to sample for training (keeps it safe for laptop)
SAMPLE_SESSIONS = 100_000

RANDOM_STATE = 42

# -------------------------
# UTIL: create session-level aggregates via DuckDB
# -------------------------
def create_session_aggregates(events_parquet, out_parquet, chunk_label=""):
    """
    Uses DuckDB to aggregate event-level data into session-level features.
    Produces out_parquet if missing.
    """
    if os.path.exists(out_parquet):
        print(f"[{chunk_label}] Aggregated file exists: {out_parquet}")
        return

    print(f"[{chunk_label}] Creating aggregated sessions → {out_parquet}")
    con = duckdb.connect()

    # This query computes common session-level features:
    # - session_start, session_end, session_duration_seconds
    # - counts of event types (views, add_to_cart, purchase)
    # - total_spent (sum price for purchase events)
    # - avg/max price of viewed items (helpful feature)
    # - unique_products per session
    query = f"""
    WITH events AS (
        SELECT *
        FROM read_parquet('{events_parquet}')
    )
    SELECT
        user_session,
        MIN(CAST(event_time AS TIMESTAMP)) AS session_start,
        MAX(CAST(event_time AS TIMESTAMP)) AS session_end,
        EXTRACT('epoch' FROM MAX(CAST(event_time AS TIMESTAMP)) - MIN(CAST(event_time AS TIMESTAMP))) AS session_duration,
        COUNT(*) AS total_events,
        SUM(CASE WHEN event_type = 'view' THEN 1 ELSE 0 END) AS n_views,
        SUM(CASE WHEN event_type = 'add_to_cart' THEN 1 ELSE 0 END) AS n_add_to_cart,
        SUM(CASE WHEN event_type = 'purchase' THEN 1 ELSE 0 END) AS n_purchases,
        SUM(CASE WHEN event_type = 'purchase' THEN price ELSE 0 END) AS total_spent,
        AVG(CASE WHEN event_type = 'view' THEN price ELSE NULL END) AS avg_price_viewed,
        MAX(CASE WHEN event_type = 'view' THEN price ELSE NULL END) AS max_price_viewed,
        COUNT(DISTINCT product_id) AS unique_products
    FROM events
    GROUP BY user_session
    """
    con.execute(query)
    # Directly run aggregation without triggering full read beforehand
    df_agg = con.execute(query).fetchdf()

    # Ensure types are sane
    df_agg["session_duration"] = df_agg["session_duration"].fillna(0).astype(float)
    df_agg["n_views"] = df_agg["n_views"].fillna(0).astype(int)
    df_agg["n_add_to_cart"] = df_agg["n_add_to_cart"].fillna(0).astype(int)
    df_agg["n_purchases"] = df_agg["n_purchases"].fillna(0).astype(int)
    df_agg["total_spent"] = df_agg["total_spent"].fillna(0.0).astype(float)
    df_agg["unique_products"] = df_agg["unique_products"].fillna(0).astype(int)
    # write out
    df_agg.to_parquet(out_parquet, index=False)
    con.close()
    print(f"[{chunk_label}] Saved aggregated sessions: {out_parquet} (rows: {len(df_agg)})")


# -------------------------
# LOAD AGGREGATES (or create if missing)
# -------------------------
def load_or_create_aggregates():
    # create if missing
    if not os.path.exists(OCT_SESSIONS):
        create_session_aggregates(OCT_EVENTS, OCT_SESSIONS, "OCT")
    if not os.path.exists(NOV_SESSIONS):
        create_session_aggregates(NOV_EVENTS, NOV_SESSIONS, "NOV")

    # read both
    df_oct = pd.read_parquet(OCT_SESSIONS)
    df_nov = pd.read_parquet(NOV_SESSIONS)
    # combine
    df = pd.concat([df_oct, df_nov], ignore_index=True)
    print(f"Loaded aggregated sessions: {len(df)} rows (Oct: {len(df_oct)}, Nov: {len(df_nov)})")
    return df


# -------------------------
# Feature engineering (session-level)
# -------------------------
def featurize(df):
    df = df.copy()
    # basic features already present
    # Add derived features:
    df["session_duration_min"] = df["session_duration"] / 60.0
    # view/cart ratios (avoid div-by-zero)
    df["views_per_product"] = df["n_views"] / (df["unique_products"].replace(0, 1))
    df["cart_rate"] = df["n_add_to_cart"] / (df["n_views"].replace(0, 1))
    # purchase flag
    df["purchase_flag"] = (df["n_purchases"] > 0).astype(int)
    # price-based features: avg_price_viewed (already), max_price_viewed
    df["avg_price_viewed"] = df["avg_price_viewed"].fillna(0.0)
    df["max_price_viewed"] = df["max_price_viewed"].fillna(0.0)

    # cap extreme session duration (optional)
    df["session_duration_min"] = df["session_duration_min"].clip(upper=24*60)  # 24 hours cap

    # select features (drop identifiers & target)
    features = [
        "total_events",
        "session_duration_min",
        "n_views",
        "n_add_to_cart",
        "unique_products",
        "views_per_product",
        "cart_rate",
        "avg_price_viewed",
        "max_price_viewed"
    ]
    # ensure cols present
    features = [c for c in features if c in df.columns]
    return df, features


# -------------------------
# TRAINING & EVAL
# -------------------------
def train_model(df, features, sample_size=SAMPLE_SESSIONS):
    # target is total_spent; we will model log1p(total_spent) to reduce skew
    df = df.copy()
    df["target"] = df["total_spent"].astype(float)

    # optional sampling to keep it light
    if sample_size is not None and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=RANDOM_STATE)

    X = df[features]
    y = np.log1p(df["target"].values)  # transform

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    print("Training rows:", X_train.shape[0], "Validation rows:", X_test.shape[0])

    if LGB_AVAILABLE:
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_test, label=y_test)
        params = {
            "objective": "regression",
            "metric": "rmse",
            "verbosity": -1,
            "seed": RANDOM_STATE,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_data_in_leaf": 20
        }
        model = lgb.train(params, train_data, num_boost_round=2000, valid_sets=[train_data, val_data],
                          early_stopping_rounds=50, verbose_eval=50)
        # predict (log space)
        y_pred_log = model.predict(X_test, num_iteration=model.best_iteration)
    else:
        # sklearn RandomForest fallback
        print("LightGBM not available — using RandomForestRegressor (slower/memory heavier).")
        rf = RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
        rf.fit(X_train, y_train)
        model = rf
        y_pred_log = model.predict(X_test)

    # back-transform
    y_test_actual = np.expm1(y_test)
    y_pred_actual = np.expm1(y_pred_log)

    # metrics
    rmse = mean_squared_error(y_test_actual, y_pred_actual, squared=False)
    mae = mean_absolute_error(y_test_actual, y_pred_actual)
    r2 = r2_score(y_test_actual, y_pred_actual)

    print("\n=== EVALUATION (on holdout) ===")
    print(f"RMSE : {rmse:,.4f}")
    print(f"MAE  : {mae:,.4f}")
    print(f"R2   : {r2:,.4f}")

    # feature importance (if available)
    try:
        if LGB_AVAILABLE:
            imp = pd.DataFrame({
                "feature": features,
                "importance": model.feature_importance(importance_type="gain")
            }).sort_values("importance", ascending=False)
        else:
            imp = pd.DataFrame({
                "feature": features,
                "importance": model.feature_importances_
            }).sort_values("importance", ascending=False)
        print("\nTop features:\n", imp.head(10).to_string(index=False))
    except Exception as e:
        print("Could not compute feature importance:", e)
        imp = None

    # save model and features
    joblib.dump({"model": model, "features": features, "lgb": LGB_AVAILABLE}, MODEL_OUT)
    with open(FEATURES_OUT, "w") as f:
        f.write("\n".join(features))

    print(f"\nModel saved to: {MODEL_OUT}")
    print(f"Features saved to: {FEATURES_OUT}")

    # optional: plot predicted vs actual scatter for small sample
    try:
        fig, ax = plt.subplots(figsize=(6,6))
        ax.scatter(y_test_actual[:2000], y_pred_actual[:2000], alpha=0.1)
        ax.plot([0, max(y_test_actual[:2000].max(), y_pred_actual[:2000].max())],
                [0, max(y_test_actual[:2000].max(), y_pred_actual[:2000].max())],
                color="red")
        ax.set_xlabel("Actual total_spent")
        ax.set_ylabel("Predicted total_spent")
        ax.set_title("Predicted vs Actual (sample)")
        plt.tight_layout()
        plt.savefig(os.path.join(MODEL_DIR, "pred_vs_actual.png"))
        print(f"Saved predicted vs actual scatter to {os.path.join(MODEL_DIR, 'pred_vs_actual.png')}")
    except Exception as e:
        print("Could not save scatter plot:", e)

    return model, features, imp


# -------------------------
# MAIN
# -------------------------
def main():
    t0 = time.time()
    df = load_or_create_aggregates()
    df, features = featurize(df)
    print("Features used:", features)
    model, features, imp = train_model(df, features)
    print("Done in", round(time.time() - t0, 1), "seconds")

if __name__ == "__main__":
    main()
