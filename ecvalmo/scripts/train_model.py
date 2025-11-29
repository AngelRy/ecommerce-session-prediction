#!/usr/bin/env python3
"""
Memory-safe, leakage-free training pipeline (Option A).

- Predicts whether a session will result in purchase (binary).
- Uses only information available BEFORE knowing the purchase:
  n_purchases and total_spent are NOT used as features.
- Deterministic time split (train = early 80%, test = last 20%)
- Streams Parquet with DuckDB in batches to stay within RAM limits.
- Batch size: 20_000 (as requested).
"""

import duckdb
import pandas as pd
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import joblib
import gc

# -------------------------
# CONFIG
# -------------------------
FILE = "/home/angels/ecommerce-session-prediction/cleanbig/data/aggregated/2019-Oct_sessions.parquet"
BATCH_SIZE = 20_000
TRAIN_FRACTION = 0.8
TARGET = "label_purchase"

# IMPORTANT: do NOT include columns that leak the target:
# - n_purchases (directly indicates purchase)
# - total_spent  (also indicates purchase)
# We'll create session_duration from session_start/session_end and include it.
FEATURES = [
    "total_events",
    "n_views",
    "n_add_to_cart",
    "avg_price_viewed",
    "max_price_viewed",
    "unique_products",
    "session_duration"   # derived from timestamps
]

# where to save artifacts
MODEL_OUT = "/home/angels/ecommerce-session-prediction/ecvalmo/model_no_leak.pkl"
SCALER_OUT = "/home/angels/ecommerce-session-prediction/ecvalmo/scaler_no_leak.pkl"


# -------------------------
# HELPERS
# -------------------------
def add_label(df):
    """Create binary label: 1 if session had any purchase (n_purchases > 0), else 0."""
    df = df.copy()
    df[TARGET] = (df.get("n_purchases", 0) > 0).astype(int)
    return df


def compute_total_rows(path):
    con = duckdb.connect()
    total = con.execute(f"SELECT COUNT(*) FROM read_parquet('{path}')").fetchone()[0]
    con.close()
    return int(total)


def stream_ordered_batches(path, offset, limit, batch_size, order_by="session_start"):
    """
    Stream rows ordered by `order_by` column deterministically, using LIMIT/OFFSET.
    Yields pandas DataFrames of up to batch_size rows.
    """
    con = duckdb.connect()
    rows_yielded = 0
    current_offset = offset

    while rows_yielded < limit:
        to_fetch = min(batch_size, limit - rows_yielded)
        # Use explicit ORDER BY to ensure deterministic time-based ordering
        query = f"""
            SELECT *
            FROM read_parquet('{path}')
            ORDER BY {order_by}
            LIMIT {to_fetch} OFFSET {current_offset}
        """
        df = con.execute(query).fetchdf()
        if df.empty:
            break
        yield df
        fetched = len(df)
        rows_yielded += fetched
        current_offset += fetched
        # free duckdb local df reference (not strictly necessary but helps)
        del df
        gc.collect()

    con.close()


# -------------------------
# BATCH PREPROCESSING
# -------------------------
def preprocess_batch(df):
    """
    Prepare a batch:
    - add label
    - compute session_duration in seconds
    - drop columns not needed
    - convert dtypes and return X (np.float32) and y (np.int8)
    """
    # label
    df = add_label(df)

    # compute session_duration from timestamps (in seconds)
    # Some rows might already have proper pandas tz-aware strings; use errors='coerce'
    df["session_start"] = pd.to_datetime(df["session_start"], errors="coerce")
    df["session_end"] = pd.to_datetime(df["session_end"], errors="coerce")
    df["session_duration"] = (df["session_end"] - df["session_start"]).dt.total_seconds().fillna(0.0)

    # Drop identifiers and raw timestamps
    df = df.drop(columns=["user_session", "session_start", "session_end"], errors="ignore")

    # Remove truly leaking columns if present
    if "n_purchases" in df.columns:
        # keep for label only; removed from features
        pass
    if "total_spent" in df.columns:
        # drop it from features if present
        df = df.drop(columns=["total_spent"], errors="ignore")

    # keep only features + target
    keep_cols = [c for c in FEATURES if c in df.columns] + [TARGET]
    df = df[keep_cols]

    # Fill NaNs
    df = df.fillna(0)

    # Convert numeric dtypes to reduce memory
    for col in df.select_dtypes(include=["int64"]).columns:
        if col != TARGET:
            df[col] = pd.to_numeric(df[col], downcast="unsigned")
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")

    # Prepare X and y
    X = df[[c for c in FEATURES if c in df.columns]].values.astype(np.float32)
    y = df[TARGET].values.astype(np.int8)

    # free df
    del df
    gc.collect()

    return X, y


# -------------------------
# TRAIN + EVAL
# -------------------------
def train_and_evaluate():
    total_rows = compute_total_rows(FILE)
    train_rows = int(total_rows * TRAIN_FRACTION)
    test_rows = total_rows - train_rows

    print(f"Total rows: {total_rows}  -> train: {train_rows}, test: {test_rows}")

    # model and scaler
    model = SGDClassifier(loss="log_loss", penalty="l2", max_iter=1, warm_start=True)
    scaler = StandardScaler()

    # TRAIN
    print("\n=== TRAINING ===")
    first_batch = True
    trained = 0
    # stream train portion (ordered by session_start)
    for raw in stream_ordered_batches(FILE, offset=0, limit=train_rows, batch_size=BATCH_SIZE):
        # preprocess
        X_batch, y_batch = preprocess_batch(raw)

        # fit scaler on first batch, then transform
        if first_batch:
            scaler.fit(X_batch)
            first_batch = False
        Xs = scaler.transform(X_batch)

        # partial fit
        model.partial_fit(Xs, y_batch, classes=np.array([0, 1]))

        trained += X_batch.shape[0]
        print(f" Trained on batch {trained}/{train_rows} rows (batch size {X_batch.shape[0]})")

        # free
        del raw, X_batch, y_batch, Xs
        gc.collect()

    print("=== TRAINING COMPLETE ===")

    # TEST
    print("\n=== TESTING ===")
    tested = 0
    all_y = []
    all_pred = []

    for raw in stream_ordered_batches(FILE, offset=train_rows, limit=test_rows, batch_size=BATCH_SIZE):
        X_batch, y_batch = preprocess_batch(raw)
        Xs = scaler.transform(X_batch)
        preds = model.predict(Xs)

        all_y.append(y_batch)
        all_pred.append(preds)

        tested += X_batch.shape[0]
        print(f" Tested batch: {tested}/{test_rows} rows (batch size {X_batch.shape[0]})")

        # free
        del raw, X_batch, y_batch, Xs, preds
        gc.collect()

    # concat results
    if len(all_y) == 0:
        print("No test data found. Exiting.")
        return

    y_true = np.concatenate(all_y)
    y_pred = np.concatenate(all_pred)

    print("\n=== FINAL METRICS ===")
    print(classification_report(y_true, y_pred, digits=4))

    # Save artifacts
    joblib.dump(model, MODEL_OUT)
    joblib.dump(scaler, SCALER_OUT)
    print(f"\nSaved model -> {MODEL_OUT}")
    print(f"Saved scaler -> {SCALER_OUT}")


# -------------------------
# ENTRYPOINT
# -------------------------
if __name__ == "__main__":
    train_and_evaluate()
