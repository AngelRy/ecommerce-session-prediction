#!/usr/bin/env python3
"""
Memory-efficient session-level aggregation for October only.
Saves output to ./data/aggregated/2019-Oct_sessions.parquet
"""

import os
import pandas as pd
import pyarrow.dataset as ds

# -------------------------
# CONFIG
# -------------------------
DATA_DIR = "/home/angels/cleanbig/data"
AGG_DIR = os.path.join(DATA_DIR, "aggregated")
os.makedirs(AGG_DIR, exist_ok=True)

DATA_FILE = os.path.join(DATA_DIR, "2019-Oct_optimized.parquet")
OUT_FILE = os.path.join(AGG_DIR, "2019-Oct_sessions.parquet")

COLUMNS = ["user_session", "event_time", "event_type", "price", "product_id"]
BATCH_SIZE = 100_000  # number of rows per batch

# -------------------------
# UTILS
# -------------------------
def aggregate_batches(file_path):
    """
    Read Parquet in batches and aggregate session-level features.
    """
    dataset = ds.dataset(file_path, format="parquet")
    batch_aggregates = []

    for batch in dataset.to_batches(batch_size=BATCH_SIZE):
        df = batch.to_pandas()
        # optimize memory
        df['event_type'] = df['event_type'].astype('category')
        df['price'] = df['price'].astype('float32')

        # session-level aggregation
        df_agg = df.groupby('user_session').agg(
            session_start=('event_time', 'min'),
            session_end=('event_time', 'max'),
            total_events=('event_type', 'count'),
            n_views=('event_type', lambda x: (x=='view').sum()),
            n_add_to_cart=('event_type', lambda x: (x=='add_to_cart').sum()),
            n_purchases=('event_type', lambda x: (x=='purchase').sum()),
            total_spent=('price', lambda x: x[df['event_type']=='purchase'].sum()),
            avg_price_viewed=('price', lambda x: x[df['event_type']=='view'].mean()),
            max_price_viewed=('price', lambda x: x[df['event_type']=='view'].max()),
            unique_products=('product_id', 'nunique')
        ).reset_index()

        batch_aggregates.append(df_agg)

    # combine all batches
    df_sessions = pd.concat(batch_aggregates, ignore_index=True)
    return df_sessions

# -------------------------
# MAIN
# -------------------------
def main():
    if os.path.exists(OUT_FILE):
        print("Aggregated file already exists:", OUT_FILE)
        return

    print("Aggregating October sessions…")
    df_sessions = aggregate_batches(DATA_FILE)

    # save
    df_sessions.to_parquet(OUT_FILE, index=False)
    print("Saved:", OUT_FILE)
    print("Rows:", len(df_sessions))

if __name__ == "__main__":
    main()
