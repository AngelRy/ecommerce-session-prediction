#!/usr/bin/env python3
import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

DATA_DIR = "./data"
CHUNK_SIZE = 200_000

# Columns to optimize
STRING_COLS = ["event_type", "category_code", "brand", "user_session"]
INT32_COLS = ["product_id", "user_id"]
INT16_COLS = ["category_id"]
FLOAT32_COLS = ["price"]

def optimize_dtypes(df):
    for col in STRING_COLS:
        if col in df.columns:
            df[col] = df[col].astype(str)  # keep as string to avoid ParquetWriter errors
    for col in INT32_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], downcast="integer")
    for col in INT16_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], downcast="integer")
    for col in FLOAT32_COLS:
        if col in df.columns:
            df[col] = df[col].astype("float32")
    if "event_time" in df.columns:
        df["event_time"] = pd.to_datetime(df["event_time"], errors="coerce")
    return df

def process_file(file_path, output_path):
    print(f"\n📂 Processing {file_path}…")
    writer = None

    for i, chunk in enumerate(pd.read_csv(file_path, chunksize=CHUNK_SIZE)):
        print(f"  - Chunk {i+1}")
        chunk = optimize_dtypes(chunk)
        table = pa.Table.from_pandas(chunk)

        if writer is None:
            writer = pq.ParquetWriter(output_path, table.schema)
        writer.write_table(table)

    if writer:
        writer.close()
    print(f"💾 Optimized Parquet file saved to {output_path}")

def main():
    files = sorted(os.listdir(DATA_DIR))
    files = [f for f in files if f.endswith(".csv")]

    for fname in files:
        input_path = os.path.join(DATA_DIR, fname)
        name, _ = os.path.splitext(fname)
        output_path = os.path.join(DATA_DIR, f"optimized_{name}.parquet")
        process_file(input_path, output_path)

if __name__ == "__main__":
    main()
