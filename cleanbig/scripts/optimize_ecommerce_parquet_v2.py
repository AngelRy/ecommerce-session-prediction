#!/usr/bin/env python3
"""
Optimized conversion of multiple large e-commerce CSVs → Parquet with dictionary encoding.
Handles categorical columns with NA correctly and efficiently.

Processes both 2019-Oct.csv and 2019-Nov.csv in one run.
"""

import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
from pathlib import Path


# ===============================================================
# 1. SCAN BOTH CSVs TO BUILD ONE GLOBAL DICTIONARY
# ===============================================================

def build_global_dicts(csv_paths, chunksize=200_000):
    cat_cols = ["event_type", "category_code", "brand"]
    global_values = {c: [] for c in cat_cols}

    print("Scanning CSVs to build global dictionaries…")

    for path in csv_paths:
        print(f"\n=== Scanning {path} ===")
        for i, chunk in enumerate(pd.read_csv(path, chunksize=chunksize)):
            print(f"  • Scan chunk {i}")

            for col in cat_cols:
                vals = chunk[col].dropna().astype(str).unique()
                global_values[col].extend(vals)

    # Deduplicate
    print("\n=== Dictionary Summary ===")
    for col in global_values:
        global_values[col] = sorted(set(global_values[col]))
        print(f"  {col}: {len(global_values[col])} unique values")

    # Create value → index maps
    dicts = {col: {v: i for i, v in enumerate(global_values[col])}
             for col in global_values}

    return global_values, dicts


# ===============================================================
# 2. ENCODE A SINGLE CSV USING THE SHARED GLOBAL DICTIONARIES
# ===============================================================

def encode_and_write(csv_path, output_path, global_values, dicts, chunksize=200_000):
    writer = None
    cat_cols = ["event_type", "category_code", "brand"]

    print(f"\nEncoding CSV → Parquet → {output_path}")

    for i, chunk in enumerate(pd.read_csv(csv_path, chunksize=chunksize)):
        print(f"  • Encode chunk {i}")

        arrays = {}

        # A) Dictionary-encoded columns
        for col in cat_cols:
            mapping = dicts[col]
            keys = global_values[col]

            # Convert raw values → global dict indices
            codes = []
            for v in chunk[col]:
                if pd.isna(v):
                    codes.append(None)
                else:
                    codes.append(mapping[str(v)])

            indices = pa.array(codes, type=pa.int32())
            dictionary = pa.array(keys, type=pa.string())

            arrays[col] = pa.DictionaryArray.from_arrays(indices, dictionary)

        # B) Other non-categorical columns
        for col in chunk.columns:
            if col not in cat_cols:
                arrays[col] = pa.array(chunk[col])

        batch = pa.RecordBatch.from_arrays(list(arrays.values()),
                                           names=list(arrays.keys()))

        if writer is None:
            writer = pq.ParquetWriter(output_path, batch.schema, compression="snappy")

        writer.write_batch(batch)

    if writer:
        writer.close()

    print(f"✔ DONE → {output_path}")


# ===============================================================
# 3. MAIN WORKFLOW
# ===============================================================

def main():
    csv_oct = "./data/2019-Oct.csv"
    csv_nov = "./data/2019-Nov.csv"

    out_oct = "./data/2019-Oct_optimized.parquet"
    out_nov = "./data/2019-Nov_optimized.parquet"

    # Build ONE dictionary from BOTH datasets
    global_values, dicts = build_global_dicts([csv_oct, csv_nov])

    # Encode October
    encode_and_write(csv_oct, out_oct, global_values, dicts)

    # Encode November
    encode_and_write(csv_nov, out_nov, global_values, dicts)


if __name__ == "__main__":
    main()
