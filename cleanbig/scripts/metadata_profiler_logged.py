#!/usr/bin/env python3
import os
import csv
import subprocess
import pandas as pd

# -------- SETTINGS --------
SAMPLE_ROWS = 50_000          # safe for 8GB RAM
PREVIEW_ROWS = 10             # head()
BYTE_SAMPLE = 10_000          # for Sniffer
# --------------------------

def file_exists(path):
    if not os.path.isfile(path):
        print(f"\n❌ File not found: {path}")
        return False
    return True

def print_file_size(path):
    size_mb = os.stat(path).st_size / (1024 * 1024)
    print(f"📏 File size: {size_mb:.2f} MB")

def count_rows_fast(path):
    print("⏳ Counting rows (this may take a while)…")
    try:
        result = subprocess.run(["wc", "-l", path], capture_output=True, text=True)
        row_count = int(result.stdout.split()[0])
        print(f"📊 Row count (fast estimate): {row_count:,}")
        return row_count
    except Exception:
        print("⚠️ wc not available → counting rows slowly…")
        row_count = 0
        with open(path, "r") as f:
            for i, _ in enumerate(f, 1):
                if i % 1_000_000 == 0:
                    print(f"  Scanned {i:,} lines…")
            row_count = i
        print(f"📊 Total rows: {row_count:,}")
        return row_count

def detect_delimiter(path):
    print("🔍 Detecting delimiter…")
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        sample = f.read(BYTE_SAMPLE)
        dialect = csv.Sniffer().sniff(sample)
    print(f"🔎 Detected delimiter: '{dialect.delimiter}'")
    return dialect.delimiter

def read_header(path, delimiter):
    print("📥 Reading header…")
    df_header = pd.read_csv(path, nrows=0, delimiter=delimiter)
    print("✅ Header read.")
    print("\n🧩 Columns:")
    for col in df_header.columns:
        print("  -", col)
    return df_header.columns.tolist()

def read_sample(path, delimiter):
    print(f"📥 Reading first {SAMPLE_ROWS:,} rows as sample…")
    try:
        df = pd.read_csv(path, nrows=SAMPLE_ROWS, delimiter=delimiter)
        print("✅ Sample loaded.")
        return df
    except Exception as e:
        print(f"❌ Could not read sample: {e}")
        return None

def summarize_missing(df):
    print("\n📉 Missing value summary (sample):")
    print(df.isna().sum())

def summarize_unique(df):
    print("\n🔢 Unique value counts (sample):")
    for col in df.columns:
        unique_vals = df[col].nunique(dropna=True)
        print(f"  {col}: {unique_vals:,} unique")

def show_preview(path, delimiter):
    print(f"\n👀 Preview (first {PREVIEW_ROWS} rows):")
    df_preview = pd.read_csv(path, nrows=PREVIEW_ROWS, delimiter=delimiter)
    print(df_preview)

def main():
    FILES = [
        "./data/2019-Oct.csv",
        "./data/2019-Nov.csv",
    ]

    for path in FILES:
        print("\n" + "=" * 60)
        print(f"📂 File: {path}")
        print("=" * 60)

        if not file_exists(path):
            continue

        print_file_size(path)
        count_rows_fast(path)

        delimiter = detect_delimiter(path)
        read_header(path, delimiter)

        sample_df = read_sample(path, delimiter)
        if sample_df is None:
            continue

        print("\n📐 Data types (inferred from sample):")
        print(sample_df.dtypes)

        summarize_missing(sample_df)
        summarize_unique(sample_df)
        show_preview(path, delimiter)

        print("\n✨ Metadata profiling complete.\n")


if __name__ == "__main__":
    main()
