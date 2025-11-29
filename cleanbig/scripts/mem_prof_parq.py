import os
import pyarrow.parquet as pq
import pandas as pd

DATA_DIR = "./data"
SAMPLE_ROWS = 1000

def profile_parquet(file_path, sample_rows=SAMPLE_ROWS):
    print(f"\n📂 Profiling {file_path}…")
    
    # Open Parquet file as pyarrow table
    parquet_file = pq.ParquetFile(file_path)
    
    # Read only first N rows into a DataFrame
    table = parquet_file.read_row_groups([0], columns=None)
    df_sample = table.to_pandas().head(sample_rows)
    
    # Memory usage per column
    mem_usage = df_sample.memory_usage(deep=True)
    total_mem = mem_usage.sum()
    
    print("Columns and dtypes:")
    print(df_sample.dtypes)
    print("\nEstimated memory usage per column (bytes):")
    print(mem_usage)
    print(f"\nTotal estimated memory usage for sample: {total_mem / 1024**2:.2f} MB")

def main():
    files = sorted(os.listdir(DATA_DIR))
    parquet_files = [f for f in files if f.endswith(".parquet")]
    
    if not parquet_files:
        print("⚠️ No Parquet files found!")
        return
    
    for fname in parquet_files:
        profile_parquet(os.path.join(DATA_DIR, fname))

if __name__ == "__main__":
    main()
