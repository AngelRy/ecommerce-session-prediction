import os
import pandas as pd
from tqdm import tqdm


# === SMART PATH SETUP ===
# Automatically locate the project root (one level above 'scripts' if it exists)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR) if os.path.basename(CURRENT_DIR) == "scripts" else CURRENT_DIR

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_FILE = os.path.join(PROJECT_ROOT, "data/aggregated_ecommerce_results.csv")
CHUNK_SIZE = 100_000            # optimized for ~8GB RAM
USE_COLS = [
    "event_time", "event_type", "product_id",
    "category_id", "category_code", "brand",
    "price", "user_id", "user_session"
]


def optimize_types(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast numeric types and convert low-cardinality objects to categories."""
    for col in df.select_dtypes(include="object").columns:
        num_unique = df[col].nunique(dropna=True)
        if num_unique / len(df) < 0.5:
            df[col] = df[col].astype("category")

    for col in df.select_dtypes(include="int").columns:
        df[col] = pd.to_numeric(df[col], downcast="unsigned")

    for col in df.select_dtypes(include="float").columns:
        df[col] = pd.to_numeric(df[col], downcast="float")

    return df


def process_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """Aggregate sales and event data per category and event type."""
    chunk["price"] = chunk["price"].fillna(0)
    agg = (
        chunk.groupby(["category_id", "event_type"], observed=True)
        .agg(
            total_sales=("price", "sum"),
            avg_price=("price", "mean"),
            num_events=("event_type", "count"),
            unique_products=("product_id", "nunique"),
        )
        .reset_index()
    )
    return agg


def process_file(file_path: str) -> pd.DataFrame:
    """Read and aggregate one file in chunks, safely."""
    print(f"\n📂 Processing file: {os.path.basename(file_path)}")

    aggregated_chunks = []
    try:
        # Initialize progress bar based on estimated number of rows
        total_rows = sum(1 for _ in open(file_path, "rb"))
        reader = pd.read_csv(
            file_path,
            usecols=USE_COLS,
            chunksize=CHUNK_SIZE,
            encoding_errors="replace",
            on_bad_lines="skip",
        )

        for chunk in tqdm(reader, total=total_rows // CHUNK_SIZE + 1, desc="Reading chunks"):
            chunk = optimize_types(chunk)
            agg = process_chunk(chunk)
            aggregated_chunks.append(agg)

    except Exception as e:
        print(f"⚠️ Error reading {file_path}: {e}")
        return pd.DataFrame()

    # Combine and re-aggregate results
    combined = pd.concat(aggregated_chunks, ignore_index=True)
    combined = (
        combined.groupby(["category_id", "event_type"], observed=True)
        .agg(
            total_sales=("total_sales", "sum"),
            avg_price=("avg_price", "mean"),
            num_events=("num_events", "sum"),
            unique_products=("unique_products", "sum"),
        )
        .reset_index()
    )

    return combined


def main():
    all_results = []

    for fname in sorted(os.listdir(DATA_DIR)):
        if fname.endswith(".csv"):
            file_path = os.path.join(DATA_DIR, fname)
            monthly_result = process_file(file_path)
            monthly_result["month"] = fname.replace(".csv", "")
            all_results.append(monthly_result)

    final_df = pd.concat(all_results, ignore_index=True)
    print("\n✅ Final aggregated result:")
    print(final_df.head(10))

    final_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n💾 Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
