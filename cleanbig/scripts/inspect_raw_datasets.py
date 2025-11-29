import os
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

FILE1 = os.path.join(DATA_DIR, "2019-Oct.csv")
FILE2 = os.path.join(DATA_DIR, "2019-Nov.csv")

N_ROWS = 300   # only read this many rows


def inspect(path, name):
    print(f"\n===== {name} =====")

    if not os.path.exists(path):
        print(f"❌ File not found: {path}")
        return

    try:
        df = pd.read_csv(path, nrows=N_ROWS)

        print("\n--- COLUMN NAMES ---")
        print(df.columns.tolist())

        print("\n--- FIRST FEW ROWS ---")
        print(df.head())

        print(f"\nLoaded only {len(df)} rows.\n")

    except Exception as e:
        print(f"Error loading {name}: {e}")


def main():
    inspect(FILE1, "2019-Oct.csv")
    inspect(FILE2, "2019-Nov.csv")


if __name__ == "__main__":
    main()
