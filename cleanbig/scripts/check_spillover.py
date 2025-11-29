import duckdb
import pandas as pd

# Make Pandas show all columns
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 100)

# Paths to Parquet files
oct_path = "./cleanbig/data/2019-Oct_optimized.parquet"
nov_path = "./cleanbig/data/2019-Nov_optimized.parquet"

# Connect to DuckDB
con = duckdb.connect()

# Tail of October
print("===== October: last 5 rows =====")
tail_oct = con.execute(f"""
    SELECT * 
    FROM read_parquet('{oct_path}') 
    ORDER BY event_time DESC 
    LIMIT 5
""").fetchdf()
print(tail_oct)

# Head of November
print("\n===== November: first 5 rows =====")
head_nov = con.execute(f"""
    SELECT * 
    FROM read_parquet('{nov_path}') 
    ORDER BY event_time ASC 
    LIMIT 5
""").fetchdf()
print(head_nov)

con.close()
