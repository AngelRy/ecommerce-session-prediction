import duckdb, os

paths = [
    "./data/2019-Oct_optimized.parquet",
    "./data/2019-Nov_optimized.parquet"
]

for p in paths:
    print("\nFILE:", p)
    print("Size MB:", round(os.path.getsize(p) / 1024**2, 2))
    print( duckdb.sql(f"SELECT COUNT(*) AS rows FROM read_parquet('{p}')").df() )



for p in paths:
    print("\nSCHEMA:", p)
    print( duckdb.sql(f"DESCRIBE SELECT * FROM read_parquet('{p}')").df() )

