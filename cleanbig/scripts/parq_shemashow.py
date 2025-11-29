for p in paths:
    print("\nSCHEMA:", p)
    display( duckdb.sql(f"DESCRIBE SELECT * FROM read_parquet('{p}')").df() )
