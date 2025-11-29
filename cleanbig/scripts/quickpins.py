import pyarrow.parquet as pq

path = "./data/2019-Oct_optimized.parquet"

# Open the Parquet file
parquet_file = pq.ParquetFile(path)

# Print schema (column names + types)
print(parquet_file.schema)
