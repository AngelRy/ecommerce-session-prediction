import os
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------------
# Paths
# ----------------------------------
DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/aggregated_ecommerce_results.csv")
FIG_DIR = os.path.join(os.path.dirname(__file__), "../figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ----------------------------------
# Load data
# ----------------------------------
df = pd.read_csv(DATA_PATH)

print("\nLoaded data:")
print(df.head())
print("\nColumns:", df.columns.tolist())

# ----------------------------------
# Fix datetime parsing
# ----------------------------------
df["month"] = pd.to_datetime(df["month"], errors="coerce")
df = df.sort_values("month")
df["month_str"] = df["month"].dt.strftime("%Y-%m")

# ----------------------------------
# Plot 1: Sales by Month
# ----------------------------------
plt.figure(figsize=(10, 5))
plt.plot(df["month_str"], df["total_sales"], marker="o")
plt.title("Total Sales per Month")
plt.xlabel("Month")
plt.ylabel("Total Sales (€)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/sales_by_month.png")
plt.close()

# ----------------------------------
# Plot 2: Top 15 Categories by Sales (category_id)
# ----------------------------------
top_categories = df.groupby("category_id")["total_sales"] \
                   .sum() \
                   .sort_values(ascending=False) \
                   .head(15)

plt.figure(figsize=(10, 6))
top_categories.plot(kind="bar")
plt.title("Top 15 Categories by Sales")
plt.xlabel("Category ID")
plt.ylabel("Total Sales (€)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/top_categories.png")
plt.close()

# ----------------------------------
# Plot 3: Event Type Breakdown Over Time
# ----------------------------------
pivot_events = df.pivot_table(
    index="month_str",
    columns="event_type",
    values="num_events",
    aggfunc="sum"
).fillna(0)

plt.figure(figsize=(12, 6))
pivot_events.plot(kind="line")
plt.title("Event Volume Over Time by Event Type")
plt.xlabel("Month")
plt.ylabel("Number of Events")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/event_type_over_time.png")
plt.close()

# ----------------------------------
# Plot 4: Category vs Month Heatmap (category_id)
# ----------------------------------
pivot_heatmap = df.pivot_table(
    index="category_id",
    columns="month_str",
    values="total_sales",
    aggfunc="sum"
).fillna(0)

plt.figure(figsize=(12, 8))
plt.imshow(pivot_heatmap, aspect="auto", cmap="Blues")
plt.colorbar(label="Sales (€)")
plt.title("Sales Heatmap: Category ID vs Month")
plt.xlabel("Month")
plt.ylabel("Category ID")
plt.xticks(range(len(pivot_heatmap.columns)), pivot_heatmap.columns, rotation=45)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/heatmap_categories.png")
plt.close()

print("\nEDA completed. Figures saved to the 'figures' directory.")
