"""Quick script to query airbnb listings from DuckDB into pandas."""
#%%
import duckdb
import pandas as pd
from pathlib import Path

DB_PATH = Path(__file__).parent / "data" / "warehouse.duckdb"

con = duckdb.connect(str(DB_PATH), read_only=True)

# Load full listings table
df = con.execute("SELECT * FROM listings").df()

print(f"Shape: {df.shape}")
print(df.dtypes)
print(df.head())

# --- example queries ---

# Top 10 by review score
top_rated = (
    df[df["review_scores_rating"].notna()]
    .sort_values("review_scores_rating", ascending=False)
    .head(10)[["name", "neighbourhood_cleansed", "room_type", "price", "review_scores_rating"]]
)
print("\nTop 10 rated:\n", top_rated.to_string(index=False))

# Average price by room type
avg_price = (
    df.assign(price_num=df["price"].str.replace(r"[$,]", "", regex=True).astype(float))
    .groupby("room_type")["price_num"]
    .mean()
    .sort_values(ascending=False)
)
print("\nAvg price by room type:\n", avg_price)

con.close()
#%%