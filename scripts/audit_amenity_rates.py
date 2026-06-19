"""Audit amenity hit rates to identify sparse features for pruning."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from geoai.models.features import AMENITY_COLS, build_feature_matrix

df = build_feature_matrix()
n = len(df)

print(f"\nAmenity hit rates across {n:,} listings\n")
print(f"{'Feature':<30} {'Count':>7} {'Hit Rate':>9}  Flag")
print("-" * 55)

rates = []
for col in AMENITY_COLS:
    count = int(df[col].sum())
    rate = count / n
    rates.append((col, count, rate))

for col, count, rate in sorted(rates, key=lambda x: x[2]):
    flag = ""
    if rate < 0.02:
        flag = "  << DROP (<2%)"
    elif rate > 0.98:
        flag = "  << DROP (>98%)"
    print(f"{col:<30} {count:>7,} {rate:>8.1%}  {flag}")

print()
drop_cols = [col for col, _, rate in rates if rate < 0.02 or rate > 0.98]
print(f"Candidates to drop ({len(drop_cols)}): {drop_cols}")
