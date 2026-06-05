# Architecture Decision Records

| ADR | Title | Status |
|-----|-------|--------|
| [001](001-duckdb-as-warehouse.md) | DuckDB as the Analytical Warehouse | Accepted |
| [002](002-polars-over-pandas.md) | Polars for Data Processing (over Pandas) | Accepted |
| [003](003-h3-for-grid-aggregation.md) | H3 Hexagons for Geographic Grid Aggregation | Accepted |
| [004](004-haversine-over-postgis.md) | Vectorized Haversine for Proximity Calculations | Accepted |
| [005](005-lightgbm-for-ml-models.md) | LightGBM for Price and Occupancy Models | Accepted |

---

ADRs use the format: Context → Options → Decision → Consequences.
New ADRs should be added here when significant architectural choices are made (ML framework, deployment target, API design, causal inference library, LLM provider).
