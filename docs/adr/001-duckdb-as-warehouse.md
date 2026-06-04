# ADR-001: DuckDB as the Analytical Warehouse

**Date:** 2026-06-04
**Status:** Accepted
**Deciders:** Rafael Belokurows

---

## Context

The platform needs a data store for ~15K Airbnb listings, ~5M calendar rows, ~500K reviews, and derived geospatial features. It must support:

- Analytical queries (aggregations, window functions, spatial joins)
- Fast ingestion from Polars DataFrames and CSV/GZ files
- No infrastructure overhead (local development, single-file deployment)
- Columnar storage for ML feature pipelines

Options evaluated:

| Option | Pros | Cons |
|--------|------|------|
| **DuckDB** | Embedded, zero-config, columnar, fast OLAP, native Polars integration, SQL | Single writer at a time, not for OLTP |
| PostgreSQL + PostGIS | Full geospatial SQL, multi-user | Requires server setup, heavy for portfolio project |
| SQLite | Embedded, simple | Row-oriented, slow on analytical queries, no native geo |
| Parquet files | Columnar, portable | No SQL, no ACID, awkward for cross-file joins |
| MotherDuck (cloud DuckDB) | Managed, shareable | Costs money, adds dependency |

---

## Decision

Use **DuckDB** as the single analytical warehouse (`data/warehouse.duckdb`).

All tables (`listings`, `calendar`, `poi_features`, `neighbourhoods`, `reviews`) live in one DuckDB file. Polars is used for data cleaning; the cleaned DataFrames are written directly into DuckDB via its native Python API (`con.execute("INSERT INTO table SELECT * FROM df")`).

---

## Consequences

**Positive:**
- Zero infrastructure setup — single `.duckdb` file, no server, no Docker
- Native Polars integration: query Polars DataFrames as SQL tables directly
- Columnar storage → fast aggregations for ML feature computation
- Full SQL including window functions, `QUALIFY`, `PIVOT` — useful for occupancy calculations
- DuckDB's spatial extension (future) can replace some GeoPandas operations

**Negative:**
- Single writer: ingestion and dashboarding cannot run concurrently on the same file without read/write contention. Mitigation: use separate connections and open read-only where possible.
- Not suitable for high-concurrency production API. Mitigation: for production, swap to MotherDuck or PostgreSQL; the SQL layer stays identical.
- `.duckdb` file is not human-readable. Mitigation: gitignored; data is reproducible from raw CSVs.

---

## References

- [DuckDB Python API](https://duckdb.org/docs/api/python/overview)
- [DuckDB Polars Integration](https://duckdb.org/docs/guides/python/polars)
