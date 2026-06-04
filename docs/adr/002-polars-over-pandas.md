# ADR-002: Polars for Data Processing (over Pandas)

**Date:** 2026-06-04
**Status:** Accepted
**Deciders:** Rafael Belokurows

---

## Context

The ingestion pipeline processes compressed CSV files (listings ~7.6MB, calendar ~14MB, reviews ~132MB). The data cleaning layer runs before loading into DuckDB. A DataFrame library is needed.

Options evaluated:

| Option | Pros | Cons |
|--------|------|------|
| **Polars** | Rust-based, lazy evaluation, fast on large files, native DuckDB integration, expressive API | Smaller ecosystem than Pandas, some ML libs expect Pandas |
| Pandas | Universal, huge ecosystem, every ML tutorial uses it | Slow on large files, high memory use, mutable by default |
| DuckDB only | SQL for everything, no Python DataFrame overhead | Harder to express complex cleaning logic (regex, list ops) |

---

## Decision

Use **Polars** for all data loading, cleaning, and transformation before DuckDB ingestion. Where downstream ML libraries require Pandas (e.g., SHAP, EconML), convert with `df.to_pandas()` at the boundary only.

---

## Consequences

**Positive:**
- Reviews file (132MB compressed) loads significantly faster than Pandas
- Lazy evaluation (`pl.LazyFrame`) enables query optimization before execution
- DuckDB can query Polars DataFrames directly — no intermediate serialization
- Immutable by default — prevents accidental mutation bugs in cleaning pipelines

**Negative:**
- SHAP, EconML, and some sklearn utilities expect Pandas. Conversion at the boundary adds a step.
- Team members familiar only with Pandas face a learning curve.
- Occasional API differences (e.g., `map_elements` vs Pandas `apply`) require attention.

---

## References

- [Polars User Guide](https://docs.pola.rs/)
- [DuckDB + Polars](https://duckdb.org/docs/guides/python/polars)
