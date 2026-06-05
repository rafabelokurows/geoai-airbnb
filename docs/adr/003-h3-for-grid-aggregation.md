# ADR-003: H3 Hexagons for Geographic Grid Aggregation

**Date:** 2026-06-04
**Status:** Accepted
**Deciders:** Rafael Belokurows

---

## Context

The platform needs a spatial grid to:

- Aggregate listing metrics (avg price, avg occupancy, count) by geographic cell
- Power H3-based heatmap layers in the Streamlit dashboard
- Enable hex-level comparison and opportunity scoring

Options evaluated:

| Option | Pros | Cons |
|--------|------|------|
| **H3 (Uber)** | Industry standard, portable, hierarchical resolutions, great Python API, used by Kepler.gl/PyDeck | Hexagonal grid doesn't align with administrative boundaries |
| Regular lat/lon grid | Simple, intuitive | Cells vary in real-world area at different latitudes; no standard library |
| Neighbourhood boundaries | Matches local knowledge | Irregular shapes, coarse granularity, admin-boundary-dependent |
| S2 cells | Also hierarchical, used by Google | Less Python tooling, less common in STR research |

---

## Decision

Use **H3 at resolution 8** (≈460m edge length, ≈0.74 km² area) for all grid aggregations.

Each listing is assigned an `h3_cell_r8` value via `h3.latlng_to_cell(lat, lon, 8)`. Per-hex aggregates (avg price, avg occupancy, listing count) are computed and stored in the `hex_aggregates` table in DuckDB.

---

## Consequences

**Positive:**
- Portable: H3 cell IDs are globally unique and deterministic — no schema changes when adding cities
- Composable: resolution 8 cells nest cleanly into resolution 7 (≈1.2km) for coarser views
- Visualization: PyDeck has native H3 layer support; no geometry conversion needed
- Comparable: standard resolution used in other STR and mobility research, enabling benchmarking

**Negative:**
- Hex cells don't align with Porto's neighbourhood boundaries — cross-boundary cells exist at edges. Mitigation: keep neighbourhood-level features separate; use H3 only for grid aggregations.
- H3 v4 API changed (`h3.latlng_to_cell` vs old `h3.geo_to_h3`) — project pins `h3>=4.0`.

---

## References

- [H3 Python API](https://uber.github.io/h3-py/)
- [H3 Resolution Table](https://h3geo.org/docs/core-library/restable/)
