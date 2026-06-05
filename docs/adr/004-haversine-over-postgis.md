# ADR-004: Vectorized Haversine for Proximity Calculations

**Date:** 2026-06-04
**Status:** Accepted
**Deciders:** Rafael Belokurows

---

## Context

Phase 2 feature engineering requires computing distances from every listing (~15K) to:
- Fixed points (city center, airport, Porto landmarks)
- Nearest POI in each category (metro, station, supermarket)
- All POIs within fixed radii (250m, 500m, 1km, 2km) for density counts

At 15K listings × 9K POIs, a naive all-pairs approach is 135M distance computations. The solution must be fast enough to run on a laptop.

Options evaluated:

| Option | Pros | Cons |
|--------|------|------|
| **Vectorized NumPy Haversine** | No extra deps, pure Python, fast with broadcasting, testable in isolation | Flat-Earth approximation breaks at >100km (irrelevant for Porto) |
| DuckDB spatial extension | SQL-native, stays in warehouse | Extra dependency, DuckDB spatial extension install not always trivial |
| PostGIS | Full geospatial SQL, industry standard | Requires PostgreSQL server — conflicts with zero-infra goal (see ADR-001) |
| GeoPandas STRtree | R-tree spatial index, faster for radius queries | Extra memory, more complex code path, GeoPandas already a dep but STRtree adds setup |
| Shapely distance | Simple API | Scalar — loops over 135M pairs |

---

## Decision

Use **vectorized NumPy Haversine** with **bounding box pre-filtering** for all proximity calculations.

For each listing, a bounding box is computed (lat ± δ, lon ± δ) to filter POIs to a candidate set before Haversine. This gives ~100× speedup over naive all-pairs on Porto's POI dataset.

Fixed-point distances (city center, airport, landmarks) use direct NumPy broadcasting — no pre-filter needed since there is only one target point.

---

## Consequences

**Positive:**
- Zero additional dependencies beyond NumPy (already required by everything else)
- Pure Python: every calculation is testable with `np.testing.assert_allclose`
- Bounding box pre-filter reduces per-listing work from ~9K to ~10–200 POI candidates
- Accuracy: Haversine error < 0.3% within Porto's 10km extent — negligible for ML features

**Negative:**
- Not suitable for global-scale distance queries. Mitigation: irrelevant — Porto fits in a 20km bounding box.
- Bounding box pre-filter is a manual approximation of a spatial index. Mitigation: equivalent in practice to STRtree for this dataset size; can swap if city scale grows.

---

## References

- Haversine formula: [Wikipedia](https://en.wikipedia.org/wiki/Haversine_formula)
- `src/geoai/features/accessibility.py` — fixed-point distances
- `src/geoai/features/poi_density.py` — bounding box + radius count
