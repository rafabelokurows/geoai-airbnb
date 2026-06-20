# ADR-007: Analytics Narratives — Hybrid Frontend/Backend Approach

**Date:** 2026-06-20
**Status:** Accepted
**Deciders:** Rafael Belokurows

---

## Context

Phase 6 extended the React + Deck.gl dashboard with richer analytics: opportunity scores per hex cell, template-based narrative text, neighbourhood revenue rankings, per-hex occupancy vs price scatter plots, and SHAP feature importance displayed in EUR terms.

The key design question was where to compute each piece of analytics — purely on the frontend (from already-loaded hexData), via new backend endpoints, or via a separate pre-computation step.

---

## Options Evaluated

| Feature | Frontend-only | Backend endpoint | Pre-computed static |
|---------|--------------|-----------------|-------------------|
| Opportunity score (revenue ↑, listing_count ↓) | ✅ All data already in hexData | N/A | Overkill |
| Hex narrative text | ✅ Template, no server needed | N/A | Overkill |
| Neighbourhood ranking | ❌ No neighbourhood column in hexData | ✅ Simple GROUP BY in Polars | Future option |
| Per-hex scatter (price vs occupancy) | ❌ Would need to filter 13k listings client-side | ✅ Filter by h3_cell_r8 server-side | Future option |
| SHAP EUR annotation | ✅ `cityAvgPrice × (exp(shap) − 1)` client-side | N/A | N/A |

---

## Decision

**Hybrid approach:** compute on the frontend where all data is already available; add backend endpoints only where server-side filtering over 13k rows is needed.

Specific choices:

1. **Opportunity score** — computed in `App.jsx` via `useMemo` using min-max normalization: `0.5 × norm_revenue + 0.5 × (1 − norm_listing_count)`. Result stored as `opportunity_score` on each hex object and passed to all child components.

2. **Hex narrative** — pure frontend template in `HexDetail.jsx`. Compares selected hex revenue vs city median (derived from `hexData`). No LLM, no backend call, no latency.

3. **Neighbourhood rankings** — new `GET /api/neighbourhoods` endpoint in `src/geoai/api/routes/neighbourhoods.py`. Polars `group_by("neighbourhood").agg(mean(estimated_annual_revenue), count(id))`, sorted descending. Required adding `l.neighbourhood` to the listings SQL query in `deps.py`.

4. **Per-hex scatter** — new `GET /api/hex/{h3_cell}/listings` endpoint returning `price` and `predicted_occupancy` per listing. Rendered as an inline SVG scatter in `HexDetail.jsx` — no charting library added.

5. **SHAP EUR annotation** — approximation `cityAvgPrice × (Math.exp(mean_abs_shap) − 1)` applied client-side. SHAP values are in log-price space; this converts the mean absolute contribution to an approximate EUR impact per night. Shown as `+€X/night` on each bar.

6. **Global SHAP state lifted to App** — previously fetched inside `AnalyticsSidebar`. Now fetched once in `App.jsx` (`fetchGlobalExplain(15)`) and passed as `shapImportance` prop to both `AnalyticsSidebar` (top 15) and `HexDetail` (top 10). Eliminates duplicate API calls.

---

## Consequences

**Positive:**
- Zero new npm packages — scatter uses inline SVG
- Opportunity score updates instantly when hexData loads (no extra round-trip)
- Neighbourhood and per-hex endpoints are lightweight Polars aggregations — sub-millisecond on loaded DataFrames
- SHAP EUR annotation gives hosts an intuitive anchor ("pool adds ~€18/night") without requiring per-listing SHAP inference at request time

**Negative:**
- SHAP EUR annotation is an approximation (uses city avg price as base, not listing-specific base price). Accuracy is ±20-30% for outlier listings.
- Narrative text is template-based — repeated pattern if many similar hex cells viewed sequentially. LLM narrative generation is a future upgrade path (captured in NEXT_SESSION.md item 4).
- Per-hex scatter makes one API call per hex click. With 672 hex cells this is acceptable; would need caching if users rapidly click many cells.

---

## References

- `src/geoai/api/routes/neighbourhoods.py`
- `src/geoai/api/routes/hex.py` (hex_listings endpoint)
- `app/src/App.jsx` (opportunity score, SHAP state)
- `app/src/components/AnalyticsSidebar.jsx`
- `app/src/components/HexDetail.jsx`
- `NEXT_SESSION.md` items 4–5 (blog posts, static pre-computation)
