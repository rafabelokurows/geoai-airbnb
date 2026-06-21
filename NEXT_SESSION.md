# Next Session TODOs

## 1. Feature Importance Analysis ✓ DONE
- Pruned 7 sparse amenity flags (<2% hit rate)
- Expanded remaining groups + added 8 description keyword features
- Price RMSE: 54.32 → 53.59 (-1.3%)

## 2. Per-Room-Type Models ✓ DONE
- Two groups: "Entire home/apt" (dedicated model) + "other" (pooled Private/Hotel/Shared room)
- Dropped `minimum_nights` from feature set
- Price RMSE: EUR53.59 → EUR52.38 (weighted aggregate)
- Occupancy MAE: 0.1643 → 0.1659 — still misses <0.15 target
  - Entire home: 0.1669 | Other: 0.1603
- Neighbourhood ranking table removed from analytics sidebar

## 3. Analytics Narratives ✓ DONE
- Opportunity score per hex (frontend computed)
- Template-based hex narrative in HexDetail
- Neighbourhood ranking table in AnalyticsSidebar (top 20, scrollable)
- Per-hex occupancy vs price scatter in HexDetail (inline SVG)
- SHAP bars with EUR annotation — top 15 in sidebar, top 10 in HexDetail

## 4. Blog Post Headlines (DATA-BACKED)

Use the SHAP feature importance values and median revenue data already in the app
to write 5–8 catchy, data-backed blog post headlines for an Airbnb host audience.
Examples of angle:
- "The one amenity that adds €X/night to your Airbnb price (according to AI)"
- "Why Airbnb hosts in [top neighbourhood] earn 2× the city median"
- "High price, low occupancy: the Airbnb trap killing your ROI"
- Pull the actual numbers from `/api/explain/global` and `/api/neighbourhoods`
  to make claims concrete and non-generic.

## 5. Static Pre-Computation for Frontend-Only Deployment

Goal: eliminate the Python backend at deploy time. Pre-compute everything to
static JSON files that the React frontend can fetch from a CDN or GitHub Pages.

What to pre-compute:
- `public/data/kpis.json` — KPI snapshot
- `public/data/hex-aggregates.json` — all hex cells with opportunity_score
- `public/data/neighbourhoods.json` — neighbourhood ranking
- `public/data/explain-global.json` — top 15 SHAP features with EUR labels
- `public/data/hex/{h3_cell}.json` — per-hex listing scatter data (one file per cell,
  ~672 files at resolution 8)
- `public/data/listings/{id}.json` — per-listing SHAP explain (lazy, generate top 500
  opportunity listings only)

Build script: `scripts/export_static.py` — runs the full pipeline, writes all JSON,
exits. CI can run this once on data refresh; frontend reads from `/data/` prefix.

Frontend change: swap `fetch('/api/...')` calls in `client.js` to `fetch('/data/...')`
when `VITE_STATIC_MODE=true` env var is set. Zero React component changes needed.

Deployment: Vite build → `dist/` → push to GitHub Pages or Cloudflare Pages.
No server required.
