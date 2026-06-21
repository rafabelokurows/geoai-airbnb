# GeoAI Short-Term Rental Intelligence Platform

> A geospatial AI platform that explains what drives Airbnb performance — combining geospatial feature engineering, machine learning, causal inference, and LLM-powered property analysis.

---

## What This Is

Short-term rental markets are deeply spatial. A listing's revenue isn't just a function of its amenities or host quality — it's a function of *where it sits*: proximity to transit, density of restaurants, walkability, competition from nearby listings, and neighborhood trajectory.

This platform engineers those spatial signals from raw Airbnb and OpenStreetMap data, feeds them into interpretable ML models, and surfaces actionable insights for anyone who needs to understand or act on STR market dynamics. The end product is not a price predictor — it's an intelligence layer that answers *why*, not just *what*.

The full stack runs locally with zero external dependencies beyond the data files. DuckDB replaces a traditional database server. OSMnx replaces a paid POI API. SHAP replaces black-box model outputs. The architecture is intentionally minimal: every component is replaceable and every intermediate artifact is inspectable.

---

## Why Porto

Porto is one of Europe's highest-signal STR markets for technical work:

- **Scale without noise**: ~15,000 active listings — large enough for robust ML, small enough to iterate fast on a laptop
- **Geographic diversity**: compact city with distinct neighborhoods (Ribeira waterfront, Bonfim gentrification frontier, Foz coastal premium) that stress-test spatial features
- **Tourism pressure**: consistently ranked top-5 European city by tourism growth, making STR dynamics economically meaningful
- **Open data quality**: InsideAirbnb maintains high-quality, regularly updated Porto snapshots — reviews, calendar, and neighborhood boundaries all included
- **OSM coverage**: dense, well-maintained OpenStreetMap data for POI extraction — transit stops, restaurants, museums, parks all reliably present
- **Regulatory context**: Porto has active STR regulation debates, making causal analysis of policy effects a realistic downstream application

The same pipeline runs on any InsideAirbnb city by swapping the city config. Porto is the reference market.

---

## Who Uses This

| Persona | Use case |
|---------|----------|
| **Property investors** | Identify undervalued micro-markets before competitors; quantify revenue uplift from specific amenities |
| **Airbnb hosts** | Understand why nearby listings earn more; get actionable improvement signals with SHAP explanations |
| **Real estate analysts** | Model STR revenue potential for acquisition underwriting; map opportunity gaps between predicted and actual yield |
| **Urban researchers** | Study how tourism infrastructure, transit access, and neighborhood change affect STR markets |
| **City planners / policy teams** | Analyze spatial concentration of STR activity; measure causal impact of regulation on supply and pricing |
| **Applied AI / ML engineers** | Reference implementation of geospatial feature engineering, SHAP explainability, and causal inference on a real dataset |

---

## Why This Is Different

Most Airbnb projects predict listing prices. This platform answers harder questions:

| Question | Module |
|----------|--------|
| What drives pricing and occupancy? | Geospatial Feature Engineering + SHAP |
| Which neighborhoods are undervalued? | Opportunity Map (predicted vs actual revenue) |
| Does adding a pool actually increase revenue? | Causal Inference (DoWhy + EconML) |
| Where should I invest? | Investment Intelligence Module |
| Why is this listing expensive? | LLM Property Analyst |

---

## Architecture

```
InsideAirbnb (Porto)          OpenStreetMap
  listings.csv.gz               OSMnx → POIs
  calendar.csv.gz     ──────►
  reviews.csv.gz
  neighbourhoods.geojson
          │
          ▼
  ┌─────────────────────┐
  │   DuckDB Warehouse  │  ← single local file, zero-config
  │  listings           │
  │  calendar_features  │
  │  listing_features   │
  └──────────┬──────────┘
             │
     ┌───────┴────────────────────┐
     │                            │
     ▼                            ▼
  ML Pipeline               Geospatial Engine
  LightGBM (price)            H3 hexagons (res-8)
  LightGBM (occupancy)        GeoPandas / OSMnx
  SHAP explainability         Haversine proximity
     │
     ▼
  predict.py (batch)
  listing_predictions
  hex_aggregates
  shap_global / hex_shap
     │
     ▼
  ┌─────────────────────┐
  │   FastAPI Backend   │  ← read-only DuckDB, <5ms responses
  │  GET /api/stats     │
  │  GET /api/hexagons  │
  │  GET /api/listings  │
  │  GET /api/shap/*    │
  └──────────┬──────────┘
             │
     React + Deck.gl Frontend
     H3HexagonLayer · ScatterplotLayer
     SHAP bars · hex detail panel
```

---

## Data Sources (Porto, Portugal)

| File | Description | Size |
|------|-------------|------|
| `data/raw/airbnb/listings.csv.gz` | ~15K Airbnb listings with full metadata | 7.6 MB |
| `data/raw/airbnb/calendar.csv.gz` | Daily availability/price per listing | 14 MB |
| `data/raw/airbnb/reviews.csv.gz` | Guest reviews with dates | 132 MB |
| `data/raw/airbnb/neighbourhoods.geojson` | Neighbourhood boundaries | 2.8 MB |
| `data/raw/osm/` | OSM POI data fetched via OSMnx | fetched at runtime |

Source: [Inside Airbnb](http://insideairbnb.com/get-the-data/)

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Data storage | DuckDB |
| Data processing | Polars, PyArrow |
| Geospatial | GeoPandas, OSMnx, H3, Shapely |
| ML | LightGBM, Scikit-Learn |
| Explainability | SHAP (TreeSHAP) |
| API | FastAPI, uvicorn |
| Frontend | React, Deck.gl (H3HexagonLayer) |
| Causal inference | DoWhy, EconML (planned) |
| AI / LLM | OpenAI, LangGraph (planned) |

---

## Quick Start

```bash
# 1. Clone and install
git clone <repo>
cd geoai_airbnb
pip install -e ".[dev]"

# 2. Download data (or place files in data/raw/airbnb/)
# See: https://insideairbnb.com/get-the-data/ → Portugal → Porto

# 3. Initialize warehouse and ingest
python scripts/ingest_porto.py

# 4. Compute geospatial features
python -m geoai.features.runner

# 5. Train ML models, evaluate, and compute predictions + SHAP
python -m geoai.models.runner

# 6. Start the API
uvicorn geoai.api.main:app --reload --port 8000

# 7. Verify endpoints
curl http://localhost:8000/api/stats
curl "http://localhost:8000/api/hexagons?mode=price"
# Interactive docs: http://localhost:8000/docs
```

---

## Project Structure

```
geoai_airbnb/
├── data/
│   ├── raw/airbnb/          # InsideAirbnb CSVs
│   ├── raw/osm/             # OSM raw cache
│   └── models/              # trained .pkl artifacts
├── src/geoai/
│   ├── config.py            # DB_PATH and constants
│   ├── database/
│   │   └── warehouse.py     # DuckDB schema init
│   ├── ingestion/
│   │   ├── airbnb.py        # listings + calendar + reviews
│   │   └── osm.py           # OSMnx POI fetcher
│   ├── features/
│   │   ├── geo.py           # H3 assignment, Haversine distances
│   │   ├── poi.py           # POI density + walkability
│   │   ├── competition.py   # nearby listing counts + price index
│   │   ├── calendar.py      # occupancy rate features
│   │   └── runner.py        # pipeline entry point
│   ├── models/
│   │   ├── features.py      # build_feature_matrix, prepare_X_y_*
│   │   ├── price.py         # LightGBM price model
│   │   ├── occupancy.py     # LightGBM occupancy model
│   │   ├── evaluate.py      # RMSE / MAE evaluation report
│   │   ├── predict.py       # batch predict → 4 DuckDB tables
│   │   └── runner.py        # train → evaluate → predict pipeline
│   ├── explainability/
│   │   └── shap_analysis.py # global + per-listing SHAP (offline)
│   └── api/
│       ├── main.py          # FastAPI app + lifespan (read-only DuckDB)
│       ├── deps.py          # get_db dependency
│       ├── schemas.py       # Pydantic response models
│       └── routes/
│           ├── stats.py     # GET /api/stats
│           ├── hexagons.py  # GET /api/hexagons[/{hex_id}]
│           ├── listings.py  # GET /api/listings
│           └── shap.py      # GET /api/shap/global, /api/shap/{hex_id}
├── tests/
│   ├── ingestion/
│   ├── features/
│   ├── models/
│   └── api/                 # FastAPI route tests (TestClient + fixture DB)
├── scripts/
│   └── ingest_porto.py      # Phase 1 ingestion runner
├── docs/
│   ├── adr/                 # Architecture Decision Records
│   └── superpowers/         # plans + specs
└── pyproject.toml
```

---

## Porting to Another City

The pipeline is city-agnostic. Porto is the reference market; swapping it out requires data and four config changes — no code rewrites.

**Prerequisites**
- InsideAirbnb snapshot for the target city: `listings.csv.gz`, `calendar.csv.gz`, `reviews.csv.gz`, `neighbourhoods.geojson` — available at [insideairbnb.com/get-the-data](https://insideairbnb.com/get-the-data/)
- The city must have reasonable OSM coverage (most European and major global cities qualify)

**Changes required**

| What | Where | Change |
|------|-------|--------|
| InsideAirbnb URLs | `src/geoai/config.py` | Replace `AIRBNB_PORTO_URL` and `CALENDAR_URL` with target city URLs |
| OSM place name | `src/geoai/config.py` | Set `OSM_CITY = "Lisbon, Portugal"` (any string `osmnx` accepts) |
| City center + landmarks | `src/geoai/config.py` | Replace `PORTO_CENTER_LAT/LON`, `PORTO_LANDMARKS`, `PORTO_AIRPORT` with city equivalents |
| Ingestion script | `scripts/ingest_porto.py` | Copy to `ingest_<city>.py`; update file paths to point at new `data/raw/airbnb/` files |
| Frontend viewport | `frontend/src/...` | Update initial map center lat/lon and zoom level |

**Steps**

```bash
# 1. Place new city files in data/raw/airbnb/
# 2. Update config.py constants above
# 3. Re-run the full pipeline — models train from scratch on new data
python scripts/ingest_<city>.py
python -m geoai.features.runner
python -m geoai.models.runner
uvicorn geoai.api.main:app --reload --port 8000
```

All geospatial logic (H3 binning, Haversine proximity, OSM POI density) is coordinate-based and city-independent. ML models train fresh on each city's data — no transfer or fine-tuning needed. The only city-specific knowledge in the codebase is in `config.py`.

**What won't transfer automatically**
- Neighborhood boundary quality depends on InsideAirbnb's per-city GeoJSON; some cities have coarser granularity
- Calendar-based occupancy features require at least one full year of calendar data for reliable estimates
- Cities with fewer than ~5,000 active listings may produce noisier model outputs

---

## Development Roadmap

| Phase | Scope | Status |
|-------|-------|--------|
| 1 | Data ingestion + DuckDB warehouse | ✅ Complete |
| 2 | Geospatial feature engineering | ✅ Complete |
| 3 | ML models (price, occupancy, revenue) | ✅ Complete |
| 4 | SHAP explainability | ✅ Complete |
| 5 | FastAPI backend (predictions + SHAP API) | ✅ Complete |
| 6 | React + Deck.gl frontend | 🔄 In Progress |
| 7 | Causal inference + LLM analyst | ⏳ Planned |