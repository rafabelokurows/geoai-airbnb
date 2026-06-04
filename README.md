# GeoAI Short-Term Rental Intelligence Platform

> A geospatial AI platform that helps investors, hosts, and analysts understand what drives Airbnb performance — combining geospatial analytics, machine learning, causal inference, and LLM-powered insights.

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
  reviews.csv.gz                Census / Tourism
  neighbourhoods.geojson
          │
          ▼
  ┌─────────────────────┐
  │   DuckDB Warehouse  │  ← single local file, zero-config
  │  listings           │
  │  calendar_features  │
  │  poi_features        │
  │  neighborhoods       │
  └──────────┬──────────┘
             │
     ┌───────┴────────┐
     │                │
     ▼                ▼
  ML Layer      Geospatial Engine
  CatBoost        H3 hexagons
  LightGBM        GeoPandas
  SHAP            PyDeck maps
     │                │
     └───────┬────────┘
             │
     Causal Inference
     DoWhy / EconML
             │
     LLM Insight Engine
     OpenAI / LangGraph
             │
     FastAPI + Streamlit
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
| Data processing | Polars |
| Geospatial | GeoPandas, OSMnx, H3, Shapely |
| ML | CatBoost, LightGBM, Scikit-Learn |
| Explainability | SHAP |
| Causal inference | DoWhy, EconML |
| AI / LLM | OpenAI, LangGraph |
| API | FastAPI |
| Visualization | PyDeck, Deck.gl |
| Dashboard | Streamlit |

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

# 4. Run dashboard (Phase 5+)
streamlit run app/dashboard.py

# 5. Run API (Phase 6+)
uvicorn src.geoai.api.main:app --reload
```

---

## Project Structure

```
geoai_airbnb/
├── data/
│   ├── raw/airbnb/          # InsideAirbnb CSVs
│   └── raw/osm/             # OSM raw cache
├── src/geoai/
│   ├── config.py            # paths and constants
│   ├── database/
│   │   └── warehouse.py     # DuckDB init + connection
│   ├── ingestion/
│   │   ├── airbnb.py        # listings + calendar + reviews
│   │   └── osm.py           # OSMnx POI fetcher
│   ├── features/            # geospatial feature engineering (Phase 2)
│   ├── ml/                  # price, occupancy, revenue models (Phase 3)
│   ├── explainability/      # SHAP analysis (Phase 4)
│   ├── causal/              # DoWhy / EconML (Phase 7)
│   ├── llm/                 # LangGraph analyst (Phase 7)
│   └── api/                 # FastAPI endpoints (Phase 6)
├── app/                     # Streamlit dashboard (Phase 5)
├── scripts/
│   └── ingest_porto.py      # Phase 1 ingestion runner
├── tests/
├── docs/
│   ├── adr/                 # Architecture Decision Records
│   └── superpowers/plans/   # implementation plans
└── pyproject.toml
```

---

## Development Roadmap

| Phase | Scope | Status |
|-------|-------|--------|
| 1 | Data ingestion + DuckDB warehouse | ✅ Complete |
| 2 | Geospatial feature engineering | ⏳ Pending |
| 3 | ML models (price, occupancy, revenue) | ⏳ Pending |
| 4 | SHAP explainability | ⏳ Pending |
| 5 | Interactive maps + Streamlit dashboard | ⏳ Pending |
| 6 | FastAPI + deployment | ⏳ Pending |
| 7 | Causal inference + LLM analyst | ⏳ Pending |

---

## Target Roles

Applied AI Engineer · Data Scientist · Geospatial Data Scientist · ML Engineer · Analytics Engineer
