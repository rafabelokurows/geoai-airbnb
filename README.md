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
