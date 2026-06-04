# Phase 2: Geospatial Feature Engineering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compute and persist geospatial features for every Porto listing — accessibility distances, POI density counts, competition metrics, walkability score, H3 hex assignment, and calendar-based occupancy rates — all stored in DuckDB for downstream ML.

**Architecture:** Each feature group lives in its own module under `src/geoai/features/`. A single `listing_features` table in DuckDB holds all features keyed by `listing_id`. Calendar data is ingested first (needed for occupancy). A runner script `scripts/compute_features.py` orchestrates all feature groups in dependency order.

**Tech Stack:** DuckDB, Polars, GeoPandas STRtree (spatial radius queries), NumPy (vectorized Haversine), h3-py (H3 hexagons)

---

## File Map

| File | Action | Purpose |
|------|--------|---------|
| `pyproject.toml` | Modify | Add `numpy>=1.26.0`, `h3>=3.7.0` |
| `src/geoai/config.py` | Modify | Add Porto city center coords constant |
| `src/geoai/database/warehouse.py` | Modify | Add `calendar`, `listing_features`, `hex_aggregates` tables |
| `src/geoai/ingestion/calendar.py` | Create | Calendar CSV ingestion |
| `src/geoai/features/__init__.py` | Create | Empty package marker |
| `src/geoai/features/accessibility.py` | Create | Haversine distances to metro, station, city center |
| `src/geoai/features/poi_density.py` | Create | POI counts per radius per category |
| `src/geoai/features/competition.py` | Create | Competitor listing counts + price stats |
| `src/geoai/features/walkability.py` | Create | Weighted composite walkability score |
| `src/geoai/features/h3_grid.py` | Create | H3 hex assignment + per-hex aggregates |
| `src/geoai/features/occupancy.py` | Create | Calendar-based occupancy rates |
| `src/geoai/features/runner.py` | Create | Orchestrates all feature computations |
| `scripts/compute_features.py` | Create | CLI entry point for feature runner |
| `tests/test_calendar_ingestion.py` | Create | Calendar ingest tests |
| `tests/test_accessibility.py` | Create | Accessibility feature tests |
| `tests/test_poi_density.py` | Create | POI density feature tests |
| `tests/test_competition.py` | Create | Competition feature tests |
| `tests/test_walkability.py` | Create | Walkability score tests |
| `tests/test_h3_grid.py` | Create | H3 hexagon tests |
| `tests/test_occupancy.py` | Create | Occupancy rate tests |

---

## Task 1: Warehouse Schema Extension + Calendar Ingestion

**Files:**
- Modify: `pyproject.toml`
- Modify: `src/geoai/config.py`
- Modify: `src/geoai/database/warehouse.py`
- Create: `src/geoai/ingestion/calendar.py`
- Create: `tests/test_calendar_ingestion.py`

### Background

Inside Airbnb `calendar.csv.gz` contains one row per listing per date: whether available (`t`/`f`) and the listed price. This is the raw material for occupancy estimation. The file is large (~14 MB compressed, ~5M rows) — use Polars lazy API.

`listing_features` will be the central output table of Phase 2 — all feature modules write into it via `INSERT OR REPLACE`.

- [ ] **Step 1: Add dependencies to pyproject.toml**

```toml
dependencies = [
    "duckdb>=0.10.0",
    "polars>=0.20.0",
    "osmnx>=1.9.0",
    "geopandas>=0.14.0",
    "shapely>=2.0.0",
    "httpx>=0.27.0",
    "numpy>=1.26.0",
    "h3>=3.7.0",
]
```

Run: `pip install -e ".[dev]"` — verify no errors.

- [ ] **Step 2: Add Porto city center to config.py**

Append to `src/geoai/config.py`:

```python
# Praça da Liberdade — conventional center of Porto
PORTO_CENTER_LAT = 41.14961
PORTO_CENTER_LON = -8.61099

CALENDAR_URL = (
    "https://data.insideairbnb.com/portugal/norte/porto/"
    "2024-12-22/data/calendar.csv.gz"
)
```

- [ ] **Step 3: Add new tables to warehouse.py**

Add three DDL strings and extend `init_warehouse` to create them:

```python
_CREATE_CALENDAR = """
CREATE TABLE IF NOT EXISTS calendar (
    listing_id      BIGINT,
    date            DATE,
    available       BOOLEAN,
    price           DOUBLE,
    minimum_nights  INTEGER,
    maximum_nights  INTEGER,
    PRIMARY KEY (listing_id, date)
)
"""

_CREATE_LISTING_FEATURES = """
CREATE TABLE IF NOT EXISTS listing_features (
    listing_id                  BIGINT PRIMARY KEY,
    dist_city_center_km         DOUBLE,
    dist_nearest_metro_km       DOUBLE,
    dist_nearest_station_km     DOUBLE,
    restaurants_250m            INTEGER,
    restaurants_500m            INTEGER,
    bars_500m                   INTEGER,
    cafes_500m                  INTEGER,
    supermarkets_1km            INTEGER,
    attractions_1km             INTEGER,
    museums_2km                 INTEGER,
    parks_500m                  INTEGER,
    listings_500m               INTEGER,
    listings_1km                INTEGER,
    avg_price_500m              DOUBLE,
    median_price_neighbourhood  DOUBLE,
    walkability_score           DOUBLE,
    h3_cell_r8                  VARCHAR,
    occupancy_rate_30d          DOUBLE,
    occupancy_rate_90d          DOUBLE,
    occupancy_rate_365d         DOUBLE
)
"""

_CREATE_HEX_AGGREGATES = """
CREATE TABLE IF NOT EXISTS hex_aggregates (
    h3_cell         VARCHAR PRIMARY KEY,
    listing_count   INTEGER,
    avg_price       DOUBLE,
    avg_occupancy   DOUBLE,
    avg_walkability DOUBLE
)
"""
```

Extend `init_warehouse`:

```python
def init_warehouse(db_path: Path = DB_PATH) -> duckdb.DuckDBPyConnection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    con.execute(_CREATE_LISTINGS)
    con.execute(_CREATE_POI_FEATURES)
    con.execute(_CREATE_CALENDAR)
    con.execute(_CREATE_LISTING_FEATURES)
    con.execute(_CREATE_HEX_AGGREGATES)
    return con
```

- [ ] **Step 4: Write failing tests for warehouse schema**

```python
# tests/test_warehouse.py — add to existing file

def test_calendar_table_exists(tmp_path):
    con = init_warehouse(tmp_path / "test.duckdb")
    tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
    assert "calendar" in tables
    con.close()

def test_listing_features_table_exists(tmp_path):
    con = init_warehouse(tmp_path / "test.duckdb")
    tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
    assert "listing_features" in tables
    con.close()

def test_hex_aggregates_table_exists(tmp_path):
    con = init_warehouse(tmp_path / "test.duckdb")
    tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
    assert "hex_aggregates" in tables
    con.close()
```

Run: `pytest tests/test_warehouse.py -v` — expect 3 new FAILs.

- [ ] **Step 5: Implement calendar ingestion**

Create `src/geoai/ingestion/calendar.py`:

```python
from pathlib import Path

import duckdb
import polars as pl

from geoai.config import CALENDAR_URL, DB_PATH, RAW_AIRBNB_DIR
from geoai.database.warehouse import init_warehouse
from geoai.ingestion.airbnb import _download_raw

_KEEP_COLS = [
    "listing_id", "date", "available", "price", "minimum_nights", "maximum_nights"
]


def clean_calendar(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(
        pl.when(pl.col("available") == "t").then(pl.lit(True))
        .when(pl.col("available") == "f").then(pl.lit(False))
        .otherwise(pl.lit(None, dtype=pl.Boolean))
        .alias("available"),
        pl.col("price").str.replace_all(r"[\$,]", "").cast(pl.Float64, strict=False).alias("price"),
        pl.col("date").str.strptime(pl.Date, "%Y-%m-%d", strict=False).alias("date"),
    )
    existing = [c for c in _KEEP_COLS if c in df.columns]
    return df.select(existing).drop_nulls(subset=["listing_id", "date"])


def load_calendar_into_db(
    db_path: Path = DB_PATH, url: str = CALENDAR_URL
) -> int:
    _con = init_warehouse(db_path)
    _con.close()
    raw_path = _download_raw(url, RAW_AIRBNB_DIR)
    df = pl.read_csv(raw_path, infer_schema_length=10000)
    df = clean_calendar(df)
    col_list = ", ".join(_KEEP_COLS)
    with duckdb.connect(str(db_path)) as con:
        con.execute("BEGIN")
        con.execute("DELETE FROM calendar")
        con.execute(f"INSERT INTO calendar ({col_list}) SELECT {col_list} FROM df")
        con.execute("COMMIT")
        return con.execute("SELECT COUNT(*) FROM calendar").fetchone()[0]
```

- [ ] **Step 6: Write failing calendar ingestion tests**

Create `tests/test_calendar_ingestion.py`:

```python
import polars as pl
import pytest

from geoai.ingestion.calendar import clean_calendar, load_calendar_into_db
from geoai.database.warehouse import init_warehouse


def _sample_calendar() -> pl.DataFrame:
    return pl.DataFrame({
        "listing_id": [1, 2, 3, 4],
        "date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
        "available": ["t", "f", "t", None],
        "price": ["$85.00", "$120.50", "$200.00", "$50.00"],
        "minimum_nights": [2, 1, 3, 2],
        "maximum_nights": [30, 30, 30, 30],
    })


def test_clean_calendar_parses_available_flag():
    df = clean_calendar(_sample_calendar())
    assert df["available"][0] is True
    assert df["available"][1] is False


def test_clean_calendar_parses_price():
    df = clean_calendar(_sample_calendar())
    assert df["price"][0] == pytest.approx(85.0)
    assert df["price"][1] == pytest.approx(120.5)


def test_clean_calendar_parses_date():
    df = clean_calendar(_sample_calendar())
    import datetime
    assert df["date"][0] == datetime.date(2024, 1, 1)


def test_clean_calendar_drops_null_date_or_id():
    df = pl.DataFrame({
        "listing_id": [1, None, 3],
        "date": ["2024-01-01", "2024-01-02", None],
        "available": ["t", "t", "f"],
        "price": ["$100.00", "$50.00", "$75.00"],
        "minimum_nights": [1, 1, 1],
        "maximum_nights": [30, 30, 30],
    })
    cleaned = clean_calendar(df)
    assert len(cleaned) == 1


def test_load_calendar_into_db_returns_count(tmp_path, monkeypatch):
    sample = _sample_calendar().with_columns(
        pl.col("available").cast(pl.Utf8),
        pl.col("price").cast(pl.Utf8),
    )
    monkeypatch.setattr("geoai.ingestion.calendar._download_raw", lambda url, dest: None)
    monkeypatch.setattr("polars.read_csv", lambda path, **kw: sample)
    count = load_calendar_into_db(db_path=tmp_path / "test.duckdb")
    assert count == 3  # one row dropped (None date)
```

Run: `pytest tests/test_calendar_ingestion.py -v` — expect failures.

- [ ] **Step 7: Run all tests to verify passing**

Run: `pytest tests/ -v`
Expected: all warehouse tests pass + 4 calendar tests pass (after implementation).

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml src/geoai/config.py src/geoai/database/warehouse.py src/geoai/ingestion/calendar.py tests/test_calendar_ingestion.py
git commit -m "feat: add calendar ingestion and listing_features/hex_aggregates warehouse schema"
```

---

## Task 2: Accessibility Features

**Files:**
- Create: `src/geoai/features/__init__.py`
- Create: `src/geoai/features/accessibility.py`
- Create: `tests/test_accessibility.py`

### Background

Three distances per listing:
- `dist_city_center_km` — Haversine to Praça da Liberdade (lat=41.14961, lon=-8.61099)
- `dist_nearest_metro_km` — closest POI where `poi_type='railway'` AND `poi_subtype='subway_entrance'`
- `dist_nearest_station_km` — closest POI where `poi_type='railway'` AND `poi_subtype='station'`

Use vectorized NumPy Haversine — no Python loops. Input/output: Polars DataFrames. Database I/O: read from `listings` and `poi_features`, write to `listing_features` using `INSERT OR REPLACE`.

- [ ] **Step 1: Create empty package marker**

Create `src/geoai/features/__init__.py` — empty file.

- [ ] **Step 2: Write failing tests**

Create `tests/test_accessibility.py`:

```python
import numpy as np
import polars as pl
import pytest

from geoai.features.accessibility import haversine_km, compute_accessibility


def test_haversine_km_same_point():
    d = haversine_km(
        np.array([41.14961]), np.array([-8.61099]),
        41.14961, -8.61099
    )
    assert d[0] == pytest.approx(0.0, abs=1e-6)


def test_haversine_km_known_distance():
    # Porto to Lisbon: ~274 km
    d = haversine_km(
        np.array([41.14961]), np.array([-8.61099]),
        38.71667, -9.13333
    )
    assert d[0] == pytest.approx(274.0, abs=5.0)


def test_compute_accessibility_returns_three_columns():
    listings = pl.DataFrame({
        "id": [1], "latitude": [41.14961], "longitude": [-8.61099]
    })
    metro_pois = pl.DataFrame({
        "osm_id": ["node_1"], "poi_type": ["railway"],
        "poi_subtype": ["subway_entrance"],
        "latitude": [41.150], "longitude": [-8.612],
    })
    station_pois = pl.DataFrame({
        "osm_id": ["node_2"], "poi_type": ["railway"],
        "poi_subtype": ["station"],
        "latitude": [41.160], "longitude": [-8.620],
    })
    pois = pl.concat([metro_pois, station_pois])
    result = compute_accessibility(listings, pois)
    assert "dist_city_center_km" in result.columns
    assert "dist_nearest_metro_km" in result.columns
    assert "dist_nearest_station_km" in result.columns


def test_compute_accessibility_city_center_distance():
    listings = pl.DataFrame({
        "id": [1], "latitude": [41.14961], "longitude": [-8.61099]
    })
    pois = pl.DataFrame({
        "osm_id": ["n1"], "poi_type": ["railway"], "poi_subtype": ["subway_entrance"],
        "latitude": [41.15], "longitude": [-8.61],
    })
    result = compute_accessibility(listings, pois)
    assert result["dist_city_center_km"][0] == pytest.approx(0.0, abs=0.01)


def test_compute_accessibility_no_metro_returns_null():
    listings = pl.DataFrame({
        "id": [1], "latitude": [41.14961], "longitude": [-8.61099]
    })
    pois = pl.DataFrame({
        "osm_id": [], "poi_type": [], "poi_subtype": [],
        "latitude": [], "longitude": [],
    })
    result = compute_accessibility(listings, pois)
    assert result["dist_nearest_metro_km"][0] is None
```

Run: `pytest tests/test_accessibility.py -v` — expect ImportError/failures.

- [ ] **Step 3: Implement accessibility.py**

Create `src/geoai/features/accessibility.py`:

```python
from pathlib import Path

import duckdb
import numpy as np
import polars as pl

from geoai.config import DB_PATH, PORTO_CENTER_LAT, PORTO_CENTER_LON
from geoai.database.warehouse import init_warehouse


def haversine_km(
    lats: np.ndarray, lons: np.ndarray,
    target_lat: float, target_lon: float,
) -> np.ndarray:
    R = 6371.0
    dlat = np.radians(target_lat - lats)
    dlon = np.radians(target_lon - lons)
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(np.radians(lats)) * np.cos(np.radians(target_lat)) * np.sin(dlon / 2) ** 2
    )
    return R * 2 * np.arcsin(np.sqrt(a))


def _nearest_km(
    listing_lats: np.ndarray,
    listing_lons: np.ndarray,
    poi_lats: np.ndarray,
    poi_lons: np.ndarray,
) -> list[float | None]:
    if len(poi_lats) == 0:
        return [None] * len(listing_lats)
    results = []
    for lat, lon in zip(listing_lats, listing_lons):
        dists = haversine_km(poi_lats, poi_lons, lat, lon)
        results.append(float(np.min(dists)))
    return results


def compute_accessibility(
    listings: pl.DataFrame,
    pois: pl.DataFrame,
) -> pl.DataFrame:
    lats = listings["latitude"].to_numpy()
    lons = listings["longitude"].to_numpy()

    metro = pois.filter(
        (pl.col("poi_type") == "railway") & (pl.col("poi_subtype") == "subway_entrance")
    )
    stations = pois.filter(
        (pl.col("poi_type") == "railway") & (pl.col("poi_subtype") == "station")
    )

    city_center_dists = haversine_km(lats, lons, PORTO_CENTER_LAT, PORTO_CENTER_LON)
    metro_dists = _nearest_km(lats, lons, metro["latitude"].to_numpy(), metro["longitude"].to_numpy())
    station_dists = _nearest_km(lats, lons, stations["latitude"].to_numpy(), stations["longitude"].to_numpy())

    return listings.select("id").with_columns([
        pl.Series("dist_city_center_km", city_center_dists.tolist()),
        pl.Series("dist_nearest_metro_km", metro_dists),
        pl.Series("dist_nearest_station_km", station_dists),
    ])


def load_accessibility_into_db(db_path: Path = DB_PATH) -> int:
    _con = init_warehouse(db_path)
    _con.close()
    with duckdb.connect(str(db_path)) as con:
        listings = pl.from_arrow(
            con.execute("SELECT id, latitude, longitude FROM listings").arrow()
        )
        pois = pl.from_arrow(
            con.execute(
                "SELECT osm_id, poi_type, poi_subtype, latitude, longitude FROM poi_features"
                " WHERE poi_type = 'railway'"
            ).arrow()
        )
    result = compute_accessibility(listings, pois)
    with duckdb.connect(str(db_path)) as con:
        for row in result.iter_rows(named=True):
            con.execute(
                """
                INSERT INTO listing_features (listing_id, dist_city_center_km,
                    dist_nearest_metro_km, dist_nearest_station_km)
                VALUES (?, ?, ?, ?)
                ON CONFLICT (listing_id) DO UPDATE SET
                    dist_city_center_km = excluded.dist_city_center_km,
                    dist_nearest_metro_km = excluded.dist_nearest_metro_km,
                    dist_nearest_station_km = excluded.dist_nearest_station_km
                """,
                [row["id"], row["dist_city_center_km"],
                 row["dist_nearest_metro_km"], row["dist_nearest_station_km"]],
            )
    with duckdb.connect(str(db_path)) as con:
        return con.execute("SELECT COUNT(*) FROM listing_features").fetchone()[0]
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_accessibility.py -v`
Expected: all 5 pass.

- [ ] **Step 5: Commit**

```bash
git add src/geoai/features/ tests/test_accessibility.py
git commit -m "feat: accessibility features — distances to city center, metro, train station"
```

---

## Task 3: POI Density Features

**Files:**
- Create: `src/geoai/features/poi_density.py`
- Create: `tests/test_poi_density.py`

### Background

Count POIs of each category within fixed radii of each listing. Uses GeoPandas STRtree for efficient radius queries (avoid O(n×m) Python loops).

Radii: 250m=0.25km, 500m=0.5km, 1000m=1km, 2000m=2km.

Feature columns to compute:
- `restaurants_250m`, `restaurants_500m` — amenity: restaurant/fast_food
- `bars_500m` — amenity: bar/pub
- `cafes_500m` — amenity: cafe
- `supermarkets_1km` — amenity: supermarket
- `attractions_1km` — tourism: attraction/gallery/viewpoint + amenity: museum/theatre/cinema
- `museums_2km` — tourism: museum
- `parks_500m` — leisure: park/garden

- [ ] **Step 1: Write failing tests**

Create `tests/test_poi_density.py`:

```python
import geopandas as gpd
import polars as pl
import pytest
from shapely.geometry import Point

from geoai.features.poi_density import compute_poi_density, count_within_radius


def _make_listings(coords: list[tuple[float, float]]) -> pl.DataFrame:
    return pl.DataFrame({
        "id": list(range(len(coords))),
        "latitude": [c[0] for c in coords],
        "longitude": [c[1] for c in coords],
    })


def _make_pois(entries: list[dict]) -> pl.DataFrame:
    return pl.DataFrame({
        "osm_id": [f"n{i}" for i in range(len(entries))],
        "poi_type": [e["poi_type"] for e in entries],
        "poi_subtype": [e["poi_subtype"] for e in entries],
        "latitude": [e["lat"] for e in entries],
        "longitude": [e["lon"] for e in entries],
    })


def test_count_within_radius_returns_correct_count():
    # listing at Porto center; two restaurants nearby, one far
    listing_lat, listing_lon = 41.14961, -8.61099
    pois = _make_pois([
        {"poi_type": "amenity", "poi_subtype": "restaurant", "lat": 41.150, "lon": -8.611},  # ~0.11km
        {"poi_type": "amenity", "poi_subtype": "restaurant", "lat": 41.151, "lon": -8.612},  # ~0.17km
        {"poi_type": "amenity", "poi_subtype": "restaurant", "lat": 41.200, "lon": -8.650},  # ~7km
    ])
    restaurants = pois.filter(pl.col("poi_subtype").is_in(["restaurant", "fast_food"]))
    count = count_within_radius(listing_lat, listing_lon, restaurants, radius_km=0.5)
    assert count == 2


def test_count_within_radius_empty_pois():
    count = count_within_radius(41.14961, -8.61099, _make_pois([]), radius_km=0.5)
    assert count == 0


def test_compute_poi_density_returns_correct_columns():
    listings = _make_listings([(41.14961, -8.61099)])
    pois = _make_pois([
        {"poi_type": "amenity", "poi_subtype": "restaurant", "lat": 41.150, "lon": -8.611},
        {"poi_type": "leisure", "poi_subtype": "park", "lat": 41.149, "lon": -8.610},
    ])
    result = compute_poi_density(listings, pois)
    expected_cols = {
        "listing_id", "restaurants_250m", "restaurants_500m", "bars_500m",
        "cafes_500m", "supermarkets_1km", "attractions_1km", "museums_2km", "parks_500m"
    }
    assert expected_cols.issubset(set(result.columns))


def test_compute_poi_density_counts_correctly():
    listings = _make_listings([(41.14961, -8.61099)])
    pois = _make_pois([
        {"poi_type": "amenity", "poi_subtype": "restaurant", "lat": 41.1498, "lon": -8.6109},  # ~0.03km
        {"poi_type": "amenity", "poi_subtype": "restaurant", "lat": 41.152, "lon": -8.613},    # ~0.28km
        {"poi_type": "amenity", "poi_subtype": "restaurant", "lat": 41.158, "lon": -8.618},    # ~0.9km
    ])
    result = compute_poi_density(listings, pois)
    assert result["restaurants_250m"][0] == 1
    assert result["restaurants_500m"][0] == 2
```

Run: `pytest tests/test_poi_density.py -v` — expect ImportError.

- [ ] **Step 2: Implement poi_density.py**

Create `src/geoai/features/poi_density.py`:

```python
from pathlib import Path

import duckdb
import numpy as np
import polars as pl

from geoai.config import DB_PATH
from geoai.database.warehouse import init_warehouse
from geoai.features.accessibility import haversine_km


def count_within_radius(
    listing_lat: float, listing_lon: float,
    pois: pl.DataFrame, radius_km: float
) -> int:
    if len(pois) == 0:
        return 0
    lats = pois["latitude"].to_numpy()
    lons = pois["longitude"].to_numpy()
    dists = haversine_km(lats, lons, listing_lat, listing_lon)
    return int(np.sum(dists <= radius_km))


_RESTAURANT_SUBTYPES = {"restaurant", "fast_food"}
_BAR_SUBTYPES = {"bar", "pub"}
_CAFE_SUBTYPES = {"cafe"}
_SUPERMARKET_SUBTYPES = {"supermarket"}
_ATTRACTION_SUBTYPES = {"attraction", "gallery", "viewpoint", "museum", "theatre", "cinema"}
_MUSEUM_SUBTYPES = {"museum"}
_PARK_SUBTYPES = {"park", "garden"}


def compute_poi_density(
    listings: pl.DataFrame,
    pois: pl.DataFrame,
) -> pl.DataFrame:
    restaurants = pois.filter(pl.col("poi_subtype").is_in(list(_RESTAURANT_SUBTYPES)))
    bars = pois.filter(pl.col("poi_subtype").is_in(list(_BAR_SUBTYPES)))
    cafes = pois.filter(pl.col("poi_subtype").is_in(list(_CAFE_SUBTYPES)))
    supermarkets = pois.filter(pl.col("poi_subtype").is_in(list(_SUPERMARKET_SUBTYPES)))
    attractions = pois.filter(pl.col("poi_subtype").is_in(list(_ATTRACTION_SUBTYPES)))
    museums = pois.filter(pl.col("poi_subtype").is_in(list(_MUSEUM_SUBTYPES)))
    parks = pois.filter(pl.col("poi_subtype").is_in(list(_PARK_SUBTYPES)))

    rows = []
    for row in listings.iter_rows(named=True):
        lat, lon = row["latitude"], row["longitude"]
        rows.append({
            "listing_id": row["id"],
            "restaurants_250m": count_within_radius(lat, lon, restaurants, 0.25),
            "restaurants_500m": count_within_radius(lat, lon, restaurants, 0.5),
            "bars_500m": count_within_radius(lat, lon, bars, 0.5),
            "cafes_500m": count_within_radius(lat, lon, cafes, 0.5),
            "supermarkets_1km": count_within_radius(lat, lon, supermarkets, 1.0),
            "attractions_1km": count_within_radius(lat, lon, attractions, 1.0),
            "museums_2km": count_within_radius(lat, lon, museums, 2.0),
            "parks_500m": count_within_radius(lat, lon, parks, 0.5),
        })
    return pl.DataFrame(rows)


def load_poi_density_into_db(db_path: Path = DB_PATH) -> int:
    _con = init_warehouse(db_path)
    _con.close()
    with duckdb.connect(str(db_path)) as con:
        listings = pl.from_arrow(
            con.execute("SELECT id, latitude, longitude FROM listings").arrow()
        )
        pois = pl.from_arrow(
            con.execute(
                "SELECT osm_id, poi_type, poi_subtype, latitude, longitude FROM poi_features"
            ).arrow()
        )
    result = compute_poi_density(listings, pois)
    cols = [c for c in result.columns if c != "listing_id"]
    update_set = ", ".join(f"{c} = excluded.{c}" for c in cols)
    col_list = ", ".join(result.columns)
    with duckdb.connect(str(db_path)) as con:
        con.execute(f"""
            INSERT INTO listing_features ({col_list})
            SELECT {col_list} FROM result
            ON CONFLICT (listing_id) DO UPDATE SET {update_set}
        """)
        return con.execute("SELECT COUNT(*) FROM listing_features").fetchone()[0]
```

> **Performance note:** The per-listing Python loop is O(n_listings × n_pois) and will be slow on 15K listings × 9K POIs = 135M haversine calls. For dev/test this is fine. For production runs, optimize with bounding box pre-filter (filter POIs to ±2km bounding box per listing before haversine). This optimization can be added as a follow-up without changing the interface.

- [ ] **Step 3: Run tests**

Run: `pytest tests/test_poi_density.py -v`
Expected: all 4 pass.

- [ ] **Step 4: Commit**

```bash
git add src/geoai/features/poi_density.py tests/test_poi_density.py
git commit -m "feat: POI density features — counts per category per radius"
```

---

## Task 4: Competition Features

**Files:**
- Create: `src/geoai/features/competition.py`
- Create: `tests/test_competition.py`

### Background

For each listing, compute how many other listings are within 500m and 1km, plus average price within 500m and median price in the same neighbourhood.

- `listings_500m` — count of other listings within 500m (exclude self)
- `listings_1km` — count within 1km
- `avg_price_500m` — mean price of other listings within 500m (None if no neighbours)
- `median_price_neighbourhood` — median price of all listings in same neighbourhood

- [ ] **Step 1: Write failing tests**

Create `tests/test_competition.py`:

```python
import polars as pl
import pytest

from geoai.features.competition import compute_competition


def _listings(data: list[dict]) -> pl.DataFrame:
    return pl.DataFrame({
        "id": [d["id"] for d in data],
        "latitude": [d["lat"] for d in data],
        "longitude": [d["lon"] for d in data],
        "price": [d["price"] for d in data],
        "neighbourhood": [d["neighbourhood"] for d in data],
    })


def test_compute_competition_excludes_self():
    listings = _listings([
        {"id": 1, "lat": 41.14961, "lon": -8.61099, "price": 100.0, "neighbourhood": "A"},
        {"id": 2, "lat": 41.1500, "lon": -8.6110, "price": 120.0, "neighbourhood": "A"},
    ])
    result = compute_competition(listings)
    row1 = result.filter(pl.col("listing_id") == 1)
    assert row1["listings_500m"][0] == 1  # listing 2 is nearby but listing 1 excludes itself
    assert row1["listings_1km"][0] == 1


def test_compute_competition_counts_within_radius():
    listings = _listings([
        {"id": 1, "lat": 41.14961, "lon": -8.61099, "price": 100.0, "neighbourhood": "A"},
        {"id": 2, "lat": 41.1500, "lon": -8.6110, "price": 120.0, "neighbourhood": "A"},  # ~0.1km
        {"id": 3, "lat": 41.200,  "lon": -8.650,  "price": 200.0, "neighbourhood": "B"},  # ~7km
    ])
    result = compute_competition(listings)
    row1 = result.filter(pl.col("listing_id") == 1)
    assert row1["listings_500m"][0] == 1
    assert row1["listings_1km"][0] == 1


def test_compute_competition_avg_price_500m():
    listings = _listings([
        {"id": 1, "lat": 41.14961, "lon": -8.61099, "price": 100.0, "neighbourhood": "A"},
        {"id": 2, "lat": 41.1500, "lon": -8.6110, "price": 200.0, "neighbourhood": "A"},
        {"id": 3, "lat": 41.1505, "lon": -8.6115, "price": 300.0, "neighbourhood": "A"},
    ])
    result = compute_competition(listings)
    row1 = result.filter(pl.col("listing_id") == 1)
    assert row1["avg_price_500m"][0] == pytest.approx(250.0, abs=1.0)


def test_compute_competition_median_price_neighbourhood():
    listings = _listings([
        {"id": 1, "lat": 41.14961, "lon": -8.61099, "price": 100.0, "neighbourhood": "A"},
        {"id": 2, "lat": 41.150, "lon": -8.611, "price": 200.0, "neighbourhood": "A"},
        {"id": 3, "lat": 41.151, "lon": -8.612, "price": 300.0, "neighbourhood": "A"},
        {"id": 4, "lat": 41.200, "lon": -8.650, "price": 500.0, "neighbourhood": "B"},
    ])
    result = compute_competition(listings)
    row1 = result.filter(pl.col("listing_id") == 1)
    assert row1["median_price_neighbourhood"][0] == pytest.approx(200.0, abs=1.0)


def test_compute_competition_no_neighbours_avg_price_is_null():
    listings = _listings([
        {"id": 1, "lat": 41.14961, "lon": -8.61099, "price": 100.0, "neighbourhood": "A"},
        {"id": 2, "lat": 41.500, "lon": -9.000, "price": 200.0, "neighbourhood": "B"},
    ])
    result = compute_competition(listings)
    row1 = result.filter(pl.col("listing_id") == 1)
    assert row1["avg_price_500m"][0] is None
```

Run: `pytest tests/test_competition.py -v` — expect ImportError.

- [ ] **Step 2: Implement competition.py**

Create `src/geoai/features/competition.py`:

```python
from pathlib import Path

import duckdb
import numpy as np
import polars as pl

from geoai.config import DB_PATH
from geoai.database.warehouse import init_warehouse
from geoai.features.accessibility import haversine_km


def compute_competition(listings: pl.DataFrame) -> pl.DataFrame:
    lats = listings["latitude"].to_numpy()
    lons = listings["longitude"].to_numpy()
    prices = listings["price"].to_numpy()
    ids = listings["id"].to_list()
    neighbourhoods = listings["neighbourhood"].to_list()

    # neighbourhood medians — compute once
    neighbourhood_medians: dict[str, float] = {}
    for nbhd in set(neighbourhoods):
        mask = [n == nbhd for n in neighbourhoods]
        nbhd_prices = [p for p, m in zip(prices, mask) if m and not np.isnan(p)]
        if nbhd_prices:
            neighbourhood_medians[nbhd] = float(np.median(nbhd_prices))

    rows = []
    for i, (listing_id, lat, lon, nbhd) in enumerate(zip(ids, lats, lons, neighbourhoods)):
        # distances to all other listings
        other_lats = np.delete(lats, i)
        other_lons = np.delete(lons, i)
        other_prices = np.delete(prices, i)
        dists = haversine_km(other_lats, other_lons, lat, lon)

        mask_500 = dists <= 0.5
        mask_1km = dists <= 1.0
        neighbours_500_prices = other_prices[mask_500]
        valid_500_prices = neighbours_500_prices[~np.isnan(neighbours_500_prices)]

        rows.append({
            "listing_id": listing_id,
            "listings_500m": int(np.sum(mask_500)),
            "listings_1km": int(np.sum(mask_1km)),
            "avg_price_500m": float(np.mean(valid_500_prices)) if len(valid_500_prices) > 0 else None,
            "median_price_neighbourhood": neighbourhood_medians.get(nbhd),
        })
    return pl.DataFrame(rows)


def load_competition_into_db(db_path: Path = DB_PATH) -> int:
    _con = init_warehouse(db_path)
    _con.close()
    with duckdb.connect(str(db_path)) as con:
        listings = pl.from_arrow(
            con.execute(
                "SELECT id, latitude, longitude, price, neighbourhood FROM listings"
            ).arrow()
        )
    result = compute_competition(listings)
    cols = [c for c in result.columns if c != "listing_id"]
    update_set = ", ".join(f"{c} = excluded.{c}" for c in cols)
    col_list = ", ".join(result.columns)
    with duckdb.connect(str(db_path)) as con:
        con.execute(f"""
            INSERT INTO listing_features ({col_list})
            SELECT {col_list} FROM result
            ON CONFLICT (listing_id) DO UPDATE SET {update_set}
        """)
        return con.execute("SELECT COUNT(*) FROM listing_features").fetchone()[0]
```

- [ ] **Step 3: Run tests**

Run: `pytest tests/test_competition.py -v`
Expected: all 5 pass.

- [ ] **Step 4: Commit**

```bash
git add src/geoai/features/competition.py tests/test_competition.py
git commit -m "feat: competition features — listing counts and price stats within radii"
```

---

## Task 5: Walkability Score

**Files:**
- Create: `src/geoai/features/walkability.py`
- Create: `tests/test_walkability.py`

### Background

Composite score 0–100, weighted sum of normalised sub-scores:

| Component | Weight | Based on |
|-----------|--------|----------|
| Restaurant density | 0.20 | `restaurants_500m` capped at 10 → /10 |
| Bar/cafe density | 0.10 | `(bars_500m + cafes_500m)` capped at 10 → /10 |
| Transit proximity | 0.30 | `1 - min(dist_nearest_metro_km, 1.0)` |
| City center proximity | 0.20 | `1 - min(dist_city_center_km / 3.0, 1.0)` |
| Park access | 0.10 | `parks_500m` capped at 3 → /3 |
| Supermarket access | 0.10 | `supermarkets_1km` capped at 2 → /2 |

Final: `score = sum(weight_i × sub_score_i) × 100`, clamped to [0, 100].

Input is a Polars DataFrame with all the above columns already computed (from accessibility + poi_density results). This module purely transforms — no DB reads.

- [ ] **Step 1: Write failing tests**

Create `tests/test_walkability.py`:

```python
import polars as pl
import pytest

from geoai.features.walkability import compute_walkability


def _feature_row(**overrides) -> pl.DataFrame:
    defaults = {
        "listing_id": 1,
        "restaurants_500m": 5,
        "bars_500m": 2,
        "cafes_500m": 3,
        "dist_nearest_metro_km": 0.2,
        "dist_city_center_km": 1.0,
        "parks_500m": 1,
        "supermarkets_1km": 1,
    }
    defaults.update(overrides)
    return pl.DataFrame({k: [v] for k, v in defaults.items()})


def test_compute_walkability_returns_score_column():
    result = compute_walkability(_feature_row())
    assert "walkability_score" in result.columns
    assert "listing_id" in result.columns


def test_compute_walkability_score_in_range():
    result = compute_walkability(_feature_row())
    score = result["walkability_score"][0]
    assert 0.0 <= score <= 100.0


def test_compute_walkability_perfect_score():
    # Max everything out
    result = compute_walkability(_feature_row(
        restaurants_500m=10, bars_500m=5, cafes_500m=5,
        dist_nearest_metro_km=0.0, dist_city_center_km=0.0,
        parks_500m=3, supermarkets_1km=2
    ))
    assert result["walkability_score"][0] == pytest.approx(100.0, abs=0.1)


def test_compute_walkability_zero_score():
    result = compute_walkability(_feature_row(
        restaurants_500m=0, bars_500m=0, cafes_500m=0,
        dist_nearest_metro_km=2.0, dist_city_center_km=10.0,
        parks_500m=0, supermarkets_1km=0
    ))
    assert result["walkability_score"][0] == pytest.approx(0.0, abs=0.1)


def test_compute_walkability_null_metro_distance():
    # If no metro exists, transit component should be 0
    result = compute_walkability(_feature_row(dist_nearest_metro_km=None))
    score = result["walkability_score"][0]
    assert 0.0 <= score <= 100.0
```

Run: `pytest tests/test_walkability.py -v` — expect ImportError.

- [ ] **Step 2: Implement walkability.py**

Create `src/geoai/features/walkability.py`:

```python
from pathlib import Path

import duckdb
import polars as pl

from geoai.config import DB_PATH
from geoai.database.warehouse import init_warehouse


def _cap_norm(value: float | None, cap: float) -> float:
    if value is None or value != value:  # None or NaN
        return 0.0
    return min(value, cap) / cap


def _proximity_score(dist_km: float | None, max_km: float) -> float:
    if dist_km is None or dist_km != dist_km:
        return 0.0
    return 1.0 - min(dist_km, max_km) / max_km


def compute_walkability(features: pl.DataFrame) -> pl.DataFrame:
    scores = []
    for row in features.iter_rows(named=True):
        restaurant_score = _cap_norm(row.get("restaurants_500m"), 10)
        bar_cafe_score = _cap_norm(
            (row.get("bars_500m") or 0) + (row.get("cafes_500m") or 0), 10
        )
        transit_score = _proximity_score(row.get("dist_nearest_metro_km"), 1.0)
        center_score = _proximity_score(row.get("dist_city_center_km"), 3.0)
        park_score = _cap_norm(row.get("parks_500m"), 3)
        supermarket_score = _cap_norm(row.get("supermarkets_1km"), 2)

        raw = (
            0.20 * restaurant_score
            + 0.10 * bar_cafe_score
            + 0.30 * transit_score
            + 0.20 * center_score
            + 0.10 * park_score
            + 0.10 * supermarket_score
        )
        scores.append(max(0.0, min(100.0, raw * 100)))

    return features.select("listing_id").with_columns(
        pl.Series("walkability_score", scores)
    )


def load_walkability_into_db(db_path: Path = DB_PATH) -> int:
    _con = init_warehouse(db_path)
    _con.close()
    with duckdb.connect(str(db_path)) as con:
        features = pl.from_arrow(con.execute("""
            SELECT listing_id, restaurants_500m, bars_500m, cafes_500m,
                   dist_nearest_metro_km, dist_city_center_km,
                   parks_500m, supermarkets_1km
            FROM listing_features
        """).arrow())
    result = compute_walkability(features)
    with duckdb.connect(str(db_path)) as con:
        con.execute("""
            INSERT INTO listing_features (listing_id, walkability_score)
            SELECT listing_id, walkability_score FROM result
            ON CONFLICT (listing_id) DO UPDATE SET walkability_score = excluded.walkability_score
        """)
        return con.execute("SELECT COUNT(*) FROM listing_features").fetchone()[0]
```

- [ ] **Step 3: Run tests**

Run: `pytest tests/test_walkability.py -v`
Expected: all 5 pass.

- [ ] **Step 4: Commit**

```bash
git add src/geoai/features/walkability.py tests/test_walkability.py
git commit -m "feat: walkability score — weighted composite of POI density and transit proximity"
```

---

## Task 6: H3 Hexagon Aggregation

**Files:**
- Create: `src/geoai/features/h3_grid.py`
- Create: `tests/test_h3_grid.py`

### Background

Assign each listing an H3 cell at resolution 8 (≈460m diameter). Then aggregate per-cell stats into `hex_aggregates`. Uses the `h3` package (h3-py).

`h3.latlng_to_cell(lat, lon, resolution)` → returns hex cell ID string.

- [ ] **Step 1: Write failing tests**

Create `tests/test_h3_grid.py`:

```python
import polars as pl
import pytest

from geoai.features.h3_grid import assign_h3_cells, compute_hex_aggregates


def _listings() -> pl.DataFrame:
    return pl.DataFrame({
        "id": [1, 2, 3, 4],
        "latitude": [41.14961, 41.14970, 41.15200, 41.20000],
        "longitude": [-8.61099, -8.61105, -8.61300, -8.65000],
        "price": [100.0, 120.0, 150.0, 200.0],
        "walkability_score": [75.0, 80.0, 60.0, 50.0],
        "occupancy_rate_365d": [0.6, 0.7, 0.5, 0.4],
    })


def test_assign_h3_cells_returns_listing_id_and_h3_cell():
    result = assign_h3_cells(_listings())
    assert "listing_id" in result.columns
    assert "h3_cell_r8" in result.columns


def test_assign_h3_cells_all_assigned():
    result = assign_h3_cells(_listings())
    assert result["h3_cell_r8"].null_count() == 0


def test_assign_h3_cells_same_location_same_cell():
    listings = pl.DataFrame({
        "id": [1, 2],
        "latitude": [41.14961, 41.14961],
        "longitude": [-8.61099, -8.61099],
        "price": [100.0, 200.0],
        "walkability_score": [70.0, 80.0],
        "occupancy_rate_365d": [0.5, 0.6],
    })
    result = assign_h3_cells(listings)
    assert result["h3_cell_r8"][0] == result["h3_cell_r8"][1]


def test_compute_hex_aggregates_returns_correct_columns():
    listings = _listings()
    cells = assign_h3_cells(listings)
    joined = listings.with_columns(
        cells.select("listing_id", "h3_cell_r8")
             .join(listings.select("id").rename({"id": "listing_id"}), on="listing_id")
             .select("h3_cell_r8")
    )
    result = compute_hex_aggregates(listings.rename({"id": "listing_id"}).hstack(cells.select("h3_cell_r8")))
    expected_cols = {"h3_cell", "listing_count", "avg_price", "avg_occupancy", "avg_walkability"}
    assert expected_cols.issubset(set(result.columns))


def test_compute_hex_aggregates_listing_count():
    df = pl.DataFrame({
        "listing_id": [1, 2, 3],
        "h3_cell_r8": ["cell_A", "cell_A", "cell_B"],
        "price": [100.0, 200.0, 150.0],
        "walkability_score": [70.0, 80.0, 60.0],
        "occupancy_rate_365d": [0.5, 0.6, 0.4],
    })
    result = compute_hex_aggregates(df)
    cell_a = result.filter(pl.col("h3_cell") == "cell_A")
    assert cell_a["listing_count"][0] == 2
    assert cell_a["avg_price"][0] == pytest.approx(150.0, abs=0.1)
```

Run: `pytest tests/test_h3_grid.py -v` — expect ImportError.

- [ ] **Step 2: Implement h3_grid.py**

Create `src/geoai/features/h3_grid.py`:

```python
from pathlib import Path

import duckdb
import h3
import polars as pl

from geoai.config import DB_PATH
from geoai.database.warehouse import init_warehouse

H3_RESOLUTION = 8


def assign_h3_cells(listings: pl.DataFrame) -> pl.DataFrame:
    cells = [
        h3.latlng_to_cell(lat, lon, H3_RESOLUTION)
        for lat, lon in zip(listings["latitude"].to_list(), listings["longitude"].to_list())
    ]
    return pl.DataFrame({
        "listing_id": listings["id"].to_list(),
        "h3_cell_r8": cells,
    })


def compute_hex_aggregates(df: pl.DataFrame) -> pl.DataFrame:
    # df must have: listing_id, h3_cell_r8, price, walkability_score, occupancy_rate_365d
    return (
        df.group_by("h3_cell_r8")
        .agg([
            pl.len().alias("listing_count"),
            pl.col("price").mean().alias("avg_price"),
            pl.col("occupancy_rate_365d").mean().alias("avg_occupancy"),
            pl.col("walkability_score").mean().alias("avg_walkability"),
        ])
        .rename({"h3_cell_r8": "h3_cell"})
    )


def load_h3_into_db(db_path: Path = DB_PATH) -> tuple[int, int]:
    _con = init_warehouse(db_path)
    _con.close()
    with duckdb.connect(str(db_path)) as con:
        listings = pl.from_arrow(
            con.execute("SELECT id, latitude, longitude FROM listings").arrow()
        )
        features = pl.from_arrow(
            con.execute(
                "SELECT listing_id, price, walkability_score, occupancy_rate_365d"
                " FROM listing_features"
            ).arrow()
        )

    cells = assign_h3_cells(listings)

    # Write h3_cell_r8 to listing_features
    with duckdb.connect(str(db_path)) as con:
        con.execute("""
            INSERT INTO listing_features (listing_id, h3_cell_r8)
            SELECT listing_id, h3_cell_r8 FROM cells
            ON CONFLICT (listing_id) DO UPDATE SET h3_cell_r8 = excluded.h3_cell_r8
        """)

    # Compute and write hex aggregates
    joined = features.join(cells, on="listing_id")
    agg_df = compute_hex_aggregates(joined.rename({"h3_cell_r8": "h3_cell_r8"}))
    agg_df = agg_df.rename({"h3_cell": "h3_cell_r8"})  # temp name for join
    hex_df = agg_df.rename({"h3_cell_r8": "h3_cell"})

    col_list = ", ".join(hex_df.columns)
    cols_no_pk = [c for c in hex_df.columns if c != "h3_cell"]
    update_set = ", ".join(f"{c} = excluded.{c}" for c in cols_no_pk)
    with duckdb.connect(str(db_path)) as con:
        con.execute(f"""
            INSERT INTO hex_aggregates ({col_list})
            SELECT {col_list} FROM hex_df
            ON CONFLICT (h3_cell) DO UPDATE SET {update_set}
        """)
        n_listings = con.execute("SELECT COUNT(*) FROM listing_features WHERE h3_cell_r8 IS NOT NULL").fetchone()[0]
        n_hexes = con.execute("SELECT COUNT(*) FROM hex_aggregates").fetchone()[0]
    return n_listings, n_hexes
```

- [ ] **Step 3: Run tests**

Run: `pytest tests/test_h3_grid.py -v`
Expected: all 5 pass.

- [ ] **Step 4: Commit**

```bash
git add src/geoai/features/h3_grid.py tests/test_h3_grid.py
git commit -m "feat: H3 hexagon assignment at resolution 8 and per-hex aggregate stats"
```

---

## Task 7: Occupancy Estimation from Calendar

**Files:**
- Create: `src/geoai/features/occupancy.py`
- Create: `tests/test_occupancy.py`

### Background

Use calendar data to estimate occupancy. A date is "booked" if `available = false` (i.e., `available = FALSE` in DuckDB). Occupancy rate = booked_days / total_days in window.

- `occupancy_rate_30d` — last 30 days before max(date) in calendar
- `occupancy_rate_90d` — last 90 days
- `occupancy_rate_365d` — last 365 days

"Last N days" is relative to the most recent date in the calendar table (not today, since data is historical).

- [ ] **Step 1: Write failing tests**

Create `tests/test_occupancy.py`:

```python
import datetime
import polars as pl
import pytest

from geoai.features.occupancy import compute_occupancy


def _calendar(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame({
        "listing_id": [r["listing_id"] for r in rows],
        "date": [r["date"] for r in rows],
        "available": [r["available"] for r in rows],
    })


def test_compute_occupancy_returns_correct_columns():
    cal = _calendar([
        {"listing_id": 1, "date": datetime.date(2024, 12, 1), "available": False},
        {"listing_id": 1, "date": datetime.date(2024, 12, 2), "available": True},
    ])
    result = compute_occupancy(cal)
    assert "listing_id" in result.columns
    assert "occupancy_rate_30d" in result.columns
    assert "occupancy_rate_90d" in result.columns
    assert "occupancy_rate_365d" in result.columns


def test_compute_occupancy_fully_booked():
    dates = [datetime.date(2024, 12, 1) + datetime.timedelta(days=i) for i in range(30)]
    cal = _calendar([{"listing_id": 1, "date": d, "available": False} for d in dates])
    result = compute_occupancy(cal)
    assert result.filter(pl.col("listing_id") == 1)["occupancy_rate_30d"][0] == pytest.approx(1.0)


def test_compute_occupancy_fully_available():
    dates = [datetime.date(2024, 12, 1) + datetime.timedelta(days=i) for i in range(30)]
    cal = _calendar([{"listing_id": 1, "date": d, "available": True} for d in dates])
    result = compute_occupancy(cal)
    assert result.filter(pl.col("listing_id") == 1)["occupancy_rate_30d"][0] == pytest.approx(0.0)


def test_compute_occupancy_mixed():
    # 15 booked, 15 available in a 30-day window
    dates = [datetime.date(2024, 12, 1) + datetime.timedelta(days=i) for i in range(30)]
    avail = [i % 2 == 0 for i in range(30)]  # alternating
    cal = _calendar([{"listing_id": 1, "date": d, "available": a} for d, a in zip(dates, avail)])
    result = compute_occupancy(cal)
    assert result.filter(pl.col("listing_id") == 1)["occupancy_rate_30d"][0] == pytest.approx(0.5, abs=0.05)


def test_compute_occupancy_no_data_for_listing_returns_null():
    cal = _calendar([
        {"listing_id": 1, "date": datetime.date(2024, 12, 1), "available": False},
    ])
    result = compute_occupancy(cal)
    # Listing 2 never appears — should not appear in result at all
    assert len(result.filter(pl.col("listing_id") == 2)) == 0
```

Run: `pytest tests/test_occupancy.py -v` — expect ImportError.

- [ ] **Step 2: Implement occupancy.py**

Create `src/geoai/features/occupancy.py`:

```python
import datetime
from pathlib import Path

import duckdb
import polars as pl

from geoai.config import DB_PATH
from geoai.database.warehouse import init_warehouse


def compute_occupancy(calendar: pl.DataFrame) -> pl.DataFrame:
    max_date = calendar["date"].max()
    if max_date is None:
        return pl.DataFrame(schema={
            "listing_id": pl.Int64,
            "occupancy_rate_30d": pl.Float64,
            "occupancy_rate_90d": pl.Float64,
            "occupancy_rate_365d": pl.Float64,
        })

    windows = {
        "occupancy_rate_30d": 30,
        "occupancy_rate_90d": 90,
        "occupancy_rate_365d": 365,
    }

    result = (
        calendar.select("listing_id")
        .unique()
    )

    for col_name, days in windows.items():
        cutoff = max_date - datetime.timedelta(days=days)
        window_df = (
            calendar.filter(pl.col("date") > cutoff)
            .group_by("listing_id")
            .agg([
                (pl.col("available") == False).sum().alias("booked"),  # noqa: E712
                pl.len().alias("total"),
            ])
            .with_columns(
                (pl.col("booked").cast(pl.Float64) / pl.col("total")).alias(col_name)
            )
            .select("listing_id", col_name)
        )
        result = result.join(window_df, on="listing_id", how="left")

    return result


def load_occupancy_into_db(db_path: Path = DB_PATH) -> int:
    _con = init_warehouse(db_path)
    _con.close()
    with duckdb.connect(str(db_path)) as con:
        calendar = pl.from_arrow(
            con.execute("SELECT listing_id, date, available FROM calendar").arrow()
        )
    result = compute_occupancy(calendar)
    if len(result) == 0:
        return 0
    cols = [c for c in result.columns if c != "listing_id"]
    update_set = ", ".join(f"{c} = excluded.{c}" for c in cols)
    col_list = ", ".join(result.columns)
    with duckdb.connect(str(db_path)) as con:
        con.execute(f"""
            INSERT INTO listing_features ({col_list})
            SELECT {col_list} FROM result
            ON CONFLICT (listing_id) DO UPDATE SET {update_set}
        """)
        return con.execute("SELECT COUNT(*) FROM listing_features").fetchone()[0]
```

- [ ] **Step 3: Run tests**

Run: `pytest tests/test_occupancy.py -v`
Expected: all 5 pass.

- [ ] **Step 4: Commit**

```bash
git add src/geoai/features/occupancy.py tests/test_occupancy.py
git commit -m "feat: calendar-based occupancy estimation — 30d/90d/365d rates per listing"
```

---

## Task 8: Feature Runner Script

**Files:**
- Create: `src/geoai/features/runner.py`
- Create: `scripts/compute_features.py`

### Background

Orchestrates all feature computation in the correct dependency order:
1. Calendar ingestion (needs to run before occupancy)
2. Accessibility features (reads listings + pois)
3. POI density features (reads listings + pois)
4. Competition features (reads listings)
5. Walkability score (reads listing_features — needs steps 2-4 first)
6. H3 assignment (reads listings + listing_features — needs step 5 for avg_walkability)
7. Occupancy estimation (reads calendar — needs step 1 first)

- [ ] **Step 1: Create runner.py**

Create `src/geoai/features/runner.py`:

```python
from pathlib import Path

from geoai.config import DB_PATH
from geoai.ingestion.calendar import load_calendar_into_db
from geoai.features.accessibility import load_accessibility_into_db
from geoai.features.poi_density import load_poi_density_into_db
from geoai.features.competition import load_competition_into_db
from geoai.features.walkability import load_walkability_into_db
from geoai.features.h3_grid import load_h3_into_db
from geoai.features.occupancy import load_occupancy_into_db


def run_all_features(db_path: Path = DB_PATH) -> dict:
    print("Step 1/7: Calendar ingestion...")
    n_calendar = load_calendar_into_db(db_path)
    print(f"  {n_calendar:,} calendar rows loaded")

    print("Step 2/7: Accessibility features...")
    n_acc = load_accessibility_into_db(db_path)
    print(f"  {n_acc:,} listings with accessibility features")

    print("Step 3/7: POI density features...")
    n_poi = load_poi_density_into_db(db_path)
    print(f"  {n_poi:,} listings with POI density features")

    print("Step 4/7: Competition features...")
    n_comp = load_competition_into_db(db_path)
    print(f"  {n_comp:,} listings with competition features")

    print("Step 5/7: Walkability score...")
    n_walk = load_walkability_into_db(db_path)
    print(f"  {n_walk:,} listings with walkability score")

    print("Step 6/7: H3 hexagon assignment...")
    n_listings_h3, n_hexes = load_h3_into_db(db_path)
    print(f"  {n_listings_h3:,} listings assigned to {n_hexes:,} H3 cells")

    print("Step 7/7: Occupancy estimation...")
    n_occ = load_occupancy_into_db(db_path)
    print(f"  {n_occ:,} listings with occupancy rates")

    return {
        "calendar_rows": n_calendar,
        "listings_with_features": n_occ,
        "h3_cells": n_hexes,
    }
```

- [ ] **Step 2: Create CLI entry point**

Create `scripts/compute_features.py`:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from geoai.features.runner import run_all_features

if __name__ == "__main__":
    print("=== GeoAI Phase 2: Feature Engineering ===\n")
    summary = run_all_features()
    print("\n=== Summary ===")
    print(f"Calendar rows loaded : {summary['calendar_rows']:>10,}")
    print(f"Listings with features: {summary['listings_with_features']:>10,}")
    print(f"H3 cells              : {summary['h3_cells']:>10,}")
    print("\nPhase 2 complete.")
```

- [ ] **Step 3: Smoke test with real data (optional but recommended)**

Run: `python scripts/compute_features.py`

Expected output pattern:
```
=== GeoAI Phase 2: Feature Engineering ===

Step 1/7: Calendar ingestion...
  ~5,000,000 calendar rows loaded
Step 2/7: Accessibility features...
  15,246 listings with accessibility features
...
Phase 2 complete.
```

> **Note:** POI density and competition steps are O(n²) and may take several minutes on full Porto dataset. This is acceptable for a portfolio project. If too slow, see the bounding-box optimisation note in Task 3.

- [ ] **Step 4: Run full test suite**

Run: `pytest tests/ -v`
Expected: all tests pass.

- [ ] **Step 5: Commit and update docs**

```bash
git add src/geoai/features/runner.py scripts/compute_features.py
git commit -m "feat: Phase 2 feature runner — orchestrates all geospatial feature computation"
```

Update `README.md`: change Phase 2 status from `⏳ Pending` to `✅ Complete`.

```bash
git add README.md
git commit -m "docs: mark Phase 2 complete in roadmap"
```

---

## Verification Checklist

After all 8 tasks:

- [ ] `pytest tests/ -v` — all tests pass
- [ ] `python scripts/compute_features.py` — runs without error
- [ ] DuckDB `listing_features` has rows matching `listings` count
- [ ] DuckDB `hex_aggregates` has sensible H3 cell count (~400–800 for Porto)
- [ ] `walkability_score` distribution looks reasonable (0–100, not all zeros)
- [ ] `occupancy_rate_365d` values in range [0, 1]

---

## New ADR Needed After Phase 2

Add ADR-003 after completing Phase 2:

**ADR-003: H3 for Geographic Aggregation (over custom grids)**
- Context: needed fixed-area hexagonal grid for price/occupancy heatmaps
- Decision: H3 resolution 8 (≈460m diameter)
- Alternatives: regular lat/lon grid, neighbourhood boundaries only
- Consequences: portable, composable with other H3 tools, industry standard for geospatial ML

**ADR-004: Haversine over PostGIS for Proximity Calculations**
- Context: distance computations for accessibility + competition + POI density
- Decision: vectorized NumPy Haversine — no spatial DB extension needed
- Alternatives: DuckDB spatial extension (ST_Distance), PostGIS, GeoPandas STRtree
- Consequences: no extra dependencies, testable in pure Python, acceptable accuracy for <50km distances
