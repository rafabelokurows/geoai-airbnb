# Phase 1: Data Warehouse Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Porto Airbnb data warehouse by ingesting raw listings from InsideAirbnb and POI data from OpenStreetMap into DuckDB, ready for feature engineering in Phase 2.

**Architecture:** A local DuckDB warehouse stores two tables (`listings`, `poi_features`). Two ingestion modules — one for Airbnb CSV data parsed with Polars, one for OSM POI data fetched with OSMnx — populate it. A shared config module owns all paths and constants.

**Tech Stack:** Python 3.11+, DuckDB, Polars, OSMnx, GeoPandas, Shapely, httpx, pytest

---

## File Structure

```
geoai_airbnb/
├── data/
│   ├── raw/
│   │   ├── airbnb/          # downloaded listings.csv.gz lands here
│   │   └── osm/             # reserved for future raw OSM dumps
│   └── warehouse.duckdb     # single DuckDB file
├── src/
│   └── geoai/
│       ├── __init__.py
│       ├── config.py         # all paths + constants
│       ├── database/
│       │   ├── __init__.py
│       │   └── warehouse.py  # init_warehouse(), get_connection()
│       └── ingestion/
│           ├── __init__.py
│           ├── airbnb.py     # download + clean + load listings
│           └── osm.py        # fetch + process + load POIs
├── tests/
│   ├── __init__.py
│   ├── test_warehouse.py
│   ├── test_airbnb_ingestion.py
│   └── test_osm_ingestion.py
├── docs/
│   └── superpowers/plans/
├── pyproject.toml
└── .gitignore
```

---

### Task 1: Project Setup

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `src/geoai/__init__.py`
- Create: `src/geoai/config.py`
- Create: `src/geoai/database/__init__.py`
- Create: `src/geoai/ingestion/__init__.py`
- Create: `tests/__init__.py`

- [ ] **Step 1: Create directory tree**

```bash
mkdir -p data/raw/airbnb data/raw/osm
mkdir -p src/geoai/database src/geoai/ingestion
mkdir -p tests
```

- [ ] **Step 2: Create `pyproject.toml`**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "geoai-airbnb"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "duckdb>=0.10.0",
    "polars>=0.20.0",
    "osmnx>=1.9.0",
    "geopandas>=0.14.0",
    "shapely>=2.0.0",
    "httpx>=0.27.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-mock>=3.12.0",
]

[tool.hatch.build.targets.wheel]
packages = ["src/geoai"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
```

- [ ] **Step 3: Create `.gitignore`**

```gitignore
data/
__pycache__/
*.pyc
.venv/
dist/
*.egg-info/
.pytest_cache/
```

- [ ] **Step 4: Create empty `__init__.py` files**

Create empty files at:
- `src/geoai/__init__.py`
- `src/geoai/database/__init__.py`
- `src/geoai/ingestion/__init__.py`
- `tests/__init__.py`

- [ ] **Step 5: Create `src/geoai/config.py`**

```python
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
RAW_AIRBNB_DIR = RAW_DIR / "airbnb"
RAW_OSM_DIR = RAW_DIR / "osm"
DB_PATH = DATA_DIR / "warehouse.duckdb"

# Verify this URL at https://insideairbnb.com/get-the-data/
# Navigate to Portugal > Norte > Porto and copy the listings.csv.gz link
AIRBNB_PORTO_URL = (
    "https://data.insideairbnb.com/portugal/norte/porto/"
    "2024-12-22/data/listings.csv.gz"
)

OSM_CITY = "Porto, Portugal"

OSM_POI_TAGS = {
    "amenity": [
        "restaurant", "bar", "cafe", "pub", "fast_food",
        "supermarket", "pharmacy",
        "museum", "theatre", "cinema",
    ],
    "tourism": ["museum", "attraction", "gallery", "viewpoint"],
    "leisure": ["park", "garden"],
    "railway": ["station", "subway_entrance"],
}
```

- [ ] **Step 6: Install the package in development mode**

```bash
pip install -e ".[dev]"
```

Expected output: `Successfully installed geoai-airbnb-0.1.0`

- [ ] **Step 7: Commit**

```bash
git init
git add pyproject.toml .gitignore src/ tests/
git commit -m "chore: initial project scaffold"
```

---

### Task 2: DuckDB Warehouse Schema

**Files:**
- Create: `src/geoai/database/warehouse.py`
- Create: `tests/test_warehouse.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_warehouse.py`:

```python
import pytest
import duckdb
from pathlib import Path
import tempfile

from geoai.database.warehouse import init_warehouse


def test_init_warehouse_creates_both_tables(tmp_path):
    db_path = tmp_path / "test.duckdb"
    con = init_warehouse(db_path)
    tables = {row[0] for row in con.execute("SHOW TABLES").fetchall()}
    assert "listings" in tables
    assert "poi_features" in tables
    con.close()


def test_listings_has_required_columns(tmp_path):
    db_path = tmp_path / "test.duckdb"
    con = init_warehouse(db_path)
    cols = {row[0] for row in con.execute("DESCRIBE listings").fetchall()}
    required = {"id", "latitude", "longitude", "price", "room_type", "neighbourhood"}
    assert required.issubset(cols)
    con.close()


def test_poi_features_has_required_columns(tmp_path):
    db_path = tmp_path / "test.duckdb"
    con = init_warehouse(db_path)
    cols = {row[0] for row in con.execute("DESCRIBE poi_features").fetchall()}
    required = {"osm_id", "poi_type", "poi_subtype", "latitude", "longitude"}
    assert required.issubset(cols)
    con.close()


def test_init_warehouse_is_idempotent(tmp_path):
    db_path = tmp_path / "test.duckdb"
    con = init_warehouse(db_path)
    con.close()
    # Second call must not raise
    con2 = init_warehouse(db_path)
    tables = {row[0] for row in con2.execute("SHOW TABLES").fetchall()}
    assert len(tables) == 2
    con2.close()
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
pytest tests/test_warehouse.py -v
```

Expected: `ImportError` or `ModuleNotFoundError` (warehouse.py doesn't exist yet)

- [ ] **Step 3: Implement `src/geoai/database/warehouse.py`**

```python
import duckdb
from pathlib import Path

from geoai.config import DB_PATH

_CREATE_LISTINGS = """
CREATE TABLE IF NOT EXISTS listings (
    id                   BIGINT PRIMARY KEY,
    name                 VARCHAR,
    latitude             DOUBLE,
    longitude            DOUBLE,
    neighbourhood        VARCHAR,
    room_type            VARCHAR,
    property_type        VARCHAR,
    accommodates         INTEGER,
    bedrooms             DOUBLE,
    beds                 DOUBLE,
    price                DOUBLE,
    minimum_nights       INTEGER,
    maximum_nights       INTEGER,
    availability_30      INTEGER,
    availability_60      INTEGER,
    availability_90      INTEGER,
    availability_365     INTEGER,
    number_of_reviews    INTEGER,
    review_scores_rating DOUBLE,
    reviews_per_month    DOUBLE,
    host_id              BIGINT,
    host_name            VARCHAR,
    host_is_superhost    BOOLEAN,
    amenities            VARCHAR,
    last_scraped         DATE
)
"""

_CREATE_POI_FEATURES = """
CREATE TABLE IF NOT EXISTS poi_features (
    osm_id       VARCHAR PRIMARY KEY,
    poi_type     VARCHAR,
    poi_subtype  VARCHAR,
    name         VARCHAR,
    latitude     DOUBLE,
    longitude    DOUBLE,
    geometry_wkt VARCHAR
)
"""


def init_warehouse(db_path: Path = DB_PATH) -> duckdb.DuckDBPyConnection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    con.execute(_CREATE_LISTINGS)
    con.execute(_CREATE_POI_FEATURES)
    return con


def get_connection(db_path: Path = DB_PATH) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(db_path))
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
pytest tests/test_warehouse.py -v
```

Expected:
```
tests/test_warehouse.py::test_init_warehouse_creates_both_tables PASSED
tests/test_warehouse.py::test_listings_has_required_columns PASSED
tests/test_warehouse.py::test_poi_features_has_required_columns PASSED
tests/test_warehouse.py::test_init_warehouse_is_idempotent PASSED
4 passed
```

- [ ] **Step 5: Commit**

```bash
git add src/geoai/database/warehouse.py tests/test_warehouse.py
git commit -m "feat: DuckDB warehouse schema with listings and poi_features tables"
```

---

### Task 3: Airbnb Data Ingestion

**Files:**
- Create: `src/geoai/ingestion/airbnb.py`
- Create: `tests/test_airbnb_ingestion.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_airbnb_ingestion.py`:

```python
import pytest
import polars as pl
from pathlib import Path
import tempfile

from geoai.ingestion.airbnb import clean_listings, load_airbnb_into_db
from geoai.database.warehouse import init_warehouse


def _sample_raw_df() -> pl.DataFrame:
    return pl.DataFrame({
        "id": [1, 2, 3, 4],
        "name": ["Cozy Studio", "Modern Flat", "River View", "Bad Listing"],
        "latitude": [41.1496, 41.1500, 41.1480, None],
        "longitude": [-8.6110, -8.6120, -8.6100, None],
        "neighbourhood_cleansed": ["Cedofeita", "Bonfim", "Ribeira", "Unknown"],
        "room_type": ["Entire home/apt", "Private room", "Entire home/apt", "Private room"],
        "property_type": ["Apartment", "Apartment", "House", "Apartment"],
        "accommodates": [2, 1, 4, 1],
        "bedrooms": [1.0, 1.0, 2.0, 1.0],
        "beds": [1.0, 1.0, 2.0, 1.0],
        "price": ["$85.00", "$1,200.00", "$120.00", "$50.00"],
        "minimum_nights": [2, 1, 3, 1],
        "maximum_nights": [30, 365, 30, 30],
        "availability_30": [10, 20, 5, 0],
        "availability_60": [20, 40, 10, 0],
        "availability_90": [30, 60, 15, 0],
        "availability_365": [120, 240, 60, 0],
        "number_of_reviews": [15, 30, 8, 0],
        "review_scores_rating": [4.8, 4.5, 4.9, None],
        "reviews_per_month": [1.2, 2.5, 0.8, None],
        "host_id": [1001, 1002, 1003, 9999],
        "host_name": ["Ana", "João", "Maria", "Ghost"],
        "host_is_superhost": ["t", "f", "t", "f"],
        "amenities": ['["Wifi","Kitchen"]', '["Wifi"]', '["Wifi","Pool"]', "[]"],
        "last_scraped": ["2024-12-22", "2024-12-22", "2024-12-22", "2024-12-22"],
    })


def test_clean_listings_parses_price_removes_symbols():
    df = _sample_raw_df()
    cleaned = clean_listings(df)
    assert cleaned["price"].dtype == pl.Float64
    assert cleaned["price"][0] == pytest.approx(85.0)
    assert cleaned["price"][1] == pytest.approx(1200.0)


def test_clean_listings_parses_superhost_flag():
    df = _sample_raw_df()
    cleaned = clean_listings(df)
    assert cleaned["host_is_superhost"][0] is True
    assert cleaned["host_is_superhost"][1] is False


def test_clean_listings_renames_neighbourhood_column():
    df = _sample_raw_df()
    cleaned = clean_listings(df)
    assert "neighbourhood" in cleaned.columns
    assert "neighbourhood_cleansed" not in cleaned.columns


def test_clean_listings_drops_rows_with_null_coordinates():
    df = _sample_raw_df()
    cleaned = clean_listings(df)
    assert len(cleaned) == 3  # row 4 has None lat/lon — dropped
    assert cleaned["id"].to_list() == [1, 2, 3]


def test_load_airbnb_into_db_returns_correct_count(tmp_path, monkeypatch):
    sample = _sample_raw_df()

    monkeypatch.setattr(
        "geoai.ingestion.airbnb._download_raw",
        lambda url, dest_dir: (dest_dir / "listings.csv").parent.mkdir(parents=True, exist_ok=True) or (dest_dir / "listings.csv"),
    )
    monkeypatch.setattr(
        "geoai.ingestion.airbnb._read_raw",
        lambda path: sample,
    )

    db_path = tmp_path / "test.duckdb"
    init_warehouse(db_path)
    count = load_airbnb_into_db(db_path=db_path)
    assert count == 3  # 3 valid rows (row 4 dropped during cleaning)
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
pytest tests/test_airbnb_ingestion.py -v
```

Expected: `ImportError` (module doesn't exist yet)

- [ ] **Step 3: Implement `src/geoai/ingestion/airbnb.py`**

```python
import httpx
import polars as pl
import duckdb
from pathlib import Path

from geoai.config import RAW_AIRBNB_DIR, DB_PATH, AIRBNB_PORTO_URL

_KEEP_COLS = [
    "id", "name", "latitude", "longitude", "neighbourhood_cleansed",
    "room_type", "property_type", "accommodates", "bedrooms", "beds",
    "price", "minimum_nights", "maximum_nights",
    "availability_30", "availability_60", "availability_90", "availability_365",
    "number_of_reviews", "review_scores_rating", "reviews_per_month",
    "host_id", "host_name", "host_is_superhost", "amenities", "last_scraped",
]


def _download_raw(url: str, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    filename = url.split("/")[-1]
    dest_path = dest_dir / filename
    if dest_path.exists():
        return dest_path
    with httpx.stream("GET", url, follow_redirects=True, timeout=120) as r:
        r.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in r.iter_bytes(chunk_size=8192):
                f.write(chunk)
    return dest_path


def _read_raw(path: Path) -> pl.DataFrame:
    return pl.read_csv(path, infer_schema_length=10000, ignore_errors=True)


def clean_listings(df: pl.DataFrame) -> pl.DataFrame:
    # Parse price: "$1,234.00" → 1234.0
    df = df.with_columns(
        pl.col("price")
        .str.replace_all(r"[\$,]", "")
        .cast(pl.Float64, strict=False)
        .alias("price")
    )
    # Parse superhost: "t"/"f" → bool
    df = df.with_columns(
        pl.col("host_is_superhost")
        .map_elements(
            lambda x: True if x == "t" else (False if x == "f" else None),
            return_dtype=pl.Boolean,
        )
        .alias("host_is_superhost")
    )
    # Select available columns from the desired set
    existing = [c for c in _KEEP_COLS if c in df.columns]
    df = df.select(existing)
    # Rename neighbourhood column
    if "neighbourhood_cleansed" in df.columns:
        df = df.rename({"neighbourhood_cleansed": "neighbourhood"})
    # Drop rows missing coordinates or id (unusable for geospatial work)
    return df.drop_nulls(subset=["id", "latitude", "longitude"])


def load_airbnb_into_db(
    db_path: Path = DB_PATH,
    url: str = AIRBNB_PORTO_URL,
) -> int:
    raw_path = _download_raw(url, RAW_AIRBNB_DIR)
    df = _read_raw(raw_path)
    df = clean_listings(df)

    # Align columns to table schema
    con = duckdb.connect(str(db_path))
    table_cols = [row[0] for row in con.execute("DESCRIBE listings").fetchall()]
    df_cols_available = [c for c in table_cols if c in df.columns]
    df = df.select(df_cols_available)

    con.execute("DELETE FROM listings")
    con.execute("INSERT INTO listings SELECT * FROM df")
    count = con.execute("SELECT COUNT(*) FROM listings").fetchone()[0]
    con.close()
    return count
```

- [ ] **Step 4: Fix monkeypatch targets in tests**

The test patches `geoai.ingestion.airbnb._download_raw` and `geoai.ingestion.airbnb._read_raw`. The mock for `_download_raw` needs to return a path. Update `test_load_airbnb_into_db_returns_correct_count` in `tests/test_airbnb_ingestion.py` to:

```python
def test_load_airbnb_into_db_returns_correct_count(tmp_path, monkeypatch):
    sample = _sample_raw_df()
    fake_path = tmp_path / "listings.csv"

    monkeypatch.setattr(
        "geoai.ingestion.airbnb._download_raw",
        lambda url, dest_dir: fake_path,
    )
    monkeypatch.setattr(
        "geoai.ingestion.airbnb._read_raw",
        lambda path: sample,
    )

    db_path = tmp_path / "test.duckdb"
    init_warehouse(db_path)
    count = load_airbnb_into_db(db_path=db_path)
    assert count == 3
```

- [ ] **Step 5: Run tests — verify they pass**

```bash
pytest tests/test_airbnb_ingestion.py -v
```

Expected:
```
tests/test_airbnb_ingestion.py::test_clean_listings_parses_price_removes_symbols PASSED
tests/test_airbnb_ingestion.py::test_clean_listings_parses_superhost_flag PASSED
tests/test_airbnb_ingestion.py::test_clean_listings_renames_neighbourhood_column PASSED
tests/test_airbnb_ingestion.py::test_clean_listings_drops_rows_with_null_coordinates PASSED
tests/test_airbnb_ingestion.py::test_load_airbnb_into_db_returns_correct_count PASSED
5 passed
```

- [ ] **Step 6: Commit**

```bash
git add src/geoai/ingestion/airbnb.py tests/test_airbnb_ingestion.py
git commit -m "feat: Airbnb listings ingestion with Polars cleaning and DuckDB load"
```

---

### Task 4: OSM POI Ingestion

**Files:**
- Create: `src/geoai/ingestion/osm.py`
- Create: `tests/test_osm_ingestion.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_osm_ingestion.py`:

```python
import pytest
import polars as pl
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, Polygon
from pathlib import Path

from geoai.ingestion.osm import process_pois, load_osm_into_db
from geoai.database.warehouse import init_warehouse


def _sample_gdf() -> gpd.GeoDataFrame:
    data = {
        "amenity": ["restaurant", "bar", None, None],
        "tourism": [None, None, "museum", None],
        "leisure": [None, None, None, "park"],
        "railway": [None, None, None, None],
        "name": ["Café Majestic", "Hard Club", "Museu do Vinho", "Jardim do Palácio"],
        "geometry": [
            Point(-8.611, 41.149),
            Point(-8.615, 41.150),
            Point(-8.612, 41.148),
            Polygon([(-8.610, 41.147), (-8.609, 41.147), (-8.609, 41.148), (-8.610, 41.147)]),
        ],
    }
    index = pd.MultiIndex.from_tuples(
        [("node", 111), ("node", 222), ("node", 333), ("way", 444)],
        names=["element_type", "osmid"],
    )
    return gpd.GeoDataFrame(data, geometry="geometry", crs="EPSG:4326", index=index)


def test_process_pois_returns_polars_dataframe():
    gdf = _sample_gdf()
    result = process_pois(gdf)
    assert isinstance(result, pl.DataFrame)


def test_process_pois_correct_row_count():
    gdf = _sample_gdf()
    result = process_pois(gdf)
    assert len(result) == 4


def test_process_pois_extracts_poi_type():
    gdf = _sample_gdf()
    result = process_pois(gdf)
    types = set(result["poi_type"].to_list())
    assert "amenity" in types
    assert "tourism" in types
    assert "leisure" in types


def test_process_pois_polygon_uses_centroid():
    gdf = _sample_gdf()
    result = process_pois(gdf)
    # way/444 is a Polygon — centroid should be inside polygon bounds
    way_row = result.filter(pl.col("osm_id") == "way_444")
    assert len(way_row) == 1
    lat = way_row["latitude"][0]
    lon = way_row["longitude"][0]
    assert 41.147 <= lat <= 41.148
    assert -8.610 <= lon <= -8.609


def test_process_pois_generates_unique_osm_ids():
    gdf = _sample_gdf()
    result = process_pois(gdf)
    ids = result["osm_id"].to_list()
    assert len(ids) == len(set(ids))


def test_load_osm_into_db_returns_count(tmp_path, monkeypatch):
    sample_gdf = _sample_gdf()

    monkeypatch.setattr(
        "geoai.ingestion.osm._fetch_osm",
        lambda city, tags: sample_gdf,
    )

    db_path = tmp_path / "test.duckdb"
    init_warehouse(db_path)
    count = load_osm_into_db(db_path=db_path)
    assert count == 4
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
pytest tests/test_osm_ingestion.py -v
```

Expected: `ImportError` (module doesn't exist yet)

- [ ] **Step 3: Implement `src/geoai/ingestion/osm.py`**

```python
import osmnx as ox
import geopandas as gpd
import polars as pl
import duckdb
from pathlib import Path
from typing import Any

from geoai.config import OSM_CITY, OSM_POI_TAGS, DB_PATH

_TAG_PRIORITY = ["amenity", "tourism", "leisure", "railway"]


def _fetch_osm(city: str, tags: dict[str, Any]) -> gpd.GeoDataFrame:
    return ox.features_from_place(city, tags=tags)


def process_pois(gdf: gpd.GeoDataFrame) -> pl.DataFrame:
    records = []
    for idx, row in gdf.iterrows():
        element_type, osm_id = idx[0], idx[1]
        uid = f"{element_type}_{osm_id}"

        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        centroid = geom if geom.geom_type == "Point" else geom.centroid

        poi_type = None
        poi_subtype = None
        for tag_key in _TAG_PRIORITY:
            val = row.get(tag_key)
            if val and str(val) not in ("nan", "None", ""):
                poi_type = tag_key
                poi_subtype = str(val)
                break

        if poi_type is None:
            continue

        name_val = row.get("name")
        records.append({
            "osm_id": uid,
            "poi_type": poi_type,
            "poi_subtype": poi_subtype,
            "name": str(name_val) if name_val and str(name_val) not in ("nan", "None") else None,
            "latitude": centroid.y,
            "longitude": centroid.x,
            "geometry_wkt": geom.wkt,
        })

    return pl.DataFrame(
        records,
        schema={
            "osm_id": pl.Utf8,
            "poi_type": pl.Utf8,
            "poi_subtype": pl.Utf8,
            "name": pl.Utf8,
            "latitude": pl.Float64,
            "longitude": pl.Float64,
            "geometry_wkt": pl.Utf8,
        },
    )


def load_osm_into_db(
    db_path: Path = DB_PATH,
    city: str = OSM_CITY,
    tags: dict = OSM_POI_TAGS,
) -> int:
    gdf = _fetch_osm(city, tags)
    df = process_pois(gdf)

    con = duckdb.connect(str(db_path))
    con.execute("DELETE FROM poi_features")
    con.execute("INSERT INTO poi_features SELECT * FROM df")
    count = con.execute("SELECT COUNT(*) FROM poi_features").fetchone()[0]
    con.close()
    return count
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
pytest tests/test_osm_ingestion.py -v
```

Expected:
```
tests/test_osm_ingestion.py::test_process_pois_returns_polars_dataframe PASSED
tests/test_osm_ingestion.py::test_process_pois_correct_row_count PASSED
tests/test_osm_ingestion.py::test_process_pois_extracts_poi_type PASSED
tests/test_osm_ingestion.py::test_process_pois_polygon_uses_centroid PASSED
tests/test_osm_ingestion.py::test_process_pois_generates_unique_osm_ids PASSED
tests/test_osm_ingestion.py::test_load_osm_into_db_returns_count PASSED
6 passed
```

- [ ] **Step 5: Commit**

```bash
git add src/geoai/ingestion/osm.py tests/test_osm_ingestion.py
git commit -m "feat: OSM POI ingestion via OSMnx with centroid extraction and DuckDB load"
```

---

### Task 5: End-to-End Smoke Test (Real Data)

This task runs the actual ingestion against real data. It requires network access. Run manually — do not add to CI.

**Files:**
- Create: `scripts/ingest_porto.py`

- [ ] **Step 1: Verify the InsideAirbnb URL is still valid**

Open `https://insideairbnb.com/get-the-data/` in a browser, find Portugal > Porto, and copy the latest `listings.csv.gz` URL. Update `AIRBNB_PORTO_URL` in `src/geoai/config.py` if it has changed.

- [ ] **Step 2: Create `scripts/ingest_porto.py`**

```python
"""Run full Phase 1 ingestion for Porto. Requires network access."""
from pathlib import Path
import sys

# Allow running as a script without installing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from geoai.database.warehouse import init_warehouse
from geoai.ingestion.airbnb import load_airbnb_into_db
from geoai.ingestion.osm import load_osm_into_db
from geoai.config import DB_PATH


def main():
    print(f"Initializing warehouse at {DB_PATH}")
    con = init_warehouse()
    con.close()

    print("Ingesting Airbnb listings (downloading ~30MB)...")
    n_listings = load_airbnb_into_db()
    print(f"  Loaded {n_listings:,} listings")

    print("Ingesting OSM POIs for Porto (network request)...")
    n_pois = load_osm_into_db()
    print(f"  Loaded {n_pois:,} POIs")

    print("\nDone. Warehouse summary:")
    from geoai.database.warehouse import get_connection
    con = get_connection()
    print(con.execute("SELECT COUNT(*) as listings FROM listings").fetchdf().to_string(index=False))
    print(con.execute("SELECT poi_type, COUNT(*) as count FROM poi_features GROUP BY poi_type ORDER BY count DESC").fetchdf().to_string(index=False))
    con.close()


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run the ingestion script**

```bash
python scripts/ingest_porto.py
```

Expected output (approximate values):
```
Initializing warehouse at .../data/warehouse.duckdb
Ingesting Airbnb listings (downloading ~30MB)...
  Loaded 15,000+ listings
Ingesting OSM POIs for Porto (network request)...
  Loaded 2,000+ POIs

Done. Warehouse summary:
 listings
   15,xxx
 poi_type  count
  amenity   xxxx
  tourism    xxx
  leisure    xxx
  railway     xx
```

If listing count is < 5,000 or POI count is < 500, something went wrong — recheck the URL in `config.py`.

- [ ] **Step 4: Run full test suite to confirm nothing broken**

```bash
pytest tests/ -v
```

Expected: All 15 tests pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/ingest_porto.py
git commit -m "feat: Phase 1 complete — Porto data warehouse with Airbnb listings and OSM POIs"
```

---

## Self-Review

**Spec coverage:**
- [x] Airbnb ingestion — Task 3
- [x] OSM ingestion — Task 4
- [x] DuckDB setup — Task 2
- [x] Deliverable: Data warehouse — Task 5 smoke test

**Placeholder scan:** None found.

**Type consistency:**
- `load_airbnb_into_db` and `load_osm_into_db` both take `db_path: Path` — consistent
- `init_warehouse` returns `duckdb.DuckDBPyConnection` — used in Task 3/4 tests correctly
- `process_pois` takes `gpd.GeoDataFrame` → returns `pl.DataFrame` — matches `load_osm_into_db` usage
- `clean_listings` takes `pl.DataFrame` → returns `pl.DataFrame` — matches `load_airbnb_into_db` usage

All consistent.
