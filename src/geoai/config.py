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

# Praça da Liberdade — conventional center of Porto
PORTO_CENTER_LAT = 41.14961
PORTO_CENTER_LON = -8.61099

CALENDAR_URL = (
    "https://data.insideairbnb.com/portugal/norte/porto/"
    "2024-12-22/data/calendar.csv.gz"
)

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
