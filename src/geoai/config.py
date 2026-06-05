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

# (lat, lon) for key Porto landmarks — hardcoded fixed points
PORTO_LANDMARKS = {
    "livraria_lello":  (41.14672, -8.61480),
    "torre_clerigos":  (41.14541, -8.61490),
    "ribeira":         (41.14082, -8.61430),
    "ponte_luis":      (41.13930, -8.60940),
    "mercado_bolhao":  (41.14910, -8.60740),
    "jardins_cristal": (41.14600, -8.62620),
}

# Francisco Sá Carneiro Airport
PORTO_AIRPORT = (41.2370, -8.6699)

OSM_POI_TAGS = {
    "amenity": [
        "restaurant", "bar", "cafe", "pub", "fast_food",
        "pharmacy",
        "museum", "theatre", "cinema",
    ],
    "shop": ["supermarket"],
    "tourism": ["museum", "attraction", "gallery", "viewpoint"],
    "leisure": ["park", "garden"],
    "railway": ["station", "subway_entrance"],
}
