from contextlib import asynccontextmanager

import duckdb
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from geoai.config import DB_PATH
from geoai.api.routes.stats import router as stats_router
from geoai.api.routes.hexagons import router as hexagons_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # read_only=True raises IOException if file doesn't exist — guard for test environments
    if DB_PATH.exists():
        app.state.db = duckdb.connect(str(DB_PATH), read_only=True)
    else:
        app.state.db = None
    yield
    if app.state.db is not None:
        app.state.db.close()


app = FastAPI(title="GeoAI Airbnb API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

app.include_router(stats_router)
app.include_router(hexagons_router)
