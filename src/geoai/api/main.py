from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from geoai.api.deps import load_app_state
from geoai.api.routes import health


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_app_state()
    yield


def create_app(load_state: bool = True) -> FastAPI:
    lifespan_ctx = lifespan if load_state else None
    app = FastAPI(title="GeoAI Airbnb API", lifespan=lifespan_ctx)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(health.router, prefix="/api")
    return app


app = create_app()
