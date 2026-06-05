# ADR-006: FastAPI for the Read-Only Prediction API

**Date:** 2026-06-05
**Status:** Accepted
**Deciders:** Rafael Belokurows

---

## Context

Phase 5 requires a REST API that serves pre-computed ML predictions, H3 hex aggregates, and SHAP values from DuckDB to a React + Deck.gl frontend. The API is read-only — all computation happens in `predict.py` before the server starts.

Requirements:
- Sub-5ms response times from DuckDB (in-process, no network hop)
- Automatic OpenAPI docs (useful for frontend development)
- Pydantic response validation
- Async-capable for future extensibility
- Minimal boilerplate

Options evaluated:

| Option | Pros | Cons |
|--------|------|------|
| **FastAPI** | Auto OpenAPI docs, Pydantic native, async, fast, lightweight | Newer (less StackOverflow) |
| Flask | Mature, simple | No async, no Pydantic, no auto docs |
| Django REST Framework | Full-featured | Overkill for a read-only local API, heavy |
| Litestar | Similar to FastAPI, newer | Smaller community, less tooling |

---

## Decision

Use **FastAPI 0.111+** with **uvicorn[standard]** as the ASGI server.

Key design choices:
- DuckDB connection opened once at startup via `lifespan` context manager in `main.py`, stored on `app.state.db`
- Connection opened in `read_only=True` mode — enforces that no writes happen through the API layer
- `DB_PATH.exists()` guard prevents `IOException` in test environments where no database file exists
- Dependency injection via `get_db(request: Request)` — allows `app.dependency_overrides` in tests without patching globals
- All endpoints return `Cache-Control: max-age=3600` — DuckDB tables only change when `predict.py` is re-run
- `Literal["price","occupancy","revenue"]` type hints on query params give FastAPI 422 validation for free

---

## Consequences

**Positive:**
- Interactive docs auto-generated at `/docs` — no separate API documentation needed
- Pydantic models (`schemas.py`) validate response shape and provide TypeScript-friendly JSON
- TestClient (via httpx) + dependency injection makes routes fully testable without a live server
- `read_only=True` DuckDB mode enforces architectural separation: ML pipeline writes, API reads

**Negative:**
- uvicorn adds a process dependency — server must be started separately from model training
- Single DuckDB connection on `app.state` is not thread-safe for concurrent writes; acceptable since the API is read-only
- `read_only=True` DuckDB raises `IOException` if the DB file doesn't exist (unlike read-write mode which creates it) — requires the `DB_PATH.exists()` lifespan guard

---

## References

- `src/geoai/api/main.py`, `src/geoai/api/deps.py`, `src/geoai/api/routes/`
- `tests/api/conftest.py` (dependency override pattern)
- [FastAPI lifespan](https://fastapi.tiangolo.com/advanced/events/)
- [DuckDB read-only connections](https://duckdb.org/docs/api/python/overview.html)
