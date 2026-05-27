"""FastAPI application entry point.

Run (local dev):
    source env/bin/activate
    uvicorn api.main:app --reload --port 8000

The Vue dev server (Vite, :5173) proxies /api here, or calls it directly with
CORS enabled below.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import dataset, infer, jobs, models, reports, sessions

app = FastAPI(
    title="Paper Engine API",
    version="0.1.0",
    description="HTTP seam between the Vue frontend and the Python engine.",
)

# Vite dev server origins.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_PREFIX = "/api"
for r in (models.router, infer.router, dataset.router, reports.router, sessions.router, jobs.router):
    app.include_router(r, prefix=API_PREFIX)


@app.get("/api/health", tags=["meta"])
def health() -> dict:
    return {"status": "ok", "service": "paper-engine-api"}
