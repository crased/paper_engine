"""Dataset stats endpoint."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from .. import services
from ..schemas import DatasetStats

router = APIRouter(prefix="/dataset", tags=["dataset"])


@router.get("/stats", response_model=DatasetStats)
def get_stats(yaml: str = "dataset.yaml") -> DatasetStats:
    try:
        return services.dataset_stats(yaml)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
