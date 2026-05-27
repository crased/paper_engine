"""Report listing + reading endpoints."""

from __future__ import annotations

from typing import List

from fastapi import APIRouter, HTTPException

from .. import services
from ..schemas import ReportContent, ReportInfo

router = APIRouter(prefix="/reports", tags=["reports"])


@router.get("", response_model=List[ReportInfo])
def get_reports() -> List[ReportInfo]:
    return services.list_reports()


@router.get("/{name}", response_model=ReportContent)
def get_report(name: str) -> ReportContent:
    try:
        return services.read_report(name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Report not found: {name}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
