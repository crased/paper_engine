"""Recorded-session listing (used by the annotate + infer pickers)."""

from __future__ import annotations

from typing import List

from fastapi import APIRouter

from .. import services

router = APIRouter(prefix="/sessions", tags=["sessions"])


@router.get("", response_model=List[str])
def get_sessions() -> List[str]:
    return services.list_sessions()
