"""Model listing endpoints."""

from __future__ import annotations

from typing import List

from fastapi import APIRouter

from .. import services
from ..schemas import ModelInfo

router = APIRouter(prefix="/models", tags=["models"])


@router.get("", response_model=List[ModelInfo])
def get_models() -> List[ModelInfo]:
    return services.list_models()
