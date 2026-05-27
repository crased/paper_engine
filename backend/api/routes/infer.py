"""Inference endpoint — run a trained model over images/a directory/a session."""

from __future__ import annotations

import anyio
from fastapi import APIRouter, HTTPException

from .. import services
from ..schemas import InferRequest, InferResponse

router = APIRouter(prefix="/infer", tags=["infer"])


@router.post("", response_model=InferResponse)
async def post_infer(req: InferRequest) -> InferResponse:
    try:
        # YOLO inference is blocking/CPU-GPU bound — run it off the event loop.
        result = await anyio.to_thread.run_sync(
            lambda: services.run_inference(
                model=req.model,
                paths=req.paths,
                directory=req.directory,
                session=req.session,
                conf=req.conf,
                limit=req.limit,
            )
        )
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return InferResponse(**result)
