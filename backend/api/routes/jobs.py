"""Long-running jobs: train + annotate, with SSE log streaming.

Each job spawns an existing pipeline CLI as a subprocess (see jobs.py). The
frontend POSTs to start a job, then opens an EventSource on /jobs/{id}/stream.
"""

from __future__ import annotations

import sys
from typing import List

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from .. import services
from ..jobs import JOB_END, manager
from ..schemas import AnnotateRequest, JobInfo, TrainRequest

router = APIRouter(prefix="/jobs", tags=["jobs"])

# The interpreter running uvicorn (the project venv) runs the CLIs too.
PYTHON = sys.executable
CWD = str(services.PROJECT_ROOT)


@router.get("", response_model=List[JobInfo])
def list_jobs() -> List[JobInfo]:
    return [j.info() for j in manager.list()]


@router.get("/{job_id}", response_model=JobInfo)
def get_job(job_id: str) -> JobInfo:
    job = manager.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job.info()


@router.post("/train", response_model=JobInfo)
async def start_train(_req: TrainRequest) -> JobInfo:
    cmd = [PYTHON, "-m", "pipeline.training_model"]
    job = await manager.start("train", cmd, CWD)
    return job.info()


@router.post("/annotate", response_model=JobInfo)
async def start_annotate(req: AnnotateRequest) -> JobInfo:
    cmd = [PYTHON, "-m", "pipeline.batch_annotator", "--session", req.session]
    if req.dataset:
        cmd += ["--dataset", req.dataset]
    cmd += ["--rpm", str(req.rpm)]
    if req.max_frames:
        cmd += ["--max-frames", str(req.max_frames)]
    if req.no_examples:
        cmd += ["--no-examples"]
    if req.dry_run:
        cmd += ["--dry-run"]
    job = await manager.start("annotate", cmd, CWD)
    return job.info()


@router.post("/{job_id}/cancel", response_model=JobInfo)
async def cancel_job(job_id: str) -> JobInfo:
    job = manager.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    await manager.cancel(job)
    return job.info()


@router.get("/{job_id}/stream")
async def stream_job(job_id: str) -> StreamingResponse:
    job = manager.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    async def event_gen():
        async for line in manager.stream(job):
            if line == JOB_END:
                yield f"event: end\ndata: {job.status}\n\n"
            else:
                # SSE: escape newlines defensively (lines are already split)
                yield f"data: {line}\n\n"

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
