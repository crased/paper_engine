"""Async job manager for long-running engine work (train, annotate).

Each job spawns one of the existing pipeline CLIs as a subprocess and streams
its stdout line-by-line. Reusing the CLIs (not re-implementing in-process) keeps
the LOCAL and DOCKER paths identical — the API runs exactly what SSH would.

Streaming to the browser is via Server-Sent Events (see routes/jobs.py).
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Dict, List, Optional

from .schemas import JobInfo

_JOB_END = "__JOB_END__"


class Job:
    def __init__(self, kind: str, cmd: List[str], cwd: str):
        self.id = uuid.uuid4().hex[:12]
        self.kind = kind
        self.cmd = cmd
        self.cwd = cwd
        self.status = "pending"  # pending | running | done | error
        self.lines: List[str] = []
        self.return_code: Optional[int] = None
        self.started = time.time()
        self.ended: Optional[float] = None
        self.subscribers: List[asyncio.Queue] = []
        self.proc: Optional[asyncio.subprocess.Process] = None

    def emit(self, line: str) -> None:
        self.lines.append(line)
        for q in list(self.subscribers):
            q.put_nowait(line)

    def info(self) -> JobInfo:
        return JobInfo(
            id=self.id,
            kind=self.kind,
            status=self.status,
            cmd=self.cmd,
            return_code=self.return_code,
            started=self.started,
            ended=self.ended,
            line_count=len(self.lines),
        )


class JobManager:
    def __init__(self) -> None:
        self.jobs: Dict[str, Job] = {}

    def get(self, job_id: str) -> Optional[Job]:
        return self.jobs.get(job_id)

    def list(self) -> List[Job]:
        return sorted(self.jobs.values(), key=lambda j: j.started, reverse=True)

    async def start(self, kind: str, cmd: List[str], cwd: str) -> Job:
        job = Job(kind, cmd, cwd)
        self.jobs[job.id] = job
        asyncio.create_task(self._run(job))
        return job

    async def cancel(self, job: Job) -> bool:
        if job.proc and job.status == "running":
            try:
                job.proc.terminate()
                return True
            except ProcessLookupError:
                return False
        return False

    async def _run(self, job: Job) -> None:
        job.status = "running"
        try:
            proc = await asyncio.create_subprocess_exec(
                *job.cmd,
                cwd=job.cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            job.proc = proc
            assert proc.stdout is not None
            async for raw in proc.stdout:
                job.emit(raw.decode(errors="replace").rstrip("\n"))
            job.return_code = await proc.wait()
            job.status = "done" if job.return_code == 0 else "error"
        except Exception as exc:  # spawn failure, etc.
            job.emit(f"[job error] {exc}")
            job.status = "error"
            job.return_code = -1
        finally:
            job.ended = time.time()
            job.emit(_JOB_END)

    async def stream(self, job: Job):
        """Yield log lines for SSE. Replays history, then live tail until end.

        Safe against missed lines: snapshot + subscribe happen with no `await`
        between them, and emit() runs on the same event loop, so nothing slips
        through the gap.
        """
        q: asyncio.Queue = asyncio.Queue()
        for ln in job.lines:  # replay history
            q.put_nowait(ln)
        if job.status in ("done", "error"):
            q.put_nowait(_JOB_END)
        else:
            job.subscribers.append(q)
        try:
            while True:
                ln = await q.get()
                yield ln
                if ln == _JOB_END:
                    break
        finally:
            if q in job.subscribers:
                job.subscribers.remove(q)


# Module-level singleton
manager = JobManager()
JOB_END = _JOB_END
