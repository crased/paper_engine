"""Paper Engine HTTP API (FastAPI).

The seam between the Vue frontend and the Python engine (`pipeline/`, `tools/`).

Two execution modes share one engine:
  - LOCAL:  Vue frontend -> this API -> pipeline/
  - DOCKER: CLI/SSH -> pipeline/ directly (no API, no GUI)

Design rule (keep the slim container slim): this package and the modules it
imports for *model running* must not import the host-coupled capture/memory
stack (tools.screencapture / tools.memory_reader / tools.mono_external) at
module load. Long-running jobs (train, annotate) are spawned as subprocesses of
the existing CLIs so the same code path serves both modes.
"""
