# Paper Engine — Web stack (Vue + FastAPI)

The TypeScript/Vue frontend + FastAPI backend that replaces the CustomTkinter GUI.

```
frontend/            Vue 3 + TS + Vite frontend  (this dir)
backend/
  api/               FastAPI backend (the seam to pipeline/)
  pipeline/, tools/  Python engine
<repo root>/
  yolo_dataset/, recordings/, reports/, runs/   data dirs (resolved via DATA_ROOT)
  env/               Python venv
```

> **Layout note:** code lives in `backend/`, data dirs stay at the repo root.
> The backend resolves code-local paths from `backend/` and data paths from the
> repo root (`DATA_ROOT = backend/..`), so it must be run from `backend/`.

## Run (local dev — two processes)

**1. Backend** (from `backend/`, venv active):

```bash
source ../env/bin/activate
cd backend
uvicorn api.main:app --reload --port 8000
```

**2. Frontend** (this dir, `frontend/`):

```bash
npm install        # first time
npm run dev        # http://localhost:5173
```

Or just run `./paperengine.sh` from the repo root to start both.

Vite proxies `/api/*` → `http://localhost:8000`, so the browser only talks to
`:5173`. SSE job logs (`/api/jobs/:id/stream`) flow through the same proxy.

## Build

```bash
npm run build      # vue-tsc typecheck + vite build -> frontend/dist/
npm run typecheck  # types only
```

## What's wired

| Page | Endpoint(s) | Status |
|---|---|---|
| Home | `/api/health`, `/api/models` | live |
| Metrics | `/api/dataset/stats`, `/api/models` | live |
| Test | `/api/infer`, `/api/sessions`, `/api/models` | live |
| Tools | `/api/jobs/{train,annotate}` (SSE), `/api/reports` | live |
| Settings | — | stub (no settings endpoints yet) |

## Two execution modes (design)

- **LOCAL**: this Vue app → FastAPI → `pipeline/`. Full interactive tool.
- **DOCKER (headless)**: no GUI/API — drive `pipeline/` CLIs over SSH. Long
  jobs here run the *same* CLIs the API spawns, so behavior is identical.

**Import-isolation rule:** the model-runner modules (`batch_annotator`,
`training_model`, `test_model`) must not import the host-coupled capture/memory
stack (`tools.screencapture`, `tools.memory_reader`, `tools.mono_external`) at
module load — that keeps the slim Docker image slim. Currently honored.

## Not yet built (next steps)

- `GET/PUT /api/settings` (LLM provider, training params) → wire Settings page
- Detection image overlays (return rendered boxes or draw client-side from bbox)
- Live capture/recorder controls (host-only, Tier B — degrade gracefully in Docker)
- Generate `src/types.ts` from `/openapi.json` instead of hand-mirroring
