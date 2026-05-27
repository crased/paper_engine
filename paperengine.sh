#!/usr/bin/env bash
# Launch Paper Engine locally: FastAPI backend (:8000) + Vue frontend (:5173).
# The backend must run from backend/ (code packages live there); data dirs stay
# at the repo root and are resolved via DATA_ROOT.
set -e
cd "$(dirname "$0")"

# shellcheck disable=SC1091
source env/bin/activate

( cd backend && exec uvicorn api.main:app --reload --port 8000 ) &
BACKEND_PID=$!
trap 'kill $BACKEND_PID 2>/dev/null' EXIT

cd frontend
[ -d node_modules ] || npm install
exec npm run dev
