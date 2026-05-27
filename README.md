# Paper Engine

> **Work in progress.** Active development, expect breaking changes.

A game automation, computer vision, and performance analysis framework. Combines Wine game execution, memory reading, YOLO object detection, and LLM-powered game analysis to build intelligent game bots and diagnose game performance issues. Designed for bug detection, automated testing, game QA, and Linux gaming diagnostics.

## What It Does

Paper Engine has two main systems:

### Game Automation Pipeline
1. **Game Execution** -- Runs game executables (Windows via Wine, Linux native, scripts)
2. **Memory Reading** -- Reads live game state from Mono/Unity games (HP, scene, loading, etc.)
3. **Screenshot Capture** -- Captures gameplay at configurable intervals
4. **LLM Annotation** -- `batch_annotator` sends recorded frames to an LLM for YOLO labelling
5. **YOLO Training** -- Trains YOLO11/YOLO26 models on annotated gameplay data
6. **Bot Generation** -- LLM generates Python bot scripts using YOLO + game controls
7. **Self-Learning** -- Automated play-annotate-train loop

### Multi-Source Game Performance Analysis
Analyzes any game's performance by collecting data from up to 6 sources:

1. **Engine configs** -- Scans game directory for .ini, .cfg, .log, benchmark data
2. **Engine detection** -- Identifies UE3, UE4, Unity, Void Engine/id Tech from directory structure and executables
3. **Community data** -- Fetches ProtonDB API for compatibility ratings and report counts
4. **Wine/DXVK prefix scan** -- Reads Proton version, DXVK state cache size, DLL overrides, environment variables
5. **Known fix detection** -- Checks for game-specific community fixes (e.g. Arkham Quixote for Batman AK)
6. **LLM analysis** -- Feeds all sources to an LLM acting as a game engine QA expert

The LLM produces a structured report with 1-5 star ratings across Engine Configuration, Runtime Environment, Known Issues & Fixes, and Performance Architecture -- with file-referenced issues and actionable fix suggestions.

## Web Interface (Vue + FastAPI)

The interface is a Vue 3 + TypeScript frontend (`web/`) backed by a FastAPI seam
(`api/`) that wraps the Python engine. The previous CustomTkinter desktop GUI is
archived outside the repo at `../paper_engine_legacy_gui/`.

```bash
# Backend -- run from backend/ (code lives there; data dirs stay at repo root)
source env/bin/activate
cd backend && uvicorn api.main:app --reload --port 8000

# Frontend (separate terminal)
cd frontend && npm install && npm run dev   # http://localhost:5173

# ...or start both at once from the repo root:
./paperengine.sh
```

Vite proxies `/api/*` to the backend; the browser only talks to `:5173`. Full
details (build, SSE job logs, the two-mode design) in [`web/README.md`](web/README.md).

**Pages:**
- **Home** -- engine status, model count, quick actions
- **Metrics** -- dataset class balance, trained-model inventory
- **Test** -- run inference over a session/directory, inspect detections
- **Tools** -- launch train/annotate jobs with live (SSE) logs, read reports
- **Settings** -- stub (backend settings endpoints pending)

### Two execution modes
- **Local** -- Vue -> FastAPI -> `pipeline/` (full interactive tool)
- **Docker (headless)** -- no GUI/API; drive the `pipeline/` CLIs over SSH. Long
  jobs run the *same* CLIs the API spawns, so behavior is identical.

## CLI

### Game Performance Report

```bash
# Run from backend/ (cd backend); session/data paths are relative to the repo root.
# Multi-source directory analysis (any game)
python -m pipeline.game_feedback --game-dir /path/to/game/
python -m pipeline.game_feedback --game-dir /path/to/game/ --prefix /path/to/wine/prefix/
python -m pipeline.game_feedback --game-dir /path/to/game/ --export report.txt
python -m pipeline.game_feedback --game-dir /path/to/game/ --json

# Session-based analysis (Mono/Unity games with memory telemetry)
python -m pipeline.game_feedback --session recordings/sessions/Cuphead_20260305_073118/
```

### Bot Pipeline

```bash
# All commands below run from backend/ (cd backend first):
uvicorn api.main:app --reload --port 8000           # Backend API (then: cd frontend && npm run dev)
python -m pipeline.gameplay_recorder --launch       # Record gameplay session
python -m pipeline.batch_annotator --session path/  # LLM-annotate session frames
python -m pipeline.training_model                   # Train YOLO model
python -m pipeline.generate_bot_script              # Generate bot script
python bot_logic/bot_scripts/cuphead_bot.py --launch  # Run unified bot
```

## Project Structure

```
paper_engine/
  backend/                        # all Python -- RUN THE BACKEND FROM HERE
    api/                          # FastAPI seam to the engine
      main.py                     # App, CORS, router mounts, /api/health
      schemas.py                  # Pydantic models (the frontend's TS contract)
      services.py                 # In-process reads + YOLO inference
      jobs.py                     # Async subprocess job runner + SSE streaming
      routes/                     # models, infer, dataset, reports, sessions, jobs
    pipeline/
      game_feedback.py            # Multi-source game analysis engine
      gameplay_recorder.py        # Screenshot + memory state capture
      batch_annotator.py          # LLM annotation of recorded sessions
      training_model.py           # YOLO training
      generate_bot_script.py      # LLM bot script generation
      self_learning.py            # Play-annotate-train orchestrator
      dataset_tools.py            # Dataset cleanup, merge, augmentation
      example_bank.py             # Few-shot visual references for annotation
      golden_rules.py             # Persistent annotation rules
      scene_context.py            # Scene descriptions for annotation grounding
      source_analyzer.py          # Static analysis of decompiled game code
    tools/
      functions.py                # Path helpers, API key (keyring + .env)
      memory_reader.py            # Linux process_vm_readv wrapper
      mono_external.py            # External Mono metadata walker
      game_state.py               # Config-driven game state reader
      mono_bridge.c / .so         # LD_PRELOAD bridge for Mono base detection
      screencapture.py            # Screenshot capture (flameshot/mss/PIL)
    bot_logic/
      bot_scripts/cuphead_bot.py  # Three-domain bot (Read-Fuse-Dispatch-Act)
      bot_scripts/domains/        # contracts, detection, navigation, strategy
      models/                     # Trained YOLO models (gitignored)
    conf/
      main_conf.ini               # LLM provider, capture settings
      training_conf.ini           # YOLO training params
      golden_rules.json           # Persistent annotation rules data
      game_configs/               # Cuphead pointer chains, scene descriptions

  frontend/                       # Vue 3 + TS + Vite  (was web/)
    src/
      api/client.ts               # Typed fetch client
      router/, stores/            # vue-router + pinia
      pages/                      # Home, Metrics, Test, Tools, Settings
      components/                 # AppSidebar, etc.
    README.md                     # Web stack run/build guide

  # Data dirs stay at the repo root; the backend resolves them via DATA_ROOT:
  yolo_dataset/                   # YOLO training data
  recordings/sessions/            # Recorded gameplay sessions
  reports/                        # Generated analysis reports (+ cache/)
  runs/  screenshots/  game/      # ML outputs, captures, game files
  env/                            # Python venv
```

## Architecture

### Memory Reading (Mono/Unity)

Four-layer system for reading live game state from any Mono-based Unity game:

1. **memory_reader.py** -- Linux `process_vm_readv` wrapper (game-agnostic)
2. **mono_external.py** -- Walks Mono runtime C structs externally (domains, assemblies, classes, fields, vtables)
3. **game_state.py** -- Config-driven reader using pointer chains (`ChainStep`). Follows static/instance fields, C# collections (Dict/List/Array)
4. **game_configs/*.py** -- Per-game field definitions (pure data, no logic)

### Three-Domain Bot

```
READ  -- Memory (HP, scene, phase) + YOLO (spatial detections)
FUSE  -- classify_phase() + fuse() -> WorldState
DISPATCH -- LEVEL_PLAYING -> Strategy, else -> Navigation
ACT   -- Key hold/release via xdg-portal RemoteDesktop
```

### Multi-Source Game Analysis

```
game_feedback.py
  _scan_game_dir()       -- Source 1: config/log files
  detect_engine()        -- Source 2: UE3/UE4/Unity/Void/id Tech detection
  fetch_protondb()       -- Source 3: ProtonDB API
  scan_wine_prefix()     -- Source 4: Proton version, DXVK cache, DLL overrides
  detect_known_fixes()   -- Source 5: game-specific community fix detection
  _build_dir_system_prompt()  -- Dynamic prompt (adapts to engine type + available data)
  generate_report_from_dir()  -- Orchestrator (calls all sources, feeds LLM)
```

## Supported Engines

| Engine | Detection Method | Config Patterns |
|--------|-----------------|-----------------|
| Unreal Engine 3 | `Engine/` + `*Game/` dirs | `**/Config/*.ini`, benchmarks |
| Unreal Engine 4 | `Content/Paks/`, `.uproject` | `**/Config/*.ini`, GameUserSettings |
| Unity | `*_Data/Managed/UnityEngine.dll` | `*_Data/*.cfg`, boot.config |
| Void Engine / id Tech | `base/` dir + GFSDK DLLs | `base/*.cfg` |
| Unknown | Fallback generic analysis | Any `.ini`, `.cfg`, `.log` found |

## Requirements

- **OS**: Linux (tested on Arch/Wayland). macOS partially supported. Windows not supported.
- **Python**: 3.8+
- **Node**: 20+ (for the Vue frontend)
- **Wine**: For running Windows games
- **LLM provider**: Anthropic Claude (default), Google Gemini, or OpenAI

### Dependencies

```bash
# Python (backend + engine)
pip install pynput python-dotenv pyyaml anthropic fastapi "uvicorn[standard]"
pip install ultralytics  # For YOLO training/inference

# Frontend
cd web && npm install
```

## Security

### API Keys
- Stored in OS keyring (SecretService/libsecret) via `tools/functions.py`
- Falls back to `.env` file (gitignored)
- Never logged, never written to reports or cache files
- Passed directly to LLM client only

### File Scanning
- Game directory scanning uses `glob()` confined to the target directory
- `relative_to()` enforces path containment
- Content capped at 60KB total with per-file line limits
- Encoding fallback chain (UTF-8, UTF-16, latin-1) prevents decode crashes

### Network
- ProtonDB API: integer AppIDs from a hardcoded static dictionary, no user input in URLs
- Static User-Agent header, 10s timeout
- All network calls fail gracefully (report continues with available data)

### Executable Security
Paper Engine executes game files with user privileges. Only run games from trusted sources. See the executable security notes below for details.

> **Never run executables from untrusted sources.** Wine .exe files, shell scripts, and Python scripts execute with full user permissions.

## Tested Games

| Game | Engine | Analysis Mode | Notes |
|------|--------|--------------|-------|
| Cuphead | Unity (Mono) | Session + Directory | Full memory telemetry, pointer chains defined |
| Batman: Arkham Knight | UE3 | Directory | 6-source analysis, Arkham Quixote detection |
| Dishonored 2 | Void Engine | Directory | GameWorks SSAO detection, virtual texturing analysis |

## License

Apache License 2.0 -- see [LICENSE](LICENSE).

## Acknowledgments

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) for object detection
- [Vue](https://vuejs.org/), [Vite](https://vite.dev/), and [FastAPI](https://fastapi.tiangolo.com/) for the web interface
- [Flameshot](https://flameshot.org/) for screenshot capture
- [ProtonDB](https://www.protondb.com/) for community compatibility data
- [PCGamingWiki](https://www.pcgamingwiki.com/) for game-specific fix knowledge
- Anthropic Claude, Google Gemini, OpenAI GPT for LLM analysis
