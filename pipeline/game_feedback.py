"""LLM-powered game performance analysis.

Two modes of operation:

**Session-based** (Mono/Unity games with memory telemetry)::

    python -m pipeline.game_feedback --session path/

**Directory-based** (any game -- multi-source analysis)::

    python -m pipeline.game_feedback --game-dir /path/to/game/
    python -m pipeline.game_feedback --game-dir /path/to/game/ --prefix /path/to/prefix/

The directory-based mode collects data from up to 6 sources:
  1. Engine config files (ini, logs, benchmarks)
  2. Engine detection (UE3/UE4/Unity from directory structure)
  3. Community data (ProtonDB API)
  4. Wine/DXVK prefix scan (Proton version, caches, DLL overrides)
  5. Known fix detection (game-specific community fixes)
  6. LLM analysis (fed all sources, acts as engine QA expert)

Or called from the GUI via ``generate_report()`` / ``generate_report_from_dir()``.
"""

from __future__ import annotations

import json
import re
import sys
import urllib.request
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# -- sys.path shim for standalone execution --
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ======================================================================
# Report dataclass
# ======================================================================


@dataclass
class Rating:
    """Single rating category (1-5 scale)."""

    category: str
    score: int  # 1-5
    label: str  # e.g. "Good", "Needs Work"
    detail: str  # one-line explanation


@dataclass
class Issue:
    """A detected issue with file reference."""

    description: str
    file: str = ""  # e.g. "pipeline/gameplay_recorder.py"
    line: str = ""  # e.g. "367" or "360-400"
    fix: str = ""  # suggested fix description


@dataclass
class Suggestion:
    """An improvement suggestion with file reference."""

    description: str
    file: str = ""  # e.g. "tools/game_state.py"
    line: str = ""  # e.g. "565"
    fix: str = ""  # concrete fix description


@dataclass
class GameReport:
    """Structured game feedback report."""

    game: str = ""
    session_name: str = ""
    generated_at: str = ""
    summary: str = ""
    ratings: List[Rating] = field(default_factory=list)
    issues: List[Issue] = field(default_factory=list)
    suggestions: List[Suggestion] = field(default_factory=list)
    raw_analysis: str = ""
    # Stats extracted from session data
    total_frames: int = 0
    duration_seconds: float = 0.0
    deaths: int = 0
    scenes_visited: List[str] = field(default_factory=list)
    hp_timeline: List[int] = field(default_factory=list)

    def overall_score(self) -> float:
        """Average of all ratings (0.0 if none)."""
        if not self.ratings:
            return 0.0
        return sum(r.score for r in self.ratings) / len(self.ratings)

    def to_text(self) -> str:
        """Format as a human-readable text report."""
        lines = []
        lines.append("=" * 60)
        lines.append("  GAME FEEDBACK REPORT")
        lines.append("=" * 60)
        lines.append(f"  Game:      {self.game or 'Unknown'}")
        lines.append(f"  Session:   {self.session_name}")
        lines.append(f"  Generated: {self.generated_at}")
        lines.append(
            f"  Duration:  {self.duration_seconds:.1f}s  |  "
            f"Frames: {self.total_frames}  |  "
            f"Deaths: {self.deaths}"
        )
        if self.scenes_visited:
            lines.append(f"  Scenes:    {', '.join(self.scenes_visited)}")
        lines.append("")

        # Ratings
        lines.append("-" * 60)
        lines.append("  RATINGS")
        lines.append("-" * 60)
        for r in self.ratings:
            stars = _stars(r.score)
            lines.append(f"  {r.category:<20s} {stars}")
            # Wrap detail text at ~70 chars
            for dline in _wrap(r.detail, 54):
                lines.append(f"    {dline}")
            lines.append("")
        if self.ratings:
            avg = self.overall_score()
            lines.append(f"  Overall              {_stars(round(avg))}")
        lines.append("")

        # Summary
        if self.summary:
            lines.append("-" * 60)
            lines.append("  SUMMARY")
            lines.append("-" * 60)
            for dline in _wrap(self.summary.strip(), 56):
                lines.append(f"  {dline}")
            lines.append("")

        # Issues
        if self.issues:
            lines.append("-" * 60)
            lines.append("  ISSUES")
            lines.append("-" * 60)
            for i, issue in enumerate(self.issues, 1):
                loc = ""
                if isinstance(issue, Issue):
                    if issue.file:
                        loc = issue.file
                        if issue.line:
                            loc += f":{issue.line}"
                    desc = issue.description
                    fix = issue.fix
                else:
                    desc = str(issue)
                    fix = ""
                lines.append(f"  {i}. {desc}")
                if loc:
                    lines.append(f"     -> {loc}")
                if fix:
                    lines.append(f"     FIX: {fix}")
                lines.append("")

        # Suggestions
        if self.suggestions:
            lines.append("-" * 60)
            lines.append("  SUGGESTED FIXES")
            lines.append("-" * 60)
            for i, sug in enumerate(self.suggestions, 1):
                loc = ""
                if isinstance(sug, Suggestion):
                    if sug.file:
                        loc = sug.file
                        if sug.line:
                            loc += f":{sug.line}"
                    desc = sug.description
                    fix = sug.fix
                else:
                    desc = str(sug)
                    fix = ""
                lines.append(f"  {i}. {desc}")
                if loc:
                    lines.append(f"     -> {loc}")
                if fix:
                    lines.append(f"     FIX: {fix}")
                lines.append("")

        lines.append("=" * 60)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serialize to plain dict."""
        d = asdict(self)
        d["overall_score"] = self.overall_score()
        return d


def _wrap(text: str, width: int = 56) -> List[str]:
    """Wrap text to width, preserving existing line breaks."""
    import textwrap

    result = []
    for paragraph in text.splitlines():
        if not paragraph.strip():
            result.append("")
        else:
            result.extend(textwrap.wrap(paragraph, width=width))
    return result or [""]


def _stars(n: int) -> str:
    """Return star bar like '[*****-----] 5/5' for n out of 5."""
    n = max(1, min(5, n))
    filled = "*" * n
    empty = "-" * (5 - n)
    return f"[{filled}{empty}] {n}/5"


# ======================================================================
# Session data loading
# ======================================================================


def load_session_frames(session_dir: Path) -> List[Dict[str, Any]]:
    """Load all frame_*.json files from a session directory, sorted."""
    frames = []
    for f in sorted(session_dir.glob("frame_*.json")):
        try:
            data = json.loads(f.read_text())
            frames.append(data)
        except Exception:
            continue
    return frames


def load_session_manifest(session_dir: Path) -> Dict[str, Any]:
    """Load session.json manifest if it exists."""
    manifest = session_dir / "session.json"
    if manifest.exists():
        try:
            return json.loads(manifest.read_text())
        except Exception:
            pass
    return {}


def find_latest_session() -> Optional[Path]:
    """Find the most recently modified session directory."""
    sessions_dir = _PROJECT_ROOT / "recordings" / "sessions"
    if not sessions_dir.exists():
        return None
    dirs = sorted(
        (d for d in sessions_dir.iterdir() if d.is_dir()),
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    # Return first with frame data
    for d in dirs:
        if list(d.glob("frame_*.json")):
            return d
    return dirs[0] if dirs else None


# ======================================================================
# Pre-analysis: extract stats from raw frames
# ======================================================================


def extract_session_stats(
    frames: List[Dict[str, Any]], manifest: Dict[str, Any]
) -> Dict[str, Any]:
    """Extract structured stats from frame data for LLM context."""
    if not frames:
        return {"error": "No frame data"}

    first = frames[0]
    last = frames[-1]
    duration = last.get("session_elapsed", 0) - first.get("session_elapsed", 0)

    # Track state changes
    scenes = set()
    max_hp = 0
    min_hp = 999
    deaths = 0
    hp_timeline = []
    scene_transitions = []
    loading_frames = 0
    prev_scene = ""
    prev_hp = -1
    damage_events = []
    time_in_combat = 0.0
    time_loading = 0.0
    prev_elapsed = first.get("session_elapsed", 0)

    # Technical diagnostics
    frame_deltas = []  # capture timing between frames
    field_init_times = {}  # when each field first becomes non-zero/non-empty
    null_read_frames = []  # frames where state looks uninitialized
    state_anomalies = []  # unexpected state transitions
    prev_in_game = False
    prev_loading = False

    for i, f in enumerate(frames):
        state = f.get("state", {})
        elapsed = f.get("session_elapsed", 0)
        dt = elapsed - prev_elapsed
        if i > 0:
            frame_deltas.append(dt)
        prev_elapsed = elapsed

        scene = state.get("scene_name", "")
        hp = state.get("hp", 0)
        hp_max = state.get("hp_max", 0)
        is_loading = state.get("is_loading", False)
        in_game = state.get("in_game", False)
        super_meter = state.get("super_meter", 0.0)
        super_max = state.get("super_meter_max", 0.0)
        level_time = state.get("level_time", 0.0)

        # -- Field initialization tracking --
        field_checks = {
            "hp": hp > 0,
            "hp_max": hp_max > 0,
            "scene_name": bool(scene),
            "in_game": in_game,
            "super_meter_max": super_max > 0,
            "level_time": level_time > 0,
        }
        for field_name, is_active in field_checks.items():
            if is_active and field_name not in field_init_times:
                field_init_times[field_name] = {
                    "frame": i,
                    "elapsed": elapsed,
                    "value": state.get(field_name),
                }

        # -- Null/uninitialized detection --
        all_zero = hp == 0 and hp_max == 0 and not scene and not in_game
        if all_zero:
            null_read_frames.append(i)

        # -- State anomaly detection --
        # in_game flickers
        if i > 0 and in_game != prev_in_game:
            state_anomalies.append(
                {
                    "frame": i,
                    "at": elapsed,
                    "type": "in_game_transition",
                    "from": prev_in_game,
                    "to": in_game,
                }
            )
        # loading flickers (rapid on/off within 2 frames)
        if i > 1 and is_loading != prev_loading:
            state_anomalies.append(
                {
                    "frame": i,
                    "at": elapsed,
                    "type": "loading_transition",
                    "from": prev_loading,
                    "to": is_loading,
                }
            )
        # HP jumps up unexpectedly (not from 0 after death)
        if prev_hp > 0 and hp > prev_hp and hp != hp_max:
            state_anomalies.append(
                {
                    "frame": i,
                    "at": elapsed,
                    "type": "hp_increase",
                    "from_hp": prev_hp,
                    "to_hp": hp,
                    "detail": "HP increased without reset to max (possible misread)",
                }
            )

        prev_in_game = in_game
        prev_loading = is_loading

        if scene:
            scenes.add(scene)
        if scene != prev_scene and scene:
            scene_transitions.append(
                {
                    "from": prev_scene,
                    "to": scene,
                    "at": elapsed,
                }
            )
            prev_scene = scene

        if hp_max > max_hp:
            max_hp = hp_max
        if hp > 0 and hp < min_hp:
            min_hp = hp

        hp_timeline.append(hp)

        # Detect damage
        if prev_hp > 0 and hp < prev_hp:
            damage_events.append(
                {
                    "at": elapsed,
                    "from_hp": prev_hp,
                    "to_hp": hp,
                    "scene": scene,
                }
            )
        # Detect death
        if prev_hp > 0 and hp <= 0:
            deaths += 1

        if is_loading:
            loading_frames += 1
            time_loading += dt
        elif "level" in scene.lower() and hp > 0:
            time_in_combat += dt

        prev_hp = hp

    # -- Frame timing statistics --
    timing_stats = {}
    if frame_deltas:
        avg_dt = sum(frame_deltas) / len(frame_deltas)
        min_dt = min(frame_deltas)
        max_dt = max(frame_deltas)
        expected = manifest.get("capture_interval", 0.5)
        jitter = [abs(d - expected) for d in frame_deltas]
        avg_jitter = sum(jitter) / len(jitter)
        late_frames = sum(1 for d in frame_deltas if d > expected * 1.5)
        timing_stats = {
            "avg_delta": avg_dt,
            "min_delta": min_dt,
            "max_delta": max_dt,
            "expected_interval": expected,
            "avg_jitter": avg_jitter,
            "late_frames": late_frames,
            "late_frame_pct": late_frames / len(frame_deltas) * 100,
        }

    return {
        "total_frames": len(frames),
        "duration_seconds": duration,
        "capture_interval": manifest.get("capture_interval", 0.5),
        "game": manifest.get("game", _guess_game_from_dir(frames)),
        "scenes_visited": sorted(scenes),
        "scene_transitions": scene_transitions,
        "max_hp": max_hp,
        "deaths": deaths,
        "damage_events": damage_events,
        "total_damage_taken": len(damage_events),
        "hp_timeline": hp_timeline,
        "loading_frames": loading_frames,
        "time_loading": time_loading,
        "time_in_combat": time_in_combat,
        "final_state": frames[-1].get("state", {}),
        "initial_state": frames[0].get("state", {}),
        # Technical diagnostics
        "timing_stats": timing_stats,
        "field_init_times": field_init_times,
        "null_read_frames": null_read_frames,
        "null_read_count": len(null_read_frames),
        "null_read_pct": len(null_read_frames) / len(frames) * 100,
        "state_anomalies": state_anomalies,
    }


def _guess_game_from_dir(frames: list) -> str:
    """Try to guess game name from session directory name."""
    # Sessions are named like Cuphead_20260305_073118
    if frames:
        return "Unknown"
    return "Unknown"


# ======================================================================
# LLM analysis
# ======================================================================

_SYSTEM_PROMPT_BASE = """\
You are a game performance analyst. You receive telemetry data captured from a \
live game session and produce a detailed technical report about the GAME's \
performance — not the tooling that captured it.

CONTEXT (use this to interpret the data, but do NOT report on the tooling itself):
- Telemetry is captured externally by reading the game's memory at fixed intervals. \
Fields that read as zero/empty early in the session mean the game has not yet \
initialized those systems — treat this as game startup time, not a tool issue.
- The game runs under Wine on Linux. Performance characteristics may differ from \
native Windows due to Wine's translation layer.
- The data includes frame timing, scene transitions, HP changes, loading states, \
and state anomalies detected during the session.

YOUR FOCUS: Analyze the GAME's performance as observed through this telemetry. \
You are evaluating how the game performs, not how well the capture tool works.

RATING CATEGORIES (1-5 stars each):
1. Loading Performance: How fast does the game load? Evaluate: total time from \
launch to gameplay, individual scene transition durations, time spent on loading \
screens vs actual gameplay, frequency of loading interruptions. Cite specific \
timestamps for each loading phase.
2. Runtime Stability: Does the game run without issues once loaded? Evaluate: \
are HP/state values consistent, does the game freeze or get stuck (state stops \
changing), does it crash (abrupt session end with abnormal final state), are \
there stuck states where the game stops progressing. Cite frame numbers and values.
3. Scene Transitions: Are scene changes clean? Evaluate: does state reset properly \
between scenes (HP back to max, level_time reset), is the loading flag behavior \
consistent (no rapid true/false flickering), do scene names follow expected game \
flow, any unexpected scene changes or regressions.
4. Responsiveness: How responsive is the game during gameplay? Evaluate: does \
level_time increment smoothly (no gaps or stalls), does super_meter progress \
during combat, are HP changes immediate (no delayed damage), any gaps in state \
updates that suggest the game is hanging or dropping frames.

For each rating give: score (1-5), a short label, and a DETAILED multi-sentence \
explanation citing specific frame numbers, timestamps, and values from the data.

{game_ref_section}

ISSUES: List specific game performance problems observed in the telemetry. \
Be detailed — cite frame numbers, timestamps, field values. Explain what the \
game is doing wrong and what game system is likely responsible. Include the \
game class/field where the issue originates (use "file" for the class name and \
"line" for the field name) and a description of the likely game-side fix.

SUGGESTIONS: Improvements the game developers could make to improve performance. \
Reference the specific game class/system. Include a concrete description of what \
should change in the game code.

SUMMARY: 3-5 sentence technical overview of the game's performance during this \
session. Focus on loading speed, stability, and responsiveness. Written for a \
game developer or QA engineer.

Format your response EXACTLY as JSON:
{{
  "ratings": [
    {{"category": "Loading Performance", "score": 3, "label": "Fair", "detail": "..."}},
    {{"category": "Runtime Stability", "score": 4, "label": "Good", "detail": "..."}},
    {{"category": "Scene Transitions", "score": 4, "label": "Good", "detail": "..."}},
    {{"category": "Responsiveness", "score": 3, "label": "Fair", "detail": "..."}}
  ],
  "issues": [
    {{
      "description": "Detailed game performance issue...",
      "file": "ClassName",
      "line": "fieldName",
      "fix": "Game-side fix description"
    }}
  ],
  "suggestions": [
    {{
      "description": "Game performance improvement...",
      "file": "ClassName",
      "line": "fieldName",
      "fix": "Game-side improvement description"
    }}
  ],
  "summary": "Game performance summary here."
}}

ONLY output the JSON. No markdown fences, no extra text."""


def _build_game_ref_section(game_name: str) -> str:
    """Build the GAME SOURCE REFERENCE section dynamically from game config.

    If a matching game config exists in conf/game_configs/, extracts the
    class/field names from its FIELDS definitions. Otherwise, uses the
    generic field names from the telemetry data.
    """
    # Try to load game-specific config
    class_fields: Dict[str, List[str]] = {}
    try:
        game_key = game_name.lower().replace(" ", "_")
        mod = __import__(f"conf.game_configs.{game_key}", fromlist=["FIELDS"])
        fields = getattr(mod, "FIELDS", [])
        for fdef in fields:
            for step in getattr(fdef, "chain", []):
                cls = getattr(step, "class_name", "") or getattr(
                    step, "owner_class", ""
                )
                field = getattr(step, "field_name", "")
                if cls and field and not field.startswith("<"):
                    class_fields.setdefault(cls, []).append(field)
                elif cls and field:
                    # Strip C# backing field syntax: <Foo>k__BackingField -> Foo
                    clean = field.replace("<", "").replace(">k__BackingField", "")
                    class_fields.setdefault(cls, []).append(clean)
    except (ImportError, AttributeError):
        pass

    if class_fields:
        lines = [
            f"GAME SOURCE REFERENCE ({game_name} — Unity/Mono C# game, reference "
            "these when identifying where in the game code an issue likely originates):"
        ]
        for cls, flds in class_fields.items():
            unique = sorted(set(flds))
            lines.append(f"- {cls}: {', '.join(unique)}")
        return "\n".join(lines)

    # Generic fallback — just describe the telemetry fields
    return (
        "GAME SOURCE REFERENCE (the specific game classes are unknown — reference "
        "the telemetry field names when identifying issues):\n"
        "- Fields available: hp, hp_max, super_meter, super_meter_max, deaths, "
        "level_time, level_ending, level_won, level_mode, in_game, scene_name, "
        "is_loading\n"
        "- Use the field name as both the 'file' and 'line' values in issues and "
        "suggestions (e.g. file='scene_name', line='is_loading')"
    )


def _build_system_prompt(game_name: str) -> str:
    """Build the full system prompt with game-specific references."""
    ref = _build_game_ref_section(game_name)
    return _SYSTEM_PROMPT_BASE.format(game_ref_section=ref)


def _build_analysis_prompt(stats: Dict[str, Any]) -> str:
    """Build the analysis prompt from extracted stats."""
    lines = [
        "Analyze this game session telemetry data:\n",
        f"Game: {stats.get('game', 'Unknown')}",
        f"Duration: {stats.get('duration_seconds', 0):.1f} seconds",
        f"Total frames captured: {stats.get('total_frames', 0)}",
        f"Capture interval: {stats.get('capture_interval', 0.5)}s",
        "",
        f"Deaths: {stats.get('deaths', 0)}",
        f"Total damage events: {stats.get('total_damage_taken', 0)}",
        f"Max HP observed: {stats.get('max_hp', 0)}",
        f"Time in combat: {stats.get('time_in_combat', 0):.1f}s",
        f"Time loading: {stats.get('time_loading', 0):.1f}s",
        f"Loading frames: {stats.get('loading_frames', 0)}",
        "",
    ]

    # -- Frame timing diagnostics --
    timing = stats.get("timing_stats", {})
    if timing:
        lines.append("CAPTURE TIMING DIAGNOSTICS:")
        lines.append(f"  Expected interval: {timing['expected_interval']}s")
        lines.append(
            f"  Actual avg delta: {timing['avg_delta']:.3f}s "
            f"(min={timing['min_delta']:.3f}s, max={timing['max_delta']:.3f}s)"
        )
        lines.append(f"  Avg jitter: {timing['avg_jitter']:.4f}s")
        lines.append(
            f"  Late frames (>1.5x interval): {timing['late_frames']} "
            f"({timing['late_frame_pct']:.1f}%)"
        )
        lines.append("")

    # -- Field initialization timeline --
    init_times = stats.get("field_init_times", {})
    if init_times:
        lines.append("FIELD INITIALIZATION TIMELINE:")
        lines.append("  (frame# @ elapsed_time: first non-zero value)")
        for field, info in sorted(init_times.items(), key=lambda x: x[1]["frame"]):
            lines.append(
                f"  {field}: frame {info['frame']} @ {info['elapsed']:.1f}s "
                f"(value={info['value']})"
            )
        lines.append("")

    # -- Null/uninitialized reads --
    null_count = stats.get("null_read_count", 0)
    null_pct = stats.get("null_read_pct", 0)
    if null_count:
        null_frames = stats.get("null_read_frames", [])
        lines.append(f"NULL/UNINITIALIZED READS: {null_count} frames ({null_pct:.1f}%)")
        # Show ranges instead of every frame
        if null_frames:
            ranges = _compress_ranges(null_frames)
            lines.append(f"  Frame ranges: {ranges}")
        lines.append("")

    # -- State anomalies --
    anomalies = stats.get("state_anomalies", [])
    if anomalies:
        lines.append(f"STATE ANOMALIES ({len(anomalies)}):")
        for a in anomalies[:25]:
            if a["type"] == "hp_increase":
                lines.append(
                    f"  frame {a['frame']} @ {a['at']:.1f}s: {a['type']} "
                    f"HP {a['from_hp']}->{a['to_hp']} -- {a.get('detail', '')}"
                )
            else:
                lines.append(
                    f"  frame {a['frame']} @ {a['at']:.1f}s: {a['type']} "
                    f"{a['from']}->{a['to']}"
                )
        if len(anomalies) > 25:
            lines.append(f"  ... and {len(anomalies) - 25} more")
        lines.append("")

    # Scene transitions
    transitions = stats.get("scene_transitions", [])
    if transitions:
        lines.append(f"SCENE TRANSITIONS ({len(transitions)}):")
        for t in transitions[:20]:
            lines.append(f"  {t['at']:.1f}s: {t['from'] or '(start)'} -> {t['to']}")
        if len(transitions) > 20:
            lines.append(f"  ... and {len(transitions) - 20} more")
        lines.append("")

    # Damage events
    damage = stats.get("damage_events", [])
    if damage:
        lines.append(f"DAMAGE EVENTS ({len(damage)}):")
        for d in damage[:15]:
            lines.append(
                f"  {d['at']:.1f}s: HP {d['from_hp']} -> {d['to_hp']} "
                f"(scene: {d['scene']})"
            )
        if len(damage) > 15:
            lines.append(f"  ... and {len(damage) - 15} more")
        lines.append("")

    # HP timeline (sampled)
    hp_tl = stats.get("hp_timeline", [])
    if hp_tl:
        # Sample ~30 points max
        step = max(1, len(hp_tl) // 30)
        sampled = hp_tl[::step]
        lines.append(f"HP TIMELINE (sampled {len(sampled)} of {len(hp_tl)} frames):")
        lines.append(f"  {sampled}")
        lines.append("")

    # Scenes visited
    scenes = stats.get("scenes_visited", [])
    if scenes:
        lines.append(f"Scenes visited: {', '.join(scenes)}")
        lines.append("")

    # Initial and final state
    initial = stats.get("initial_state", {})
    if initial:
        lines.append("INITIAL GAME STATE (frame 0):")
        for k, v in initial.items():
            lines.append(f"  {k}: {v}")
        lines.append("")

    final = stats.get("final_state", {})
    if final:
        lines.append("FINAL GAME STATE (last frame):")
        for k, v in final.items():
            lines.append(f"  {k}: {v}")

    return "\n".join(lines)


def _compress_ranges(nums: List[int]) -> str:
    """Compress [0,1,2,3,7,8,9] into '0-3, 7-9'."""
    if not nums:
        return ""
    ranges = []
    start = nums[0]
    end = nums[0]
    for n in nums[1:]:
        if n == end + 1:
            end = n
        else:
            ranges.append(f"{start}-{end}" if start != end else str(start))
            start = end = n
    ranges.append(f"{start}-{end}" if start != end else str(start))
    return ", ".join(ranges)


def _parse_llm_response(text: str) -> Dict[str, Any]:
    """Parse the LLM JSON response, handling markdown fences."""
    text = text.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.splitlines()
        # Remove first and last line if they're fences
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON in the text
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
    return {}


# ======================================================================
# Main entry point
# ======================================================================


def generate_report(
    session_dir: Optional[Path] = None,
    log_fn=None,
) -> GameReport:
    """Generate a full game feedback report for a session.

    Args:
        session_dir: Path to session directory. If None, uses latest.
        log_fn: Optional callback for progress logging (e.g. gui log).

    Returns:
        GameReport with ratings, issues, suggestions.
    """

    def _log(msg):
        if log_fn:
            log_fn(msg)
        else:
            print(msg)

    report = GameReport()
    report.generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # -- Find session --
    if session_dir is None:
        session_dir = find_latest_session()
        if session_dir is None:
            _log("ERROR: No recorded sessions found.\n")
            _log("Record a game session first (Test > Launch Game).\n")
            report.summary = "No session data available."
            return report

    session_dir = Path(session_dir)
    report.session_name = session_dir.name
    _log(f"Analyzing session: {session_dir.name}\n")

    # -- Load data --
    manifest = load_session_manifest(session_dir)
    frames = load_session_frames(session_dir)
    _log(f"  Loaded {len(frames)} frames\n")

    if not frames:
        _log("  WARNING: No frame data in session. Using directory metadata only.\n")
        # Still produce a minimal report
        frame_pngs = list(session_dir.glob("frame_*.png"))
        report.total_frames = len(frame_pngs)
        report.game = manifest.get("game", session_dir.name.split("_")[0])
        report.summary = (
            f"Session has {len(frame_pngs)} screenshots but no state data. "
            "Memory reader may not have been connected."
        )
        report.issues.append(
            Issue(
                "No game state data recorded -- memory reader was not attached.",
                file="pipeline/gameplay_recorder.py",
                line="400",
            )
        )
        report.suggestions.append(
            Suggestion(
                "Ensure the game is running and memory reader is connected before recording.",
                file="pipeline/gameplay_recorder.py",
                line="293",
                fix="Check that GameStateReader is initialized before capture loop starts.",
            )
        )
        report.ratings.append(
            Rating(
                "Loading Performance",
                1,
                "Critical",
                "Cannot assess -- no telemetry data captured.",
            )
        )
        report.ratings.append(
            Rating(
                "Runtime Stability",
                1,
                "Critical",
                "Cannot assess -- game state was never read.",
            )
        )
        report.ratings.append(
            Rating("Scene Transitions", 1, "Critical", "No scene data available.")
        )
        report.ratings.append(
            Rating(
                "Responsiveness",
                1,
                "Critical",
                "Cannot assess -- no game state timeline.",
            )
        )
        return report

    # -- Extract stats --
    _log("  Extracting session statistics...\n")
    stats = extract_session_stats(frames, manifest)
    report.game = stats.get("game", session_dir.name.split("_")[0])
    report.total_frames = stats["total_frames"]
    report.duration_seconds = stats["duration_seconds"]
    report.deaths = stats["deaths"]
    report.scenes_visited = stats["scenes_visited"]
    report.hp_timeline = stats["hp_timeline"]

    # -- Check if we have meaningful data --
    has_gameplay = any(f.get("state", {}).get("scene_name", "") for f in frames)

    if not has_gameplay:
        _log("  WARNING: All frames show empty game state (game not loaded).\n")
        _log("  Producing report from timing data only.\n")
        report.summary = (
            f"Session recorded {stats['total_frames']} frames over "
            f"{stats['duration_seconds']:.0f}s but the game never loaded "
            f"into a scene. The memory reader was connected but the game "
            f"had not started gameplay."
        )
        report.issues.append(
            Issue(
                "Game never entered a playable scene during recording. All pointer "
                "chains returned null for the entire session.",
                file="tools/game_state.py",
                line="565",
            )
        )
        report.issues.append(
            Issue(
                "All frames show in_game=false and empty scene_name. Mono objects "
                "for PlayerData and SceneLoader were never allocated.",
                file="conf/game_configs/cuphead.py",
                line="148",
            )
        )
        report.suggestions.append(
            Suggestion(
                "Wait for the game to fully load before starting capture.",
                file="pipeline/gameplay_recorder.py",
                line="367",
                fix="Add a pre-capture check that polls GameStateReader.read_all() "
                "until at least scene_name is non-empty before entering the main loop.",
            )
        )
        report.suggestions.append(
            Suggestion(
                "Check Wine compatibility if the game fails to launch.",
                file="tools/cuphead_memory.py",
                fix="Verify Wine process spawns correctly and mono_bridge.so is preloaded.",
            )
        )
        report.ratings.append(
            Rating(
                "Loading Performance",
                1,
                "Critical",
                f"Game never finished loading in {stats['duration_seconds']:.0f}s. "
                "No scene was ever entered -- the game may have stalled during startup.",
            )
        )
        report.ratings.append(
            Rating(
                "Runtime Stability",
                2,
                "Poor",
                "Game process was running but never reached a playable state. "
                "All game systems remained uninitialized for the entire session.",
            )
        )
        report.ratings.append(
            Rating(
                "Scene Transitions",
                1,
                "Critical",
                "No scene transitions occurred. SceneLoader.SceneName remained "
                "empty throughout the session.",
            )
        )
        report.ratings.append(
            Rating(
                "Responsiveness",
                1,
                "Critical",
                "Cannot assess -- game never reached gameplay.",
            )
        )
        return report

    # -- Call LLM for analysis --
    _log("  Sending to LLM for analysis...\n")
    try:
        from conf.config_parser import main_conf as config
        from tools.functions import get_api_key
        from pipeline.generate_bot_script import get_llm_client, call_llm

        provider = config.LLM_PROVIDER
        model = config.LLM_MODEL
        api_key = get_api_key()

        if not api_key:
            _log("  ERROR: No API key configured. Set one in Settings.\n")
            report.summary = "LLM analysis unavailable -- no API key configured."
            report.issues.append(
                Issue(
                    "No LLM API key configured. Cannot run AI-powered analysis.",
                    file="conf/main_conf.ini",
                    line="21",
                )
            )
            report.suggestions.append(
                Suggestion(
                    "Configure an API key in Settings to enable AI-powered analysis.",
                    file="tools/functions.py",
                    line="8",
                    fix="Store API key via: secret-tool store --label='Paper Engine' "
                    "service paper_engine user api_key",
                )
            )
            # Still include the stats-based info
            _fill_stats_only(report, stats)
            return report

        client = get_llm_client(provider, api_key)
        prompt = _build_analysis_prompt(stats)
        system_prompt = _build_system_prompt(report.game)

        _log(f"  Using {provider}/{model}...\n")
        response = call_llm(
            client,
            provider,
            model,
            system_prompt + "\n\n" + prompt,
            max_tokens=8192,
        )
        report.raw_analysis = response

        # Parse structured response
        parsed = _parse_llm_response(response)
        if parsed:
            for r in parsed.get("ratings", []):
                report.ratings.append(
                    Rating(
                        category=r.get("category", "Unknown"),
                        score=int(r.get("score", 3)),
                        label=r.get("label", "Fair"),
                        detail=r.get("detail", ""),
                    )
                )
            for iss in parsed.get("issues", []):
                if isinstance(iss, dict):
                    report.issues.append(
                        Issue(
                            description=iss.get("description", ""),
                            file=iss.get("file", ""),
                            line=str(iss.get("line", "")),
                            fix=iss.get("fix", ""),
                        )
                    )
                else:
                    report.issues.append(Issue(description=str(iss)))
            for sug in parsed.get("suggestions", []):
                if isinstance(sug, dict):
                    report.suggestions.append(
                        Suggestion(
                            description=sug.get("description", ""),
                            file=sug.get("file", ""),
                            line=str(sug.get("line", "")),
                            fix=sug.get("fix", ""),
                        )
                    )
                else:
                    report.suggestions.append(Suggestion(description=str(sug)))
            report.summary = parsed.get("summary", "")
            _log("  Analysis complete.\n")
        else:
            _log("  WARNING: Could not parse LLM response as JSON.\n")
            report.summary = response[:500]

    except ImportError as e:
        _log(f"  ERROR: Missing dependency: {e}\n")
        report.summary = f"LLM analysis unavailable: {e}"
        _fill_stats_only(report, stats)
    except Exception as e:
        _log(f"  ERROR: LLM call failed: {e}\n")
        report.summary = f"LLM analysis failed: {e}"
        _fill_stats_only(report, stats)

    return report


def _fill_stats_only(report: GameReport, stats: Dict[str, Any]):
    """Fill report with stats-only ratings when LLM is unavailable."""
    duration = stats.get("duration_seconds", 0)
    loading_pct = stats.get("time_loading", 0) / max(duration, 1) * 100
    init_times = stats.get("field_init_times", {})
    anomalies = stats.get("state_anomalies", [])
    transitions = stats.get("scene_transitions", [])

    # Loading Performance — how long until gameplay
    first_gameplay = init_times.get("scene_name", {}).get("elapsed", duration)
    if first_gameplay > 60:
        lp_score, lp_label = 1, "Critical"
    elif first_gameplay > 30:
        lp_score, lp_label = 2, "Poor"
    elif first_gameplay > 15:
        lp_score, lp_label = 3, "Fair"
    elif first_gameplay > 5:
        lp_score, lp_label = 4, "Good"
    else:
        lp_score, lp_label = 5, "Excellent"
    report.ratings.append(
        Rating(
            "Loading Performance",
            lp_score,
            lp_label,
            f"Time to first scene: {first_gameplay:.1f}s. "
            f"{loading_pct:.0f}% of session spent loading.",
        )
    )

    # Runtime Stability — anomalies and state consistency
    anomaly_count = len(anomalies)
    deaths = stats.get("deaths", 0)
    if anomaly_count > 10:
        rs_score, rs_label = 2, "Poor"
    elif anomaly_count > 3:
        rs_score, rs_label = 3, "Fair"
    else:
        rs_score, rs_label = 4, "Good"
    report.ratings.append(
        Rating(
            "Runtime Stability",
            rs_score,
            rs_label,
            f"{anomaly_count} state anomalies detected, {deaths} deaths. "
            f"{stats.get('total_frames')} frames captured over {duration:.0f}s.",
        )
    )

    # Scene Transitions — flow and reset behavior
    num_transitions = len(transitions)
    if num_transitions == 0:
        st_score, st_label = 1, "Critical"
    elif num_transitions < 2:
        st_score, st_label = 3, "Fair"
    else:
        st_score, st_label = 4, "Good"
    scenes = stats.get("scenes_visited", [])
    report.ratings.append(
        Rating(
            "Scene Transitions",
            st_score,
            st_label,
            f"{num_transitions} transitions across {len(scenes)} scenes.",
        )
    )

    # Responsiveness — does the game react during gameplay
    combat_time = stats.get("time_in_combat", 0)
    damage_events = stats.get("total_damage_taken", 0)
    if combat_time > 10:
        rp_score, rp_label = 4, "Good"
    elif combat_time > 0:
        rp_score, rp_label = 3, "Fair"
    else:
        rp_score, rp_label = 2, "Poor"
    report.ratings.append(
        Rating(
            "Responsiveness",
            rp_score,
            rp_label,
            f"{combat_time:.1f}s of active gameplay, "
            f"{damage_events} damage events recorded.",
        )
    )


def export_report(report: GameReport, path: Path) -> Path:
    """Write report to a text file. Returns the path."""
    path.write_text(report.to_text())
    return path


# ======================================================================
# Game directory analysis (non-session, config/log based)
# ======================================================================

# File patterns to scan, in priority order.  Each tuple is
# (glob, max_lines) — we read up to max_lines from each match.
_SCAN_PATTERNS: List[tuple] = [
    # Benchmark / log files
    ("**/benchmark*.log", 200),
    ("**/Logs/*.log", 300),
    # UE system settings (merged effective config)
    ("**/Config/*SystemSettings.ini", 200),
    # Engine config
    ("**/Config/*Engine.ini", 120),
    # Platform overrides
    ("**/Config/PC/*.ini", 80),
    # Launcher / hardware detection
    ("**/Config/Launcher.ini", 250),
    # Default configs (for comparison)
    ("**/Config/Default*Settings.ini", 150),
    ("**/Config/Default*Engine.ini", 100),
    # Base engine config
    ("**/Engine/Config/Base*Settings.ini", 120),
    ("**/Engine/Config/BaseEngine.ini", 120),
    # Console variables
    ("**/Config/ConsoleVariables.ini", 50),
    # id Tech / Void Engine (Dishonored 2, DOOM, etc.)
    ("base/*.cfg", 100),
    ("base/*.txt", 50),
    # Unity player settings
    ("*_Data/*.cfg", 50),
    ("*_Data/boot.config", 20),
    # Generic game settings
    ("*.cfg", 100),
    ("**/GameUserSettings.ini", 200),
    ("**/Input.ini", 50),
    # Version info
    ("version.txt", 10),
    ("**/goggame*.info", 50),
]

# Max total chars to feed into the prompt (leave room for system prompt)
_MAX_FILE_CONTENT = 60000


def _scan_game_dir(game_dir: Path, log_fn=None) -> Dict[str, str]:
    """Scan a game install directory for performance-relevant files.

    Returns dict of {relative_path: file_content_string}.
    """

    def _log(msg):
        if log_fn:
            log_fn(msg)
        else:
            print(msg)

    found: Dict[str, str] = {}
    total_chars = 0

    for pattern, max_lines in _SCAN_PATTERNS:
        matches = sorted(game_dir.glob(pattern))
        for fpath in matches:
            if not fpath.is_file():
                continue
            rel = str(fpath.relative_to(game_dir))
            if rel in found:
                continue
            try:
                # Try UTF-8 first, fall back to UTF-16, then latin-1
                text = None
                for enc in ("utf-8", "utf-16", "latin-1"):
                    try:
                        text = fpath.read_text(encoding=enc)
                        break
                    except (UnicodeDecodeError, UnicodeError):
                        continue
                if text is None:
                    continue

                # Truncate to max_lines
                lines = text.splitlines()[:max_lines]
                content = "\n".join(lines)

                if total_chars + len(content) > _MAX_FILE_CONTENT:
                    remaining = _MAX_FILE_CONTENT - total_chars
                    if remaining < 200:
                        _log(f"  Skipping {rel} (content limit reached)\n")
                        continue
                    content = content[:remaining] + "\n... (truncated)"

                found[rel] = content
                total_chars += len(content)
                _log(f"  Found: {rel} ({len(lines)} lines)\n")

                if total_chars >= _MAX_FILE_CONTENT:
                    break
            except Exception:
                continue
        if total_chars >= _MAX_FILE_CONTENT:
            break

    return found


# ======================================================================
# Source 2: Engine detection
# ======================================================================


def detect_engine(game_dir: Path, log_fn=None) -> Dict[str, Any]:
    """Identify game engine from directory structure, executables, and DLLs."""

    def _log(msg: str):
        if log_fn:
            log_fn(msg)
        else:
            print(msg, end="")

    result: Dict[str, Any] = {
        "engine": "unknown",
        "engine_version": "",
        "architecture": "",
        "key_markers": [],
        "game_module": "",
    }

    # -- Check for Unreal Engine --
    engine_dir = game_dir / "Engine"
    game_subdirs = [
        d for d in game_dir.iterdir() if d.is_dir() and d.name.endswith("Game")
    ]

    if engine_dir.is_dir() and game_subdirs:
        game_module = game_subdirs[0].name
        result["game_module"] = game_module

        # UE4 markers: Content/Paks, .uproject files
        has_paks = any(game_dir.glob("**/Content/Paks/*.pak"))
        has_uproject = any(game_dir.glob("*.uproject"))

        if has_paks or has_uproject:
            result["engine"] = "Unreal Engine 4"
            result["key_markers"].append("Content/Paks or .uproject file")
        else:
            result["engine"] = "Unreal Engine 3"
            result["key_markers"].append(f"{game_module}/ directory (UE3 game module)")

    # -- Check for Unity --
    elif not engine_dir.is_dir():
        data_dirs = [
            d for d in game_dir.iterdir() if d.is_dir() and d.name.endswith("_Data")
        ]
        managed_dll = any(game_dir.glob("*_Data/Managed/UnityEngine.dll"))
        mono_dir = any(game_dir.glob("MonoBleedingEdge"))

        if data_dirs and (managed_dll or mono_dir):
            result["engine"] = "Unity"
            result["game_module"] = data_dirs[0].name
            result["key_markers"].append(f"{data_dirs[0].name}/ directory")
            if managed_dll:
                result["key_markers"].append("Managed/UnityEngine.dll")
            if mono_dir:
                result["key_markers"].append("MonoBleedingEdge/ (Mono runtime)")
            il2cpp = any(game_dir.glob("**/GameAssembly.dll"))
            if il2cpp:
                result["engine_version"] = "IL2CPP"
            elif mono_dir:
                result["engine_version"] = "Mono"

    # -- Check for id Tech / Void Engine --
    if result["engine"] == "unknown":
        base_dir = game_dir / "base"
        has_cfg = base_dir.is_dir() and any(base_dir.glob("*.cfg"))
        # GameWorks DLLs are a strong signal for Void Engine (Arkane)
        gfsdk_dlls = list(game_dir.glob("GFSDK_*.dll"))
        root_exes = list(game_dir.glob("*.exe"))

        if has_cfg and root_exes:
            # Distinguish Void Engine (Arkane) from classic id Tech
            if gfsdk_dlls:
                result["engine"] = "Void Engine (id Tech)"
                result["key_markers"].append("base/ directory with .cfg files")
                result["key_markers"].append(
                    f"GameWorks DLLs: {', '.join(d.name for d in gfsdk_dlls[:3])}"
                )
            else:
                result["engine"] = "id Tech"
                result["key_markers"].append("base/ directory with .cfg files")
            result["game_module"] = root_exes[0].stem

    # -- Detect architecture from executables --
    exes = list(game_dir.glob("*.exe")) or list(game_dir.glob("Binaries/**/*.exe"))
    if exes:
        try:
            with open(exes[0], "rb") as f:
                f.seek(0x3C)
                pe_offset = int.from_bytes(f.read(4), "little")
                f.seek(pe_offset + 4)
                machine = int.from_bytes(f.read(2), "little")
                if machine == 0x8664:
                    result["architecture"] = "x86-64"
                elif machine == 0x14C:
                    result["architecture"] = "x86"
                result["key_markers"].append(f"Executable: {exes[0].name}")
        except Exception:
            pass

    if result["engine"] != "unknown":
        _log(
            f"  Engine: {result['engine']}"
            f"{' (' + result['engine_version'] + ')' if result['engine_version'] else ''}"
            f" [{result['architecture']}]\n"
        )
    else:
        _log("  Engine: unknown\n")

    return result


# ======================================================================
# Source 3: Community data (ProtonDB)
# ======================================================================

_STEAM_APPIDS: Dict[str, int] = {
    "batman arkham knight": 208650,
    "batman arkham city": 200260,
    "batman arkham asylum": 35140,
    "batman arkham origins": 209000,
    "cuphead": 268910,
    "dark souls iii": 374320,
    "dark souls remastered": 570940,
    "elden ring": 1245620,
    "red dead redemption 2": 1174180,
    "cyberpunk 2077": 1091500,
    "the witcher 3": 292030,
    "dishonored 2": 403640,
    "dishonored": 205100,
    "fallout 4": 377160,
    "skyrim special edition": 489830,
    "grand theft auto v": 271590,
    "no man's sky": 275850,
    "doom eternal": 782330,
    "horizon zero dawn": 1151640,
    "god of war": 1593500,
    "spider-man remastered": 1817070,
    "days gone": 1259420,
    "death stranding": 1190460,
    "monster hunter world": 582010,
    "resident evil village": 1196590,
    "sekiro": 814380,
    "control": 870780,
    "hades": 1145360,
    "celeste": 504230,
    "hollow knight": 367520,
}


def _fetch_url_json(url: str, timeout: int = 10) -> Optional[Dict]:
    """Fetch JSON from a URL.  Returns None on any failure."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "PaperEngine/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None


def fetch_protondb(game_name: str, log_fn=None) -> Dict[str, Any]:
    """Fetch ProtonDB compatibility summary for a game."""

    def _log(msg: str):
        if log_fn:
            log_fn(msg)
        else:
            print(msg, end="")

    normalized = game_name.lower().strip()
    app_id = _STEAM_APPIDS.get(normalized)
    if app_id is None:
        # Fuzzy match
        for name, aid in _STEAM_APPIDS.items():
            if name in normalized or normalized in name:
                app_id = aid
                break
    if not app_id:
        _log(f"  ProtonDB: No Steam AppID known for '{game_name}'\n")
        return {}

    _log(f"  ProtonDB: Fetching summary for AppID {app_id}...\n")
    url = f"https://www.protondb.com/api/v1/reports/summaries/{app_id}.json"
    data = _fetch_url_json(url)

    if not data:
        _log("  ProtonDB: Failed to fetch data\n")
        return {}

    result = {
        "tier": data.get("tier", "unknown"),
        "trending_tier": data.get("trendingTier", ""),
        "total_reports": data.get("total", 0),
        "confidence": data.get("confidence", ""),
        "score": data.get("score", 0),
        "best_reported": data.get("bestReportedTier", ""),
        "steam_app_id": app_id,
    }
    _log(
        f"  ProtonDB: {result['tier'].upper()} "
        f"(trending {result['trending_tier']}, "
        f"{result['total_reports']} reports)\n"
    )
    return result


# ======================================================================
# Source 4: Wine / DXVK prefix scan
# ======================================================================


def find_wine_prefix(game_name: str) -> Optional[Path]:
    """Auto-detect Wine prefix.  Checks Heroic and Steam paths."""
    home = Path.home()

    # Heroic: ~/Games/Heroic/Prefixes/default/<game>/
    heroic_base = home / "Games" / "Heroic" / "Prefixes" / "default"
    if heroic_base.is_dir():
        exact = heroic_base / game_name
        if exact.is_dir():
            return exact
        # Fuzzy match
        norm = game_name.lower().replace(" ", "").replace(":", "").replace("-", "")
        for d in heroic_base.iterdir():
            if d.is_dir():
                d_norm = (
                    d.name.lower().replace(" ", "").replace(":", "").replace("-", "")
                )
                if norm in d_norm or d_norm in norm:
                    return d

    # Steam: ~/.steam/steam/steamapps/compatdata/<appid>/pfx/
    app_id = _STEAM_APPIDS.get(game_name.lower().strip())
    if app_id:
        steam_pfx = (
            home / ".steam" / "steam" / "steamapps" / "compatdata" / str(app_id) / "pfx"
        )
        if steam_pfx.is_dir():
            return steam_pfx

    return None


def _parse_wine_registry_section(reg_path: Path, section_key: str) -> Dict[str, str]:
    """Extract key=value pairs from a Wine registry section."""
    values: Dict[str, str] = {}
    if not reg_path.exists():
        return values
    try:
        text = reg_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return values

    in_section = False
    target = f"[{section_key}]"

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("["):
            bracket_end = stripped.find("]")
            if bracket_end > 0:
                in_section = stripped[: bracket_end + 1] == target
                continue
        if not in_section:
            continue
        if stripped.startswith(("#", "@")) or not stripped:
            continue
        m = re.match(r'"([^"]+)"="([^"]*)"', stripped)
        if m:
            values[m.group(1)] = m.group(2)
        else:
            m2 = re.match(r'"([^"]+)"=(\w+):(.+)', stripped)
            if m2:
                values[m2.group(1)] = f"{m2.group(2)}:{m2.group(3)}"
    return values


def scan_wine_prefix(prefix_dir: Path, log_fn=None) -> Dict[str, Any]:
    """Scan a Wine prefix for runtime environment information."""

    def _log(msg: str):
        if log_fn:
            log_fn(msg)
        else:
            print(msg, end="")

    result: Dict[str, Any] = {
        "prefix_path": str(prefix_dir),
        "proton_version": "",
        "dxvk_cache": {},
        "dll_overrides": {},
        "env_vars": {},
        "wine_config": {},
    }

    # -- Proton version --
    version_file = prefix_dir / "version"
    if version_file.exists():
        try:
            result["proton_version"] = version_file.read_text().strip()
            _log(f"  Proton: {result['proton_version']}\n")
        except Exception:
            pass

    # -- DXVK cache --
    dxvk_dir = (
        prefix_dir / "drive_c" / "users" / "steamuser" / "AppData" / "Local" / "dxvk"
    )
    if dxvk_dir.is_dir():
        cache_files = [f for f in dxvk_dir.iterdir() if f.is_file()]
        total_size = sum(f.stat().st_size for f in cache_files)
        result["dxvk_cache"] = {
            "exists": True,
            "files": [f.name for f in cache_files],
            "total_size_mb": round(total_size / (1024 * 1024), 1),
            "file_count": len(cache_files),
        }
        _log(
            f"  DXVK cache: {result['dxvk_cache']['total_size_mb']} MB "
            f"({len(cache_files)} files)\n"
        )
    else:
        result["dxvk_cache"] = {"exists": False}
        _log("  DXVK cache: not found\n")

    # -- DLL overrides from registry --
    user_reg = prefix_dir / "user.reg"
    overrides = _parse_wine_registry_section(
        user_reg, "Software\\\\Wine\\\\DllOverrides"
    )
    if overrides:
        result["dll_overrides"] = overrides
        _standard = (
            "msvc",
            "vcomp",
            "vcruntime",
            "ucrtbase",
            "api-ms-win",
            "atl",
            "concrt",
            "vccorlib",
        )
        _skip_names = {"atiadlxx", "atidxx64", "nvcuda"}
        notable = {
            k: v
            for k, v in overrides.items()
            if not k.startswith(_standard) and k not in _skip_names
        }
        if notable:
            _log(
                f"  DLL overrides ({len(notable)} notable): "
                f"{', '.join(notable.keys())}\n"
            )
        else:
            _log(f"  DLL overrides: {len(overrides)} (standard Proton defaults)\n")

    # -- Environment variables --
    vol_env = _parse_wine_registry_section(user_reg, "Volatile Environment")
    if vol_env:
        result["env_vars"] = vol_env
        dxvk_vars = {k: v for k, v in vol_env.items() if "DXVK" in k or "VKD3D" in k}
        if dxvk_vars:
            _log(f"  DXVK env: {dxvk_vars}\n")

    # -- dxgi.dll / d3d11.dll check --
    sys32 = prefix_dir / "drive_c" / "windows" / "system32"
    for dll_name in ("dxgi", "d3d11"):
        dll_path = sys32 / f"{dll_name}.dll"
        if dll_path.exists():
            size_mb = dll_path.stat().st_size / (1024 * 1024)
            is_dxvk = size_mb > 2.0
            result["wine_config"][dll_name] = {
                "size_mb": round(size_mb, 1),
                "is_dxvk": is_dxvk,
            }
            if dll_name == "dxgi":
                _log(
                    f"  dxgi.dll: {'DXVK' if is_dxvk else 'Wine built-in'} "
                    f"({size_mb:.1f} MB)\n"
                )

    return result


# ======================================================================
# Source 5: Known fix detection
# ======================================================================


def detect_known_fixes(
    game_name: str,
    game_dir: Path,
    prefix_dir: Optional[Path],
    engine_info: Dict[str, Any],
    config_files: Dict[str, str],
    log_fn=None,
) -> Dict[str, Any]:
    """Check for known community fixes and common performance issues."""

    def _log(msg: str):
        if log_fn:
            log_fn(msg)
        else:
            print(msg, end="")

    result: Dict[str, Any] = {"installed": [], "missing": [], "checks": []}
    normalized = game_name.lower().strip()
    engine = engine_info.get("engine", "unknown").lower()

    # -- Game-specific --
    if "batman" in normalized and "knight" in normalized:
        _check_batman_ak(game_dir, prefix_dir, config_files, result, _log)

    # -- Generic engine --
    if "unreal engine 3" in engine:
        _check_ue3_generic(config_files, result, _log)

    # -- Generic Wine/Proton --
    if prefix_dir:
        _check_wine_generic(prefix_dir, result, _log)

    _log(
        f"  Known fixes: {len(result['installed'])} installed, "
        f"{len(result['missing'])} missing\n"
    )
    return result


def _check_batman_ak(
    game_dir: Path,
    prefix_dir: Optional[Path],
    config_files: Dict[str, str],
    result: Dict,
    log_fn,
):
    """Batman: Arkham Knight specific fix detection."""
    # 1. Arkham Quixote (custom dxgi.dll in game Binaries)
    quixote_path = game_dir / "Binaries" / "Win64" / "dxgi.dll"
    if quixote_path.exists():
        size_mb = quixote_path.stat().st_size / (1024 * 1024)
        if size_mb < 2.0:
            result["installed"].append(
                {
                    "name": "Arkham Quixote",
                    "description": "Custom dxgi.dll that recycles texture allocations "
                    "to fix streaming stuttering",
                    "note": "WARNING: May conflict with DXVK's dxgi.dll under Proton.",
                }
            )
        else:
            result["checks"].append(
                {
                    "name": "Arkham Quixote",
                    "status": "dxgi.dll in Binaries/ but appears to be DXVK copy",
                }
            )
    else:
        result["missing"].append(
            {
                "name": "Arkham Quixote",
                "description": "Custom dxgi.dll that recycles texture allocations "
                "to fix the notorious streaming stuttering",
                "severity": "recommended",
                "note": "Under Proton/DXVK this fix is complex -- DXVK replaces "
                "dxgi.dll system-wide, so the mod may need adaptation.",
                "url": "https://community.pcgamingwiki.com/files/file/2113-arkham-quixote/",
            }
        )

    # 2. GameWorks Interactive Smoke (biggest GPU hog)
    smoke_disabled = any(
        "bEnableInteractiveSmoke=False" in c for c in config_files.values()
    )
    if smoke_disabled:
        result["installed"].append(
            {
                "name": "GameWorks Smoke Disabled",
                "description": "Interactive smoke/fog disabled -- the single "
                "biggest GPU perf improvement for AK",
            }
        )
    else:
        result["missing"].append(
            {
                "name": "GameWorks Smoke Disabled",
                "description": "Interactive smoke/fog is the #1 GPU hog in AK",
                "severity": "critical",
                "fix": "Set bEnableInteractiveSmoke=False in "
                "BmGame/Config/BmSystemSettings.ini",
            }
        )

    # 3. GameWorks Paper Debris
    paper_disabled = any(
        "bEnableInteractivePaperDebris=False" in c for c in config_files.values()
    )
    if paper_disabled:
        result["installed"].append(
            {
                "name": "GameWorks Paper Debris Disabled",
                "description": "Interactive paper debris disabled -- reduces GPU load",
            }
        )

    # 4. SSD info check
    result["checks"].append(
        {
            "name": "SSD Storage",
            "status": "info",
            "note": "AK's texture streaming was designed for HDDs. Running from "
            f"SSD significantly reduces stutters. Game at: {game_dir}",
        }
    )


def _check_ue3_generic(config_files: Dict[str, str], result: Dict, log_fn):
    """Generic UE3 performance checks."""
    for path, content in config_files.items():
        pool_match = re.search(r"PoolSize\s*=\s*(\d+)", content)
        if pool_match:
            pool_mb = int(pool_match.group(1))
            if pool_mb < 512:
                result["checks"].append(
                    {
                        "name": "Low Texture Pool",
                        "status": "warning",
                        "note": f"Texture pool {pool_mb}MB in {path} -- may cause "
                        "excessive streaming on high-res displays",
                    }
                )


def _check_wine_generic(prefix_dir: Path, result: Dict, log_fn):
    """Generic Wine/Proton checks."""
    version_file = prefix_dir / "version"
    if version_file.exists():
        try:
            version = version_file.read_text().strip()
            if any(
                old in version.lower() for old in ("proton 5", "proton 6", "proton-4")
            ):
                result["checks"].append(
                    {
                        "name": "Old Proton Version",
                        "status": "warning",
                        "note": f"Running {version} -- consider GE-Proton 9+",
                    }
                )
        except Exception:
            pass

    dxvk_dir = (
        prefix_dir / "drive_c" / "users" / "steamuser" / "AppData" / "Local" / "dxvk"
    )
    if not dxvk_dir.is_dir() or not list(dxvk_dir.glob("*.dxvk.*")):
        result["checks"].append(
            {
                "name": "No DXVK Cache",
                "status": "warning",
                "note": "No DXVK state cache -- first run will stutter "
                "from shader compilation.",
            }
        )


# ======================================================================
# Multi-source LLM prompt builder
# ======================================================================


def _build_dir_system_prompt(
    engine_info: Dict[str, Any],
    community: Dict[str, Any],
    prefix_info: Dict[str, Any],
    fixes: Dict[str, Any],
) -> str:
    """Build a dynamic system prompt based on available data sources."""
    engine = engine_info.get("engine", "unknown")

    # -- Engine expertise block --
    if "unreal engine 3" in engine.lower():
        engine_expertise = (
            "YOUR ENGINE EXPERTISE (Unreal Engine 3):\n"
            "You have deep knowledge of UE3 internals. Key areas:\n"
            "- Texture streaming: UE3's streaming manager is the #1 stuttering "
            "source in ports. Analyze StreamingPoolSize, texture group LODs, "
            "AllowedTextureRequests, LimitTextureIncreasesInFlight. UE3 streaming "
            "was designed for HDDs -- SSDs change the profile entirely.\n"
            "- GameWorks integration: Many UE3 games (WB/Rocksteady) bolt on "
            "NVIDIA GameWorks. Interactive smoke, rain occlusion, enhanced light "
            "shafts, APEX destruction are common GPU killers.\n"
            "- Shadow system: Projected shadow volumes. MaxShadowResolution, "
            "hardware vs software filtering, conservative bounds are critical.\n"
            "- PhysX/APEX: CPU-side PhysX and APEX destruction budgets.\n"
            "- Frame rate management: bSmoothFrameRate interacts poorly with "
            "VSync and external limiters.\n"
            "- Post-processing cost hierarchy: AO > light shafts > DOF > bloom.\n"
            "- Wine/Proton: D3D11->Vulkan via DXVK adds overhead. Shader "
            "compilation stutter is a known issue. DXVK state cache size "
            "indicates pipeline compilation coverage.\n"
        )
    elif "unreal engine 4" in engine.lower():
        engine_expertise = (
            "YOUR ENGINE EXPERTISE (Unreal Engine 4):\n"
            "You have deep knowledge of UE4 internals. Key areas:\n"
            "- Shader compilation: Massive shader permutation system causes "
            "stutter. PSO caching is critical.\n"
            "- Level streaming: World Composition can cause hitches.\n"
            "- Texture streaming: Improved over UE3 but pool sizes still matter.\n"
            "- CVar overrides in Engine.ini and GameUserSettings.ini.\n"
        )
    elif "void" in engine.lower() or "id tech" in engine.lower():
        engine_expertise = (
            "YOUR ENGINE EXPERTISE (Void Engine / id Tech):\n"
            "You have deep knowledge of id Tech and Void Engine internals. Key areas:\n"
            "- Megatexture/virtual texturing: Void Engine uses virtual texturing "
            "derived from id Tech 5. Texture pop-in and streaming hitches are "
            "common, especially on HDDs or under memory pressure.\n"
            "- GPU compute: Void Engine uses compute shaders extensively for "
            "SSAO, shadow filtering, and post-processing. GFSDK_SSAO integration "
            "adds NVIDIA-specific overhead.\n"
            "- CPU threading: Void Engine improved on id Tech's job system but "
            "Dishonored 2's launch had severe CPU bottlenecks. Thread scheduling "
            "and draw call batching are known weak points.\n"
            "- Shader compilation: Like UE4, first-run stutter from shader "
            "compilation is common. DXVK state cache coverage matters.\n"
            "- Memory management: base/ directory .cfg files control engine "
            "CVars. User settings typically stored in AppData.\n"
            "- Wine/Proton: D3D11->Vulkan via DXVK. Void Engine games have "
            "historically had mixed Proton compat due to heavy compute usage.\n"
        )
    elif "unity" in engine.lower():
        engine_expertise = (
            "YOUR ENGINE EXPERTISE (Unity):\n"
            "- Mono vs IL2CPP: Mono has higher GC pressure and JIT overhead.\n"
            "- GC pressure: Boehm GC causes frame spikes.\n"
            "- Asset bundle loading: Synchronous loads cause hitches.\n"
            "- Rendering pipeline: Built-in vs URP vs HDRP differ hugely.\n"
        )
    else:
        engine_expertise = (
            "YOUR ENGINE EXPERTISE:\n"
            "Analyze config files and identify the engine from internal markers. "
            "Apply knowledge of common engine patterns.\n"
        )

    # -- Data source manifest --
    sources = ["Engine configuration files (configs, logs, benchmarks)"]
    if engine_info.get("engine") != "unknown":
        sources.append(f"Engine detection: {engine}")
    if community:
        tier = community.get("tier", "")
        sources.append(
            f"ProtonDB: {tier.upper()}, {community.get('total_reports', 0)} reports"
        )
    if prefix_info and prefix_info.get("proton_version"):
        sources.append(f"Wine/DXVK prefix: {prefix_info['proton_version']}")
    if fixes and (fixes.get("installed") or fixes.get("missing")):
        sources.append(
            f"Known fixes: {len(fixes.get('installed', []))} installed, "
            f"{len(fixes.get('missing', []))} missing"
        )
    sources_text = "\n".join(f"  {i + 1}. {s}" for i, s in enumerate(sources))

    # -- Assemble (plain string, no .format()) --
    json_example = (
        "Format your response EXACTLY as JSON:\n"
        "{\n"
        '  "ratings": [\n'
        '    {"category": "Engine Configuration", "score": 3, "label": "Fair", "detail": "..."},\n'
        '    {"category": "Runtime Environment", "score": 4, "label": "Good", "detail": "..."},\n'
        '    {"category": "Known Issues & Fixes", "score": 2, "label": "Poor", "detail": "..."},\n'
        '    {"category": "Performance Architecture", "score": 3, "label": "Fair", "detail": "..."}\n'
        "  ],\n"
        '  "issues": [\n'
        "    {\n"
        '      "description": "Detailed issue with engine-level explanation...",\n'
        '      "file": "path/to/config.ini",\n'
        '      "line": "SettingName=Value",\n'
        '      "fix": "Specific fix with values and reasoning"\n'
        "    }\n"
        "  ],\n"
        '  "suggestions": [\n'
        "    {\n"
        '      "description": "Improvement with architectural context...",\n'
        '      "file": "path/to/config.ini",\n'
        '      "line": "SettingName",\n'
        '      "fix": "Specific implementation with values"\n'
        "    }\n"
        "  ],\n"
        '  "summary": "Comprehensive performance summary."\n'
        "}\n\n"
        "ONLY output the JSON. No markdown fences, no extra text."
    )

    prompt = (
        "You are a senior game engine QA specialist and performance architect "
        "with deep expertise in UE3, UE4, UE5, Unity, CryEngine, and id Tech "
        "internals, plus extensive Linux gaming experience via Wine/Proton/DXVK.\n\n"
        "You are NOT a settings auditor. You are a game development expert who "
        "understands engine internals, rendering pipelines, memory management, "
        "and the architectural decisions that cause performance problems. You "
        "know the source code of these engines and how they work under the hood.\n\n"
        f"{engine_expertise}\n"
        f"DATA SOURCES AVAILABLE:\n{sources_text}\n\n"
        "RATING CATEGORIES (1-5 stars each):\n"
        "1. Engine Configuration: Settings optimization, graphics quality vs "
        "performance tradeoffs, texture streaming health, shadow quality, "
        "post-processing overhead, frame rate management. Cite specific settings, "
        "values, and file paths.\n"
        "2. Runtime Environment: Proton/Wine version suitability, DXVK setup, "
        "DLL configuration, shader cache state, compatibility layer health. "
        "If no prefix data available, assess from engine's known Proton compat.\n"
        "3. Known Issues & Fixes: Community fixes applied/missing, outstanding "
        "issues with documented solutions. Reference fixes by name and impact. "
        "Incorporate ProtonDB data if available.\n"
        "4. Performance Architecture: Engine-level or game-level architectural "
        "bottlenecks beyond settings. Streaming architecture, GPU/CPU budget, "
        "port quality issues, fundamental design decisions. Use engine expertise "
        "to identify issues a settings change alone cannot fix.\n\n"
        "For each rating: score (1-5), short label, DETAILED multi-sentence "
        "explanation citing settings, file paths, engine internals, and "
        "architectural patterns. Explain WHY at the engine level.\n\n"
        "ISSUES: Specific problems across ALL data sources -- not just "
        "misconfigurations but architectural problems, engine bugs, "
        "compatibility layer issues, missing community fixes. Cite exact file "
        "paths and settings. Explain engine-level cause. Provide concrete fix.\n\n"
        "SUGGESTIONS: Go beyond config tweaks. Engine-level optimizations, "
        "community fix installations, runtime environment changes, "
        "architectural workarounds. Be specific and actionable.\n\n"
        "SUMMARY: 4-6 sentences for a game developer or Linux gaming power "
        "user. Cover engine, known weaknesses, runtime environment health, "
        "most impactful improvements.\n\n"
        f"{json_example}"
    )
    return prompt


def _build_multi_source_user_prompt(
    game_name: str,
    config_files: Dict[str, str],
    engine_info: Dict[str, Any],
    community: Dict[str, Any],
    prefix_info: Dict[str, Any],
    fixes: Dict[str, Any],
) -> str:
    """Build the user prompt with all collected data organized by source."""
    sections: List[str] = [
        f"GAME: {game_name}\n",
    ]

    # -- Source 1: Engine configs --
    if config_files:
        sections.append("=" * 60)
        sections.append("SOURCE 1: ENGINE CONFIGURATION FILES")
        sections.append("=" * 60)
        for rel_path, content in config_files.items():
            sections.append(f"\n--- FILE: {rel_path} ---")
            sections.append(content)

    # -- Source 2: Engine detection --
    if engine_info.get("engine") != "unknown":
        sections.append("\n" + "=" * 60)
        sections.append("SOURCE 2: ENGINE DETECTION")
        sections.append("=" * 60)
        sections.append(f"Engine: {engine_info['engine']}")
        if engine_info.get("engine_version"):
            sections.append(f"Backend: {engine_info['engine_version']}")
        if engine_info.get("architecture"):
            sections.append(f"Architecture: {engine_info['architecture']}")
        if engine_info.get("game_module"):
            sections.append(f"Game module: {engine_info['game_module']}")
        if engine_info.get("key_markers"):
            sections.append(f"Markers: {', '.join(engine_info['key_markers'])}")

    # -- Source 3: Community data --
    if community:
        sections.append("\n" + "=" * 60)
        sections.append("SOURCE 3: COMMUNITY DATA (ProtonDB)")
        sections.append("=" * 60)
        sections.append(f"Tier: {community.get('tier', 'N/A').upper()}")
        sections.append(f"Trending: {community.get('trending_tier', 'N/A')}")
        sections.append(f"Total reports: {community.get('total_reports', 0)}")
        sections.append(f"Confidence: {community.get('confidence', 'N/A')}")
        sections.append(f"Best reported: {community.get('best_reported', 'N/A')}")

    # -- Source 4: Wine/DXVK prefix --
    if prefix_info and prefix_info.get("proton_version"):
        sections.append("\n" + "=" * 60)
        sections.append("SOURCE 4: WINE/DXVK RUNTIME ENVIRONMENT")
        sections.append("=" * 60)
        sections.append(f"Proton: {prefix_info['proton_version']}")
        sections.append(f"Prefix: {prefix_info.get('prefix_path', 'N/A')}")

        cache = prefix_info.get("dxvk_cache", {})
        if cache.get("exists"):
            sections.append(
                f"DXVK state cache: {cache['total_size_mb']} MB "
                f"({cache['file_count']} files)"
            )
        else:
            sections.append("DXVK state cache: NOT FOUND (expect shader stutter)")

        overrides = prefix_info.get("dll_overrides", {})
        if overrides:
            _standard = (
                "msvc",
                "vcomp",
                "vcruntime",
                "ucrtbase",
                "api-ms-win",
                "atl",
                "concrt",
                "vccorlib",
            )
            _skip = {"atiadlxx", "atidxx64", "nvcuda"}
            notable = {
                k: v
                for k, v in overrides.items()
                if not k.startswith(_standard) and k not in _skip
            }
            if notable:
                sections.append(f"Notable DLL overrides: {json.dumps(notable)}")
            else:
                sections.append(
                    f"DLL overrides: {len(overrides)} (standard Proton defaults)"
                )

        env_vars = prefix_info.get("env_vars", {})
        dxvk_env = {
            k: v
            for k, v in env_vars.items()
            if "DXVK" in k or "VKD3D" in k or "WINE" in k
        }
        if dxvk_env:
            sections.append(f"DXVK/Wine env: {json.dumps(dxvk_env)}")

        wc = prefix_info.get("wine_config", {})
        if "dxgi" in wc:
            d = wc["dxgi"]
            sections.append(
                f"dxgi.dll: {'DXVK' if d['is_dxvk'] else 'Wine'} ({d['size_mb']} MB)"
            )
        if "d3d11" in wc:
            d = wc["d3d11"]
            sections.append(
                f"d3d11.dll: {'DXVK' if d['is_dxvk'] else 'Wine'} ({d['size_mb']} MB)"
            )

    # -- Source 5: Known fixes --
    if fixes and (
        fixes.get("installed") or fixes.get("missing") or fixes.get("checks")
    ):
        sections.append("\n" + "=" * 60)
        sections.append("SOURCE 5: KNOWN FIX STATUS")
        sections.append("=" * 60)

        for fix in fixes.get("installed", []):
            sections.append(f"[INSTALLED] {fix['name']}: {fix['description']}")
            if fix.get("note"):
                sections.append(f"  NOTE: {fix['note']}")

        for fix in fixes.get("missing", []):
            sev = fix.get("severity", "info").upper()
            sections.append(f"[MISSING - {sev}] {fix['name']}: {fix['description']}")
            if fix.get("fix"):
                sections.append(f"  FIX: {fix['fix']}")
            if fix.get("note"):
                sections.append(f"  NOTE: {fix['note']}")
            if fix.get("url"):
                sections.append(f"  URL: {fix['url']}")

        for chk in fixes.get("checks", []):
            st = chk.get("status", "info").upper()
            sections.append(f"[CHECK - {st}] {chk['name']}: {chk.get('note', '')}")

    return "\n".join(sections)


def generate_report_from_dir(
    game_dir: Path,
    prefix_dir: Optional[Path] = None,
    log_fn=None,
) -> GameReport:
    """Generate a multi-source performance report from a game install.

    Collects data from up to 6 sources:
      1. Engine config files (ini, logs, benchmarks)
      2. Engine detection (UE3/UE4/Unity from directory structure)
      3. Community data (ProtonDB API)
      4. Wine/DXVK prefix scan (Proton version, caches, overrides)
      5. Known fix detection (game-specific community fixes)
      6. LLM analysis (fed all sources above)

    Args:
        game_dir: Path to the game's install directory.
        prefix_dir: Optional Wine prefix path.  Auto-detected if None.
        log_fn: Optional callback for progress logging.

    Returns:
        GameReport with ratings, issues, suggestions.
    """

    def _log(msg):
        if log_fn:
            log_fn(msg)
        else:
            print(msg)

    report = GameReport()
    report.generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    game_dir = Path(game_dir)
    if not game_dir.is_dir():
        _log(f"ERROR: Directory not found: {game_dir}\n")
        report.summary = f"Directory not found: {game_dir}"
        return report

    report.game = game_dir.name
    report.session_name = f"multi-source:{game_dir.name}"
    _log(f"Scanning game directory: {game_dir.name}\n")

    # ---- Source 1: Config files ----
    config_files = _scan_game_dir(game_dir, log_fn=log_fn)
    if config_files:
        _log(
            f"  Collected {len(config_files)} files "
            f"({sum(len(v) for v in config_files.values())} chars)\n"
        )
    else:
        _log("  WARNING: No performance-relevant files found.\n")

    # ---- Source 2: Engine detection ----
    _log("\n  Detecting engine...\n")
    engine_info = detect_engine(game_dir, log_fn=log_fn)

    # ---- Source 3: Community data ----
    _log("\n  Fetching community data...\n")
    community = fetch_protondb(report.game, log_fn=log_fn)

    # ---- Source 4: Wine/DXVK prefix ----
    if prefix_dir is None:
        _log("\n  Auto-detecting Wine prefix...\n")
        prefix_dir = find_wine_prefix(report.game)
        if prefix_dir:
            _log(f"  Found prefix: {prefix_dir}\n")
        else:
            _log("  No Wine prefix found (use --prefix to specify)\n")

    prefix_info: Dict[str, Any] = {}
    if prefix_dir and Path(prefix_dir).is_dir():
        _log("\n  Scanning Wine prefix...\n")
        prefix_info = scan_wine_prefix(Path(prefix_dir), log_fn=log_fn)

    # ---- Source 5: Known fix detection ----
    _log("\n  Checking known fixes...\n")
    fixes = detect_known_fixes(
        report.game,
        game_dir,
        prefix_dir,
        engine_info,
        config_files,
        log_fn=log_fn,
    )

    # ---- Source 6: LLM analysis ----
    has_any_data = bool(
        config_files
        or engine_info.get("engine") != "unknown"
        or community
        or prefix_info
        or fixes.get("installed")
        or fixes.get("missing")
    )
    if not has_any_data:
        report.summary = "No data collected for analysis."
        return report

    _log("\n  Building multi-source prompt...\n")
    system_prompt = _build_dir_system_prompt(
        engine_info,
        community,
        prefix_info,
        fixes,
    )
    user_prompt = _build_multi_source_user_prompt(
        report.game,
        config_files,
        engine_info,
        community,
        prefix_info,
        fixes,
    )

    _log("  Sending to LLM for analysis...\n")
    try:
        from conf.config_parser import main_conf as config
        from tools.functions import get_api_key
        from pipeline.generate_bot_script import get_llm_client, call_llm

        provider = config.LLM_PROVIDER
        model = config.LLM_MODEL
        api_key = get_api_key()

        if not api_key:
            _log("  ERROR: No API key configured.\n")
            report.summary = "LLM analysis unavailable -- no API key."
            return report

        client = get_llm_client(provider, api_key)
        _log(f"  Using {provider}/{model}...\n")

        response = call_llm(
            client,
            provider,
            model,
            system_prompt + "\n\n" + user_prompt,
            max_tokens=8192,
        )
        report.raw_analysis = response

        parsed = _parse_llm_response(response)
        if parsed:
            for r in parsed.get("ratings", []):
                report.ratings.append(
                    Rating(
                        category=r.get("category", "Unknown"),
                        score=int(r.get("score", 3)),
                        label=r.get("label", "Fair"),
                        detail=r.get("detail", ""),
                    )
                )
            for iss in parsed.get("issues", []):
                if isinstance(iss, dict):
                    report.issues.append(
                        Issue(
                            description=iss.get("description", ""),
                            file=iss.get("file", ""),
                            line=str(iss.get("line", "")),
                            fix=iss.get("fix", ""),
                        )
                    )
                else:
                    report.issues.append(Issue(description=str(iss)))
            for sug in parsed.get("suggestions", []):
                if isinstance(sug, dict):
                    report.suggestions.append(
                        Suggestion(
                            description=sug.get("description", ""),
                            file=sug.get("file", ""),
                            line=str(sug.get("line", "")),
                            fix=sug.get("fix", ""),
                        )
                    )
                else:
                    report.suggestions.append(Suggestion(description=str(sug)))
            report.summary = parsed.get("summary", "")
            _log("  Analysis complete.\n")
        else:
            _log("  WARNING: Could not parse LLM response as JSON.\n")
            report.summary = response[:500]

    except Exception as e:
        _log(f"  ERROR: LLM call failed: {e}\n")
        report.summary = f"LLM analysis failed: {e}"

    return report


# ======================================================================
# CLI
# ======================================================================


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Game feedback report")
    parser.add_argument(
        "--session", type=str, default=None, help="Path to session directory"
    )
    parser.add_argument(
        "--game-dir",
        type=str,
        default=None,
        help="Path to a game install directory (config/log analysis, no session needed)",
    )
    parser.add_argument(
        "--export", type=str, default=None, help="Export report to text file"
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="Path to Wine prefix directory (auto-detected if not specified)",
    )
    parser.add_argument(
        "--json", action="store_true", help="Output as JSON instead of text"
    )
    args = parser.parse_args()

    # When outputting JSON, send progress logs to stderr so stdout is clean JSON
    log_fn = None
    if args.json:
        log_fn = lambda msg: sys.stderr.write(msg)

    if args.game_dir:
        prefix = Path(args.prefix) if args.prefix else None
        report = generate_report_from_dir(
            game_dir=Path(args.game_dir),
            prefix_dir=prefix,
            log_fn=log_fn,
        )
    else:
        session_dir = Path(args.session) if args.session else None
        report = generate_report(session_dir=session_dir, log_fn=log_fn)

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        text = report.to_text()
        print(text)

    if args.export:
        out = export_report(report, Path(args.export))
        print(f"\nExported to: {out}")


if __name__ == "__main__":
    main()
