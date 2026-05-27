"""
Shared data contracts between all three domains.

Every domain communicates through these dataclasses. No domain
imports from another domain — they all import from here.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, Optional


# ═══════════════════════════════════════════════════════════════════════════
# GAME PHASE (moved from cuphead_bot.py — the router key)
# ═══════════════════════════════════════════════════════════════════════════


class GamePhase(Enum):
    UNKNOWN = auto()
    LOADING = auto()
    MAIN_MENU = auto()
    WORLD_MAP = auto()
    LEVEL_PLAYING = auto()
    LEVEL_DYING = auto()
    LEVEL_ENDING = auto()
    LEVEL_WON = auto()
    PAUSED = auto()


# ═══════════════════════════════════════════════════════════════════════════
# YOLO CLASS GROUPS — filter detections per domain
# ═══════════════════════════════════════════════════════════════════════════

# Matches yolo_dataset/dataset.yaml (13 classes)
# See: yolo_dataset/dataset.yaml for canonical ID→name mapping
GAMEPLAY_CLASSES = {0, 1, 2, 3}  # player, enemy, projectile, platform
UI_CLASSES = {
    4,
    5,
    6,
    7,
    8,
    9,
}  # MenuStart, OptionMenu, DlcMenu, MenuExit, GameExit, ControlsMenu
MAP_CLASSES = {10, 11}  # collectable, Misson node


# ═══════════════════════════════════════════════════════════════════════════
# DETECTION — output of Domain 1 (YOLO thread)
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class Detection:
    """Single bounding box from YOLO."""

    class_id: int
    class_name: str
    bbox: tuple[float, float, float, float]  # x_center, y_center, w, h (normalized 0-1)
    confidence: float


@dataclass
class DetectionFrame:
    """One frame's worth of YOLO output. Written by detection thread, read by others."""

    detections: list[Detection] = field(default_factory=list)
    timestamp: float = 0.0  # time.monotonic() when inference completed

    def age_ms(self) -> float:
        """How stale this frame is, in milliseconds."""
        return (time.monotonic() - self.timestamp) * 1000.0

    def by_group(self, class_ids: set[int]) -> list[Detection]:
        """Filter detections to a specific class group."""
        return [d for d in self.detections if d.class_id in class_ids]


# ═══════════════════════════════════════════════════════════════════════════
# WORLD STATE — fused memory + detection snapshot (input to all domains)
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class WorldState:
    """Everything a domain needs to make a decision. Built each tick by the router."""

    # Memory (authoritative, <1ms) — filled by fuse()
    phase: GamePhase = GamePhase.UNKNOWN
    hp: int = 0
    hp_max: int = 0
    super_meter: float = 0.0
    super_meter_max: float = 0.0
    deaths: int = 0
    level_time: float = 0.0
    level_ending: bool = False
    level_won: bool = False
    level_mode: int = -1
    in_game: bool = False
    scene_name: str = ""
    is_loading: bool = False
    num_times_hit: int = 0
    num_parries: int = 0
    just_got_hit: bool = False
    memory_ok: bool = False
    raw_memory: Dict[str, Any] = field(default_factory=dict)

    # Vision (latest available from detection thread)
    detection_frame: Optional[DetectionFrame] = None


# ═══════════════════════════════════════════════════════════════════════════
# ACTION COMMAND — output of Domain 2 or Domain 3
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class ActionCommand:
    """What keys to hold/release. Passed from a domain back to the router for execution."""

    keys: Dict[int, bool] = field(
        default_factory=dict
    )  # evdev keycode → True=hold / False=release


# ═══════════════════════════════════════════════════════════════════════════
# PHASE CLASSIFICATION — pure function, lives here because both router
# and domains may need it
# ═══════════════════════════════════════════════════════════════════════════


def classify_phase(state: Dict[str, Any]) -> GamePhase:
    """Determine game phase from memory state dict."""
    scene = state.get("scene_name", "")
    is_loading = state.get("is_loading", False)
    in_game = state.get("in_game", False)
    hp = state.get("hp", 0)
    level_ending = state.get("level_ending", False)
    level_won = state.get("level_won", False)

    if is_loading:
        return GamePhase.LOADING

    if not scene:
        return GamePhase.UNKNOWN

    scene_lower = scene.lower()

    if "title" in scene_lower or "slot_select" in scene_lower:
        return GamePhase.MAIN_MENU

    if "map_world" in scene_lower:
        return GamePhase.WORLD_MAP

    if "level" in scene_lower:
        if level_won:
            return GamePhase.LEVEL_WON
        if level_ending:
            return GamePhase.LEVEL_ENDING
        if hp <= 0:
            return GamePhase.LEVEL_DYING
        return GamePhase.LEVEL_PLAYING

    if in_game:
        return GamePhase.WORLD_MAP

    return GamePhase.UNKNOWN


# ═══════════════════════════════════════════════════════════════════════════
# FUSE — Memory state → WorldState
# ═══════════════════════════════════════════════════════════════════════════


def fuse(
    memory_state: Dict[str, Any], prev_ws: Optional[WorldState] = None
) -> WorldState:
    """Build WorldState from memory.  Detects hit events by comparing to prev."""
    ws = WorldState()

    if not memory_state:
        return ws

    ws.memory_ok = True
    ws.raw_memory = memory_state
    ws.phase = classify_phase(memory_state)
    ws.hp = memory_state.get("hp", 0)
    ws.hp_max = memory_state.get("hp_max", 0)
    ws.super_meter = memory_state.get("super_meter", 0.0)
    ws.super_meter_max = memory_state.get("super_meter_max", 0.0)
    ws.deaths = memory_state.get("deaths", 0)
    ws.level_time = memory_state.get("level_time", 0.0)
    ws.level_ending = memory_state.get("level_ending", False)
    ws.level_won = memory_state.get("level_won", False)
    ws.level_mode = memory_state.get("level_mode", -1)
    ws.in_game = memory_state.get("in_game", False)
    ws.scene_name = memory_state.get("scene_name", "")
    ws.is_loading = memory_state.get("is_loading", False)
    ws.num_times_hit = memory_state.get("num_times_hit", 0)
    ws.num_parries = memory_state.get("num_parries", 0)

    # Detect hit events by comparing to previous state
    if prev_ws is not None and prev_ws.memory_ok:
        hp_dropped = ws.hp < prev_ws.hp and prev_ws.hp > 0
        hit_count_rose = ws.num_times_hit > prev_ws.num_times_hit
        ws.just_got_hit = hp_dropped or hit_count_rose

    return ws
