"""
Domain 3 — Strategy / Tactical (combat decisions)

Handles LEVEL_PLAYING only. Memory provides HP, super, hit detection.
YOLO provides enemy/projectile/player positions for spatial awareness.

Priority layers (highest wins):
    Layer 0: SURVIVE  — HP=0 release all; super if critical
    Layer 1: EVADE    — Recently hit -> jump + dash + dodge away from threat
    Layer 2: ENGAGE   — Aim toward nearest/largest enemy (YOLO-informed)
    Layer 3: ADVANCE  — Default: run right + shoot

Consumes: WorldState (memory state + gameplay detections from YOLO)
Produces: ActionCommand + updated evasion tick counter
"""

from __future__ import annotations

import logging
from typing import Optional

from bot_logic.bot_scripts.domains.contracts import (
    ActionCommand,
    Detection,
    DetectionFrame,
    GamePhase,
    WorldState,
    GAMEPLAY_CLASSES,
)

logger = logging.getLogger(__name__)

# How many ticks to stay in evasive mode after getting hit
_EVASION_DURATION_TICKS = 8

# Detection frame staleness threshold (ms)
_MAX_DETECTION_AGE_MS = 500.0


class StrategyDomain:
    """Priority-layered combat decision engine for LEVEL_PLAYING."""

    def __init__(self, kbd):
        """
        Args:
            kbd: KeyboardController from cuphead_bot.py — provides keycode constants
                 and hold/release methods.
        """
        self._kbd = kbd
        self._evasion_ticks_left: int = 0

    def _default_actions(self) -> ActionCommand:
        """All keys released — safe baseline for every layer."""
        return ActionCommand(
            keys={
                self._kbd.MOVE_LEFT: False,
                self._kbd.MOVE_RIGHT: False,
                self._kbd.MOVE_UP: False,
                self._kbd.MOVE_DOWN: False,
                self._kbd.JUMP: False,
                self._kbd.SHOOT: False,
                self._kbd.LOCK_AIM: False,
                self._kbd.DASH: False,
                self._kbd.EX_SUPER: False,
                self._kbd.SWITCH_WEAPON: False,
                self._kbd.CONFIRM: False,
            }
        )

    # --- public API ---

    def decide(self, ws: WorldState) -> ActionCommand:
        """Priority-layered decision: first layer to activate wins."""
        result = self._survive(ws)
        if result is not None:
            return result
        result = self._evade(ws)
        if result is not None:
            return result
        result = self._engage(ws)
        if result is not None:
            return result
        return self._advance(ws)

    # --- priority layers ---

    def _survive(self, ws: WorldState) -> Optional[ActionCommand]:
        """Layer 0: dead -> release all. Critical HP + full super -> use super."""
        if ws.hp <= 0:
            self._evasion_ticks_left = 0
            return self._default_actions()

        if (
            ws.super_meter_max > 0
            and ws.super_meter >= ws.super_meter_max
            and ws.hp == 1
        ):
            actions = self._default_actions()
            actions.keys[self._kbd.EX_SUPER] = True
            return actions

        return None

    def _evade(self, ws: WorldState) -> Optional[ActionCommand]:
        """Layer 1: dodge on hit. Uses YOLO threats for direction, falls back to alternating."""
        if ws.just_got_hit:
            self._evasion_ticks_left = _EVASION_DURATION_TICKS

        if self._evasion_ticks_left <= 0:
            return None

        self._evasion_ticks_left -= 1
        actions = self._default_actions()

        # Pick dodge direction — YOLO-informed if available
        detections = self._get_gameplay_detections(ws)
        threat = self._find_nearest_threat(detections)
        if threat is not None:
            # Dodge away from threat: threat on right -> move left, and vice versa
            if threat.bbox[0] > 0.5:
                actions.keys[self._kbd.MOVE_LEFT] = True
            else:
                actions.keys[self._kbd.MOVE_RIGHT] = True
        else:
            # No YOLO data — alternate direction by tick parity
            if self._evasion_ticks_left % 2 == 0:
                actions.keys[self._kbd.MOVE_LEFT] = True
            else:
                actions.keys[self._kbd.MOVE_RIGHT] = True

        actions.keys[self._kbd.JUMP] = True
        actions.keys[self._kbd.DASH] = True
        actions.keys[self._kbd.SHOOT] = True
        return actions

    def _engage(self, ws: WorldState) -> Optional[ActionCommand]:
        """Layer 2: aim toward largest enemy if YOLO detections are available."""
        detections = self._get_gameplay_detections(ws)
        enemies = [d for d in detections if d.class_id == 1]  # class 1 = enemy
        if not enemies:
            return None

        # Target the largest enemy (bbox area as proxy for closest/most important)
        target = max(enemies, key=lambda d: d.bbox[2] * d.bbox[3])
        actions = self._default_actions()

        if target.bbox[0] < 0.4:
            actions.keys[self._kbd.MOVE_LEFT] = True
        elif target.bbox[0] > 0.6:
            actions.keys[self._kbd.MOVE_RIGHT] = True

        actions.keys[self._kbd.SHOOT] = True
        return actions

    def _advance(self, ws: WorldState) -> ActionCommand:
        """Layer 3: default behavior — run right + shoot."""
        actions = self._default_actions()
        actions.keys[self._kbd.MOVE_RIGHT] = True
        actions.keys[self._kbd.SHOOT] = True
        return actions

    # --- helpers ---

    def _get_gameplay_detections(self, ws: WorldState) -> list[Detection]:
        """Return fresh gameplay detections, or empty list if stale/missing."""
        if (
            ws.detection_frame is not None
            and ws.detection_frame.age_ms() < _MAX_DETECTION_AGE_MS
        ):
            return ws.detection_frame.by_group(GAMEPLAY_CLASSES)
        return []

    def _find_nearest_threat(self, detections: list[Detection]) -> Optional[Detection]:
        """Find the most immediate threat (enemy or projectile) by bbox area."""
        threats = [d for d in detections if d.class_id in {1, 2}]  # enemy, projectile
        if not threats:
            return None

        # Largest bbox area = closest to camera / most dangerous
        return max(threats, key=lambda d: d.bbox[2] * d.bbox[3])
