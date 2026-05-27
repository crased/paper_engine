"""
Domain 2 — Navigation (menus, world map, UI interaction)

Handles all non-gameplay phases. Memory tells us WHAT screen we're on,
YOLO tells us WHERE the interactive elements are.

Consumes: WorldState (phase from memory + UI/map detections from YOLO)
Produces: ActionCommand (which keys to press)
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from bot_logic.bot_scripts.domains.contracts import (
    ActionCommand,
    DetectionFrame,
    GamePhase,
    WorldState,
    UI_CLASSES,
    MAP_CLASSES,
)

logger = logging.getLogger(__name__)


class NavigationDomain:
    """Handles MAIN_MENU, WORLD_MAP, LEVEL_ENDING, LEVEL_WON, LEVEL_DYING, LOADING."""

    def __init__(self, kbd):
        """
        Args:
            kbd: KeyboardController from cuphead_bot.py — provides keycode constants
                 (MOVE_LEFT, CONFIRM, etc.) and hold/release methods.

        Reference for kbd keycode attributes — see cuphead_bot.py KeyboardController:
            kbd.MOVE_LEFT, kbd.MOVE_RIGHT, kbd.MOVE_UP, kbd.MOVE_DOWN,
            kbd.JUMP, kbd.SHOOT, kbd.LOCK_AIM, kbd.DASH,
            kbd.EX_SUPER, kbd.SWITCH_WEAPON, kbd.CONFIRM, kbd.PAUSE

        Docs:
            ActionCommand dataclass: see contracts.py — keys is Dict[int, bool]
                where int = evdev keycode, True = hold, False = release
            evdev keycodes: https://docs.python.org/3/library/struct.html (raw codes)
                or see cuphead_bot.py lines ~120-135 for the KEY_* constants used
        """
        self._kbd = kbd

    def decide(self, ws: WorldState) -> ActionCommand:
        actions = ActionCommand(keys={
            self._kbd.MOVE_LEFT: False, self._kbd.MOVE_RIGHT: False,
            self._kbd.MOVE_UP: False, self._kbd.MOVE_UP: False,
            self._kbd.MOVE_JUMP: False, self._kbd.MOVE_JUMP: False,
            self._kbd.CONFIRM: False, self._kbd.Confirm: False,
        })    
        if ws.phase == GamePhase.LOADING: return self._handle_loading(ws)
        if ws.phase == GamePhase.MAIN_MENU: return self._handle_main_menu(ws)
        if ws.phase == GamePhase.WORLD_MAP: return self._handle_world_map(ws)
        if ws.phase == GamePhase.LEVEL_ENDING: return self._handle_dyi
        #
        # TODO: dispatch by ws.phase:
        #   if ws.phase == GamePhase.LOADING:     return self._handle_loading(ws)
        #   if ws.phase == GamePhase.MAIN_MENU:   return self._handle_main_menu(ws)
        #   if ws.phase == GamePhase.WORLD_MAP:   return self._handle_world_map(ws)
        #   if ws.phase in (GamePhase.LEVEL_ENDING, GamePhase.LEVEL_WON):
        #                                         return self._handle_level_end(ws)
        #   if ws.phase == GamePhase.LEVEL_DYING: return self._handle_dying(ws)
        #   return self._handle_unknown(ws)
        #
        # Docs:
        #   GamePhase enum values: see contracts.py lines 21-30
        #   WorldState fields: see contracts.py lines 82-106
        #   Reference implementation: cuphead_bot.py decide() lines ~815-841 (non-gameplay branches)
        pass

    def _handle_loading(self, ws: WorldState) -> ActionCommand:
        # TODO: release all keys — nothing to do while loading
        #   return ActionCommand(keys={...all False...})
        #
        # Reference: cuphead_bot.py line ~817-818
        pass

    def _handle_main_menu(self, ws: WorldState) -> ActionCommand:
        # TODO: check ws.detection_frame for UI_CLASSES detections
        #   if ws.detection_frame and ws.detection_frame.age_ms() < 500:
        #       ui_dets = ws.detection_frame.by_group(UI_CLASSES)
        #       ... use bbox positions to decide which button is selected ...
        #
        #   fallback (no detections): blind CONFIRM like current bot
        #       actions.keys[self._kbd.CONFIRM] = True
        #
        # Docs:
        #   DetectionFrame.by_group(): see contracts.py line 71-73
        #   DetectionFrame.age_ms(): see contracts.py line 67-69
        #   Detection.bbox format: (x_center, y_center, w, h) normalized 0-1
        #       see contracts.py line 56
        #   UI_CLASSES = {5,6,7,8,9,10} — MenuStart, OptionMenu, DlcMenu, MenuExit,
        #       GameExit, ControlsMenu (see contracts.py and yolo_dataset/dataset.yaml)
        #   time.time(): https://docs.python.org/3/library/time.html#time.time
        #
        # Reference: cuphead_bot.py line ~823-825
        pass

    def _handle_world_map(self, ws: WorldState) -> ActionCommand:
        # TODO: check ws.detection_frame for MAP_CLASSES detections
        #   if ws.detection_frame and ws.detection_frame.age_ms() < 500:
        #       map_dets = ws.detection_frame.by_group(MAP_CLASSES)
        #       ... find "Misson node" (class 12), navigate toward its bbox center ...
        #
        #   fallback: move right + periodic CONFIRM (current behavior)
        #       actions.keys[self._kbd.MOVE_RIGHT] = True
        #       if int(time.time()) % 3 == 0:
        #           actions.keys[self._kbd.CONFIRM] = True
        #
        # Docs:
        #   MAP_CLASSES = {11,12} — collectable, Misson node
        #       (see contracts.py and yolo_dataset/dataset.yaml)
        #   Detection.bbox: (x_center, y_center, w, h) normalized
        #       x_center < 0.5 means left of screen, > 0.5 means right
        #   time.time(): https://docs.python.org/3/library/time.html#time.time
        #
        # Reference: cuphead_bot.py lines ~827-831
        pass

    def _handle_level_end(self, ws: WorldState) -> ActionCommand:
        # TODO: press CONFIRM to advance past results/victory screen
        #   actions.keys[self._kbd.CONFIRM] = True
        #
        # Reference: cuphead_bot.py lines ~833-835
        pass

    def _handle_dying(self, ws: WorldState) -> ActionCommand:
        # TODO: release all keys, wait for retry prompt detection, then CONFIRM
        #   return ActionCommand(keys={...all False...})
        #
        # Reference: cuphead_bot.py lines ~837-838
        pass

    def _handle_unknown(self, ws: WorldState) -> ActionCommand:
        # TODO: release all keys — safe default when state is unclear
        #   return ActionCommand(keys={...all False...})
        #
        #   optionally: if stuck for >N seconds, try pressing ESC or CONFIRM
        #   (would need a self._unknown_since timestamp to track duration)
        #
        # Docs:
        #   time.monotonic(): https://docs.python.org/3/library/time.html#time.monotonic
        #
        # Reference: cuphead_bot.py lines ~820-821
        pass
