"""
Cuphead-specific memory reader — high-level API for game state.

Ties together the parent-child process launch, MonoExternal metadata walker,
and game-specific pointer chains to read Cuphead's game state via memory.

Usage:
    from tools.cuphead_memory import CupheadMemory

    mem = CupheadMemory()
    mem.launch()        # Launch game as child process
    mem.attach()        # Find Mono, resolve classes

    # Read game state
    state = mem.read_state()
    print(f"HP: {state.hp}/{state.hp_max}")
    print(f"Level time: {state.level_time:.1f}s")

    # Or read individual values
    hp = mem.read_player_hp()
    level_time = mem.read_level_time()

    # Cleanup
    mem.close()
"""

from __future__ import annotations

import logging
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from tools.memory_reader import ProcessMemory, find_wine_pid
from tools.mono_external import (
    MonoExternal,
    MonoClassInfo,
    MonoArrayOffsets,
    MonoTypeEnum,
    CSharpDictOffsets,
)

logger = logging.getLogger(__name__)


@dataclass
class CupheadState:
    """Snapshot of Cuphead's game state read from memory."""

    # Player stats
    hp: int = 0
    hp_max: int = 0
    super_meter: float = 0.0
    super_meter_max: float = 0.0
    deaths: int = 0

    # Level info
    level_time: float = 0.0
    level_ending: bool = False
    level_won: bool = False
    level_started: bool = False
    level_mode: int = -1  # 0=Easy, 1=Normal, 2=Hard

    # Game state
    in_game: bool = False
    scene_name: str = ""
    is_loading: bool = False

    # Meta
    timestamp: float = 0.0
    read_ok: bool = False

    def summary(self) -> str:
        if not self.read_ok:
            return "No data (read failed)"
        parts = []
        if self.in_game:
            parts.append(f"HP:{self.hp}/{self.hp_max}")
            parts.append(f"Super:{self.super_meter:.0f}/{self.super_meter_max:.0f}")
            parts.append(f"Deaths:{self.deaths}")
            parts.append(f"Time:{self.level_time:.1f}s")
            if self.level_ending:
                parts.append("ENDING")
            if self.level_won:
                parts.append("WON")
            mode_names = {0: "Easy", 1: "Normal", 2: "Hard"}
            parts.append(f"Mode:{mode_names.get(self.level_mode, '?')}")
        else:
            parts.append("Not in game")
        if self.scene_name:
            parts.append(f"Scene:{self.scene_name}")
        if self.is_loading:
            parts.append("LOADING")
        return " | ".join(parts)


# ---------------------------------------------------------------------------
# Cuphead class/field specs (what to look up in Mono metadata)
# ---------------------------------------------------------------------------

# Classes we need from Assembly-CSharp
_GAME_CLASSES = [
    ("", "PlayerStatsManager"),
    ("", "PlayerManager"),
    ("", "Level"),
    ("", "PlayerData"),
    ("", "SceneLoader"),
    ("", "AbstractPlayerController"),
]


class CupheadMemory:
    """High-level Cuphead game state reader via process memory.

    This class manages the full pipeline:
    1. Launch Cuphead.exe via Wine as a child process
    2. Find mono.dll in the process's memory map
    3. Walk Mono metadata to find game classes and field offsets
    4. Read game state by chasing pointer chains

    All memory reads use process_vm_readv (parent -> child, no ptrace needed).
    """

    def __init__(self):
        self._proc: Optional[subprocess.Popen] = None
        self._pm: Optional[ProcessMemory] = None
        self._mono: Optional[MonoExternal] = None
        self._pid: Optional[int] = None

        # Resolved classes
        self._classes: dict[str, Optional[MonoClassInfo]] = {}

        # Cached pointer addresses (refreshed each read)
        self._level_current: int = 0
        self._player_data: int = 0

    @property
    def pid(self) -> Optional[int]:
        return self._pid

    @property
    def process_memory(self) -> Optional[ProcessMemory]:
        return self._pm

    @property
    def mono(self) -> Optional[MonoExternal]:
        return self._mono

    def launch(
        self,
        exe_path: Optional[str | Path] = None,
        wine_bin: str = "wine",
        bridge: bool = True,
        wait_for_mono: bool = True,
        timeout: float = 60.0,
    ) -> bool:
        """Launch Cuphead via Wine as a child process.

        Args:
            exe_path: Path to Cuphead.exe. Auto-detected if None.
            wine_bin: Wine binary name.
            bridge: If True, use LD_PRELOAD with mono_bridge.so
                    (enables prctl for external process access too).
            wait_for_mono: Wait for mono.dll to appear in process maps.
            timeout: Max seconds to wait.

        Returns:
            True if the game launched and mono.dll was found.
        """
        if exe_path is None:
            project_root = Path(__file__).resolve().parent.parent
            exe_path = project_root / "game" / "CupHead" / "Cuphead.exe"

        exe_path = Path(exe_path)
        if not exe_path.exists():
            logger.error("Game executable not found: %s", exe_path)
            return False

        game_dir = exe_path.parent
        env = os.environ.copy()

        if bridge:
            bridge_so = Path(__file__).resolve().parent / "mono_bridge.so"
            if bridge_so.exists():
                existing = env.get("LD_PRELOAD", "")
                env["LD_PRELOAD"] = (
                    f"{bridge_so}:{existing}" if existing else str(bridge_so)
                )
                logger.info("Using LD_PRELOAD: %s", env["LD_PRELOAD"])
            else:
                logger.warning("mono_bridge.so not found at %s, skipping", bridge_so)

        # Suppress Wine debug noise (fixme, warn, err, etc.)
        env["WINEDEBUG"] = "-all"

        logger.info("Launching %s via %s ...", exe_path.name, wine_bin)
        self._proc = subprocess.Popen(
            [wine_bin, str(exe_path)],
            cwd=str(game_dir),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        logger.info("Wine launcher PID: %d", self._proc.pid)

        # Wait for the actual game process
        start = time.monotonic()
        self._pid = None

        while time.monotonic() - start < timeout:
            pid = find_wine_pid("Cuphead.exe")
            if pid is not None:
                self._pid = pid
                self._pm = ProcessMemory(pid)
                logger.info(
                    "Game process: PID %d (%.1fs)", pid, time.monotonic() - start
                )
                break
            time.sleep(0.5)

        if self._pid is None:
            logger.error("Game process not found after %.0fs", timeout)
            return False

        if not wait_for_mono:
            return True

        # Wait for mono.dll to be loaded
        while time.monotonic() - start < timeout:
            self._pm.refresh_regions()
            mono_base = self._pm.get_module_base("mono.dll")
            if mono_base is not None:
                logger.info(
                    "mono.dll loaded at 0x%x (%.1fs)",
                    mono_base,
                    time.monotonic() - start,
                )
                return True
            time.sleep(0.5)

        logger.error("mono.dll not found after %.0fs", timeout)
        return False

    def attach_to_running(self, pid: Optional[int] = None) -> bool:
        """Attach to an already-running Cuphead process.

        This works if:
        - The game was launched as a child of this process, OR
        - prctl(PR_SET_PTRACER, ANY) was called (via mono_bridge.so), OR
        - ptrace_scope=0

        Args:
            pid: PID of the game process. Auto-detected if None.

        Returns:
            True if attached and mono.dll found.
        """
        if pid is None:
            pid = find_wine_pid("Cuphead.exe")
        if pid is None:
            logger.error("Cuphead.exe not found")
            return False

        self._pid = pid
        self._pm = ProcessMemory(pid)

        # Verify we can read
        if not self._pm.is_alive():
            logger.error("Process %d not alive", pid)
            return False

        mono_base = self._pm.get_module_base("mono.dll")
        if mono_base is None:
            logger.error("mono.dll not found for PID %d", pid)
            return False

        logger.info("Attached to PID %d, mono.dll at 0x%x", pid, mono_base)
        return True

    def attach(self) -> bool:
        """Initialize MonoExternal and resolve game classes.

        Call this after launch() or attach_to_running().

        Returns:
            True if Mono metadata was successfully read.
        """
        if self._pm is None:
            logger.error("Not connected to game. Call launch() first.")
            return False

        mono_base = self._pm.get_module_base("mono.dll")
        if mono_base is None:
            logger.error("mono.dll not found")
            return False

        self._mono = MonoExternal(self._pm, mono_base)
        if not self._mono.attach():
            logger.error("Failed to attach to Mono runtime")
            return False

        logger.info("Mono attached. Assemblies: %s", self._mono.list_assemblies())

        # Resolve game classes
        logger.info("Resolving game classes...")
        self._classes = self._mono.find_classes_batch("Assembly-CSharp", _GAME_CLASSES)

        found = [k for k, v in self._classes.items() if v is not None]
        missing = [k for k, v in self._classes.items() if v is None]
        logger.info("Found: %s", found)
        if missing:
            logger.warning("Missing: %s", missing)

        return bool(found)

    def _get_class(self, name: str) -> Optional[MonoClassInfo]:
        """Get a resolved class by name."""
        return self._classes.get(name)

    # -- Pointer chain helpers --

    def _read_player_stats_manager(self) -> Optional[int]:
        """Chase the pointer chain to find the PlayerStatsManager instance.

        Chain: PlayerManager.players (static Dictionary<PlayerId, AbstractPlayerController>)
               -> valueSlots[0] (AbstractPlayerController)
               -> _stats (PlayerStatsManager field in AbstractPlayerController)

        The 'players' field is a Dictionary<PlayerId, AbstractPlayerController>,
        NOT a List<T> as originally assumed. The Mono Dictionary layout is:
            +0x28: V[] valueSlots (MonoArray of values)
            +0x38: int count

        Returns the address of the PlayerStatsManager object, or None.
        """
        pm_class = self._get_class("PlayerManager")
        if pm_class is None or self._mono is None or self._pm is None:
            return None

        # PlayerManager.players is a static field (Dictionary<PlayerId, APC>)
        players_ptr = self._mono.read_static_field_ptr(pm_class, "players")
        if not players_ptr:
            logger.debug("PlayerManager.players is null")
            return None

        # Read Dictionary.count at +0x38
        dict_count = self._pm.read_int32(players_ptr + CSharpDictOffsets.COUNT)
        if dict_count is None or dict_count < 1:
            logger.debug(
                "players dict empty (count=%s, dict=0x%x)", dict_count, players_ptr
            )
            return None

        # Read Dictionary.valueSlots (MonoArray*) at +0x28
        value_slots = self._pm.read_pointer(players_ptr + CSharpDictOffsets.VALUE_SLOTS)
        if not value_slots:
            logger.debug("players dict valueSlots is null (dict=0x%x)", players_ptr)
            return None

        # Read Dictionary.touchedSlots at +0x30 (how many slots have been used)
        touched = self._pm.read_int32(players_ptr + CSharpDictOffsets.TOUCHED_SLOTS)
        if touched is None or touched < 1:
            logger.debug(
                "players dict touchedSlots=%s (dict=0x%x)", touched, players_ptr
            )
            return None

        # Iterate through valueSlots to find the first non-null player object.
        # In a Mono Dictionary, entries are at slot indices 0..touchedSlots-1,
        # but some may be null/removed. We want the first valid player.
        for i in range(min(touched, 4)):
            player = self._pm.read_pointer(value_slots + MonoArrayOffsets.DATA + i * 8)
            if player:
                logger.debug(
                    "players dict: count=%d, touched=%d, slot[%d]=0x%x",
                    dict_count,
                    touched,
                    i,
                    player,
                )
                return self._chase_player0_to_stats(player)

        logger.debug(
            "players dict: all valueSlots null (count=%d, touched=%d, arr=0x%x)",
            dict_count,
            touched,
            value_slots,
        )
        return None

    def _chase_player0_to_stats(self, player0: int) -> Optional[int]:
        """Given a player0 (AbstractPlayerController) pointer, chase to _stats.

        Returns the address of the PlayerStatsManager object, or None.
        """
        if self._pm is None or self._mono is None:
            return None

        logger.debug("player0: 0x%x", player0)

        # Identify what player0 actually is
        if logger.isEnabledFor(logging.DEBUG):
            vtable_ptr = self._pm.read_pointer(player0)
            if vtable_ptr:
                klass_ptr = self._pm.read_pointer(vtable_ptr)
                if klass_ptr:
                    name_ptr = self._pm.read_pointer(klass_ptr + 0x50)
                    obj_name = self._mono._read_cstring(name_ptr) if name_ptr else None
                    logger.debug(
                        "player0 type: %s (vtable=0x%x, klass=0x%x)",
                        obj_name or "?",
                        vtable_ptr,
                        klass_ptr,
                    )

        apc_class = self._get_class("AbstractPlayerController")
        if apc_class is None:
            return None

        stats_field = apc_class.get_field("_stats")
        if stats_field is None:
            for f in apc_class.fields:
                if "stats" in f.name.lower():
                    stats_field = f
                    break
        if stats_field is None:
            logger.warning("_stats field not found in AbstractPlayerController")
            logger.debug("Fields: %s", [f.name for f in apc_class.fields])
            return None

        stats_ptr = self._pm.read_pointer(player0 + stats_field.offset)
        if not stats_ptr:
            logger.debug(
                "player0._stats is null (player0=0x%x, offset=0x%x)",
                player0,
                stats_field.offset,
            )
            return None

        logger.debug(
            "PSM chain: player0=0x%x -> _stats=0x%x (offset=0x%x)",
            player0,
            stats_ptr,
            stats_field.offset,
        )
        return stats_ptr

    def _read_level_current(self) -> Optional[int]:
        """Read Level.Current (static backing field).

        Returns address of the Level instance, or None.
        """
        level_class = self._get_class("Level")
        if level_class is None or self._mono is None:
            return None

        # <Current>k__BackingField is a static field
        current = self._mono.read_static_field_ptr(
            level_class, "<Current>k__BackingField"
        )
        return current if current else None

    # -- Public game state readers --

    def read_player_hp(self) -> Optional[int]:
        """Read current player HP."""
        stats = self._read_player_stats_manager()
        if stats is None:
            return None

        psm = self._get_class("PlayerStatsManager")
        if psm is None:
            return None

        return self._mono.read_instance_field_int(stats, psm, "<Health>k__BackingField")

    def read_player_hp_max(self) -> Optional[int]:
        """Read max player HP."""
        stats = self._read_player_stats_manager()
        if stats is None:
            return None

        psm = self._get_class("PlayerStatsManager")
        if psm is None:
            return None

        return self._mono.read_instance_field_int(
            stats, psm, "<HealthMax>k__BackingField"
        )

    def read_super_meter(self) -> Optional[float]:
        """Read current super meter value."""
        stats = self._read_player_stats_manager()
        if stats is None:
            return None

        psm = self._get_class("PlayerStatsManager")
        if psm is None:
            return None

        return self._mono.read_instance_field_float(
            stats, psm, "<SuperMeter>k__BackingField"
        )

    def read_super_meter_max(self) -> Optional[float]:
        """Read max super meter value."""
        stats = self._read_player_stats_manager()
        if stats is None:
            return None

        psm = self._get_class("PlayerStatsManager")
        if psm is None:
            return None

        return self._mono.read_instance_field_float(
            stats, psm, "<SuperMeterMax>k__BackingField"
        )

    def read_deaths(self) -> Optional[int]:
        """Read player death count."""
        stats = self._read_player_stats_manager()
        if stats is None:
            return None

        psm = self._get_class("PlayerStatsManager")
        if psm is None:
            return None

        return self._mono.read_instance_field_int(stats, psm, "<Deaths>k__BackingField")

    def read_level_time(self) -> Optional[float]:
        """Read level timer (seconds)."""
        level = self._read_level_current()
        if level is None:
            return None

        level_class = self._get_class("Level")
        if level_class is None:
            return None

        # Find the LevelTime field (might be named differently)
        field = level_class.get_field("LevelTime")
        if field is None:
            # Try backing field pattern
            field = level_class.get_field("<LevelTime>k__BackingField")
        if field is None:
            # Search for time-related float fields
            for f in level_class.fields:
                if "time" in f.name.lower() and f.type_code == 0x0C:
                    field = f
                    break
        if field is None:
            return None

        return self._pm.read_float(level + field.offset)

    def read_level_ending(self) -> Optional[bool]:
        """Read if level is ending."""
        level = self._read_level_current()
        if level is None:
            return None

        level_class = self._get_class("Level")
        if level_class is None:
            return None

        field = level_class.get_field("Ending")
        if field is None:
            field = level_class.get_field("<Ending>k__BackingField")
        if field is None:
            return None

        val = self._pm.read_uint8(level + field.offset)
        return bool(val) if val is not None else None

    def read_level_won(self) -> Optional[bool]:
        """Read if level was won.

        Won is a STATIC field on Level (not instance), so we read
        from the static data block, not from Level.Current instance.
        """
        level_class = self._get_class("Level")
        if level_class is None or self._mono is None:
            return None

        return self._mono.read_static_field_bool(level_class, "<Won>k__BackingField")

    def read_level_mode(self) -> Optional[int]:
        """Read level mode (0=Easy, 1=Normal, 2=Hard).

        CurrentMode is a STATIC field on Level (not instance), so we read
        from the static data block, not from Level.Current instance.
        """
        level_class = self._get_class("Level")
        if level_class is None or self._mono is None:
            return None

        # <CurrentMode>k__BackingField is a static field (valuetype enum)
        # Read as int32 from static data block
        return self._mono.read_static_field_int(
            level_class, "<CurrentMode>k__BackingField"
        )

    def read_in_game(self) -> Optional[bool]:
        """Read if a save file is loaded (PlayerData.inGame)."""
        pd_class = self._get_class("PlayerData")
        if pd_class is None or self._mono is None:
            return None

        return self._mono.read_static_field_bool(pd_class, "inGame")

    def read_scene_name(self) -> Optional[str]:
        """Read current scene name from SceneLoader."""
        sl_class = self._get_class("SceneLoader")
        if sl_class is None or self._mono is None:
            return None

        # SceneName backing field (static string at +0x20)
        str_ptr = self._mono.read_static_field_ptr(
            sl_class, "<SceneName>k__BackingField"
        )
        if str_ptr is None or str_ptr == 0:
            return None

        return self._mono._read_mono_string(str_ptr)

    def read_is_loading(self) -> Optional[bool]:
        """Read if the game is loading a scene."""
        sl_class = self._get_class("SceneLoader")
        if sl_class is None or self._mono is None:
            return None

        # Try currentlyLoading (static bool) first
        loading = self._mono.read_static_field_bool(sl_class, "currentlyLoading")
        if loading is not None:
            return loading

        # Fallback: doneLoadingSceneAsync is an instance field, not useful here
        return None

    # -- Aggregate state read --

    def read_state(self) -> CupheadState:
        """Read all game state in one call.

        Returns a CupheadState dataclass. Individual fields may be None/default
        if the relevant game object doesn't exist (e.g., not in a level).
        """
        state = CupheadState(timestamp=time.monotonic())

        def _v(val, default):
            """Return val if not None, else default (avoids truthiness bugs)."""
            return val if val is not None else default

        try:
            state.in_game = _v(self.read_in_game(), False)
            state.scene_name = _v(self.read_scene_name(), "")
            state.is_loading = _v(self.read_is_loading(), False)

            # Player stats (only available when in a level)
            state.hp = _v(self.read_player_hp(), 0)
            state.hp_max = _v(self.read_player_hp_max(), 0)
            state.super_meter = _v(self.read_super_meter(), 0.0)
            state.super_meter_max = _v(self.read_super_meter_max(), 0.0)
            state.deaths = _v(self.read_deaths(), 0)

            # Level info
            state.level_time = _v(self.read_level_time(), 0.0)
            state.level_ending = _v(self.read_level_ending(), False)
            state.level_won = _v(self.read_level_won(), False)
            state.level_mode = _v(self.read_level_mode(), -1)

            state.read_ok = True
        except (OSError, ValueError, OverflowError) as e:
            logger.debug("Expected error reading game state: %s", e)
            state.read_ok = False
        except Exception as e:
            logger.error("Unexpected error reading game state: %s", e)
            state.read_ok = False

        return state

    # -- Lifecycle --

    def close(self):
        """Clean up."""
        if self._proc is not None:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.debug("Game didn't terminate in time, killing")
                try:
                    self._proc.kill()
                    self._proc.wait(timeout=3)
                except Exception as e:
                    logger.debug("kill() failed: %s", e)
            except Exception as e:
                logger.debug("terminate() failed: %s", e)
            self._proc = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        # Don't kill the game process on GC — it's annoying during development
        pass


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    log_level = logging.DEBUG if "--debug" in sys.argv else logging.INFO
    logging.basicConfig(level=log_level, format="%(levelname)s: %(message)s")

    mem = CupheadMemory()

    if "--launch" in sys.argv:
        print("Launching Cuphead...")
        if not mem.launch():
            print("Launch failed")
            sys.exit(1)
        print("Game launched, waiting a few seconds for Mono to initialize...")
        time.sleep(5)
    else:
        print("Attaching to running Cuphead...")
        if not mem.attach_to_running():
            print("Cuphead not running. Use --launch to start it.")
            sys.exit(1)

    print("Initializing Mono reader...")
    if not mem.attach():
        print("Failed to read Mono metadata")
        sys.exit(1)

    print("\nClasses found:")
    for name, klass in mem._classes.items():
        if klass:
            static_fields = [f for f in klass.fields if f.is_static]
            instance_fields = [f for f in klass.fields if not f.is_static]
            print(
                f"  {name}: {len(klass.fields)} fields "
                f"({len(instance_fields)} instance, {len(static_fields)} static), "
                f"vtable=0x{klass.vtable_addr:x}, "
                f"static=0x{klass.static_data_addr:x}"
            )
            for f in klass.fields:
                flags = []
                if f.is_static:
                    flags.append("STATIC")
                if f.is_literal:
                    flags.append("CONST")
                flag_str = f" [{','.join(flags)}]" if flags else ""
                print(
                    f"    +0x{f.offset:04x}: {f.name} ({f.type_name}) attrs=0x{f.attrs:04x}{flag_str}"
                )

    if "--diag" in sys.argv:
        print("\n=== DIAGNOSTICS: PlayerManager static data ===")
        pm_class = mem._get_class("PlayerManager")
        if pm_class and pm_class.static_data_addr and mem._pm:
            static_base = pm_class.static_data_addr
            print(f"Static data base: 0x{static_base:x}")

            # Dump 0x90 bytes of the static data block
            raw = mem._pm.read_bytes(static_base, 0x90)
            if raw:
                print("Raw static data hex dump:")
                for off in range(0, len(raw), 16):
                    chunk = raw[off : off + 16]
                    hex_part = " ".join(f"{b:02x}" for b in chunk)
                    ascii_part = "".join(
                        chr(b) if 32 <= b < 127 else "." for b in chunk
                    )
                    print(f"  +0x{off:04x}: {hex_part:<48s} {ascii_part}")

            # For each non-const static field, read and show the value
            print("\nStatic field values:")
            for f in pm_class.fields:
                if not f.is_static:
                    continue
                if f.is_literal:
                    print(
                        f"  +0x{f.offset:04x}: {f.name} = <const/literal, no storage>"
                    )
                    continue

                addr = static_base + f.offset
                if f.type_code in (0x12, 0x1D, 0x15, 0x14, 0x1C):
                    # pointer type
                    val = mem._pm.read_pointer(addr)
                    print(
                        f"  +0x{f.offset:04x}: {f.name} ({f.type_name}) = 0x{val or 0:x}"
                    )

                    # If this is the players field, dig deeper
                    if f.name == "players" and val and mem._mono:
                        print(
                            f"\n    === Deep inspection of 'players' at 0x{val:x} ==="
                        )
                        obj_raw = mem._pm.read_bytes(val, 0x40)
                        if obj_raw:
                            print("    Raw bytes:")
                            for roff in range(0, len(obj_raw), 16):
                                chunk = obj_raw[roff : roff + 16]
                                hx = " ".join(f"{b:02x}" for b in chunk)
                                print(f"      +0x{roff:02x}: {hx}")

                        # Read vtable -> klass -> name to identify type
                        vtable_ptr = mem._pm.read_pointer(val)
                        if vtable_ptr:
                            klass_ptr = mem._pm.read_pointer(vtable_ptr)
                            if klass_ptr:
                                name_ptr = mem._pm.read_pointer(klass_ptr + 0x50)
                                obj_name = (
                                    mem._mono._read_cstring(name_ptr)
                                    if name_ptr
                                    else None
                                )
                                ns_ptr = mem._pm.read_pointer(klass_ptr + 0x58)
                                obj_ns = (
                                    mem._mono._read_cstring(ns_ptr) if ns_ptr else None
                                )
                                parent_ptr = mem._pm.read_pointer(klass_ptr + 0x30)
                                parent_name = None
                                if parent_ptr:
                                    pn_ptr = mem._pm.read_pointer(parent_ptr + 0x50)
                                    parent_name = (
                                        mem._mono._read_cstring(pn_ptr)
                                        if pn_ptr
                                        else None
                                    )
                                print(
                                    f"    Object type: {obj_ns or ''}.{obj_name or '?'}"
                                    f" (parent: {parent_name or '?'})"
                                    f" vtable=0x{vtable_ptr:x}, klass=0x{klass_ptr:x}"
                                )

                                # Read field_count and fields of the actual object class
                                fc = mem._pm.read_uint32(klass_ptr + 0x9C)
                                print(f"    Field count: {fc}")
                                fields_ptr = mem._pm.read_pointer(klass_ptr + 0xB0)
                                if fields_ptr and fc and fc < 100:
                                    for fi in range(min(fc, 20)):
                                        fa = fields_ptr + fi * 0x20
                                        fn_ptr = mem._pm.read_pointer(fa + 0x08)
                                        fn = (
                                            mem._mono._read_cstring(fn_ptr)
                                            if fn_ptr
                                            else f"field_{fi}"
                                        )
                                        fo = mem._pm.read_uint32(fa + 0x18)
                                        ft_ptr = mem._pm.read_pointer(fa + 0x00)
                                        ft = 0
                                        fa_val = 0
                                        if ft_ptr:
                                            tc = mem._pm.read_int8(ft_ptr + 0x0A)
                                            ft = (tc & 0xFF) if tc is not None else 0
                                            a = mem._pm.read_uint16(ft_ptr + 0x08)
                                            fa_val = a if a is not None else 0
                                        s = " [STATIC]" if (fa_val & 0x10) else ""
                                        tn = MonoTypeEnum.name(ft)
                                        print(
                                            f"      +0x{fo or 0:04x}: {fn} ({tn}) attrs=0x{fa_val:04x}{s}"
                                        )

                        # Try interpreting as List<T>
                        items = mem._pm.read_pointer(val + 0x10)
                        size = mem._pm.read_int32(val + 0x18)
                        print(f"    As List<T>: _items=0x{items or 0:x}, _size={size}")

                        # Try interpreting as MonoArray
                        arr_bounds = mem._pm.read_pointer(val + 0x10)
                        arr_len = mem._pm.read_int32(val + 0x18)
                        arr_data0 = mem._pm.read_pointer(val + 0x20)
                        print(
                            f"    As MonoArray: bounds=0x{arr_bounds or 0:x},"
                            f" max_length={arr_len},"
                            f" data[0]=0x{arr_data0 or 0:x}"
                        )

                elif f.type_code == 0x02:
                    val = mem._pm.read_uint8(addr)
                    print(
                        f"  +0x{f.offset:04x}: {f.name} ({f.type_name}) = {bool(val) if val is not None else None}"
                    )
                elif f.type_code in (0x08, 0x09):
                    val = mem._pm.read_int32(addr)
                    print(f"  +0x{f.offset:04x}: {f.name} ({f.type_name}) = {val}")
                elif f.type_code == 0x0C:
                    val = mem._pm.read_float(addr)
                    print(f"  +0x{f.offset:04x}: {f.name} ({f.type_name}) = {val}")
                else:
                    val = mem._pm.read_uint64(addr)
                    print(
                        f"  +0x{f.offset:04x}: {f.name} ({f.type_name}) = 0x{val or 0:x}"
                    )

        print("\n=== DIAGNOSTICS COMPLETE ===")
        sys.exit(0)

    # Choose between legacy (hardcoded) and generic (config-driven) reader
    if "--generic" in sys.argv:
        from tools.game_state import GameStateReader
        from conf.game_configs.cuphead import FIELDS as CUPHEAD_FIELDS, ASSEMBLY, format_state

        print("\n--- Using generic GameStateReader ---")
        assert mem._mono is not None, "Mono not initialized"
        reader = GameStateReader(mem._mono, CUPHEAD_FIELDS, ASSEMBLY)
        reader.resolve_classes()
        print(reader.describe())

        print("\nReading game state (Ctrl+C to stop)...")
        try:
            while True:
                state = reader.read_all()
                print(f"\r{format_state(state):<120s}", end="", flush=True)
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nStopped")
    else:
        print("\nReading game state (Ctrl+C to stop)...")
        print("  (use --generic to test the config-driven GameStateReader)")
        try:
            while True:
                state = mem.read_state()
                print(f"\r{state.summary():<120s}", end="", flush=True)
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nStopped")
