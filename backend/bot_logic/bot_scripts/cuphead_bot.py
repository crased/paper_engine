"""
Cuphead Bot — Read → Fuse → Decide → Act (Three-Domain Router)

Memory reading + optional YOLO vision. Reads game state via
process_vm_readv, fuses with YOLO detections, routes decisions
to NavigationDomain (menus/map) or StrategyDomain (combat).
Input injection uses xdg-desktop-portal RemoteDesktop.

Architecture:
    READ  (<1ms) — Memory state via GameStateReader
    FUSE  (<1ms) — Memory → WorldState + attach DetectionFrame
    ROUTE (<1ms) — LEVEL_PLAYING → StrategyDomain, else → NavigationDomain
    ACT   (<1ms) — Key hold/release via portal RemoteDesktop

    Background: DetectionDomain runs YOLO inference in a daemon thread

Usage:
    python bot_scripts/cuphead_bot.py --launch
    python bot_scripts/cuphead_bot.py --launch --no-play   # observe only
    python bot_scripts/cuphead_bot.py --launch --verbose    # debug logging
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, Set

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════


class Config:
    """Bot configuration.  CLI args override these defaults."""

    GAME_PATH = str(PROJECT_ROOT / "game")
    TARGET_MONITOR = "DP-1"
    TARGET_FPS = 30
    FRAME_TIME = 1.0 / TARGET_FPS
    ASSEMBLY_LOAD_TIMEOUT = 60.0


# ═══════════════════════════════════════════════════════════════════════════
# KEYBOARD CONTROLLER (xdg-portal RemoteDesktop)
# ═══════════════════════════════════════════════════════════════════════════

# Linux evdev keycodes (from input-event-codes.h)
KEY_ESC = 1
KEY_ENTER = 28
KEY_A = 30
KEY_S = 31
KEY_D = 32
KEY_W = 17
KEY_X = 45
KEY_C = 46
KEY_V = 47
KEY_SPACE = 57
KEY_LEFTSHIFT = 42
KEY_LEFTCTRL = 29


class _PortalSession:
    """
    Manages an xdg-desktop-portal RemoteDesktop session in a background thread.

    The portal provides compositor-level input injection that works on
    Wayland regardless of window focus.  Runs its own asyncio event loop
    in a daemon thread; public methods are thread-safe.
    """

    PORTAL_BUS = "org.freedesktop.portal.Desktop"
    PORTAL_PATH = "/org/freedesktop/portal/desktop"
    RD_IFACE = "org.freedesktop.portal.RemoteDesktop"
    DEVICE_KEYBOARD = 1

    def __init__(self):
        self._session_handle: Optional[str] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._bus = None
        self._rd_iface = None
        self._ready = threading.Event()
        self._error: Optional[str] = None

    def start(self, timeout: float = 15.0) -> bool:
        """Start the portal session.  Blocks until ready or timeout."""
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout):
            logger.error("Portal session timed out after %.0fs", timeout)
            return False
        if self._error:
            logger.error("Portal session failed: %s", self._error)
            return False
        logger.info("Portal RemoteDesktop session active")
        return True

    def stop(self):
        """Tear down the portal session."""
        self._session_handle = None
        if self._loop and self._loop.is_running():

            async def _drain():
                await asyncio.sleep(0.1)

            try:
                fut = asyncio.run_coroutine_threadsafe(_drain(), self._loop)
                fut.result(timeout=1.0)
            except Exception:
                pass
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=3)

    @property
    def active(self) -> bool:
        return self._session_handle is not None and self._error is None

    def notify_key(self, keycode: int, state: int):
        """Send a keyboard event.  state: 1=press, 0=release."""
        if not self.active or self._loop is None:
            return
        asyncio.run_coroutine_threadsafe(
            self._async_notify_key(keycode, state), self._loop
        )

    def _run_loop(self):
        """Entry point for the background thread."""
        try:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._loop.run_until_complete(self._setup_session())
            if self.active:
                self._loop.run_forever()
        except Exception as exc:
            logger.error("Portal thread error: %s", exc)
            self._error = str(exc)
            self._ready.set()
        finally:
            if self._bus:
                try:
                    self._loop.run_until_complete(self._cleanup())
                except Exception:
                    pass

    async def _cleanup(self):
        try:
            self._bus.disconnect()
        except Exception:
            pass

    async def _setup_session(self):
        from dbus_fast.aio import MessageBus
        from dbus_fast import Variant, MessageType

        try:
            self._bus = await MessageBus().connect()
            unique = self._bus.unique_name.replace(".", "_").replace(":", "")
            counter = [0]

            async def portal_request(method: str, *args):
                counter[0] += 1
                token = f"pe_{counter[0]}"
                req_path = f"/org/freedesktop/portal/desktop/request/{unique}/{token}"
                fut = self._loop.create_future()

                def on_msg(msg):
                    if msg.message_type == MessageType.SIGNAL and msg.path == req_path:
                        resp = msg.body[0]
                        details = msg.body[1] if len(msg.body) > 1 else {}
                        if not fut.done():
                            fut.set_result((resp, details))

                self._bus.add_message_handler(on_msg)

                mutable_args = list(args)
                opts = mutable_args[-1] if mutable_args else {}
                if isinstance(opts, dict):
                    opts["handle_token"] = Variant("s", token)
                    mutable_args[-1] = opts

                intro = await self._bus.introspect(self.PORTAL_BUS, self.PORTAL_PATH)
                proxy = self._bus.get_proxy_object(
                    self.PORTAL_BUS, self.PORTAL_PATH, intro
                )
                iface = proxy.get_interface(self.RD_IFACE)
                fn = getattr(iface, f"call_{method}")
                await fn(*mutable_args)

                try:
                    resp, details = await asyncio.wait_for(fut, timeout=10.0)
                finally:
                    self._bus.remove_message_handler(on_msg)
                return resp, details

            # 1. CreateSession
            import random as _rnd

            session_token = f"pe_{_rnd.randint(10000, 99999)}"
            resp, details = await portal_request(
                "create_session",
                {"session_handle_token": Variant("s", session_token)},
            )
            if resp != 0:
                self._error = f"CreateSession failed (resp={resp})"
                self._ready.set()
                return

            sh = details.get("session_handle")
            self._session_handle = sh.value if hasattr(sh, "value") else sh

            # 2. SelectDevices (keyboard)
            resp, details = await portal_request(
                "select_devices",
                self._session_handle,
                {"types": Variant("u", self.DEVICE_KEYBOARD)},
            )
            if resp != 0:
                self._error = f"SelectDevices failed (resp={resp})"
                self._ready.set()
                return

            # 3. Start
            resp, details = await portal_request(
                "start",
                self._session_handle,
                "",
                {},
            )
            if resp != 0:
                self._error = f"Start failed/cancelled (resp={resp})"
                self._ready.set()
                return

            # Cache the interface proxy for fast key sending
            intro = await self._bus.introspect(self.PORTAL_BUS, self.PORTAL_PATH)
            proxy = self._bus.get_proxy_object(self.PORTAL_BUS, self.PORTAL_PATH, intro)
            self._rd_iface = proxy.get_interface(self.RD_IFACE)

            self._ready.set()

        except Exception as exc:
            self._error = str(exc)
            self._ready.set()

    async def _async_notify_key(self, keycode: int, state: int):
        if self._rd_iface is None or self._session_handle is None:
            return
        try:
            await self._rd_iface.call_notify_keyboard_keycode(
                self._session_handle, {}, keycode, state
            )
        except Exception as exc:
            logger.debug("Portal key notify failed: %s", exc)


class KeyboardController:
    """Keyboard input via xdg-portal RemoteDesktop.  Uses evdev keycodes."""

    def __init__(self, portal: _PortalSession):
        self._portal = portal
        # Cuphead default controls
        self.MOVE_LEFT = KEY_A
        self.MOVE_RIGHT = KEY_D
        self.MOVE_UP = KEY_W
        self.MOVE_DOWN = KEY_S
        self.JUMP = KEY_SPACE
        self.SHOOT = KEY_X
        self.LOCK_AIM = KEY_LEFTSHIFT
        self.DASH = KEY_LEFTCTRL
        self.EX_SUPER = KEY_C
        self.SWITCH_WEAPON = KEY_V
        self.PAUSE = KEY_ESC
        self.CONFIRM = KEY_ENTER

    def hold_key(self, key: int):
        self._portal.notify_key(key, 1)

    def release_key(self, key: int):
        self._portal.notify_key(key, 0)

    def tap_key(self, key: int):
        self._portal.notify_key(key, 1)
        self._portal.notify_key(key, 0)


class EmergencyStop:
    """ESC key listener via evdev — reads physical keyboard at /dev/input."""

    def __init__(self):
        self.running = True
        self._thread: Optional[threading.Thread] = None

    def start(self):
        self._thread = threading.Thread(target=self._listen, daemon=True)
        self._thread.start()
        logger.info("Emergency stop listener started (ESC to stop)")

    def stop(self):
        self.running = False

    def _listen(self):
        try:
            import evdev
            import select

            devices = []
            for path in evdev.list_devices():
                dev = evdev.InputDevice(path)
                caps = dev.capabilities()
                if 1 in caps and 1 in caps[1]:
                    devices.append(dev)
                else:
                    dev.close()

            if not devices:
                logger.warning("No keyboard devices found for ESC listener")
                return

            logger.debug(
                "ESC listener watching %d devices: %s",
                len(devices),
                [d.name for d in devices],
            )

            fds = {dev.fd: dev for dev in devices}
            while self.running:
                r, _, _ = select.select(list(fds.keys()), [], [], 0.5)
                for fd in r:
                    dev = fds[fd]
                    try:
                        for event in dev.read():
                            if event.type == 1 and event.code == 1 and event.value == 1:
                                print("\nESC pressed — stopping bot...")
                                self.running = False
                                return
                    except OSError:
                        pass

            for dev in devices:
                dev.close()

        except ImportError:
            logger.warning("evdev not available for ESC listener; use Ctrl+C")
        except Exception as exc:
            logger.warning("ESC listener error: %s", exc)


# ═══════════════════════════════════════════════════════════════════════════
# WINDOW MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════


def _strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def _run_kwin_script(js_code: str, marker: str, timeout: float = 1.0) -> Optional[str]:
    """Load and execute a KWin JS script via D-Bus, read output from journal."""
    import tempfile
    from datetime import datetime, timezone

    js_path = None
    try:
        since_ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".js", delete=False) as f:
            f.write(js_code)
            js_path = f.name

        load_result = subprocess.run(
            [
                "dbus-send",
                "--session",
                "--print-reply",
                "--dest=org.kde.KWin",
                "/Scripting",
                "org.kde.kwin.Scripting.loadScript",
                f"string:{js_path}",
            ],
            capture_output=True,
            text=True,
            timeout=3,
        )
        if load_result.returncode != 0:
            return None

        script_id = None
        for line in load_result.stdout.splitlines():
            line = line.strip()
            if line.startswith("int32"):
                script_id = line.split()[-1]
                break
        if script_id is None:
            return None

        subprocess.run(
            [
                "dbus-send",
                "--session",
                "--print-reply",
                "--dest=org.kde.KWin",
                f"/Scripting/Script{script_id}",
                "org.kde.kwin.Script.run",
            ],
            capture_output=True,
            text=True,
            timeout=3,
        )

        time.sleep(timeout)
        journal_result = subprocess.run(
            [
                "journalctl",
                "--user",
                "-u",
                "plasma-kwin_wayland",
                "--no-pager",
                "-n",
                "30",
                "--since",
                since_ts,
                "--output=cat",
            ],
            capture_output=True,
            text=True,
            timeout=3,
        )

        last_match = None
        for line in journal_result.stdout.splitlines():
            if marker in line:
                last_match = line.split(marker, 1)[1].strip()
        return last_match

    except Exception as e:
        logger.debug("_run_kwin_script failed: %s", e)
        return None
    finally:
        if js_path:
            try:
                os.unlink(js_path)
            except OSError:
                pass


def find_wine_window() -> Optional[str]:
    """Find Wine window geometry as 'WxH+X+Y'.  KDE > Sway > xdotool."""
    # --- KDE Plasma 6 / KWin Wayland ---
    if (
        os.environ.get("XDG_SESSION_TYPE", "").lower() == "wayland"
        and subprocess.run(["which", "dbus-send"], capture_output=True).returncode == 0
    ):
        js_code = r"""
var wins = workspace.windowList();
var found = [];
for (var i = 0; i < wins.length; i++) {
    var w = wins[i];
    var cls = String(w.resourceClass || "").toLowerCase();
    var name = String(w.resourceName || "").toLowerCase();
    if (cls.indexOf(".exe") >= 0 || name.indexOf(".exe") >= 0 ||
        cls.indexOf("wine") >= 0 || name.indexOf("wine") >= 0) {
        found.push(JSON.stringify({
            x: Math.round(w.frameGeometry.x),
            y: Math.round(w.frameGeometry.y),
            w: Math.round(w.frameGeometry.width),
            h: Math.round(w.frameGeometry.height),
            cls: cls, name: name, caption: String(w.caption || "")
        }));
    }
}
console.log("PE_WINE_WINDOWS:" + found.join("|||"));
"""
        data_str = _run_kwin_script(js_code, "PE_WINE_WINDOWS:")
        if data_str:
            entries = data_str.split("|||")
            best, best_area = None, 0
            for entry in entries:
                try:
                    info = json.loads(entry)
                    area = info["w"] * info["h"]
                    if area > best_area:
                        best_area = area
                        best = info
                except (json.JSONDecodeError, KeyError):
                    continue
            if best:
                geometry = f"{best['w']}x{best['h']}+{best['x']}+{best['y']}"
                logger.info("Wine window via KDE D-Bus: %s", geometry)
                return geometry

    # --- xdotool fallback ---
    if subprocess.run(["which", "xdotool"], capture_output=True).returncode == 0:
        try:
            for search_term in ["wine", ".exe", "Cuphead"]:
                search = subprocess.run(
                    ["xdotool", "search", "--onlyvisible", "--name", search_term],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                wids = search.stdout.strip().splitlines()
                if wids:
                    best_geom, max_area = None, 0
                    for wid in wids:
                        try:
                            geo = subprocess.run(
                                ["xdotool", "getwindowgeometry", "--shell", wid],
                                capture_output=True,
                                text=True,
                                timeout=2,
                            )
                            vals = {}
                            for line in geo.stdout.splitlines():
                                if "=" in line:
                                    k, v = line.split("=", 1)
                                    vals[k] = v
                            if all(k in vals for k in ("WIDTH", "HEIGHT", "X", "Y")):
                                w = int(vals["WIDTH"])
                                h = int(vals["HEIGHT"])
                                area = w * h
                                if area > max_area:
                                    max_area = area
                                    best_geom = f"{w}x{h}+{vals['X']}+{vals['Y']}"
                        except (subprocess.CalledProcessError, ValueError):
                            continue
                    if best_geom:
                        logger.info("Wine window via xdotool: %s", best_geom)
                        return best_geom
        except Exception as e:
            logger.debug("xdotool failed: %s", e)

    logger.warning("Failed to find Wine window geometry")
    return None


def _get_monitor_geometry(output_name: str) -> Optional[Dict[str, int]]:
    """Query KDE kscreen-doctor for a monitor's position and size."""
    try:
        result = subprocess.run(
            ["kscreen-doctor", "--outputs"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        current_output = None
        for line in result.stdout.splitlines():
            stripped = _strip_ansi(line).strip()
            if stripped.startswith("Output:"):
                parts = stripped.split()
                current_output = parts[2] if len(parts) >= 3 else None
            if current_output == output_name and "Geometry:" in stripped:
                geo_part = stripped.split("Geometry:")[1].strip()
                pos, size = geo_part.split()
                x, y = map(int, pos.split(","))
                w, h = map(int, size.split("x"))
                return {"x": x, "y": y, "width": w, "height": h}
    except Exception as e:
        logger.warning("Could not query monitor geometry: %s", e)
    return None


def move_wine_to_monitor(output_name: str, retries: int = 3) -> bool:
    """Move Wine window to a specific monitor via KWin D-Bus."""
    monitor = _get_monitor_geometry(output_name)
    if not monitor:
        logger.warning("Monitor '%s' not found", output_name)
        return False

    mx, my, mw, mh = monitor["x"], monitor["y"], monitor["width"], monitor["height"]

    js_code = f"""
var wins = workspace.windowList();
var moved = false;
for (var i = 0; i < wins.length; i++) {{
    var w = wins[i];
    var cls = String(w.resourceClass || "").toLowerCase();
    var name = String(w.resourceName || "").toLowerCase();
    if ((cls.indexOf(".exe") >= 0 || name.indexOf(".exe") >= 0 ||
         cls.indexOf("wine") >= 0 || name.indexOf("wine") >= 0) && !moved) {{
        w.frameGeometry.x = {mx};
        w.frameGeometry.y = {my};
        w.frameGeometry.width = {mw};
        w.frameGeometry.height = {mh};
        console.log("PE_MOVED_WINDOW:" + String(w.caption) + " to {output_name}");
        moved = true;
    }}
}}
if (!moved) {{ console.log("PE_MOVED_WINDOW:NONE"); }}
"""
    for attempt in range(retries):
        result_str = _run_kwin_script(js_code, "PE_MOVED_WINDOW:")
        if result_str is not None and result_str != "NONE":
            logger.info("Moved window '%s' to %s", result_str, output_name)
            return True
        time.sleep(3)

    logger.warning("Could not move Wine window to %s", output_name)
    return False


def _is_wine_still_running() -> bool:
    """Check if Wine processes are alive."""
    for name in ["wineserver", "wine-preloader", "wine64-preloader", ".exe"]:
        try:
            result = subprocess.run(
                ["pgrep", "-f", name],
                capture_output=True,
                text=True,
                timeout=2,
            )
            if result.stdout.strip():
                return True
        except (
            subprocess.CalledProcessError,
            FileNotFoundError,
            subprocess.TimeoutExpired,
        ):
            pass
    return False


# ═══════════════════════════════════════════════════════════════════════════
# CUPHEAD BOT
# ═══════════════════════════════════════════════════════════════════════════


class CupheadBot:
    """
    Cuphead bot — three-domain router.  READ → FUSE → ROUTE → ACT at 30fps.

    Routes LEVEL_PLAYING to StrategyDomain, all other phases to
    NavigationDomain.  DetectionDomain runs YOLO in a background thread
    and attaches spatial detections to WorldState each tick.
    """

    def __init__(
        self,
        enable_play: bool = True,
        target_fps: int = 30,
        target_monitor: Optional[str] = "DP-1",
    ):
        self.enable_play = enable_play
        self.target_fps = target_fps
        self.frame_time = 1.0 / target_fps
        self.target_monitor = target_monitor

        self._portal: Optional[_PortalSession] = None
        self.kbd: Optional[KeyboardController] = None
        self.emergency_stop = EmergencyStop()

        self._held_keys: Set = set()
        self.running = True
        self._tick_count = 0

    def _execute_actions(self, actions: Dict):
        """Apply action dict with hold/release state tracking."""
        if self.kbd is None:
            return
        for key, should_hold in actions.items():
            if key in (self.kbd.CONFIRM, self.kbd.PAUSE):
                if should_hold:
                    self.kbd.tap_key(key)
                continue

            if should_hold and key not in self._held_keys:
                self.kbd.hold_key(key)
                self._held_keys.add(key)
            elif not should_hold and key in self._held_keys:
                self.kbd.release_key(key)
                self._held_keys.discard(key)

    def _release_all_keys(self):
        if self.kbd is None:
            return
        for key in list(self._held_keys):
            self.kbd.release_key(key)
        self._held_keys.clear()

    def run(self):
        """Launch game, attach memory, run READ → FUSE → DECIDE → ACT loop."""
        from tools.cuphead_memory import CupheadMemory
        from tools.game_state import GameStateReader
        from conf.game_configs.cuphead import FIELDS, ASSEMBLY, GAME_NAME, format_state
        from bot_logic.bot_scripts.domains.contracts import (
            GamePhase,
            WorldState,
            fuse,
        )
        from bot_logic.bot_scripts.domains.navigation import NavigationDomain
        from bot_logic.bot_scripts.domains.strategy import StrategyDomain
        from bot_logic.bot_scripts.domains.detection import DetectionDomain

        detection = None
        try:
            # 1. Launch game
            logger.info("Launching Cuphead...")
            mem = CupheadMemory()
            if not mem.launch():
                logger.error("Failed to launch game")
                return

            # 2. Wait for Assembly-CSharp
            logger.info("Waiting for game assemblies...")
            attach_start = time.time()
            attached = False
            while time.time() - attach_start < Config.ASSEMBLY_LOAD_TIMEOUT:
                mem._pm.refresh_regions()
                if mem.attach():
                    assemblies = mem._mono.list_assemblies()
                    if ASSEMBLY in assemblies:
                        logger.info(
                            "Assembly-CSharp loaded (%d assemblies, %.1fs)",
                            len(assemblies),
                            time.time() - attach_start,
                        )
                        attached = True
                        break
                time.sleep(2.0)

            if not attached:
                logger.error(
                    "Assembly-CSharp not loaded after %.0fs",
                    Config.ASSEMBLY_LOAD_TIMEOUT,
                )
                return

            # 3. Set up GameStateReader
            reader = GameStateReader(mem._mono, FIELDS, ASSEMBLY)
            resolved = reader.resolve_classes()
            found = [k for k, v in resolved.items() if v is not None]
            missing = [k for k, v in resolved.items() if v is None]
            logger.info("Memory reader: %d fields, classes: %s", len(FIELDS), found)
            if missing:
                logger.warning("Missing classes: %s", missing)

            # 4. Find and move game window
            logger.info("Waiting for game window...")
            geometry = None
            for _ in range(30):
                geometry = find_wine_window()
                if geometry:
                    break
                time.sleep(1.0)

            if geometry:
                logger.info("Window found: %s", geometry)
            else:
                logger.warning("Window not found")

            if self.target_monitor and geometry:
                moved = move_wine_to_monitor(self.target_monitor)
                if moved:
                    time.sleep(1.0)
                    new_geom = find_wine_window()
                    if new_geom:
                        geometry = new_geom

            # 5. Set up portal input
            if self.enable_play:
                logger.info("Setting up portal RemoteDesktop session...")
                self._portal = _PortalSession()
                if not self._portal.start(timeout=15.0):
                    logger.error("Portal session failed — running observe-only")
                    self.enable_play = False
                else:
                    self.kbd = KeyboardController(self._portal)

            # 6. Set up three-domain system
            nav = NavigationDomain(self.kbd) if self.kbd else None
            strat = StrategyDomain(self.kbd) if self.kbd else None

            # Optional: start YOLO detection if a trained model exists
            model_candidates = sorted(
                (PROJECT_ROOT / "models").glob("*/weights/best.pt"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if model_candidates:
                model_path = str(model_candidates[0])
                logger.info("Starting detection domain: %s", model_path)
                try:
                    detection = DetectionDomain(model_path)
                    detection.start()
                except Exception as exc:
                    logger.warning("Detection domain failed to start: %s", exc)
                    detection = None
            else:
                logger.info("No YOLO model found — running memory-only")

            # 7. Main loop
            self.emergency_stop.start()

            logger.info(
                "Bot loop starting (play=%s, fps=%d)", self.enable_play, self.target_fps
            )
            print(f"Bot running — play={self.enable_play}. Press ESC to stop.")

            start_time = time.time()
            last_status_time = 0.0
            prev_phase = GamePhase.UNKNOWN
            prev_ws: Optional[WorldState] = None

            while (
                self.running
                and self.emergency_stop.running
                and _is_wine_still_running()
            ):
                loop_start = time.time()

                # READ
                try:
                    memory_state = reader.read_all()
                except Exception as exc:
                    logger.warning("State read error: %s", exc)
                    memory_state = {}

                # FUSE
                ws = fuse(memory_state, prev_ws)

                # Attach latest YOLO detections (if available)
                if detection:
                    ws.detection_frame = detection.get_latest()

                if ws.phase != prev_phase:
                    logger.info("Phase: %s -> %s", prev_phase.name, ws.phase.name)
                    prev_phase = ws.phase

                # DECIDE + ACT — route by phase
                if self.enable_play and self.kbd and nav and strat:
                    if ws.phase == GamePhase.LEVEL_PLAYING:
                        action = strat.decide(ws)
                    else:
                        action = nav.decide(ws)
                    self._execute_actions(action.keys)

                prev_ws = ws

                # STATUS (every 2s)
                if loop_start - last_status_time >= 2.0:
                    elapsed = loop_start - start_time
                    phase_name = ws.phase.name if ws.memory_ok else "NO_STATE"
                    status = format_state(memory_state) if memory_state else "no state"
                    tps = self._tick_count / max(elapsed, 0.01)
                    held = (
                        ",".join(
                            k
                            for k, v in [
                                ("L", self.kbd.MOVE_LEFT in self._held_keys),
                                ("R", self.kbd.MOVE_RIGHT in self._held_keys),
                                ("J", self.kbd.JUMP in self._held_keys),
                                ("S", self.kbd.SHOOT in self._held_keys),
                                ("D", self.kbd.DASH in self._held_keys),
                            ]
                            if v
                        )
                        if self.kbd
                        else "-"
                    )
                    print(
                        f"\r[{elapsed:6.1f}s] {tps:4.0f}tps "
                        f"{phase_name:<15s} keys=[{held}]  {status}",
                        end="",
                        flush=True,
                    )
                    last_status_time = loop_start

                self._tick_count += 1

                # FPS limiting
                elapsed_tick = time.time() - loop_start
                sleep_time = self.frame_time - elapsed_tick
                if sleep_time > 0:
                    time.sleep(sleep_time)

            print()
            if not self.emergency_stop.running:
                logger.info("Stopped by ESC key")
            elif not _is_wine_still_running():
                logger.info("Game process exited")

        except KeyboardInterrupt:
            print()
            logger.info("Interrupted by user")

        except Exception as e:
            logger.error("Unexpected error: %s", e)
            import traceback

            traceback.print_exc()

        finally:
            self._release_all_keys()
            if detection:
                detection.stop()
            self.emergency_stop.stop()
            if self._portal:
                self._portal.stop()

            try:
                elapsed = time.time() - start_time
            except NameError:
                elapsed = 0.0
            logger.info("Bot ran for %.1fs, %d ticks", elapsed, self._tick_count)
            print("--- Cuphead bot finished ---")


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Cuphead Memory Bot")
    parser.add_argument(
        "--launch", action="store_true", required=True, help="Launch Cuphead via Wine"
    )
    parser.add_argument(
        "--no-play", action="store_true", help="Don't send inputs — observe only"
    )
    parser.add_argument(
        "--fps", type=int, default=30, help="Bot tick rate (default: 30)"
    )
    parser.add_argument(
        "--monitor", type=str, default="DP-1", help="Target monitor (default: DP-1)"
    )
    parser.add_argument("--verbose", action="store_true", help="Debug logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    bot = CupheadBot(
        enable_play=not args.no_play,
        target_fps=args.fps,
        target_monitor=args.monitor,
    )
    bot.run()


if __name__ == "__main__":
    main()
