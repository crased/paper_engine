"""
GameplayRecorder — captures synchronized screenshots + memory state during gameplay.

Creates a session directory under recordings/ with:
  - frame_NNNNNN.png         (screenshot)
  - frame_NNNNNN.json        (memory state + timestamp)
  - session.json             (manifest: game, config, timing, frame count)

Usage:
    # As part of a larger system (e.g. bot loop):
    recorder = GameplayRecorder(state_reader, screen_capture)
    recorder.start_session()
    ...
    recorder.tick()   # call every frame; internally rate-limits captures
    ...
    recorder.end_session()

    # Standalone CLI (launches game, records until Ctrl-C):
    python -m pipeline.gameplay_recorder --launch
"""

from __future__ import annotations

import io
import json
import logging
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Project root (same pattern as other pipeline/*.py files)
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Screen capture (extracted from bot, made standalone)
# ---------------------------------------------------------------------------


def _resolve_flameshot_screen(window_geometry: Optional[str]) -> Optional[str]:
    """Map a window geometry (WxH+X+Y) to a flameshot ``-n`` screen number.

    Parses ``kscreen-doctor -o`` to find which output contains the window's
    top-left corner, then returns the 0-based screen index that flameshot
    expects.  Returns *None* if the mapping cannot be determined.
    """
    if window_geometry is None:
        return None

    match = re.match(r"(\d+)x(\d+)\+(-?\d+)\+(-?\d+)", window_geometry)
    if not match:
        return None
    win_x, win_y = int(match.group(3)), int(match.group(4))

    try:
        result = subprocess.run(
            ["kscreen-doctor", "-o"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        clean = re.sub(r"\x1b\[[0-9;]*m", "", result.stdout)

        outputs: list[tuple[int, str, int, int, int, int]] = []
        current_num = -1
        current_name = ""
        for line in clean.split("\n"):
            m = re.match(r"\s*Output:\s*(\d+)\s+(\S+)", line)
            if m:
                current_num = int(m.group(1))
                current_name = m.group(2)
                continue
            m = re.match(r"\s*Geometry:\s*(-?\d+),(-?\d+)\s+(\d+)x(\d+)", line)
            if m and current_num >= 0:
                gx, gy = int(m.group(1)), int(m.group(2))
                gw, gh = int(m.group(3)), int(m.group(4))
                outputs.append((current_num, current_name, gx, gy, gw, gh))
                current_num = -1

        if not outputs:
            return None

        outputs.sort(key=lambda o: o[0])
        for idx, (_num, name, gx, gy, gw, gh) in enumerate(outputs):
            if gx <= win_x < gx + gw and gy <= win_y < gy + gh:
                logger.info(
                    "Window at +%d+%d → output %s (flameshot -n %d)",
                    win_x,
                    win_y,
                    name,
                    idx,
                )
                return str(idx)

        return None
    except Exception as exc:
        logger.debug("kscreen-doctor failed: %s", exc)
        return None


class FrameCapture:
    """
    Lightweight screen capture returning numpy RGB arrays.
    Supports Wayland (flameshot) and X11 (mss).
    Extracted from bot_scripts/cuphead_bot.py ScreenCapture for reuse.

    On Wayland, uses ``flameshot screen --raw -n <screen>`` to target the
    correct monitor.  The ``--region`` flag is avoided because it hangs on
    KDE Plasma 6 Wayland.
    """

    def __init__(self):
        self._use_flameshot = (
            sys.platform == "linux"
            and os.environ.get("XDG_SESSION_TYPE", "").lower() == "wayland"
        )
        self._window_geometry: Optional[str] = None
        self._screen_number: Optional[str] = None  # flameshot -n value
        self._sct = None

        if not self._use_flameshot:
            try:
                import mss as _mss

                self._sct = _mss.mss()
            except ImportError:
                logger.warning("mss not installed; capture unavailable on X11")
        else:
            if (
                subprocess.run(["which", "flameshot"], capture_output=True).returncode
                != 0
            ):
                raise RuntimeError("flameshot not found — required for Wayland capture")

    def set_window_geometry(self, geometry: str):
        """Set target window geometry.  Format: WxH+X+Y (X/Y can be negative)."""
        self._window_geometry = geometry
        self._screen_number = _resolve_flameshot_screen(geometry)
        logger.info(
            "FrameCapture: geometry=%s, flameshot screen=%s",
            geometry,
            self._screen_number,
        )

    def capture(self) -> Optional[np.ndarray]:
        """Return a screenshot as an RGB numpy array (H, W, 3) uint8, or None."""
        if self._use_flameshot:
            return self._capture_flameshot()
        return self._capture_mss()

    # -- internals --

    def _capture_flameshot(self) -> Optional[np.ndarray]:
        try:
            cmd = ["flameshot", "screen", "--raw"]

            # Use -n to target the correct monitor.
            # --region is broken on KDE Plasma 6 Wayland (hangs until timeout).
            if self._screen_number is not None:
                cmd.extend(["-n", self._screen_number])

            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=5,
                check=True,
            )
            if not result.stdout:
                return None
            img = Image.open(io.BytesIO(result.stdout)).convert("RGB")
            frame = np.array(img)

            # If the frame doesn't match the expected window size, crop.
            if self._window_geometry:
                match = re.match(
                    r"(\d+)x(\d+)\+(-?\d+)\+(-?\d+)", self._window_geometry
                )
                if match:
                    expected_w, expected_h = int(match.group(1)), int(match.group(2))
                    actual_h, actual_w = frame.shape[:2]
                    if actual_w > expected_w + 50 or actual_h > expected_h + 50:
                        logger.debug(
                            "Frame bigger than expected (%dx%d vs %dx%d), cropping",
                            actual_w,
                            actual_h,
                            expected_w,
                            expected_h,
                        )
                        return self._crop_to_geometry(frame)

            return frame
        except (
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired,
            FileNotFoundError,
        ) as exc:
            logger.warning("flameshot capture failed: %s", exc)
            return None
        except Exception as exc:
            logger.error("flameshot unexpected error: %s", exc)
            return None

    def _capture_mss(self) -> Optional[np.ndarray]:
        if self._sct is None:
            return None
        try:
            monitors = self._sct.monitors
            if len(monitors) < 2:
                return None
            monitor = monitors[1]  # primary monitor
            grab = self._sct.grab(monitor)
            frame = np.array(grab)[:, :, :3][:, :, ::-1]  # BGRA -> RGB
            if self._window_geometry:
                return self._crop_to_geometry(frame)
            return frame
        except Exception as exc:
            logger.warning("mss capture failed: %s", exc)
            return None

    def _crop_to_geometry(self, frame: np.ndarray) -> np.ndarray:
        if not self._window_geometry:
            return frame
        match = re.match(r"(\d+)x(\d+)\+(-?\d+)\+(-?\d+)", self._window_geometry)
        if not match:
            logger.warning(
                "Invalid geometry '%s', returning uncropped", self._window_geometry
            )
            return frame
        width, height, x, y = map(int, match.groups())
        x = max(0, x)
        y = max(0, y)
        max_y, max_x, _ = frame.shape
        y1, y2 = max(0, y), min(max_y, y + height)
        x1, x2 = max(0, x), min(max_x, x + width)
        return frame[y1:y2, x1:x2]


# ---------------------------------------------------------------------------
# Session metadata
# ---------------------------------------------------------------------------


@dataclass
class SessionInfo:
    """Metadata written as session.json at end of recording."""

    game: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    frame_count: int = 0
    capture_interval: float = 0.5
    resolution: Optional[List[int]] = None  # [width, height] of first frame
    window_geometry: Optional[str] = None
    fields_recorded: List[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# GameplayRecorder
# ---------------------------------------------------------------------------


class GameplayRecorder:
    """
    Synchronised screenshot + memory-state recorder.

    Call ``tick()`` from your main loop (bot, manual play, etc).
    The recorder internally rate-limits captures to ``interval`` seconds.
    """

    def __init__(
        self,
        state_reader,  # GameStateReader (or None for screenshot-only)
        frame_capture,  # FrameCapture or any object with .capture() -> Optional[ndarray]
        interval: float = 0.5,  # seconds between captures
        output_root: Optional[str] = None,
        game_name: str = "unknown",
    ):
        self._reader = state_reader
        self._capture = frame_capture
        self._interval = max(0.05, interval)  # floor at 50ms
        self._game_name = game_name

        if output_root is None:
            self._output_root = _PROJECT_ROOT / "recordings" / "sessions"
        else:
            self._output_root = Path(output_root)

        # Session state
        self._session_dir: Optional[Path] = None
        self._session_info: Optional[SessionInfo] = None
        self._frame_idx: int = 0
        self._last_capture_time: float = 0.0
        self._running: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def session_dir(self) -> Optional[Path]:
        return self._session_dir

    @property
    def frame_count(self) -> int:
        return self._frame_idx

    def start_session(self, notes: str = "") -> Path:
        """Create a new session directory and start recording."""
        ts = time.strftime("%Y%m%d_%H%M%S")
        session_name = f"{self._game_name}_{ts}"
        self._session_dir = self._output_root / session_name
        self._session_dir.mkdir(parents=True, exist_ok=True)

        field_names = []
        if self._reader is not None:
            try:
                field_names = self._reader.list_fields()
            except Exception:
                pass

        self._session_info = SessionInfo(
            game=self._game_name,
            start_time=time.time(),
            capture_interval=self._interval,
            window_geometry=self._capture._window_geometry,
            fields_recorded=field_names,
            notes=notes,
        )
        self._frame_idx = 0
        self._last_capture_time = 0.0
        self._running = True
        logger.info("Recording session started: %s", self._session_dir)
        return self._session_dir

    def tick(self) -> bool:
        """
        Call every iteration of the game loop.
        Returns True if a frame was captured this tick, False if skipped (rate limit).
        """
        if not self._running:
            return False

        now = time.time()
        if now - self._last_capture_time < self._interval:
            return False

        self._last_capture_time = now
        return self._capture_frame(now)

    def end_session(self) -> Optional[Path]:
        """Finalise and write session.json.  Returns session directory path."""
        if not self._running or self._session_dir is None:
            return None

        self._running = False
        self._session_info.end_time = time.time()
        self._session_info.frame_count = self._frame_idx

        manifest_path = self._session_dir / "session.json"
        with open(manifest_path, "w") as f:
            json.dump(self._session_info.to_dict(), f, indent=2)

        duration = self._session_info.end_time - self._session_info.start_time
        logger.info(
            "Session ended: %d frames in %.1fs (%.1f fps) → %s",
            self._frame_idx,
            duration,
            self._frame_idx / max(duration, 0.001),
            self._session_dir,
        )
        return self._session_dir

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _capture_frame(self, timestamp: float) -> bool:
        """Capture one frame + state and save to disk."""
        # 1. Screenshot
        frame = self._capture.capture()
        if frame is None:
            logger.debug("Frame capture returned None, skipping")
            return False

        # 2. Memory state
        state: Dict[str, Any] = {}
        if self._reader is not None:
            try:
                state = self._reader.read_all()
            except Exception as exc:
                logger.warning("State read failed: %s", exc)
                state = {"_read_error": str(exc)}

        # 3. Build metadata
        meta = {
            "frame_index": self._frame_idx,
            "timestamp": timestamp,
            "session_elapsed": timestamp - self._session_info.start_time,
            "state": state,
        }

        # 4. Save to disk
        frame_name = f"frame_{self._frame_idx:06d}"
        img_path = self._session_dir / f"{frame_name}.png"
        meta_path = self._session_dir / f"{frame_name}.json"

        # Record resolution from first frame
        if self._frame_idx == 0:
            h, w = frame.shape[:2]
            self._session_info.resolution = [w, h]

        try:
            img = Image.fromarray(frame)
            img.save(img_path, format="PNG", optimize=False)
        except Exception as exc:
            logger.error("Failed to save frame image: %s", exc)
            return False

        try:
            with open(meta_path, "w") as f:
                json.dump(meta, f, separators=(",", ":"))
        except Exception as exc:
            logger.error("Failed to save frame metadata: %s", exc)
            # Image was saved, don't count as complete failure
            pass

        self._frame_idx += 1
        if self._frame_idx % 50 == 0:
            logger.info("Recorded %d frames", self._frame_idx)

        return True


# ---------------------------------------------------------------------------
# Standalone CLI: launch game + record
# ---------------------------------------------------------------------------


def _find_wine_window() -> Optional[str]:
    """Find Wine/Cuphead window geometry via KWin D-Bus scripting."""
    # Try KWin first (Plasma 6 Wayland)
    try:
        kwin_script = """
        var clients = workspace.windowList();
        for (var i = 0; i < clients.length; i++) {
            var c = clients[i];
            var rc = String(c.resourceClass).toLowerCase();
            var rn = String(c.resourceName).toLowerCase();
            if (rc.indexOf(".exe") !== -1 || rc.indexOf("wine") !== -1 ||
                rn.indexOf(".exe") !== -1 || rn.indexOf("wine") !== -1) {
                var g = c.frameGeometry;
                print(g.width + "x" + g.height + "+" + g.x + "+" + g.y);
                break;
            }
        }
        """
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".js", delete=False) as f:
            f.write(kwin_script)
            script_path = f.name

        # Register script
        reg = subprocess.run(
            [
                "dbus-send",
                "--session",
                "--dest=org.kde.KWin",
                "--print-reply",
                "/Scripting",
                "org.kde.kwin.Scripting.loadScript",
                f"string:{script_path}",
                "string:paper_engine_find_window",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if reg.returncode != 0:
            return None

        # Extract script ID
        for line in reg.stdout.splitlines():
            if "int32" in line:
                script_id = line.strip().split()[-1]
                break
        else:
            return None

        # Run script
        subprocess.run(
            [
                "dbus-send",
                "--session",
                "--dest=org.kde.KWin",
                "--print-reply",
                f"/Scripting/Script{script_id}",
                "org.kde.kwin.Script.run",
            ],
            capture_output=True,
            timeout=5,
        )
        time.sleep(0.3)

        # Read from journal
        journal = subprocess.run(
            [
                "journalctl",
                "--user",
                "-u",
                "plasma-kwin_wayland",
                "-n",
                "20",
                "--no-pager",
                "-o",
                "cat",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        # Unload script
        subprocess.run(
            [
                "dbus-send",
                "--session",
                "--dest=org.kde.KWin",
                "--print-reply",
                f"/Scripting/Script{script_id}",
                "org.kde.kwin.Script.stop",
            ],
            capture_output=True,
            timeout=5,
        )
        subprocess.run(
            [
                "dbus-send",
                "--session",
                "--dest=org.kde.KWin",
                "--print-reply",
                "/Scripting",
                "org.kde.kwin.Scripting.unloadScript",
                "string:paper_engine_find_window",
            ],
            capture_output=True,
            timeout=5,
        )
        os.unlink(script_path)

        # Parse geometry from journal output
        for line in reversed(journal.stdout.splitlines()):
            match = re.search(r"(\d+x\d+\+\-?\d+\+\-?\d+)", line)
            if match:
                return match.group(1)

    except Exception as exc:
        logger.debug("KWin window detection failed: %s", exc)

    return None


def main():
    """Standalone: launch Cuphead, attach memory reader, record gameplay."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Record gameplay: screenshots + memory state"
    )
    parser.add_argument("--launch", action="store_true", help="Launch Cuphead via Wine")
    parser.add_argument(
        "--pid", type=int, default=None, help="Attach to existing game PID"
    )
    parser.add_argument(
        "--interval", type=float, default=0.5, help="Capture interval in seconds"
    )
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    parser.add_argument(
        "--no-memory", action="store_true", help="Screenshot-only, no memory reading"
    )
    parser.add_argument(
        "--duration", type=float, default=0, help="Stop after N seconds (0=unlimited)"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    state_reader = None
    mem = None

    if not args.no_memory:
        from tools.cuphead_memory import CupheadMemory
        from tools.game_state import GameStateReader
        from conf.game_configs.cuphead import FIELDS, ASSEMBLY, GAME_NAME

        mem = CupheadMemory()

        if args.launch:
            logger.info("Launching Cuphead...")
            if not mem.launch():
                logger.error("Failed to launch game")
                sys.exit(1)
        elif args.pid:
            from tools.memory_reader import ProcessMemory
            from tools.mono_external import MonoExternal

            mem._pid = args.pid
            mem._pm = ProcessMemory(args.pid)
        else:
            logger.error("Must specify --launch or --pid")
            sys.exit(1)

        logger.info("Attaching to Mono runtime...")
        if not mem.attach():
            logger.error("Failed to attach to Mono")
            sys.exit(1)

        state_reader = GameStateReader(mem._mono, FIELDS, ASSEMBLY)
        state_reader.resolve_classes()
        game_name = GAME_NAME
        logger.info("Memory reader ready, %d fields configured", len(FIELDS))
    else:
        game_name = "unknown"
        if args.launch:
            # Launch game but don't attach memory
            from tools.cuphead_memory import CupheadMemory

            mem = CupheadMemory()
            if not mem.launch():
                logger.error("Failed to launch game")
                sys.exit(1)
            from conf.game_configs.cuphead import GAME_NAME

            game_name = GAME_NAME

    # Wait for window to appear
    logger.info("Waiting for game window...")
    geometry = None
    for _ in range(30):
        geometry = _find_wine_window()
        if geometry:
            break
        time.sleep(1.0)

    if not geometry:
        logger.warning("Could not detect game window geometry; capturing full screen")

    # Set up capture
    frame_capture = FrameCapture()
    if geometry:
        frame_capture.set_window_geometry(geometry)

    # Create recorder
    recorder = GameplayRecorder(
        state_reader=state_reader,
        frame_capture=frame_capture,
        interval=args.interval,
        output_root=args.output,
        game_name=game_name,
    )

    session_dir = recorder.start_session(
        notes=f"CLI recording, interval={args.interval}s"
    )
    logger.info("Recording to %s  (Ctrl-C to stop)", session_dir)

    start = time.time()
    try:
        while True:
            captured = recorder.tick()
            if captured and recorder.frame_count % 10 == 0:
                # Print a status line
                elapsed = time.time() - start
                if state_reader:
                    state = state_reader.read_all()
                    from conf.game_configs.cuphead import format_state

                    status = format_state(state)
                else:
                    status = "screenshot-only"
                print(
                    f"\r[{elapsed:6.1f}s] frames={recorder.frame_count:5d}  {status}",
                    end="",
                    flush=True,
                )

            if args.duration > 0 and (time.time() - start) >= args.duration:
                logger.info("Duration limit reached (%.0fs)", args.duration)
                break

            # Sleep a bit less than the interval to avoid drift
            time.sleep(min(0.05, args.interval / 4))

    except KeyboardInterrupt:
        print()  # newline after \r status
        logger.info("Interrupted by user")

    session_path = recorder.end_session()
    if session_path:
        logger.info("Session saved to %s", session_path)
        # Print summary
        manifest = session_path / "session.json"
        if manifest.exists():
            with open(manifest) as f:
                info = json.load(f)
            duration = info["end_time"] - info["start_time"]
            print(f"\nRecording summary:")
            print(f"  Game:       {info['game']}")
            print(f"  Frames:     {info['frame_count']}")
            print(f"  Duration:   {duration:.1f}s")
            print(f"  Resolution: {info.get('resolution', 'N/A')}")
            print(f"  Directory:  {session_path}")


if __name__ == "__main__":
    main()
