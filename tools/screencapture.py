from datetime import datetime
import os
import sys
import subprocess
from conf.config_parser import screencapture_conf as config


def create_screenshots_directory():
    """
    Creates a 'screenshots/captures' directory if it doesn't exist.
    Returns the path to the directory.
    """
    directory = os.path.join("screenshots", "captures")
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created '{directory}' directory")
    return directory


def check_screenshot_tool():
    """
    Check which screenshot tool is available on the system.
    Returns the preferred tool name or None if none are available.
    """
    tools = ["flameshot", "scrot", "import"]  # Priority order
    for tool in tools:
        try:
            result = subprocess.run(["which", tool], capture_output=True, text=True)
            if result.returncode == 0:
                return tool
        except Exception:
            continue
    return None


def find_wine_window():
    """
    Find the Wine game window and return its geometry as "WxH+x+y".

    Tries multiple backends in order:
      1. Sway (swaymsg -t get_tree)
      2. KDE Plasma / KWin (D-Bus scripting via loadScript)

    Returns None if the window cannot be found or we're on macOS/Windows.
    """
    if sys.platform in ("darwin", "win32"):
        return None

    # Try Sway first
    geom = _find_wine_window_sway()
    if geom:
        return geom

    # Try KDE Plasma / KWin
    geom = _find_wine_window_kwin()
    if geom:
        return geom

    return None


def _find_wine_window_sway():
    """Find Wine window via swaymsg (Sway WM)."""
    try:
        result = subprocess.run(["which", "swaymsg"], capture_output=True, text=True)
        if result.returncode != 0:
            return None

        result = subprocess.run(
            ["swaymsg", "-t", "get_tree"], capture_output=True, text=True, check=True
        )
        import json

        tree = json.loads(result.stdout)

        def search(node):
            app_id = node.get("app_id", "") or ""
            if "wine" in app_id.lower():
                r = node.get("rect", {})
                return f"{r.get('width', 1920)}x{r.get('height', 1080)}+{r.get('x', 0)}+{r.get('y', 0)}"
            for child in node.get("nodes", []) + node.get("floating_nodes", []):
                found = search(child)
                if found:
                    return found
            return None

        return search(tree)
    except Exception:
        return None


def _find_wine_window_kwin():
    """Find Wine window via KDE KWin D-Bus scripting.

    Loads a small JavaScript snippet into KWin that prints window info
    to the journal, then parses the output.
    """
    import tempfile
    import json

    # JavaScript that KWin executes to list windows matching Wine/explorer/.exe/cuphead.
    # Broadened keywords so we don't miss the game window if KWin registers it
    # under the game's executable name instead of "wine" or "explorer".
    js_code = r"""
var wins = workspace.windowList();
var found = [];
var keywords = ["wine", "explorer", ".exe", "cuphead"];
for (var i = 0; i < wins.length; i++) {
    var w = wins[i];
    var cls = String(w.resourceClass || "").toLowerCase();
    var name = String(w.resourceName || "").toLowerCase();
    var cap = String(w.caption || "").toLowerCase();
    var matched = false;
    for (var k = 0; k < keywords.length; k++) {
        if (cls.indexOf(keywords[k]) >= 0 || name.indexOf(keywords[k]) >= 0 ||
            cap.indexOf(keywords[k]) >= 0) {
            matched = true;
            break;
        }
    }
    if (matched) {
        found.push(JSON.stringify({
            x: Math.round(w.x), y: Math.round(w.y),
            w: Math.round(w.width), h: Math.round(w.height),
            cls: cls, name: name, caption: String(w.caption || "")
        }));
    }
}
console.log("PE_WINE_WINDOWS:" + found.join("|||"));
"""

    try:
        import time
        from datetime import datetime, timezone

        # Record timestamp BEFORE running the script so we only read fresh journal entries
        since_ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        # Write JS to temp file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".js", delete=False) as f:
            f.write(js_code)
            js_path = f.name

        # Load script into KWin
        load_result = subprocess.run(
            [
                "dbus-send",
                "--session",
                "--dest=org.kde.KWin",
                "--print-reply",
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

        # Start the script
        subprocess.run(
            [
                "dbus-send",
                "--session",
                "--dest=org.kde.KWin",
                "--print-reply",
                "/Scripting",
                "org.kde.kwin.Scripting.start",
            ],
            capture_output=True,
            text=True,
            timeout=3,
        )

        # Give KWin a moment to execute
        time.sleep(0.5)

        # Read output from journal — use --since to avoid stale entries
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

        # Clean up temp file
        os.unlink(js_path)

        # Use the LAST matching line (most recent execution) to avoid stale results
        last_data_str = None
        for line in journal_result.stdout.splitlines():
            if "PE_WINE_WINDOWS:" in line:
                last_data_str = line.split("PE_WINE_WINDOWS:", 1)[1].strip()

        if last_data_str is None or not last_data_str:
            return None

        entries = last_data_str.split("|||")
        # Pick the largest Wine window (likely the game)
        best = None
        best_area = 0
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
            return f"{best['w']}x{best['h']}+{best['x']}+{best['y']}"
        return None
    except Exception:
        return None


def take_screenshot(directory, window_geometry=None):
    """
    Takes a screenshot and saves it with a timestamp.

    Args:
        directory: Directory to save screenshots
        window_geometry: Optional geometry string for flameshot (e.g., "1920x1080+0+0")
    """
    # Generate filename with current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"screenshot_{timestamp}.png"
    filepath = os.path.join(directory, filename)

    if sys.platform == "darwin":
        # macOS - try flameshot first (cross-platform consistency), fallback to screencapture
        screenshot_taken = False

        # Try flameshot (same as Linux for consistency)
        try:
            # Always use non-interactive "flameshot screen --raw" (never "flameshot gui")
            with open(filepath, "wb") as f:
                subprocess.run(["flameshot", "screen", "--raw"], stdout=f, check=True)
            print(f"✓ Screenshot (flameshot): {filename}")
            screenshot_taken = True
            return str(filepath)
        except (FileNotFoundError, subprocess.CalledProcessError) as e:
            print(f"⚠️  Flameshot failed: {e}")
            print("Falling back to native screencapture...")

        # Fall back to native macOS screencapture
        if not screenshot_taken:
            try:
                cmd = ["screencapture", "-x", filepath]  # -x: no sound
                subprocess.run(cmd, check=True)
                print(f"✓ Screenshot (macOS screencapture): {filename}")
                return str(filepath)
            except (FileNotFoundError, subprocess.CalledProcessError) as e:
                print(f"✗ Error: All screenshot tools failed: {e}")
                print("Install flameshot: brew install flameshot")
                return None

    elif sys.platform == "win32":
        from PIL import ImageGrab

        try:
            # Capture the screen
            screenshot = ImageGrab.grab()
            # Save the screenshot
            screenshot.save(filepath)
            print(f"Screenshot saved: {filename}")
        except Exception as e:
            print(f"Error taking screenshot: {e}")
    else:
        # Linux
        # Try flameshot first (default), then fall back to alternatives
        screenshot_taken = False

        # Try flameshot
        try:
            # Check if we should capture a specific window
            # Always use non-interactive "flameshot screen --raw" (never "flameshot gui")
            with open(filepath, "wb") as f:
                subprocess.run(["flameshot", "screen", "--raw"], stdout=f, check=True)
            print(f"✓ Screenshot (flameshot): {filename}")
            screenshot_taken = True
            return str(filepath)
        except (FileNotFoundError, subprocess.CalledProcessError) as e:
            print(f"⚠️  Flameshot failed: {e}")
            print("Trying alternative screenshot tool...")

        # Fall back to scrot if flameshot failed
        if not screenshot_taken:
            try:
                cmd = ["scrot", filepath]
                subprocess.run(cmd, check=True)
                print(f"✓ Screenshot (scrot): {filename}")
                screenshot_taken = True
                return str(filepath)
            except (FileNotFoundError, subprocess.CalledProcessError):
                print("⚠️  scrot not available or failed")

        # Fall back to ImageMagick import if scrot failed
        if not screenshot_taken:
            try:
                cmd = ["import", "-window", "root", filepath]
                subprocess.run(cmd, check=True)
                print(f"✓ Screenshot (ImageMagick): {filename}")
                screenshot_taken = True
                return str(filepath)
            except (FileNotFoundError, subprocess.CalledProcessError):
                print("⚠️  ImageMagick import not available or failed")

        # If all tools failed
        if not screenshot_taken:
            print(
                "✗ Error: No screenshot tool available (tried: flameshot, scrot, ImageMagick)"
            )
            print(
                "Install one with: sudo apt install flameshot  OR  sudo apt install scrot"
            )
            return None
