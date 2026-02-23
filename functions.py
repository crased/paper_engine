import os
import sys
import subprocess
from pathlib import Path
import configparser


def launch_label_studio(env):
    """Launch Label Studio for annotation.

    Args:
        env: Environment dictionary with LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED set

    Returns:
        subprocess.Popen object for the label-studio process
    """
    try:
        command = ["label-studio", "start", "--port", "8080"]
        label_process = subprocess.Popen(command, env=env, start_new_session=True)
        print("\nLabel Studio started on http://localhost:8080")
        return label_process
    except FileNotFoundError:
        raise FileNotFoundError(
            "label-studio command not found. Install with: pip install label-studio"
        )
    except subprocess.SubprocessError as e:
        raise RuntimeError(
            f"Failed to start Label Studio: {e}\n"
            "Try running manually: label-studio start --port 8080"
        )


def get_title(game_path):
    """Extract game title from executable filename.

    Supports .exe, .sh, .py, and no-extension executables.
    """
    game_path_obj = Path(game_path)

    # If game_path is a file, extract title from it directly
    if game_path_obj.is_file():
        return game_path_obj.stem if game_path_obj.suffix else game_path_obj.name

    # If game_path is a directory, find executable in it
    for file in game_path_obj.iterdir():
        if file.is_file():
            # Check for common executable extensions or execute permission
            if file.suffix in [".exe", ".sh", ".py"] or (
                not file.suffix and os.access(file, os.X_OK)
            ):
                return file.stem if file.suffix else file.name

    # Fallback: use directory name
    return game_path_obj.name


def _load_custom_filters():
    """
    Load custom executable filters from conf/executable_filters.ini
    Returns tuple of (custom_excluded_patterns, custom_position_filters)

    Returns empty lists if config doesn't exist or has errors.
    Encapsulated to prevent mutation of defaults.
    """
    config_path = Path("conf/executable_filters.ini")

    # Return empty lists if config doesn't exist (use defaults only)
    if not config_path.exists():
        return ([], [])

    try:
        parser = configparser.ConfigParser()
        parser.read(config_path)

        custom_excluded = []
        custom_position = []

        # Read substring patterns from [BLACKLIST] section
        if parser.has_section("BLACKLIST"):
            for key in parser.options("BLACKLIST"):
                value = parser.get("BLACKLIST", key).strip()
                # Remove inline comments
                if "#" in value:
                    value = value.split("#")[0].strip()
                if value:  # Only add non-empty values
                    custom_excluded.append(value.lower())

        # Read position patterns from [BLACKLIST_Position] section
        # Format: pattern_name = pattern,position
        if parser.has_section("BLACKLIST_Position"):
            for key in parser.options("BLACKLIST_Position"):
                value = parser.get("BLACKLIST_Position", key).strip()
                # Remove inline comments
                if "#" in value:
                    value = value.split("#")[0].strip()
                if value and "," in value:
                    try:
                        pattern, pos = value.split(",", 1)
                        pattern = pattern.strip().lower()
                        pos = int(pos.strip())
                        if pattern and pos >= 0:  # Validate
                            custom_position.append((pattern, pos))
                    except (ValueError, IndexError):
                        print(f"⚠️  Invalid position filter in config: {key} = {value}")
                        continue

        if custom_excluded or custom_position:
            print(
                f"✓ Loaded {len(custom_excluded)} custom substring filters and {len(custom_position)} position filters"
            )

        # Return immutable tuples to prevent mutation
        return (tuple(custom_excluded), tuple(custom_position))

    except Exception as e:
        print(f"⚠️  Could not load custom filters from {config_path}: {e}")
        print("   Using default filters only.")
        return ([], [])


def path_finder(game_path):
    game_path = Path(game_path).resolve()  # Resolve to absolute path
    # You may have to change games x permissions level to continue.
    if not game_path.exists():
        print(f"Game folder '{game_path}' not found!")
        return None

    # Find executable files by extension:
    # 1. *.exe files (Windows games via Wine)
    # 2. *.sh scripts (shell scripts)
    # 3. *.py scripts (Python games)
    executable_files = []

    # Find .exe files (including symlinks)
    for exe in game_path.rglob("*.exe"):
        executable_files.append(exe)

    # Find .sh scripts (including symlinks)
    for sh in game_path.rglob("*.sh"):
        executable_files.append(sh)

    # Find .py scripts (including symlinks)
    for py in game_path.rglob("*.py"):
        executable_files.append(py)

    if not executable_files:
        print("No game executables found in game folder.")
        return None

    # DEFAULT filters (immutable - use tuple for safety)
    DEFAULT_EXCLUDED_PATTERNS = (
        "crash",
        "uninstall",
        "setup",
        "config",
        "launcher",
        "update",
        "installer",
        "unity",
        "unreal",
        "helper",
        "reporter",
    )

    # DEFAULT position patterns (immutable)
    DEFAULT_POSITION_FILTERS = (
        (
            "unins",
            0,
        ),  # Inno Setup uninstallers: unins000.exe, unins001.exe (industry standard)
        ("setup", 0),  # Common installer: setup.exe (default for most installer tools)
        ("install", 0),  # Installers: install.exe, installer.exe
        (
            "vcredist",
            0,
        ),  # Visual C++ redistributables: vcredist_x64.exe, vcredist_x86.exe
        ("ue4prereq", 0),  # Unreal Engine prerequisites: UE4PrereqSetup_x64.exe
        ("directx", 0),  # DirectX installers: DirectX_Setup.exe
    )

    # Load user-defined custom filters (if config exists)
    custom_excluded, custom_position = _load_custom_filters()

    # Combine filters (creates new lists, doesn't mutate originals)
    excluded_patterns = list(DEFAULT_EXCLUDED_PATTERNS) + list(custom_excluded)
    position_filters = list(DEFAULT_POSITION_FILTERS) + list(custom_position)

    filtered_exe_files = []
    for exe in executable_files:
        exe_lower = exe.name.lower()
        should_exclude = False

        # Check substring patterns
        if any(pattern in exe_lower for pattern in excluded_patterns):
            should_exclude = True

        # Check position-based patterns
        for pattern, start_pos in position_filters:
            if len(exe_lower) > start_pos + len(pattern):
                if exe_lower[start_pos : start_pos + len(pattern)] == pattern:
                    should_exclude = True
                    break

        if not should_exclude:
            filtered_exe_files.append(exe)

    # If filtering removed all files, use original list
    if not filtered_exe_files:
        filtered_exe_files = executable_files

    # Smart prioritization: executable matching folder name goes first
    def prioritize_exe(exe):
        folder_name = exe.parent.name.lower()
        exe_name = exe.stem.lower()

        # Priority 1: Exact match with folder name
        if exe_name == folder_name:
            return (0, -exe.stat().st_size)
        # Priority 2: Folder name is in exe name
        elif folder_name in exe_name:
            return (1, -exe.stat().st_size)
        # Priority 3: Sort by file size (larger = likely main game)
        else:
            return (2, -exe.stat().st_size)

    filtered_exe_files.sort(key=prioritize_exe)

    # Auto-select the best match (first after priority sort)
    if filtered_exe_files:
        if len(filtered_exe_files) > 1:
            print("\nMultiple game executables found:")
            for idx, exe in enumerate(filtered_exe_files, 1):
                size_mb = exe.stat().st_size / (1024 * 1024)
                marker = " <-- selected" if idx == 1 else ""
                print(f"  {idx}) {exe} ({size_mb:.1f} MB){marker}")
            print(f"\nAuto-selected best match: {filtered_exe_files[0].name}")
        return filtered_exe_files[0]

    return None


def delete_last_screenshot(screenshots_dir="screenshots"):
    """Delete the most recent screenshot from the screenshots directory.

    Args:
        screenshots_dir: Path to screenshots directory (default: "screenshots")

    Returns:
        bool: True if screenshot was deleted, False otherwise
    """
    screenshots_path = Path(screenshots_dir)

    if not screenshots_path.exists():
        print(f"Screenshots directory '{screenshots_dir}' not found!")
        return False

    # Get PNG files matching screenshot pattern
    screenshot_files = list(screenshots_path.glob("screenshot_*.png"))

    if not screenshot_files:
        print("No screenshots found to delete.")
        return False

    # Skip deletion if only one screenshot exists
    if len(screenshot_files) == 1:
        print("Only one screenshot exists, skipping deletion.")
        return False

    # Sort by filename (contains timestamp: screenshot_YYYYMMDD_HHMMSS.png)
    # This is more reliable than modification time
    screenshot_files.sort()
    last_screenshot = screenshot_files[-1]

    # Delete the file
    try:
        last_screenshot.unlink()
        print(f"✓ Deleted most recent screenshot: {last_screenshot.name}")
        return True
    except FileNotFoundError:
        print(f"Screenshot file disappeared: {last_screenshot.name}")
        return False
    except PermissionError:
        print(f"Permission denied deleting: {last_screenshot.name}")
        return False
    except Exception as e:
        print(f"Unexpected error deleting screenshot: {e}")
        return False
