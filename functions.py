import os
import sys
import subprocess
from pathlib import Path
from pynput import keyboard
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
        print("\nERROR: label-studio command not found.")
        print("Install with: pip install label-studio")
        print("Then ensure it's in your PATH.")
        sys.exit(1)
    except subprocess.SubprocessError as e:
        print(f"\nERROR: Failed to start Label Studio: {e}")
        print("Try running manually: label-studio start --port 8080")
        sys.exit(1)

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
           if file.suffix in ['.exe', '.sh', '.py'] or (not file.suffix and os.access(file, os.X_OK)):
               return file.stem if file.suffix else file.name

   # Fallback: use directory name
   return game_path_obj.name
def path_finder(game_path):
   game_path = Path("game/").resolve()  # Resolve to absolute path
   # You may have to change games x permissions level to continue.
   if not game_path.exists():
     print(f"Game folder '{game_path}' not found!")
     return None

   # Find executable files by multiple methods:
   # 1. *.exe files (Windows games via Wine)
   # 2. *.sh scripts (shell scripts)
   # 3. *.py scripts (Python games)
   # 4. Files with execute permission (Linux native executables)
   executable_files = []

   # Security: Validate that all found files are within game/ directory
   def is_safe_path(file_path):
       """Ensure file is within game/ directory (prevent symlink attacks)."""
       try:
           resolved = file_path.resolve()
           return resolved.is_relative_to(game_path)
       except (ValueError, OSError):
           return False

   # Find .exe files
   for exe in game_path.rglob("*.exe"):
       if is_safe_path(exe):
           executable_files.append(exe)

   # Find .sh scripts
   for sh in game_path.rglob("*.sh"):
       if is_safe_path(sh):
           executable_files.append(sh)

   # Find .py scripts
   for py in game_path.rglob("*.py"):
       if is_safe_path(py):
           executable_files.append(py)

   # Find Linux native executables (files with execute permission, no extension)
   for file in game_path.rglob("*"):
       if file.is_file() and os.access(file, os.X_OK) and not file.suffix:
           if is_safe_path(file):
               executable_files.append(file)

   if not executable_files:
       print("No game executables found in game folder.")
       return None

   # Filter out common non-game executables
   excluded_patterns = [
       'crash', 'uninstall', 'setup', 'config', 'launcher', 'update',
       'installer', 'unity', 'unreal', 'helper', 'reporter'
   ]

   filtered_exe_files = []
   for exe in executable_files:
       exe_lower = exe.name.lower()
       if not any(pattern in exe_lower for pattern in excluded_patterns):
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

   # If only one executable found after filtering, auto-select it
   if len(filtered_exe_files) == 1:
       return filtered_exe_files[0]

   # If multiple executables, let user choose
   print("\nMultiple game executables found:")
   for idx, exe in enumerate(filtered_exe_files, 1):
       size_mb = exe.stat().st_size / (1024 * 1024)
       print(f"  {idx}) {exe} ({size_mb:.1f} MB)")

   while True:
       try:
           choice = input(f"\nSelect game (1-{len(filtered_exe_files)}): ").strip()
           selected_idx = int(choice) - 1
           if 0 <= selected_idx < len(filtered_exe_files):
               return filtered_exe_files[selected_idx]
           else:
               print(f"Please enter a number between 1 and {len(filtered_exe_files)}")
       except ValueError:
           print("Please enter a valid number")
       except KeyboardInterrupt:
           print("\nCancelled.")
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




