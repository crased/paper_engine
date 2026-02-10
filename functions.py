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
   for file in Path(game_path).iterdir():
     if file.name.endswith(".exe"):
       title = file.name.rstrip(".exe").strip()
   return title
def path_finder(game_path):
   game_path = Path("game/")
   # You may have to change games x permissions level to continue.
   if not game_path.exists():
     print(f"Game folder '{game_path}' not found!")
     return None

   # Recursively find all .exe files in game/ directory and subdirectories
   exe_files = list(game_path.rglob("*.exe"))

   if not exe_files:
       print("No .exe files found in game folder.")
       return None

   # If only one .exe found, auto-select it
   if len(exe_files) == 1:
       return exe_files[0]

   # If multiple .exe files, let user choose
   print("\nMultiple game executables found:")
   for idx, exe in enumerate(exe_files, 1):
       print(f"  {idx}) {exe}")

   while True:
       try:
           choice = input(f"\nSelect game (1-{len(exe_files)}): ").strip()
           selected_idx = int(choice) - 1
           if 0 <= selected_idx < len(exe_files):
               return exe_files[selected_idx]
           else:
               print(f"Please enter a number between 1 and {len(exe_files)}")
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




