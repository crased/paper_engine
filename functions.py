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
#you may have to change games x permisions level to continue.
   if not game_path.exists():
     print(f"Game folder '{game_path}' not found!")
     return None   
   
   exe_files = list(game_path.glob("*.exe"))
   
   if not exe_files:
       print("No .exe files found in game folder.")
       return None
   exe_path = exe_files[0]
   return exe_path # Assuming the first .exe file is the game executables

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




