import os
import game
import subprocess
from pathlib import Path

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
      



