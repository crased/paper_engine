#import neccesary files
#from screenshot import take_screenshot, create_screenshots_directory
#from bot_script import game_bot
#import game
#then run bot_script on supported game this is something i want other people to build.
#for showcase though ill do like doom or something.
#while bot_script  == True.
#activate screengrab.
#if screengrab.py == true.
#set delay to 500ms. this is the delay for screen grab activation.
#record time should be set in screengrab.py.
#after screengrab time == 500.
#return screengrab to footage folder.
from screencapture import take_screenshot, create_screenshots_directory, find_wine_window
from functions import launch_label_studio, get_title, path_finder
from pathlib import Path
import subprocess
import os
import time
import sys
import shutil
from conf.config_parser import main_conf as config

def check_python_packages():
    """Check if required Python packages are installed.

    Returns:
        dict: {package_name: (is_installed: bool, error_msg: str|None)}
    """
    results = {}

    # Check pynput
    try:
        import pynput
        results['pynput'] = (True, None)
    except ImportError as e:
        results['pynput'] = (False, str(e))

    # Check PIL (Pillow)
    try:
        import PIL
        results['PIL'] = (True, None)
    except ImportError as e:
        results['PIL'] = (False, str(e))

    return results

def check_system_tools():
    """Check if required system tools are available in PATH.

    Returns:
        dict: {tool_name: (is_found: bool, path_or_error: str)}
    """
    results = {}
    tools = ['wine', 'flameshot', 'label-studio']

    for tool in tools:
        path = shutil.which(tool)
        if path:
            results[tool] = (True, path)
        else:
            results[tool] = (False, f'{tool} not found in PATH')

    return results

def validate_dependencies():
    """Comprehensive dependency check at startup.

    Collects all missing dependencies and reports them together.

    Returns:
        bool: True if all dependencies are met, False otherwise
    """
    python_packages = check_python_packages()
    system_tools = check_system_tools()

    missing_packages = [pkg for pkg, (installed, _) in python_packages.items() if not installed]
    missing_tools = [tool for tool, (found, _) in system_tools.items() if not found]

    if not missing_packages and not missing_tools:
        return True

    # Print comprehensive error message
    print("\nERROR: Missing required dependencies:\n")

    if missing_packages:
        print("Python packages:")
        for pkg in missing_packages:
            print(f"  - {pkg}: pip install {pkg if pkg != 'PIL' else 'Pillow'}")
        print()

    if missing_tools:
        print("System tools:")
        for tool in missing_tools:
            if tool == 'wine':
                print(f"  - wine:")
                print("      Ubuntu/Debian: sudo apt install wine")
                print("      Arch: sudo pacman -S wine")
                print("      Fedora: sudo dnf install wine")
            elif tool == 'flameshot':
                print(f"  - flameshot:")
                print("      Ubuntu/Debian: sudo apt install flameshot")
                print("      Arch: sudo pacman -S flameshot")
                print("      Fedora: sudo dnf install flameshot")
            elif tool == 'label-studio':
                print(f"  - label-studio: pip install label-studio")
        print()

    print("Please install missing dependencies and try again.")
    return False

def main():
   # Validate all dependencies at startup
   if not validate_dependencies():
       sys.exit(1)

   # Set up environment for Label Studio
   env = os.environ.copy()
   env.update(config.LABEL_STUDIO_ENV)

   # Validate game directory exists
   game_path = config.GAME_PATH
   if not Path(game_path).exists():
       if config.AUTO_CREATE_DIRECTORIES:
           print(f"\nCreating game directory: {game_path}")
           Path(game_path).mkdir(parents=True, exist_ok=True)
           print(f"Please place a Windows game executable (.exe) in {game_path}")
           sys.exit(1)
       else:
           print(f"\nERROR: Game directory '{game_path}' does not exist.")
           sys.exit(1)

   # Find game executable
   exe_path = path_finder(game_path)
   if exe_path is None:
       print("\nERROR: No .exe files found in game/ directory.")
       print("Please place a Windows game executable (.exe) in the game/ folder.")
       sys.exit(1)

   game_title = get_title(game_path)

   # --- Step 1: Launch game (optional) ---
   time.sleep(0.5)
   choice = input("\nLaunch game? (Y/N): ").strip().upper()
   if choice == "Y":
       try:
           game_process = subprocess.Popen(["wine", "explorer", f"/desktop=game,{config.WINE_DESKTOP_RESOLUTION}", str(exe_path)])
           print(f"\nStarting game: {exe_path}")
           print("Screenshots will be captured every 5 seconds...")
       except FileNotFoundError:
           print("\nERROR: Wine executable not found.")
           print("Install wine:")
           print("  Ubuntu/Debian: sudo apt install wine")
           print("  Arch: sudo pacman -S wine")
           print("  Fedora: sudo dnf install wine")
           sys.exit(1)
       except PermissionError:
           print(f"\nERROR: Permission denied when executing: {exe_path}")
           print(f"Try: chmod +x {exe_path}")
           sys.exit(1)
       except subprocess.SubprocessError as e:
           print(f"\nERROR: Failed to start game process: {e}")
           print(f"Game: {exe_path}")
           print("Check that the .exe file is valid and wine is properly configured.")
           sys.exit(1)

       # Wait for game to initialize
       time.sleep(config.GAME_INITIALIZATION_WAIT)

       # Detect game window geometry once after init
       window_geometry = find_wine_window()
       if window_geometry:
           print(f"Locked onto game window: {window_geometry}")
       else:
           print("Could not detect game window, falling back to full screen capture.")

       # Step 1.5: Screenshot loop — runs while game is active
       screenshots_dir = create_screenshots_directory()
       imgs_before = len([f for f in os.listdir(screenshots_dir) if f.endswith('.png')])

       while game_process.poll() is None:
           time.sleep(config.SCREENSHOT_INTERVAL)
           take_screenshot(screenshots_dir, window_geometry)

       imgs_after = len([f for f in os.listdir(screenshots_dir) if f.endswith('.png')])
       print(f"\nGame process ended. Added {imgs_after - imgs_before} images. Total: {imgs_after}")
   else:
       print("Skipping game launch.")

   # --- Step 2: Launch Label Studio (optional) ---
   time.sleep(0.5)
   choice = input("\nLaunch Label Studio for annotation? (Y/N): ").strip().upper()
   if choice == "Y":
       launch_label_studio(env)
   else:
       print("Skipping Label Studio.")

   # --- Step 3: Train YOLO model (optional) ---
   time.sleep(0.5)
   choice = input("\nTrain YOLO model? (Y/N): ").strip().upper()
   if choice == "Y":
       import training_model
       training_model.main()
   else:
       print("Skipping model training.")









































































































































if __name__ == "__main__":
    main()