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
from functions import launch_label_studio, get_title, path_finder, delete_last_screenshot
from pathlib import Path
import subprocess
import os
import time
import sys
import shutil
import configparser
from conf.config_parser import main_conf as config

def update_llm_config(provider):
    """Update conf/main_conf.ini with chosen LLM provider and default model.

    Args:
        provider: LLM package name ('anthropic', 'openai', or 'google-generativeai')
    """
    config_file = Path(__file__).parent / "conf" / "main_conf.ini"

    # Map package names to provider names and default models
    provider_map = {
        'anthropic': {
            'provider': 'anthropic',
            'model': 'claude-sonnet-4-5-20250514'
        },
        'openai': {
            'provider': 'openai',
            'model': 'gpt-4'
        },
        'google-genai': {
            'provider': 'google',
            'model': 'gemini-2.5-flash'
        }
    }

    if provider not in provider_map:
        print(f"⚠️  Unknown provider: {provider}, config not updated")
        return

    try:
        parser = configparser.ConfigParser()
        parser.read(config_file)

        # Update LLM section
        if not parser.has_section('LLM'):
            parser.add_section('LLM')

        parser.set('LLM', 'llm_provider', provider_map[provider]['provider'])
        parser.set('LLM', 'llm_model', provider_map[provider]['model'])

        # Preserve max_tokens_search if it exists
        if not parser.has_option('LLM', 'max_tokens_search'):
            parser.set('LLM', 'max_tokens_search', '4096')

        # Write back to file
        with open(config_file, 'w') as f:
            parser.write(f)

        print(f"✓ Updated config: {provider_map[provider]['provider']} / {provider_map[provider]['model']}")
    except Exception as e:
        print(f"⚠️  Could not update config file: {e}")

def check_python_packages():
    """Check if required Python packages are installed.

    Returns:
        dict: {package_name: (is_installed: bool, error_msg: str|None)}
    """
    results = {}

    # Core dependencies (always required)
    core_packages = {
        'pynput': 'pynput',
        'PIL': 'Pillow',
        'mss': 'mss',
        'dotenv': 'python-dotenv',
        'yaml': 'pyyaml',
    }

    # Optional: YOLO training dependencies
    training_packages = {
        'ultralytics': 'ultralytics',
        'torch': 'torch',
    }

    # Optional: LLM provider dependencies (user chooses which)
    llm_packages = {
        'anthropic': 'anthropic',
        'openai': 'openai',
        'google.genai': 'google-genai',
    }

    # Check core packages
    for import_name, pip_name in core_packages.items():
        try:
            __import__(import_name)
            results[pip_name] = (True, None)
        except ImportError as e:
            results[pip_name] = (False, str(e))

    # Check training packages
    for import_name, pip_name in training_packages.items():
        try:
            __import__(import_name)
            results[pip_name] = (True, None)
        except ImportError as e:
            results[pip_name] = (False, str(e))

    # Check LLM packages
    for import_name, pip_name in llm_packages.items():
        try:
            __import__(import_name)
            results[pip_name] = (True, None)
        except ImportError as e:
            results[pip_name] = (False, str(e))

    return results

def check_system_tools():
    """Check if required system tools are available in PATH.

    Returns:
        dict: {tool_name: (is_found: bool, path_or_error: str)}
    """
    results = {}

    # Platform-specific tools
    if sys.platform == "darwin":
        # macOS: wine, flameshot (recommended), label-studio
        # Note: screencapture is built-in fallback
        tools = ['wine', 'flameshot', 'label-studio']
    elif sys.platform == "win32":
        # Windows: label-studio only (no wine needed, screenshot built-in)
        tools = ['label-studio']
    else:
        # Linux: all tools
        tools = ['wine', 'flameshot', 'label-studio']

    for tool in tools:
        path = shutil.which(tool)
        if path:
            results[tool] = (True, path)
        else:
            results[tool] = (False, f'{tool} not found in PATH')

    return results

def install_python_package(package):
    """Install a Python package using pip.

    Args:
        package: Package name to install

    Returns:
        bool: True if installation succeeded, False otherwise
    """
    print(f"\n  Installing {package}...")
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'install', package],
            capture_output=True,
            text=True,
            check=True
        )
        print(f"  ✓ {package} installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Failed to install {package}: {e}")
        return False

def validate_dependencies():
    """Comprehensive dependency check at startup.

    Collects all missing dependencies and reports them together.
    Offers to install missing Python packages.

    Returns:
        bool: True if all dependencies are met, False otherwise
    """
    python_packages = check_python_packages()
    system_tools = check_system_tools()

    # Categorize missing packages
    core_packages = ['pynput', 'python-dotenv', 'pyyaml']
    training_packages = ['ultralytics', 'torch']
    llm_packages = ['anthropic', 'openai', 'google-generativeai']

    missing_core = [pkg for pkg in core_packages if pkg in python_packages and not python_packages[pkg][0]]
    missing_training = [pkg for pkg in training_packages if pkg in python_packages and not python_packages[pkg][0]]
    missing_llm = [pkg for pkg in llm_packages if pkg in python_packages and not python_packages[pkg][0]]
    missing_tools = [tool for tool, (found, _) in system_tools.items() if not found]

    # Check if at least one LLM provider is already installed
    installed_llm = [pkg for pkg in llm_packages if pkg in python_packages and python_packages[pkg][0]]
    has_llm_provider = len(installed_llm) > 0

    # If everything is installed, we're good
    if not missing_core and not missing_training and not missing_llm and not missing_tools:
        return True

    print("\n" + "="*60)
    print("DEPENDENCY CHECK")
    print("="*60)

    # Handle missing core packages
    if missing_core:
        print("\n⚠️  Missing CORE dependencies (required):")
        for pkg in missing_core:
            print(f"  - {pkg}")

        choice = input("\nInstall missing core packages? (Y/N): ").strip().upper()
        if choice == "Y":
            print("\nInstalling core packages...")
            failed = []
            for pkg in missing_core:
                if not install_python_package(pkg):
                    failed.append(pkg)

            if failed:
                print(f"\n✗ Failed to install: {', '.join(failed)}")
                print("Please install these manually and try again.")
                return False
            print("\n✓ Core packages installed successfully!")
        else:
            print("\nCannot proceed without core dependencies.")
            return False

    # Handle missing training packages
    if missing_training:
        print("\n⚠️  Missing TRAINING dependencies (optional for YOLO training):")
        for pkg in missing_training:
            print(f"  - {pkg}")

        choice = input("\nInstall training packages? (Y/N): ").strip().upper()
        if choice == "Y":
            print("\nInstalling training packages (this may take a while)...")
            for pkg in missing_training:
                install_python_package(pkg)
            print("\n✓ Training packages installation complete!")
        else:
            print("Skipping training packages. YOLO training will not be available.")

    # Handle missing LLM packages (skip if at least one is already installed)
    if missing_llm and not has_llm_provider:
        print("\n⚠️  Missing LLM provider dependencies (optional for bot generation):")
        print("  - anthropic (Claude)")
        print("  - openai (GPT)")
        print("  - google-generativeai (Gemini)")

        print("\nWhich LLM providers do you want to install?")
        print("  1) Anthropic (Claude) - Recommended")
        print("  2) OpenAI (GPT)")
        print("  3) Google (Gemini)")
        print("  4) All providers")
        print("  5) Skip LLM installation")

        choice = input("\nEnter choice (1-5): ").strip()

        llm_to_install = []
        selected_provider = None  # Track which single provider was chosen

        if choice == "1" and 'anthropic' in missing_llm:
            llm_to_install = ['anthropic']
            selected_provider = 'anthropic'
        elif choice == "2" and 'openai' in missing_llm:
            llm_to_install = ['openai']
            selected_provider = 'openai'
        elif choice == "3" and 'google-generativeai' in missing_llm:
            llm_to_install = ['google-generativeai']
            selected_provider = 'google-generativeai'
        elif choice == "4":
            llm_to_install = missing_llm
            # For "all providers", default to anthropic
            selected_provider = 'anthropic' if 'anthropic' in missing_llm else missing_llm[0]
        elif choice == "5":
            print("Skipping LLM packages. Bot generation will not be available.")
        else:
            print("Invalid choice. Skipping LLM installation.")

        if llm_to_install:
            print(f"\nInstalling LLM packages: {', '.join(llm_to_install)}...")
            for pkg in llm_to_install:
                install_python_package(pkg)
            print("\n✓ LLM packages installation complete!")

            # Update config to match the chosen provider
            if selected_provider:
                update_llm_config(selected_provider)
    elif has_llm_provider:
        print(f"\n✓ LLM provider already installed: {', '.join(installed_llm)}")

        # Check if config matches installed provider
        # Priority: anthropic > openai > google
        primary_provider = None
        if 'anthropic' in installed_llm:
            primary_provider = 'anthropic'
        elif 'openai' in installed_llm:
            primary_provider = 'openai'
        elif 'google-generativeai' in installed_llm:
            primary_provider = 'google-generativeai'

        # Verify config matches
        if primary_provider:
            provider_map = {
                'anthropic': 'anthropic',
                'openai': 'openai',
                'google-generativeai': 'google'
            }
            expected_provider = provider_map[primary_provider]

            # Read current config
            try:
                from conf.config_parser import main_conf
                current_provider = main_conf.get('LLM_PROVIDER', '')

                if current_provider != expected_provider:
                    print(f"⚠️  Config mismatch: installed={primary_provider}, config={current_provider}")
                    print(f"   Updating config to match installed provider...")
                    update_llm_config(primary_provider)
            except Exception as e:
                print(f"⚠️  Could not verify config: {e}")

    # Handle missing system tools
    if missing_tools:
        print("\n⚠️  Missing SYSTEM tools:")
        for tool in missing_tools:
            if tool == 'wine':
                print(f"  - wine:")
                if sys.platform == "darwin":
                    print("      macOS: brew install wine-stable")
                    print("      Alternative: Install CrossOver or PlayOnMac")
                else:
                    print("      Ubuntu/Debian: sudo apt install wine")
                    print("      Arch: sudo pacman -S wine")
                    print("      Fedora: sudo dnf install wine")
            elif tool == 'flameshot':
                print(f"  - flameshot (screenshot tool):")
                if sys.platform == "darwin":
                    print("      macOS: brew install flameshot")
                    print("      Note: Built-in screencapture works as fallback")
                else:
                    print("      Ubuntu/Debian: sudo apt install flameshot")
                    print("      Arch: sudo pacman -S flameshot")
                    print("      Fedora: sudo dnf install flameshot")
            elif tool == 'label-studio':
                print(f"  - label-studio: pip install label-studio")
                if input("\nInstall label-studio? (Y/N): ").strip().upper() == "Y":
                    install_python_package('label-studio')

        print("\nPlease install missing system tools and restart the program.")
        return False

    print("\n" + "="*60)
    print("✓ All required dependencies are installed!")
    print("="*60)
    return True

def main():
   # Validate all dependencies at startup
   if not validate_dependencies():
       sys.exit(1)

   # Welcome message
   print("\n" + "="*60)
   print("          PAPER ENGINE - Pre-Release v1.0")
   print("="*60)
   print("\nThank you for using Paper Engine!")
   print("This is a pre-release version - feedback is appreciated.")
   print("\nPaper Engine: Game automation and computer vision toolkit")
   print("Repository: https://github.com/crased/paper_engine")
   print("\n" + "="*60 + "\n")

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
           # Determine launch method based on file type
           exe_path_obj = Path(exe_path)

           # Security check: Validate file is within game/ directory
           game_dir = Path("game/").resolve()
           try:
               resolved_path = exe_path_obj.resolve()
               if not resolved_path.is_relative_to(game_dir):
                   print(f"\nSECURITY ERROR: File is outside game/ directory: {exe_path}")
                   print("This could be a symlink attack. Only run files within game/")
                   sys.exit(1)
           except (ValueError, OSError) as e:
               print(f"\nSECURITY ERROR: Cannot validate file path: {e}")
               sys.exit(1)

           # Security check: Ensure file is readable and exists
           if not exe_path_obj.exists():
               print(f"\nERROR: Game file does not exist: {exe_path}")
               sys.exit(1)

           if not os.access(exe_path_obj, os.R_OK):
               print(f"\nERROR: Cannot read game file: {exe_path}")
               print("Check file permissions.")
               sys.exit(1)

           if exe_path_obj.suffix.lower() == '.exe':
               # Windows executable - use Wine
               game_process = subprocess.Popen(["wine", "explorer", f"/desktop=game,{config.WINE_DESKTOP_RESOLUTION}", str(exe_path)])
               print(f"\nStarting game via Wine: {exe_path}")
           elif exe_path_obj.suffix.lower() == '.sh':
               # Shell script - use bash (warn if not executable)
               if not os.access(exe_path_obj, os.X_OK):
                   print(f"\n⚠️  WARNING: Script is not marked as executable: {exe_path}")
                   print(f"Consider running: chmod +x {exe_path}")
               game_process = subprocess.Popen(["bash", str(exe_path)])
               print(f"\nStarting game script: {exe_path}")
           elif exe_path_obj.suffix.lower() == '.py':
               # Python script - use python (warn if not executable)
               if not os.access(exe_path_obj, os.X_OK):
                   print(f"\n⚠️  WARNING: Script is not marked as executable: {exe_path}")
                   print(f"Consider running: chmod +x {exe_path}")
               game_process = subprocess.Popen(["python", str(exe_path)])
               print(f"\nStarting Python game: {exe_path}")

           print("Screenshots will be captured every 5 seconds...")
       except FileNotFoundError as e:
           print(f"\nERROR: Executable or required tool not found: {e}")
           if exe_path_obj.suffix.lower() == '.exe':
               print("Install wine:")
               if sys.platform == "darwin":
                   print("  macOS: brew install wine-stable")
                   print("  Alternative: Install CrossOver or PlayOnMac")
               else:
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
           if exe_path_obj.suffix.lower() == '.exe':
               print("Check that the .exe file is valid and wine is properly configured.")
           else:
               print("Check that the file has execute permissions and is a valid executable.")
           sys.exit(1)

       # Wait for game to initialize
       time.sleep(config.GAME_INITIALIZATION_WAIT)

       # Detect game window geometry once after init (Sway only)
       window_geometry = find_wine_window()
       if window_geometry:
           print(f"✓ Locked onto game window: {window_geometry}")
       else:
           print("✓ Using full screen capture mode")

       # Step 1.5: Screenshot loop — runs while game is active
       screenshots_dir = create_screenshots_directory()
       imgs_before = len([f for f in os.listdir(screenshots_dir) if f.endswith('.png')])
       screenshots_taken = 0
       SCREENSHOT_SOFT_LIMIT = 201

       while game_process.poll() is None:
           time.sleep(config.SCREENSHOT_INTERVAL)
           take_screenshot(screenshots_dir, window_geometry)
           screenshots_taken += 1

           # Soft stop at 201 screenshots
           if screenshots_taken >= SCREENSHOT_SOFT_LIMIT:
               print(f"\n⚠️  Reached {SCREENSHOT_SOFT_LIMIT} screenshots.")
               print(f"Total screenshots in folder: {len([f for f in os.listdir(screenshots_dir) if f.endswith('.png')])}")
               continue_choice = input("Continue capturing? (Y/N): ").strip().upper()
               if continue_choice != "Y":
                   print("Stopping screenshot capture. Closing game...")
                   game_process.terminate()
                   break
               else:
                   print("Continuing screenshot capture...")
                   screenshots_taken = 0  # Reset counter for next batch

       imgs_after = len([f for f in os.listdir(screenshots_dir) if f.endswith('.png')])

       # Delete the last screenshot (most recent)
       delete_last_screenshot(screenshots_dir)

       imgs_final = len([f for f in os.listdir(screenshots_dir) if f.endswith('.png')])
       print(f"\nGame process ended. Added {imgs_final - imgs_before} images. Total: {imgs_final}")
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

   # --- Step 3.5: Test trained model (optional) ---
   time.sleep(0.5)
   choice = input("\nTest trained model on screenshots? (Y/N): ").strip().upper()
   if choice == "Y":
       import test_model
       # Use default parameters: best.pt model, screenshots directory, conf=0.25
       test_model.test_model()
   else:
       print("Skipping model testing.")

   # --- Step 4: Generate bot script (optional) ---
   time.sleep(0.5)
   choice = input("\nGenerate AI bot script? (Y/N): ").strip().upper()
   if choice == "Y":
       from generate_bot_script import generate_bot_script, save_bot_script, read_controls_from_config, search_game_controls, save_controls_to_config

       # Try to read controls from config
       print(f"\nReading controls configuration for {game_title}...")
       existing_controls = read_controls_from_config(game_title)

       if existing_controls:
           print("✓ Found existing controls configuration")
           print("Improving existing controls with latest information...")
           controls_info = search_game_controls(game_title, existing_controls)

           if controls_info:
               config_path = save_controls_to_config(game_title, game_path, controls_info)
               print(f"✓ Controls improved and saved to: {config_path}")
           else:
               print("\n✗ Failed to improve controls")
               print("  Using existing controls from config")
               controls_info = existing_controls
       else:
           print("No controls configuration found. Searching automatically...")
           print(f"\nSearching web for {game_title} controls...")
           controls_info = search_game_controls(game_title)

           if controls_info:
               config_path = save_controls_to_config(game_title, game_path, controls_info)
               print(f"✓ Controls saved to: {config_path}")
           else:
               print("\n✗ Failed to retrieve game controls")
               print("  You can manually create a controls config file in conf/")
               print("Skipping bot generation.")
               controls_info = None

       if controls_info:
           print("✓ Controls loaded")
           print("\nGenerating Python bot script...")
           print("This may take 1-2 minutes...\n")

           script_code = generate_bot_script(game_title, controls_info)

           if script_code:
               script_path = save_bot_script(game_title, script_code)
               print(f"\n" + "="*60)
               print("✓ BOT SCRIPT GENERATED SUCCESSFULLY!")
               print("="*60)
               print(f"Script saved to: {script_path}")
               print(f"\nTo run your bot:")
               print(f"  1. Start {game_title}")
               print(f"  2. Run: python {script_path}")
               print(f"  3. Press ESC to stop the bot")
               print("="*60)
           else:
               print("\n✗ Failed to generate bot script")
               print("  Check your API key configuration in .env")
   else:
       print("Skipping bot generation.")









































































































































if __name__ == "__main__":
    main()