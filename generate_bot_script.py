import os
import sys
from pathlib import Path
from functions import get_title, path_finder
from conf.config_parser import main_conf as config


def create_env_file_if_missing():
    """Create .env file if it doesn't exist"""
    env_path = Path(".env")

    if not env_path.exists():
        print("\nNo .env file found. Creating one...")
        with open(env_path, "w") as f:
            f.write("# Paper Engine - Environment Variables\n")
            f.write("# Default: Google Gemini (Free tier available)\n")
            f.write(
                "# Get your free API key at: https://aistudio.google.com/apikey\n\n"
            )
            f.write("API_KEY=your-api-key-here\n\n")
            f.write("# Optional: Switch to advanced models\n")
            f.write("# Anthropic Claude: https://console.anthropic.com/settings/keys\n")
            f.write("# OpenAI GPT: https://platform.openai.com/api-keys\n")
            f.write("# Configure provider in conf/main_conf.ini [LLM] section\n")
        print(f"Created .env file at: {env_path.absolute()}")
        print("\n" + "=" * 60)
        print("API KEY SETUP")
        print("=" * 60)
        print("\nDefault: Google Gemini (FREE)")
        print("  1. Visit: https://aistudio.google.com/apikey")
        print("  2. Click 'Create API Key'")
        print("  3. Copy your key")
        print(f"  4. Edit {env_path.absolute()} and replace 'your-api-key-here'")
        print("\nOptional: Use advanced models (Anthropic/OpenAI)")
        print("  - See README.md for instructions")
        print("=" * 60)
        return False
    return True


def _ensure_env_loaded():
    """Load .env file, creating it first if missing. Safe to call multiple times."""
    create_env_file_if_missing()
    from dotenv import load_dotenv

    load_dotenv()


def get_llm_client(provider, api_key):
    """
    Get the appropriate LLM client based on provider.
    Imports each SDK lazily so only the selected provider needs to be installed.

    Args:
        provider: LLM provider name (anthropic, openai, google)
        api_key: API key for the provider

    Returns:
        Client instance for the provider
    """
    if provider.lower() == "anthropic":
        try:
            from anthropic import Anthropic
        except ImportError:
            raise ImportError(
                "anthropic package not installed. Run: pip install anthropic"
            )
        return Anthropic(api_key=api_key)
    elif provider.lower() == "openai":
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")
        return OpenAI(api_key=api_key)
    elif provider.lower() == "google":
        try:
            from google import genai
        except ImportError:
            raise ImportError(
                "google-genai package not installed. Run: pip install google-genai"
            )
        return genai.Client(api_key=api_key)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")


def call_llm(client, provider, model, prompt, max_tokens=4096):
    """
    Call LLM API with unified interface

    Args:
        client: LLM client instance
        provider: Provider name (anthropic, openai, google)
        model: Model name
        prompt: Prompt text
        max_tokens: Maximum tokens to generate

    Returns:
        str: Generated text response
    """
    try:
        if provider.lower() == "anthropic":
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text

        elif provider.lower() == "openai":
            response = client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content

        elif provider.lower() == "google":
            response = client.models.generate_content(model=model, contents=prompt)
            return response.text

        else:
            raise ValueError(f"Unsupported provider: {provider}")

    except Exception as e:
        raise Exception(f"LLM API call failed: {e}")


def search_game_controls(game_title, existing_controls=None):
    """
    Search for game controls using configured LLM
    If existing controls are provided, improves them instead of starting from scratch

    Args:
        game_title: Name of the game
        existing_controls: Optional existing controls text to improve

    Returns:
        str: Game controls information (new or improved)
    """
    _ensure_env_loaded()

    # Get LLM configuration
    llm_provider = config.LLM_PROVIDER
    llm_model = config.LLM_MODEL
    max_tokens = config.MAX_TOKENS_SEARCH

    # Get API key from environment
    api_key = os.environ.get("API_KEY")

    if not api_key or api_key == "your-api-key-here":
        print("\n" + "=" * 60)
        print(f"ERROR: API_KEY Not Configured")
        print("=" * 60)
        print("\nSetup Instructions:")
        print(f"1. Current LLM provider: {llm_provider}")
        if llm_provider.lower() == "anthropic":
            print("   Get API key at: https://console.anthropic.com/settings/keys")
        print("2. Edit .env and add your API key")
        print("3. Use the Configuration dialog in gui.py to set up your provider")
        print("4. Or configure provider in conf/main_conf.ini (llm_provider)")
        print("=" * 60 + "\n")
        return None

    # Get LLM client
    try:
        client = get_llm_client(llm_provider, api_key)
    except Exception as e:
        print(f"ERROR: Failed to initialize {llm_provider} client: {e}")
        return None

    if existing_controls:
        print(
            f"\nImproving existing {game_title} controls using {llm_provider}/{llm_model}..."
        )
        prompt = f"""You are a playtest Research professional. Verify and correct the controls documentation for "{game_title}".

EXISTING CONTROLS:
{existing_controls}

STRICT ACCURACY REQUIREMENTS:
- CRITICAL: Every key binding MUST be 100% accurate to the actual game defaults
- Cross-reference your knowledge with known PC keyboard defaults for "{game_title}"
- If ANY control is uncertain, mark it as [UNCERTAIN] or omit it entirely
- Incorrect information is WORSE than missing information
- Do NOT guess, assume, or approximate
- Only include controls you can verify are correct

Verification checklist:
1. Check each existing key binding is correct for {game_title}
2. Correct any wrong keys with the actual defaults
3. Add only verified missing controls
4. Flag or remove anything uncertain

Output format (factual only, no commentary):
**Movement:** A/D/W/S - [actions]
**Actions:** [Key] - [what it does]
**Special:** [Key combo] - [what it does]
**Menu:** [Key] - [what it does]

VERIFY BEFORE INCLUDING. Accuracy over completeness."""
    else:
        print(
            f"\nSearching for {game_title} controls using {llm_provider}/{llm_model}..."
        )
        prompt = f"""You are a playtest Research professional. Search the web for the EXACT DEFAULT KEYBOARD CONTROLS for "{game_title}" from official documentation, game settings screens, or verified gaming wikis.

STRICT ACCURACY REQUIREMENTS:
- Search for official "{game_title}" PC keyboard control documentation
- Every key binding MUST match the actual default keyboard settings
- Reference: Check game manual, official wiki, or in-game control settings
- If you cannot find verified information, mark as [UNCERTAIN] or omit
- Common mistake: Don't confuse controller buttons with keyboard keys
- Incorrect information is WORSE than missing information

Example of what to search for:
- "{game_title} default keyboard controls"
- "{game_title} PC keyboard bindings"
- "{game_title} keyboard settings"

Output format (factual only):
**Movement:**
A - Move Left
D - Move Right

**Actions:**
Z - Jump (example)
X - Shoot (example)

**Special:**
[verified combo] - [action]

**Menu:**
[verified keys]

SEARCH FIRST. VERIFY. Then provide. Accuracy over completeness."""

    try:
        controls_info = call_llm(client, llm_provider, llm_model, prompt, max_tokens)
        return controls_info

    except Exception as e:
        print(f"ERROR: Failed to search for controls: {e}")
        return None


def save_controls_to_config(game_title, game_path, controls_info):
    """
    Save game controls to .ini config file

    Args:
        game_title: Name of the game
        game_path: Path to game directory
        controls_info: Controls information text

    Returns:
        Path: Path to saved config file
    """
    # Create conf directory if it doesn't exist
    conf_dir = Path("conf")
    conf_dir.mkdir(exist_ok=True)

    # Sanitize game title for filename
    safe_title = "".join(
        c for c in game_title if c.isalnum() or c in (" ", "_")
    ).strip()
    safe_title = safe_title.replace(" ", "_").lower()

    config_path = conf_dir / f"{safe_title}_controls.ini"

    # Write to .ini file
    with open(config_path, "w") as f:
        f.write(f"# ============================================\n")
        f.write(f"# {game_title} - Keyboard Controls\n")
        f.write(f"# ============================================\n")
        f.write(f"#\n")
        f.write(f"# ⚠️  WARNING: AI-GENERATED CONTENT\n")
        f.write(f"# This file was automatically generated by an LLM.\n")
        f.write(f"# LLMs can make mistakes or provide inaccurate information.\n")
        f.write(f"#\n")
        f.write(f"# ✓ ALWAYS verify controls against the actual game\n")
        f.write(f"# ✓ Check in-game settings/controls menu\n")
        f.write(f"# ✓ Edit this file to correct any errors\n")
        f.write(f"#\n")
        f.write(f"# ============================================\n\n")

        f.write(f"[GameInfo]\n")
        f.write(f"game_name = {game_title}\n")
        f.write(f"game_path = {game_path}\n\n")
        f.write(f"[Controls]\n")
        f.write(f"# AI-generated controls (verify accuracy):\n\n")

        # Write controls info as comments for reference
        for line in controls_info.split("\n"):
            f.write(f"# {line}\n")

    return config_path


def read_controls_from_config(game_title):
    """
    Read game controls from .ini config file

    Args:
        game_title: Name of the game

    Returns:
        str: Controls information from .ini file, or None if not found
    """
    conf_dir = Path("conf")
    safe_title = "".join(
        c for c in game_title if c.isalnum() or c in (" ", "_")
    ).strip()
    safe_title = safe_title.replace(" ", "_").lower()
    config_path = conf_dir / f"{safe_title}_controls.ini"

    if not config_path.exists():
        return None

    # Read the controls section
    with open(config_path, "r") as f:
        content = f.read()
        # Extract everything after [Controls]
        if "[Controls]" in content:
            controls_section = content.split("[Controls]")[1]
            # Remove comment markers for cleaner reading
            controls_text = "\n".join(
                line.lstrip("# ") for line in controls_section.split("\n")
            )
            return controls_text.strip()
    return None


def generate_bot_script_part1(
    game_title, controls_info, llm_provider, llm_model, client
):
    """
    Generate Part 1: Core infrastructure (model, capture, keyboard control)

    Args:
        game_title: Name of the game
        controls_info: Game controls information
        llm_provider: LLM provider name
        llm_model: LLM model name
        client: LLM client instance

    Returns:
        str: Part 1 Python code
    """
    print(f"\n[Part 1/2] Generating core infrastructure...")

    prompt = f"""Create the CORE INFRASTRUCTURE for a "{game_title}" bot (Part 1 of 2).

GAME CONTROLS:
{controls_info}

Generate ONLY Part 1 - Core Infrastructure:

1. **Imports and Configuration**:
```python
import time
import sys
import numpy as np
import mss
from pynput.keyboard import Controller, Listener, Key
from ultralytics import YOLO
from typing import List, Dict, Tuple, Optional
```

2. **Configuration Class**:
```python
class Config:
    MODEL_PATH = 'runs/detect/paper_engine_model/weights/best.pt'
    CONFIDENCE_THRESHOLD = 0.5
    SCREEN_REGION = {{"top": 0, "left": 0, "width": 1920, "height": 1080}}
    TARGET_FPS = 30
    FRAME_TIME = 1.0 / TARGET_FPS
```

3. **Detection Data Structure**:
```python
class Detection:
    def __init__(self, class_name, confidence, bbox):
        self.class_name = class_name
        self.confidence = confidence
        self.bbox = bbox  # (x1, y1, x2, y2)
        self.center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
```

4. **YOLODetector Class** - Handles model and detection:
```python
class YOLODetector:
    def __init__(self, model_path, confidence_threshold):
        # Initialize

    def load_model(self) -> bool:
        # Load YOLO model from model_path
        # Return True if successful

    def detect_objects(self, frame: np.ndarray) -> List[Detection]:
        # Run inference on frame
        # Parse results into Detection objects
        # Return list of detections
```

5. **ScreenCapture Class** - Handles screen capture:
```python
class ScreenCapture:
    def __init__(self, region):
        # Initialize mss

    def capture(self) -> Optional[np.ndarray]:
        # Capture screen using mss
        # Convert BGRA to RGB
        # Return numpy array
```

6. **KeyboardController Class** - Handles keyboard control:
Parse game controls from above and create this class:
```python
class KeyboardController:
    def __init__(self):
        self.keyboard = Controller()
        # Define key mappings from game controls
        self.MOVE_LEFT = 'a'  # Parse from controls
        self.MOVE_RIGHT = 'd'
        self.JUMP = 'z'
        self.SHOOT = 'x'
        # ... all other controls

    def press_key(self, key, duration=0.1):
        # Press and release key

    def hold_key(self, key):
        # Press and hold

    def release_key(self, key):
        # Release key

    def tap_key(self, key):
        # Quick tap
```

7. **Emergency Stop Listener**:
```python
class EmergencyStop:
    def __init__(self):
        self.running = True
        self.listener = None

    def setup(self):
        # Create pynput Listener for ESC key
        # Set running=False when ESC pressed

    def start(self):
        # Start listener

    def stop(self):
        # Stop listener
```

Return ONLY this Part 1 code. Keep it under 300 lines. Make it complete and functional.
Include docstrings. NO placeholder comments like "# TODO" or "# Implement this"."""

    try:
        code = call_llm(client, llm_provider, llm_model, prompt, max_tokens=6000)

        if "```python" in code:
            code = code.split("```python")[1].split("```")[0].strip()
        elif "```" in code:
            code = code.split("```")[1].split("```")[0].strip()

        return code
    except Exception as e:
        print(f"ERROR: Failed to generate Part 1: {e}")
        return None


def generate_bot_script_part2(
    game_title, controls_info, llm_provider, llm_model, client
):
    """
    Generate Part 2: Decision logic, actions, and main loop

    Args:
        game_title: Name of the game
        controls_info: Game controls information
        llm_provider: LLM provider name
        llm_model: LLM model name
        client: LLM client instance

    Returns:
        str: Part 2 Python code
    """
    print(f"\n[Part 2/2] Generating decision logic and main loop...")

    prompt = f"""Create the GAME LOGIC AND EXECUTION for a "{game_title}" bot (Part 2 of 2).

This continues from Part 1 which has: YOLODetector, ScreenCapture, KeyboardController, EmergencyStop classes.

GAME CONTROLS:
{controls_info}

Generate ONLY Part 2 - Game Logic:

1. **GameBot Main Class**:
```python
class GameBot:
    def __init__(self):
        # Initialize all components from Part 1
        self.detector = YOLODetector(Config.MODEL_PATH, Config.CONFIDENCE_THRESHOLD)
        self.screen = ScreenCapture(Config.SCREEN_REGION)
        self.keyboard = KeyboardController()
        self.emergency_stop = EmergencyStop()
        self.running = True

    def categorize_detections(self, detections: List[Detection]) -> Dict:
        # Categorize into: enemies, projectiles, items, obstacles, pink_objects
        # Return dict with lists for each category

    def make_decision(self, detections: List[Detection]) -> str:
        # Simple decision tree:
        # - If projectile close → return "dodge"
        # - If enemy detected → return "attack"
        # - If item detected → return "collect"
        # - Else → return "explore"
        # Return action string

    def execute_action(self, action: str):
        # Map action string to keyboard methods:
        # "dodge" → dash
        # "attack" → shoot + move toward enemy
        # "collect" → move toward item
        # "explore" → move right + shoot

        # Use self.keyboard methods to execute

    def run(self):
        # Main game loop:
        # 1. Load model
        # 2. Start emergency stop listener
        # 3. Loop while running:
        #    - Capture screen
        #    - Detect objects
        #    - Make decision
        #    - Execute action
        #    - FPS limiting (sleep)
        # 4. Cleanup on exit
```

2. **Main Execution Block**:
```python
if __name__ == "__main__":
    print("=" * 60)
    print(f"{{game_title}} AI Bot")
    print("=" * 60)
    print("Instructions:")
    print("1. Start the game")
    print("2. Enter a level")
    print("3. Press ESC anytime to stop\\n")

    bot = GameBot()

    try:
        bot.run()
    except KeyboardInterrupt:
        print("\\nBot stopped by user")
    except Exception as e:
        print(f"\\nError: {{e}}")
    finally:
        print("\\nBot shutdown complete")
        sys.exit(0)
```

Return ONLY this Part 2 code. Keep it under 300 lines. Make it complete and functional.
Focus on SIMPLE, WORKING logic. Include docstrings."""

    try:
        code = call_llm(client, llm_provider, llm_model, prompt, max_tokens=8000)

        if "```python" in code:
            code = code.split("```python")[1].split("```")[0].strip()
        elif "```" in code:
            code = code.split("```")[1].split("```")[0].strip()

        return code
    except Exception as e:
        print(f"ERROR: Failed to generate Part 2: {e}")
        return None


def generate_bot_script(game_title, controls_info):
    """
    Generate a complete Python bot script using two-stage LLM generation

    Args:
        game_title: Name of the game
        controls_info: Game controls information

    Returns:
        str: Complete generated Python bot script code
    """
    _ensure_env_loaded()

    # Get LLM configuration
    llm_model = config.LLM_MODEL
    llm_provider = config.LLM_PROVIDER
    api_key = os.environ.get("API_KEY")

    if not api_key or api_key == "your-api-key-here":
        print(f"\nERROR: API_KEY not configured for provider: {llm_provider}")
        print("Edit .env file and add your API key")
        return None

    # Get appropriate client
    try:
        client = get_llm_client(llm_provider, api_key)
    except Exception as e:
        print(f"ERROR: Failed to initialize {llm_provider} client: {e}")
        return None

    print(f"\nGenerating bot script for {game_title} using {llm_model}...")
    print(f"Provider: {llm_provider}")
    print("Using two-stage generation for better readability...")

    # Generate Part 1: Core infrastructure
    part1_code = generate_bot_script_part1(
        game_title, controls_info, llm_provider, llm_model, client
    )
    if not part1_code:
        return None

    print(f"✓ Part 1 complete ({len(part1_code.split(chr(10)))} lines)")

    # Generate Part 2: Game logic and execution
    part2_code = generate_bot_script_part2(
        game_title, controls_info, llm_provider, llm_model, client
    )
    if not part2_code:
        return None

    print(f"✓ Part 2 complete ({len(part2_code.split(chr(10)))} lines)")

    # Combine both parts
    header = f'''"""
{game_title} AI Bot - YOLO Object Detection with pynput Keyboard Control

Auto-generated bot script that uses:
- YOLO for real-time object detection
- pynput for keyboard control
- mss for screen capture

Requirements:
    pip install ultralytics mss pynput numpy opencv-python

Usage:
    1. Start {game_title}
    2. Run: python {game_title.lower().replace(" ", "_")}_bot.py
    3. Press ESC to stop

Generated in two parts for readability:
    Part 1: Core infrastructure (YOLO, screen capture, keyboard control)
    Part 2: Game logic (decision making, actions, main loop)
"""

'''

    complete_script = (
        header
        + part1_code
        + "\n\n\n"
        + "# "
        + "=" * 77
        + "\n"
        + "# PART 2: GAME LOGIC AND EXECUTION\n"
        + "# "
        + "=" * 77
        + "\n\n"
        + part2_code
    )

    total_lines = len(complete_script.split("\n"))
    print(f"\n✓ Combined script complete ({total_lines} lines total)")

    return complete_script


def save_bot_script(game_title, script_code):
    """
    Save the generated bot script to file

    Args:
        game_title: Name of the game
        script_code: Python bot script code

    Returns:
        Path: Path to saved bot script
    """
    # Create bot_scripts directory
    scripts_dir = Path("bot_scripts")
    scripts_dir.mkdir(exist_ok=True)

    # Sanitize filename
    safe_title = "".join(
        c for c in game_title if c.isalnum() or c in (" ", "_")
    ).strip()
    safe_title = safe_title.replace(" ", "_").lower()
    script_path = scripts_dir / f"{safe_title}_bot.py"

    # Write script
    with open(script_path, "w") as f:
        f.write(script_code)

    return script_path


def main():
    """Main execution (standalone CLI entry point)"""
    _ensure_env_loaded()

    print("=" * 60)
    print(f"Game Controls Finder - Powered by {config.LLM_MODEL}")
    print("=" * 60)

    # Get game title using path_finder to properly detect exe
    try:
        game_exe = path_finder(config.GAME_PATH)
        if not game_exe:
            print(f"\nERROR: No game executable found in {config.GAME_PATH}")
            sys.exit(1)

        game_title = get_title(game_exe)
        game_path = game_exe
        print(f"\nGame detected: {game_title}")
        print(f"Executable: {game_exe}")
    except Exception as e:
        print(f"\nERROR: Could not detect game: {e}")
        sys.exit(1)

    # Check if controls already exist
    existing_controls = read_controls_from_config(game_title)

    if existing_controls:
        print(f"\n✓ Found existing controls configuration")
        print("Improving existing controls with latest information...")
        controls_info = search_game_controls(game_title, existing_controls)
    else:
        print(f"\nNo existing controls found. Searching from scratch...")
        controls_info = search_game_controls(game_title)

    if not controls_info:
        print("\nFailed to retrieve game controls.")
        sys.exit(1)

    # Display controls
    print("\n" + "=" * 60)
    print("GAME CONTROLS FOUND:")
    print("=" * 60)
    print(controls_info)
    print("=" * 60)

    # Save to config file
    config_path = save_controls_to_config(game_title, game_path, controls_info)

    print(f"\n✓ Controls saved to: {config_path}")

    # Ask if user wants to generate bot script
    print("\n" + "=" * 60)
    print("Bot Script Generation")
    print("=" * 60)
    print("\nGenerate a Python bot script using these controls?")
    print("This will create a bot that uses the trained YOLO model and these controls.")

    generate_choice = input("\nGenerate bot script? (Y/N): ").strip().upper()

    if generate_choice in ("Y", "y"):
        # Generate bot script
        bot_script = generate_bot_script(game_title, controls_info)

        if bot_script:
            # Save bot script
            script_path = save_bot_script(game_title, bot_script)
            print(f"\n✓ Bot script saved to: {script_path}")
            print(f"\nTo run the bot:")
            print(f"  python {script_path}")
            print("\nMake sure you have trained your YOLO model first!")
        else:
            print("\nFailed to generate bot script.")
    else:
        print("\nSkipping bot script generation.")

    print("\nDone!")


if __name__ == "__main__":
    main()
