import os
import sys
from pathlib import Path
from google import genai
from dotenv import load_dotenv
from functions import get_title
from conf.config_parser import main_conf as config


def create_env_file_if_missing():
    """Create .env file if it doesn't exist"""
    env_path = Path(".env")

    if not env_path.exists():
        print("\nNo .env file found. Creating one...")
        with open(env_path, 'w') as f:
            f.write("# Paper Engine - Environment Variables\n")
            f.write("# Get your API key at: https://console.anthropic.com/settings/keys\n\n")
            f.write("ANTHROPIC_API_KEY=your-api-key-here\n")
        print(f"✓ Created .env file at: {env_path.absolute()}")
        print("\nPlease edit .env and add your Anthropic API key, then run this script again.")
        return False
    return True


# Create .env if missing
if not create_env_file_if_missing():
    sys.exit(0)

# Load environment variables from .env file
load_dotenv()


def get_llm_client(provider, api_key):
    """
    Get the appropriate LLM client based on provider

    Args:
        provider: LLM provider name (anthropic, openai, google)
        api_key: API key for the provider

    Returns:
        Client instance for the provider
    """
    if provider.lower() == "anthropic":
        return Anthropic(api_key=api_key)
    elif provider.lower() == "openai":
        return OpenAI(api_key=api_key)
    elif provider.lower() == "google":
        client = genai.Client(api_key=api_key)
        return client
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
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text

        elif provider.lower() == "openai":
            response = client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content

        elif provider.lower() == "google":
            response = client.models.generate_content(
                model=model,
                contents=prompt
            )
            return response.text

        else:
            raise ValueError(f"Unsupported provider: {provider}")

    except Exception as e:
        raise Exception(f"LLM API call failed: {e}")


def search_game_controls(game_title):
    """
    Search for game controls using configured LLM

    Args:
        game_title: Name of the game

    Returns:
        str: Game controls information
    """
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
        print("   (If .env doesn't exist, run: python main.py first)")
        print("3. Configure provider in conf/main_conf.ini (llm_provider)")
        print("=" * 60 + "\n")
        return None

    # Get LLM client
    try:
        client = get_llm_client(llm_provider, api_key)
    except Exception as e:
        print(f"ERROR: Failed to initialize {llm_provider} client: {e}")
        return None

    print(f"\nSearching for {game_title} controls using {llm_provider}/{llm_model}...")

    prompt = f"""Search the web for "{game_title}" game controls and keyboard mappings.

Provide a comprehensive list of:
1. Movement controls (WASD, arrow keys, etc.)
2. Action buttons (jump, shoot, interact, etc.)
3. Special moves or combos
4. Menu/UI controls
5. Any other important keybinds

Format as a clear, organized list."""

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
    safe_title = "".join(c for c in game_title if c.isalnum() or c in (' ', '_')).strip()
    safe_title = safe_title.replace(' ', '_').lower()

    config_path = conf_dir / f"{safe_title}_controls.ini"

    # Write to .ini file
    with open(config_path, 'w') as f:
        f.write(f"# Game Controls for {game_title}\n")
        f.write(f"# Auto-generated by Claude Opus\n\n")
        f.write(f"[GameInfo]\n")
        f.write(f"game_name = {game_title}\n")
        f.write(f"game_path = {game_path}\n\n")
        f.write(f"[Controls]\n")
        f.write(f"# Raw controls information from web search:\n")

        # Write controls info as comments for reference
        for line in controls_info.split('\n'):
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
    safe_title = "".join(c for c in game_title if c.isalnum() or c in (' ', '_')).strip()
    safe_title = safe_title.replace(' ', '_').lower()
    config_path = conf_dir / f"{safe_title}_controls.ini"

    if not config_path.exists():
        return None

    # Read the controls section
    with open(config_path, 'r') as f:
        content = f.read()
        # Extract everything after [Controls]
        if "[Controls]" in content:
            controls_section = content.split("[Controls]")[1]
            # Remove comment markers for cleaner reading
            controls_text = "\n".join(line.lstrip("# ") for line in controls_section.split("\n"))
            return controls_text.strip()
    return None


def generate_bot_script_part1(game_title, controls_info, llm_provider, llm_model, client):
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


def generate_bot_script_part2(game_title, controls_info, llm_provider, llm_model, client):
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
    part1_code = generate_bot_script_part1(game_title, controls_info, llm_provider, llm_model, client)
    if not part1_code:
        return None

    print(f"✓ Part 1 complete ({len(part1_code.split(chr(10)))} lines)")

    # Generate Part 2: Game logic and execution
    part2_code = generate_bot_script_part2(game_title, controls_info, llm_provider, llm_model, client)
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

    complete_script = header + part1_code + "\n\n\n" + "# " + "="*77 + "\n" + "# PART 2: GAME LOGIC AND EXECUTION\n" + "# " + "="*77 + "\n\n" + part2_code

    total_lines = len(complete_script.split("\n"))
    print(f"\n✓ Combined script complete ({total_lines} lines total)")

    return complete_script


def generate_bot_script_old(game_title, controls_info):
    """
    OLD SINGLE-STAGE GENERATION (kept for reference)
    Generate a Python bot script using LLM

    Args:
        game_title: Name of the game
        controls_info: Game controls information

    Returns:
        str: Generated Python bot script code
    """
    # Get LLM configuration
    llm_model = config.LLM_MODEL
    llm_provider = config.LLM_PROVIDER
    api_key = os.environ.get("API_KEY")

    if not api_key or api_key == "your-api-key-here":
        print(f"\nERROR: API_KEY not configured for provider: {llm_provider}")
        print("Edit .env file and add your API key")
        return None

    client = Anthropic(api_key=api_key)

    print(f"\nGenerating bot script for {game_title} using {llm_model}...")

    prompt = f"""Create a complete, functional Python bot script for "{game_title}" that uses YOLO object detection and pynput keyboard control.

GAME CONTROLS REFERENCE:
{controls_info}

REQUIREMENTS - The script MUST include:

1. **YOLO Model Integration**:
   - Load model from: 'runs/detect/paper_engine_model/weights/best.pt'
   - Run inference on each frame
   - Parse detections (class names, confidence, bounding boxes)
   - Handle empty detections gracefully

2. **Screen Capture**:
   - Use mss library for fast screenshot capture
   - Capture game window region (configurable)
   - Convert to format YOLO expects (RGB numpy array)

3. **Keyboard Control with pynput**:
   - Create keyboard Controller instance
   - Parse the game controls from above
   - Create helper methods for each action:
     * press_key(key, duration=0.1) - press and release
     * hold_key(key) - press and hold
     * release_key(key) - release held key
   - Map game actions to keyboard keys based on controls

4. **Decision Making Logic**:
   - Analyze YOLO detections each frame
   - Create decision tree based on what's detected:
     * If enemy detected → attack or dodge
     * If obstacle detected → jump or avoid
     * If item detected → move towards it
     * If nothing detected → explore/move forward
   - Use bounding box positions to decide direction

5. **Bot Class Structure**:
```python
class GameBot:
    def __init__(self):
        self.model = None  # YOLO model
        self.keyboard = Controller()
        self.running = True
        self.screen_region = {{"top": 0, "left": 0, "width": 1920, "height": 1080}}

    def load_model(self):
        # Load YOLO model

    def capture_screen(self):
        # Capture screenshot using mss
        # Return numpy array

    def detect_objects(self, frame):
        # Run YOLO inference
        # Return list of detections with class, confidence, bbox

    def make_decision(self, detections):
        # Analyze detections
        # Return action to take

    def execute_action(self, action):
        # Use pynput to press keys
        # Based on game controls

    def run(self):
        # Main game loop
        # Capture → Detect → Decide → Act
        # Include FPS limiting (30 FPS)
```

6. **Emergency Stop**:
   - Use pynput Listener to detect ESC key
   - Set self.running = False when ESC pressed
   - Clean exit from game loop

7. **Key Implementation Details**:
   - FPS limit: time.sleep() between frames
   - Key press duration: ~0.1 seconds for tap, longer for holds
   - Confidence threshold: 0.5 for detections
   - Add print statements for debugging (what was detected, what action taken)

8. **Example Action Mapping** (adapt to actual controls):
```python
def move_left(self):
    self.keyboard.press('a')
    time.sleep(0.1)
    self.keyboard.release('a')
```

Return ONLY the complete, functional Python code with:
- All imports at the top
- GameBot class with all methods implemented
- if __name__ == "__main__" block to run the bot
- Comprehensive comments explaining logic
- Error handling for model loading and screen capture

Make it production-ready and directly executable."""

    try:
        response = client.messages.create(
            model=llm_model,
            max_tokens=8192,
            messages=[{
                "role": "user",
                "content": prompt
            }]
        )

        script_code = response.content[0].text

        # Extract Python code if wrapped in markdown
        if "```python" in script_code:
            script_code = script_code.split("```python")[1].split("```")[0].strip()
        elif "```" in script_code:
            script_code = script_code.split("```")[1].split("```")[0].strip()

        return script_code

    except Exception as e:
        print(f"ERROR: Failed to generate bot script: {e}")
        return None


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
    safe_title = "".join(c for c in game_title if c.isalnum() or c in (' ', '_')).strip()
    safe_title = safe_title.replace(' ', '_').lower()
    script_path = scripts_dir / f"{safe_title}_bot.py"

    # Write script
    with open(script_path, 'w') as f:
        f.write(script_code)

    return script_path


def main():
    """Main execution"""
    print("=" * 60)
    print(f"Game Controls Finder - Powered by {config.LLM_MODEL}")
    print("=" * 60)

    # Get game title
    try:
        game_path = Path(config.GAME_PATH)
        game_title = get_title(game_path)
        print(f"\nGame detected: {game_title}")
    except Exception as e:
        print(f"\nERROR: Could not detect game: {e}")
        sys.exit(1)

    # Search for controls
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
