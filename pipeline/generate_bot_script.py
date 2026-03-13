import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.functions import get_title, path_finder, get_api_key
from conf.config_parser import main_conf as config


def read_dataset_class_names():
    """Read YOLO class names from the trained model weights (source of truth).

    The trained model may have more classes than the original data.yaml if the
    dataset was expanded after the yaml was written.  Reading from the model
    guarantees the bot script uses the exact class names that will appear at
    inference time.

    Falls back to dataset/data.yaml or dataset.yaml if the model is not found.

    Returns:
        dict: Mapping of class_id (int) -> class_name (str), or empty dict if not found.
    """
    # Primary: read from trained model weights
    model_path = Path("runs/detect/paper_engine_model/weights/best.pt")
    if model_path.exists():
        try:
            from ultralytics import YOLO

            model = YOLO(str(model_path))
            names = model.names  # dict {int: str}
            if names:
                print(f"Read {len(names)} class names from trained model: {model_path}")
                return {int(k): str(v) for k, v in names.items()}
        except Exception as e:
            print(f"Warning: Could not read class names from model {model_path}: {e}")

    # Fallback: read from dataset YAML
    dataset_dir = Path("dataset")
    yaml_file = None
    for yaml_name in ["data.yaml", "dataset.yaml"]:
        potential = dataset_dir / yaml_name
        if potential.exists():
            yaml_file = potential
            break

    if not yaml_file:
        return {}

    try:
        import yaml

        with open(yaml_file, "r") as f:
            data = yaml.safe_load(f)
        names = data.get("names", {})
        # Ensure keys are ints and values are strings
        if isinstance(names, dict):
            return {int(k): str(v) for k, v in names.items()}
        elif isinstance(names, list):
            return {i: str(v) for i, v in enumerate(names)}
        return {}
    except Exception as e:
        print(f"Warning: Could not read class names from {yaml_file}: {e}")
        return {}


def create_env_file_if_missing():
    """Create .env file if it doesn't exist"""
    env_path = Path(".env")

    if not env_path.exists():
        print("\nNo .env file found. Creating one...")
        with open(env_path, "w") as f:
            f.write("# Paper Engine - Environment Variables\n")
            f.write("# Configure provider in conf/main_conf.ini [LLM] section\n\n")
            f.write("# --- API Key via OS Keyring (recommended) ---\n")
            f.write("# Store your key once:\n")
            f.write(
                '#   secret-tool store --label="Paper Engine" service paper_engine user api_key\n'
            )
            f.write("# Then set these labels so the app can find it:\n")
            f.write("KEYRING_SERVICE=paper_engine\n")
            f.write("KEYRING_USER=api_key\n\n")
            f.write(
                "# --- Legacy: plain-text key (fallback if keyring unavailable) ---\n"
            )
            f.write("# API_KEY=your-api-key-here\n")
        os.chmod(env_path, 0o600)  # Owner read/write only
        print(f"Created .env file at: {env_path.absolute()}")
        print("\n" + "=" * 60)
        print("API KEY SETUP")
        print("=" * 60)
        print("\nRecommended (OS keyring — key never stored on disk):")
        print("  1. Visit: https://aistudio.google.com/apikey")
        print("  2. Click 'Create API Key' and copy it")
        print("  3. Run:")
        print(
            '     secret-tool store --label="Paper Engine" service paper_engine user api_key'
        )
        print("  4. Paste your key when prompted")
        print("\nLegacy (plain-text .env):")
        print("  Uncomment API_KEY= in .env and paste your key")
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
    # Get LLM configuration
    llm_provider = config.LLM_PROVIDER
    llm_model = config.LLM_MODEL
    max_tokens = config.MAX_TOKENS_SEARCH

    # Get API key from keyring (preferred) or env (legacy fallback)
    api_key = get_api_key()

    if not api_key:
        print("\n" + "=" * 60)
        print("ERROR: API Key Not Configured")
        print("=" * 60)
        print("\nSetup (recommended — stored in OS keyring, never on disk):")
        print("  1. pip install keyring")
        print(
            '  2. secret-tool store --label="Paper Engine" service paper_engine user api_key'
        )
        print("  3. Set KEYRING_SERVICE=paper_engine and KEYRING_USER=api_key in .env")
        print("\nLegacy (plain-text .env):")
        print("  Set API_KEY=<your-key> in .env")
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
        # Store relative path to avoid leaking absolute filesystem paths
        try:
            rel_path = Path(game_path).relative_to(Path.cwd())
        except ValueError:
            rel_path = Path(game_path).name
        f.write(f"game_path = {rel_path}\n\n")
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
    game_title, controls_info, llm_provider, llm_model, client, class_names=None
):
    """
    Generate Part 1: Core infrastructure (model, capture, keyboard control)

    Args:
        game_title: Name of the game
        controls_info: Game controls information
        class_names: Optional dict of {class_id: class_name} from dataset
        llm_provider: LLM provider name
        llm_model: LLM model name
        client: LLM client instance

    Returns:
        str: Part 1 Python code
    """
    print(f"\n[Part 1/2] Generating core infrastructure...")

    # Build class names section for prompt
    if class_names:
        class_list = "\n".join(
            f"  {cid}: '{cname}'" for cid, cname in sorted(class_names.items())
        )
        class_names_section = f"""
YOLO DATASET CLASS NAMES (from trained model — use these EXACT names in code):
{class_list}

IMPORTANT: The YOLO model outputs these exact class names (lowercase). Your code
MUST use these exact strings when checking detection.class_name. Do NOT invent
class names like "Player", "Enemy", "Projectile" etc. — use the names above."""
    else:
        class_names_section = ""

    prompt = f"""Create the CORE INFRASTRUCTURE for a "{game_title}" bot (Part 1 of 2).

GAME CONTROLS:
{controls_info}
{class_names_section}

Generate ONLY Part 1 - Core Infrastructure:

1. **Imports and Configuration**:
```python
import time
import sys
import os
import subprocess
import io
import numpy as np
from pynput.keyboard import Controller, Listener, Key
from ultralytics import YOLO
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from PIL import Image

# Resolve project root (bot_scripts/ is one level below)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
```

2. **Configuration Class**:
```python
class Config:
    MODEL_PATH = str(PROJECT_ROOT / 'runs/detect/paper_engine_model/weights/best.pt')
    CONFIDENCE_THRESHOLD = 0.5
    SCREEN_REGION = {{"top": 0, "left": 0, "width": 1920, "height": 1080}}
    TARGET_FPS = 30
    FRAME_TIME = 1.0 / TARGET_FPS
    GAME_PATH = str(PROJECT_ROOT / 'game')
    WINE_DESKTOP_RESOLUTION = '1920x1080'
    GAME_INIT_WAIT = 10
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
        # Load YOLO model: self.model = YOLO(self.model_path)
        # Return True if successful

    def detect_objects(self, frame: np.ndarray) -> List[Detection]:
        # Run inference: self.model.predict(source=frame, conf=..., verbose=False)
        # Parse results into Detection objects
        # Return list of detections
```

5. **ScreenCapture Class** - MUST support both Wayland and X11:
This is CRITICAL. On Wayland (Linux), mss/XGetImage fails. You MUST:
- Detect Wayland via `os.environ.get("XDG_SESSION_TYPE") == "wayland"`
- On Wayland: use flameshot subprocess to capture (pipe PNG to stdout via `-r` flag)
- On X11/macOS/Windows: use mss as fallback
- Include a `set_window_geometry(geometry: str)` method for targeted window capture

```python
class ScreenCapture:
    def __init__(self, region):
        self.region = region
        self._use_flameshot = (sys.platform == "linux" and
            os.environ.get("XDG_SESSION_TYPE", "").lower() == "wayland")
        self._window_geometry = None
        if not self._use_flameshot:
            import mss as _mss
            self._sct = _mss.mss()

    def set_window_geometry(self, geometry):
        self._window_geometry = geometry

    def capture(self) -> Optional[np.ndarray]:
        if self._use_flameshot:
            return self._capture_flameshot()
        return self._capture_mss()

    def _capture_flameshot(self):
        # ALWAYS use "flameshot screen --raw" (non-interactive). NEVER use "flameshot gui" (blocks).
        # Do NOT pass --region to flameshot (broken on v13+ / Qt6 — hangs).
        # Instead: capture full screen, then crop in Python using _crop_to_geometry().
        # subprocess.run(["flameshot", "screen", "--raw"], capture_output=True, timeout=5)
        # Image.open(io.BytesIO(result.stdout)).convert("RGB") -> np.array -> crop

    def _crop_to_geometry(self, frame):
        # Parse self._window_geometry "WxH+X+Y" -> crop frame[y:y+h, x:x+w]

    def _capture_mss(self):
        # self._sct.grab(self.region) -> BGRA -> numpy RGB
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

    def press_key(self, key, duration=0.05):
        # Press, sleep(duration), release

    def hold_key(self, key):
        # Press and hold

    def release_key(self, key):
        # Release key

    def tap_key(self, key):
        # Quick press + release (no sleep)
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

8. **Game Launching Utilities** (standalone functions, NOT inside a class):
```python
def _is_wine_still_running() -> bool:
    # Use pgrep -f to check for active Wine process names:
    # "wineserver", "wine-preloader", "wine64-preloader", ".exe"
    # Return True if any match found, False otherwise
    # IMPORTANT: The initial 'wine explorer' PID exits quickly while the
    # actual game continues as a child process.  Never use game_proc.poll()
    # to check if the game is still running — use this function instead.

def find_wine_window() -> Optional[str]:
    # Try multiple backends to find Wine window geometry:
    # 1. Sway: swaymsg -t get_tree (check "which swaymsg" first)
    # 2. KDE Plasma 6 / KWin Wayland — D-Bus scripting (two-step: load then start)
    #    CRITICAL D-Bus details (interface is lowercase "kwin", NOT "KWin"):
    #      Step A: Record timestamp via datetime.now(timezone.utc)
    #      Step B: Write JS to a temp file
    #      Step C: dbus-send --session --print-reply --dest=org.kde.KWin /Scripting
    #              org.kde.kwin.Scripting.loadScript string:<path_to_js>
    #      Step D: dbus-send --session --print-reply --dest=org.kde.KWin /Scripting
    #              org.kde.kwin.Scripting.start
    #      Step E: time.sleep(0.5) then read journalctl --user -u plasma-kwin_wayland
    #              --since <timestamp> --output=cat --no-pager -n 30
    #    The JS should: call workspace.windowList(), iterate windows, check
    #    resourceClass/resourceName/caption for "wine", "explorer", ".exe",
    #    or the game name; pick the largest; console.log a marker + JSON
    #    with x,y,w,h; parse from journal lines
    # Return "WxH+x+y" format string, or None

def launch_game() -> Optional[subprocess.Popen]:
    # Import path_finder from project: sys.path.insert(0, str(PROJECT_ROOT))
    # Find exe in Config.GAME_PATH using path_finder()
    # Launch .exe via: ["wine", "explorer", f"/desktop=game,{{Config.WINE_DESKTOP_RESOLUTION}}", str(exe_path)]
    # CRITICAL: resolution MUST be comma-joined to /desktop flag, NOT a separate argument
    # Set cwd=exe_path.parent so the game can find its data files
    # Do NOT suppress stderr (no stdout=DEVNULL) so Wine errors are visible
    # Launch .sh via bash, .py via python
    # Return Popen process or None
    # DO NOT sleep inside this function — the caller (GameBot.run) handles the wait
```

CRITICAL:
- Every class must be FULLY FUNCTIONAL with real implementations, NOT mocks or stubs.
- YOLODetector MUST use `self.model = YOLO(self.model_path)` and `self.model.predict()`.
- ScreenCapture MUST detect Wayland and use flameshot subprocess. Only use mss on X11.
- KeyboardController MUST use pynput `Controller()` to actually press/release keys.
- EmergencyStop MUST use pynput `Listener` to detect ESC key press.
- NO mock data, NO simulated detections, NO print-only stubs.

Return ONLY this Part 1 code. Keep it under 400 lines. Make it complete and functional.
Include docstrings. NO placeholder comments like "# TODO" or "# Implement this"."""

    try:
        code = call_llm(client, llm_provider, llm_model, prompt, max_tokens=8000)

        if "```python" in code:
            code = code.split("```python")[1].split("```")[0].strip()
        elif "```" in code:
            code = code.split("```")[1].split("```")[0].strip()

        return code
    except Exception as e:
        print(f"ERROR: Failed to generate Part 1: {e}")
        return None


def generate_bot_script_part2(
    game_title,
    controls_info,
    llm_provider,
    llm_model,
    client,
    part1_code,
    class_names=None,
):
    """
    Generate Part 2: Decision logic, actions, and main loop

    Args:
        game_title: Name of the game
        controls_info: Game controls information
        llm_provider: LLM provider name
        llm_model: LLM model name
        client: LLM client instance
        part1_code: The actual Part 1 code to reference
        class_names: Optional dict of {class_id: class_name} from dataset

    Returns:
        str: Part 2 Python code
    """
    print(f"\n[Part 2/2] Generating decision logic and main loop...")

    # Build class names section for Part 2 prompt
    if class_names:
        class_list = "\n".join(
            f"  {cid}: '{cname}'" for cid, cname in sorted(class_names.items())
        )
        class_names_section = f"""
YOLO DATASET CLASS NAMES (from the trained model — these are the EXACT strings
that detection.class_name will contain at runtime):
{class_list}

CRITICAL: You MUST use these exact class name strings (lowercase) in
categorize_detections(). For example, if the player class is 'cuphead', check
`d.class_name == "cuphead"`, NOT `d.class_name == "Player"`.
Do NOT invent or guess class names."""
    else:
        class_names_section = ""

    prompt = f"""Create the GAME LOGIC AND EXECUTION for a "{game_title}" bot (Part 2 of 2).

CRITICAL RULES:
- DO NOT redefine ANY class from Part 1 (Config, Detection, YOLODetector, ScreenCapture, KeyboardController, EmergencyStop)
- DO NOT create mock/placeholder/stub versions of Part 1 classes
- DO NOT add any import statements that Part 1 already includes
- Your code will be appended directly after Part 1 in the same file
- All Part 1 classes are ALREADY DEFINED and FULLY FUNCTIONAL — just USE them

Here is the EXACT Part 1 code that will precede your code:

```python
{part1_code}
```

GAME CONTROLS:
{controls_info}
{class_names_section}

Generate ONLY Part 2 — the GameBot class and main block. Use the Part 1 classes exactly as defined above.

Part 1 provides these APIs you MUST use (do not redefine):
- `Config.MODEL_PATH`, `Config.CONFIDENCE_THRESHOLD`, `Config.SCREEN_REGION`, `Config.TARGET_FPS`, `Config.FRAME_TIME`, `Config.GAME_INIT_WAIT`
- `Detection(class_name, confidence, bbox)` with `.class_name`, `.confidence`, `.bbox`, `.center`
- `YOLODetector(model_path, confidence_threshold)` with `.load_model()` -> bool, `.detect_objects(frame)` -> List[Detection]
- `ScreenCapture(region)` with `.capture()` -> numpy array, `.set_window_geometry(geom)` for targeted capture
- `KeyboardController()` with `.press_key()`, `.hold_key()`, `.release_key()`, `.tap_key()` and key constants like `.MOVE_LEFT`, `.MOVE_RIGHT`, `.JUMP`, `.SHOOT`, `.DASH`, etc.
- `EmergencyStop()` with `.setup()`, `.start()`, `.stop()`, `.running` flag
- `launch_game()` -> Optional[subprocess.Popen] — launches the game executable
- `find_wine_window()` -> Optional[str] — finds Wine window geometry ("WxH+x+y")
- `_is_wine_still_running()` -> bool — checks if any Wine processes are active (use this instead of game_proc.poll())

Generate this structure:

```python
game_title = "{game_title}"

class GameBot:
    def __init__(self):
        self.detector = YOLODetector(Config.MODEL_PATH, Config.CONFIDENCE_THRESHOLD)
        self.screen = ScreenCapture(Config.SCREEN_REGION)
        self.keyboard = KeyboardController()
        self.emergency_stop = EmergencyStop()
        self.running = True

    def categorize_detections(self, detections: List[Detection]) -> Dict:
        # Categorize into: enemies, projectiles, items, obstacles, pink_objects
        # Use detection.class_name to categorize

    def make_decision(self, detections_categorized: Dict) -> str:
        # Priority: dodge > attack > collect > explore
        # Return action string or dict of booleans

    def execute_action(self, action):
        # Use self.keyboard methods (hold_key, release_key, tap_key)
        # Use self.keyboard key constants (MOVE_LEFT, MOVE_RIGHT, JUMP, SHOOT, DASH, etc.)

    def run(self):
        # 1. Launch game: game_proc = launch_game()
        # 2. Wait for initialisation: time.sleep(Config.GAME_INIT_WAIT)
        # 3. Detect window: geom = find_wine_window(); self.screen.set_window_geometry(geom)
        # 4. Load model: self.detector.load_model()
        # 5. Emergency stop: self.emergency_stop.setup() then .start()
        # 6. Main loop: capture -> detect_objects -> categorize -> decide -> execute
        # 7. Check _is_wine_still_running() to stop if game exits (do NOT use game_proc.poll() — the initial wine PID exits early while the game keeps running as a child process)
        # 8. FPS limiting with Config.FRAME_TIME
        # 9. Cleanup in finally block: release all held keys

if __name__ == "__main__":
    # Print instructions, create GameBot, run with try/except/finally
```

Return ONLY the Part 2 code. Keep it under 300 lines. NO import statements. NO class redefinitions.
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


def _validate_generated_script(script, class_names=None):
    """Check the generated bot script for common LLM generation issues.

    Prints warnings for any problems found.  Does not modify the script.

    Args:
        script: The full generated script text.
        class_names: Optional dict of {class_id: class_name} from dataset.
    """
    warnings = []

    # Check for mock/placeholder patterns
    mock_patterns = [
        (
            "mock_frame",
            "ScreenCapture.capture() returns a mock string instead of a real frame",
        ),
        (
            "_mock_detection",
            "YOLODetector uses hardcoded mock detections instead of real inference",
        ),
        ("Placeholder", "Contains placeholder class comments — may not be functional"),
        (
            "Simulates",
            "Contains 'Simulates' — class may be a stub instead of real implementation",
        ),
        ('"path/to/your/', "Config.MODEL_PATH still has a placeholder path"),
    ]

    for pattern, message in mock_patterns:
        if pattern in script:
            warnings.append(f"  - {message} (found: '{pattern}')")

    # Check that key real implementations exist
    required_patterns = [
        ("YOLO(", "YOLO model is never loaded — YOLODetector may be a stub"),
        (
            "mss.mss()",
            "mss screen capture is never initialised — ScreenCapture may be a stub",
        ),
        (
            "Controller()",
            "pynput Controller is never created — KeyboardController may be a stub",
        ),
    ]

    for pattern, message in required_patterns:
        if pattern not in script:
            warnings.append(f"  - {message} (missing: '{pattern}')")

    # Check that generated script uses actual dataset class names
    if class_names:
        actual_names = set(class_names.values())
        # Common wrong class names the LLM might invent
        invented_names = [
            '"Player"',
            '"Enemy"',
            '"Projectile"',
            '"PinkObject"',
            '"Obstacle"',
            '"Item"',
            '"Boss"',
        ]
        for bad_name in invented_names:
            if bad_name in script:
                # Check it's not just in a comment
                stripped_name = bad_name.strip('"')
                if stripped_name.lower() not in {n.lower() for n in actual_names}:
                    warnings.append(
                        f"  - Script uses invented class name {bad_name} instead of "
                        f"actual dataset names: {sorted(actual_names)}"
                    )

        # Check that at least some actual class names appear in the script
        found_any = any(
            f'"{name}"' in script or f"'{name}'" in script for name in actual_names
        )
        if not found_any:
            warnings.append(
                f"  - None of the actual dataset class names found in script: {sorted(actual_names)}"
            )

    if warnings:
        print("\n⚠️  Generated script validation warnings:")
        for w in warnings:
            print(w)
        print("  Consider regenerating or manually fixing the script.\n")
    else:
        print("✓ Script validation passed — no mock/placeholder patterns detected")


def _strip_duplicate_definitions(complete_script, part1_code):
    """Remove duplicate class/import definitions that Part 2 may have regenerated.

    Scans Part 1 for class names and import lines, then removes any duplicate
    definitions that appear after the Part 2 separator.  This guards against
    the LLM redefining Config, Detection, YOLODetector, etc. in Part 2.

    Args:
        complete_script: The full combined script text.
        part1_code: The Part 1 code string (used to find class names).

    Returns:
        str: The cleaned script with duplicates removed.
    """
    import re

    # Find all class names defined in Part 1
    part1_classes = set(re.findall(r"^class\s+(\w+)", part1_code, re.MULTILINE))

    # Locate the Part 2 separator
    separator = "# PART 2: GAME LOGIC AND EXECUTION"
    sep_idx = complete_script.find(separator)
    if sep_idx == -1:
        return complete_script

    part1_section = complete_script[:sep_idx]
    part2_section = complete_script[sep_idx:]

    # Collect Part 1 import lines (normalised) for dedup
    part1_imports = set()
    for line in part1_code.splitlines():
        stripped = line.strip()
        if stripped.startswith(("import ", "from ")):
            part1_imports.add(stripped)

    # Remove duplicate class blocks from Part 2
    # A class block starts with "class Foo" and continues until the next
    # top-level definition or end of string.
    for cls_name in part1_classes:
        # Match "class <Name>(...):  ...  until next top-level class/def/if or end"
        pattern = (
            rf"^class\s+{re.escape(cls_name)}\b[^\n]*:\n"
            rf"(?:(?:[ \t]+[^\n]*|[ \t]*)\n)*"
        )
        part2_section = re.sub(pattern, "", part2_section, flags=re.MULTILINE)

    # Remove duplicate import lines from Part 2
    cleaned_lines = []
    for line in part2_section.splitlines():
        stripped = line.strip()
        if stripped.startswith(("import ", "from ")) and stripped in part1_imports:
            continue  # skip duplicate import
        cleaned_lines.append(line)
    part2_section = "\n".join(cleaned_lines)

    # Remove excessive blank lines that may result from stripping
    part2_section = re.sub(r"\n{4,}", "\n\n\n", part2_section)

    result = part1_section + part2_section

    # Report what was stripped
    stripped_classes = []
    for cls_name in part1_classes:
        if cls_name != "GameBot" and f"class {cls_name}" not in part2_section:
            stripped_classes.append(cls_name)
    if stripped_classes:
        print(
            f"  Stripped {len(stripped_classes)} duplicate class(es) from Part 2: "
            + ", ".join(sorted(stripped_classes))
        )

    return result


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
    api_key = get_api_key()

    if not api_key:
        print(f"\nERROR: API key not configured for provider: {llm_provider}")
        print(
            "Store with: secret-tool store --label='Paper Engine' service paper_engine user api_key"
        )
        return None

    # Get appropriate client
    try:
        client = get_llm_client(llm_provider, api_key)
    except Exception as e:
        print(f"ERROR: Failed to initialize {llm_provider} client: {e}")
        return None

    # Read actual YOLO class names from dataset
    class_names = read_dataset_class_names()
    if class_names:
        class_names_str = ", ".join(
            f"{cid}: '{cname}'" for cid, cname in sorted(class_names.items())
        )
        print(f"Dataset classes: {class_names_str}")
    else:
        class_names_str = ""
        print("Warning: No dataset class names found. LLM will guess class names.")

    print(f"\nGenerating bot script for {game_title} using {llm_model}...")
    print(f"Provider: {llm_provider}")
    print("Using two-stage generation for better readability...")

    # Generate Part 1: Core infrastructure
    part1_code = generate_bot_script_part1(
        game_title,
        controls_info,
        llm_provider,
        llm_model,
        client,
        class_names=class_names,
    )
    if not part1_code:
        return None

    print(f"✓ Part 1 complete ({len(part1_code.split(chr(10)))} lines)")

    # Generate Part 2: Game logic and execution (pass Part 1 code as context)
    part2_code = generate_bot_script_part2(
        game_title,
        controls_info,
        llm_provider,
        llm_model,
        client,
        part1_code,
        class_names=class_names,
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

    # Strip duplicate class definitions that the LLM may have generated
    complete_script = _strip_duplicate_definitions(complete_script, part1_code)

    # Validate the final script for common generation issues
    _validate_generated_script(complete_script, class_names=class_names)

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
