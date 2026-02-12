"""
Cuphead AI Bot - YOLO Object Detection with pynput Keyboard Control

Auto-generated bot script that uses:
- YOLO for real-time object detection
- pynput for keyboard control
- mss for screen capture

Requirements:
    pip install ultralytics mss pynput numpy opencv-python

Usage:
    1. Start Cuphead
    2. Run: python cuphead_bot.py
    3. Press ESC to stop

Generated in two parts for readability:
    Part 1: Core infrastructure (YOLO, screen capture, keyboard control)
    Part 2: Game logic (decision making, actions, main loop)
"""

import time
import sys
import numpy as np
import mss
from pynput.keyboard import Controller, Listener, Key
from ultralytics import YOLO
from typing import List, Dict, Tuple, Optional

class Config:
    """
    Configuration class for the Cuphead bot, defining paths, thresholds, and screen parameters.
    """
    MODEL_PATH: str = 'runs/detect/paper_engine_model/weights/best.pt'
    CONFIDENCE_THRESHOLD: float = 0.5
    SCREEN_REGION: Dict[str, int] = {"top": 0, "left": 0, "width": 1920, "height": 1080}
    TARGET_FPS: int = 30
    FRAME_TIME: float = 1.0 / TARGET_FPS

class Detection:
    """
    Represents a single object detection, storing its class, confidence, and bounding box.
    """
    def __init__(self, class_name: str, confidence: float, bbox: Tuple[int, int, int, int]):
        """
        Initializes a Detection object.

        Args:
            class_name (str): The name of the detected class (e.g., 'Cuphead', 'Boss_Projectile').
            confidence (float): The confidence score of the detection (0.0 to 1.0).
            bbox (Tuple[int, int, int, int]): The bounding box coordinates (x1, y1, x2, y2).
        """
        self.class_name = class_name
        self.confidence = confidence
        self.bbox = bbox  # (x1, y1, x2, y2)
        self.center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)

    def __repr__(self):
        return (f"Detection(class_name='{self.class_name}', confidence={self.confidence:.2f}, "
                f"bbox={self.bbox}, center={self.center})")


class YOLODetector:
    """
    Handles loading and running inference with a YOLOv8 model for object detection.
    """
    def __init__(self, model_path: str, confidence_threshold: float):
        """
        Initializes the YOLODetector with the model path and confidence threshold.

        Args:
            model_path (str): Path to the trained YOLO model weights.
            confidence_threshold (float): Minimum confidence to consider a detection valid.
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.model: Optional[YOLO] = None
        self.class_names: Optional[Dict[int, str]] = None

    def load_model(self) -> bool:
        """
        Loads the YOLO model from the specified path.

        Returns:
            bool: True if the model was loaded successfully, False otherwise.
        """
        try:
            self.model = YOLO(self.model_path)
            self.class_names = self.model.names
            print(f"YOLO model loaded from {self.model_path}")
            return True
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            return False

    def detect_objects(self, frame: np.ndarray) -> List[Detection]:
        """
        Runs inference on the provided frame and returns a list of detected objects.

        Args:
            frame (np.ndarray): The input image frame (RGB format).

        Returns:
            List[Detection]: A list of Detection objects representing recognized entities.
        """
        if self.model is None:
            print("Model not loaded. Call load_model() first.")
            return []

        results = self.model.predict(frame, conf=self.confidence_threshold, verbose=False)
        detections: List[Detection] = []

        for r in results:
            boxes = r.boxes
            for i in range(len(boxes)):
                conf = boxes.conf[i].item()
                cls = int(boxes.cls[i].item())
                bbox_xyxy = boxes.xyxy[i].cpu().numpy().astype(int)

                class_name = self.class_names.get(cls, f"Unknown_{cls}")
                
                detections.append(
                    Detection(
                        class_name=class_name,
                        confidence=conf,
                        bbox=tuple(bbox_xyxy)
                    )
                )
        return detections


class ScreenCapture:
    """
    Handles capturing screenshots of a specified region using the mss library.
    """
    def __init__(self, region: Dict[str, int]):
        """
        Initializes the ScreenCapture object with the desired screen region.

        Args:
            region (Dict[str, int]): A dictionary defining the capture region
                                     with keys "top", "left", "width", "height".
        """
        self.sct = mss.mss()
        self.region = region

    def capture(self) -> Optional[np.ndarray]:
        """
        Captures a screenshot of the defined region.

        The captured image is converted from BGRA to RGB format as a NumPy array.

        Returns:
            Optional[np.ndarray]: A NumPy array representing the captured screen in RGB format,
                                  or None if capture fails.
        """
        try:
            sct_img = self.sct.grab(self.region)
            # Convert to numpy array, drop alpha channel, and convert BGR to RGB
            frame = np.array(sct_img, dtype=np.uint8)[:, :, :3]  # BGRA to BGR
            frame = frame[:, :, ::-1]  # BGR to RGB
            return frame
        except mss.exception.ScreenShotError as e:
            print(f"Screen capture error: {e}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred during screen capture: {e}")
            return None


class KeyboardController:
    """
    Manages keyboard inputs for the Cuphead game, mapping game actions to pynput keys.
    """
    def __init__(self):
        """
        Initializes the KeyboardController and defines key mappings based on Cuphead's controls.
        """
        self.keyboard = Controller()

        # Movement Controls
        self.MOVE_LEFT = 'a'
        self.MOVE_RIGHT = 'd'
        self.AIM_UP = 'w'
        self.AIM_DOWN = 's'

        # Action Buttons
        self.JUMP = Key.space
        self.SHOOT = 'j'
        self.DASH = 'l'
        self.LOCK_AIM = 'k'

        # Special Moves or Combos
        # Parry is a sequence (Jump -> Jump again airborne) handled by logic, not a direct key
        self.EX_MOVE = 'h'
        self.SUPER_ART = 'h' # Same key as EX Move, differentiated by EX Meter status

        # Menu/UI Controls
        self.PAUSE_MENU = Key.esc
        # Menu navigation (WASD or Arrow Keys) and selection (Enter) are not part of core gameplay bot
        # but could be added if menu interaction is needed.
        self.SELECT_CONFIRM = Key.enter
        self.BACK_CLOSE_MENU = Key.esc


    def press_key(self, key: str | Key, duration: float = 0.1):
        """
        Presses a key, holds it for a specified duration, then releases it.

        Args:
            key (str | Key): The key to press (e.g., 'a', Key.space).
            duration (float): The duration in seconds to hold the key.
        """
        self.keyboard.press(key)
        time.sleep(duration)
        self.keyboard.release(key)

    def hold_key(self, key: str | Key):
        """
        Presses and holds a specified key. It must be manually released later.

        Args:
            key (str | Key): The key to hold.
        """
        self.keyboard.press(key)

    def release_key(self, key: str | Key):
        """
        Releases a previously held key.

        Args:
            key (str | Key): The key to release.
        """
        self.keyboard.release(key)

    def tap_key(self, key: str | Key):
        """
        Performs a quick tap of a key (press and immediate release).

        Args:
            key (str | Key): The key to tap.
        """
        self.press_key(key, duration=0.01) # A very short duration for a tap


class EmergencyStop:
    """
    Provides a mechanism to gracefully stop the bot by listening for the ESC key.
    """
    def __init__(self):
        """
        Initializes the EmergencyStop handler.
        """
        self.running: bool = True
        self.listener: Optional[Listener] = None

    def _on_press(self, key):
        """
        Callback function for pynput Listener when a key is pressed.
        Sets self.running to False if the ESC key is pressed.
        """
        try:
            if key == Key.esc:
                print("\nEmergency ESC key pressed. Stopping bot...")
                self.running = False
                return False  # Stop the listener
        except AttributeError:
            # Handle special keys that don't have a char attribute
            pass

    def setup(self):
        """
        Sets up the pynput Listener to monitor for the ESC key press.
        """
        self.listener = Listener(on_press=self._on_press)

    def start(self):
        """
        Starts the keyboard listener in a non-blocking way.
        """
        if self.listener:
            self.listener.start()
            print("Emergency stop listener started. Press ESC to stop the bot.")
        else:
            print("Emergency stop listener not set up. Call setup() first.")

    def stop(self):
        """
        Stops the keyboard listener.
        """
        if self.listener:
            self.listener.stop()
            self.listener.join() # Wait for the listener thread to finish
            print("Emergency stop listener stopped.")


# =============================================================================
# PART 2: GAME LOGIC AND EXECUTION
# =============================================================================

import time
import sys
import threading
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any

# --- MOCK CLASSES FROM PART 1 (FOR INDEPENDENT TESTING) ---
# In a real scenario, these would be imported from your Part 1 file.
# These mocks provide minimal functionality to allow GameBot to run without
# requiring the full Yolo/ScreenCapture setup.

@dataclass
class Detection:
    """Represents a detected object in the game."""
    label: str
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    center_x: int = None
    center_y: int = None

    def __post_init__(self):
        if self.center_x is None:
            self.center_x = (self.bbox[0] + self.bbox[2]) // 2
        if self.center_y is None:
            self.center_y = (self.bbox[1] + self.bbox[3]) // 2

class YOLODetector:
    """Mock YOLODetector for Part 2. In a real scenario, this loads a YOLO model."""
    def __init__(self, model_path: str, confidence_threshold: float):
        print(f"Mock YOLODetector initialized: model='{model_path}', conf={confidence_threshold}")

    def detect(self, screenshot: Any) -> List[Detection]:
        """
        Mock detection method. Returns sample detections for testing purposes.
        In a real scenario, this would run the YOLO model on the screenshot
        and return actual detected objects.
        """
        current_time = time.time()
        
        # Simulate a projectile moving horizontally
        proj_x = 200 + int(300 * ((current_time / 2.0) % 1)) # Moves between 200 and 500 every 2 seconds
        
        # Simulate an enemy moving slightly
        enemy_x = 550 + int(50 * ((current_time / 3.0) % 1))
        
        # Simulate a pink object appearing periodically
        pink_object_detections = []
        if int(current_time * 2) % 10 == 0: # Appears briefly every 5 seconds
            pink_object_detections = [Detection(label='pink_object', bbox=(380, 200, 420, 240), confidence=0.9, center_x=400, center_y=220)]

        return [
            Detection(label='projectile', bbox=(proj_x, 400, proj_x + 30, 430), confidence=0.8, center_x=proj_x + 15, center_y=415),
            Detection(label='enemy', bbox=(enemy_x, 300, enemy_x + 80, 400), confidence=0.9, center_x=enemy_x + 40, center_y=350),
            Detection(label='item', bbox=(100, 500, 120, 520), confidence=0.7, center_x=110, center_y=510),
        ] + pink_object_detections

class ScreenCapture:
    """Mock ScreenCapture for Part 2. In a real scenario, this captures game frames."""
    def __init__(self, region: Tuple[int, int, int, int]):
        self.region = region
        print(f"Mock ScreenCapture initialized for region: {region}")

    def capture(self):
        """Mock capture method. Returns a dummy object instead of actual image data."""
        # In a real scenario, this would capture a screenshot using libraries like mss or pyautogui.
        return "mock_screenshot_data"

class KeyboardController:
    """Mock KeyboardController for Part 2. In a real scenario, this controls keyboard inputs."""
    def __init__(self):
        self.held_keys = set()
        print("Mock KeyboardController initialized")

    def press(self, key: str):
        if key not in self.held_keys:
            print(f"[KEYBOARD] PRESS: {key}")
            self.held_keys.add(key)

    def release(self, key: str):
        if key in self.held_keys:
            print(f"[KEYBOARD] RELEASE: {key}")
            self.held_keys.remove(key)

    def tap(self, key: str, duration: float = 0.05):
        print(f"[KEYBOARD] TAP: {key} (duration: {duration:.2f}s)")
        # In a real scenario, this would press, sleep, then release the key.
        pass

    def release_all(self):
        if self.held_keys:
            print(f"[KEYBOARD] RELEASE ALL: {list(self.held_keys)}")
            self.held_keys.clear()

class EmergencyStop:
    """Mock EmergencyStop for Part 2. In a real scenario, listens for an emergency key press."""
    def __init__(self):
        self._stop_event = threading.Event()
        self._listener_thread = None
        print("Mock EmergencyStop initialized")

    def _listen_for_stop(self, stop_callback):
        """Simulates listening for a stop signal."""
        print("Mock EmergencyStop: Listener started. Bot will stop via Ctrl+C in console.")
        while not self._stop_event.is_set():
            time.sleep(0.1) # Simulate checking for stop signal
        stop_callback()

    def start_listener(self, stop_callback):
        """Starts the emergency stop listener in a separate thread."""
        self._listener_thread = threading.Thread(target=self._listen_for_stop, args=(stop_callback,), daemon=True)
        self._listener_thread.start()

    def stop_listener(self):
        """Signals the listener thread to stop."""
        print("Mock EmergencyStop: Signaling listener to stop.")
        self._stop_event.set()
        if self._listener_thread and self._listener_thread.is_alive():
            self._listener_thread.join(timeout=1) # Wait for thread to finish
            if self._listener_thread.is_alive():
                print("Warning: EmergencyStop listener thread did not terminate cleanly.")
# --- END MOCK CLASSES ---


# Configuration and constants
class Config:
    """Configuration settings for the Cuphead AI Bot."""
    MODEL_PATH: str = "path/to/cuphead_yolov8n.pt"  # Replace with your actual model path
    CONFIDENCE_THRESHOLD: float = 0.5  # Minimum confidence for a detection to be considered

    # Screen region for capture (x, y, width, height)
    # Adjust this based on your game window position and size.
    # Example: (0, 0, 800, 600) for a 800x600 window at top-left.
    SCREEN_REGION: Tuple[int, int, int, int] = (0, 0, 800, 600) 

    FPS: int = 15  # Target frames per second for the bot's logic loop

    # Distance thresholds (pixels) relative to Cuphead's assumed position
    # These values will require fine-tuning for optimal gameplay.
    DODGE_DISTANCE_THRESHOLD: int = 120  # How close a projectile needs to be to trigger a dodge
    ENEMY_ATTACK_DISTANCE_THRESHOLD: int = 500 # Range to consider attacking an enemy
    ITEM_COLLECT_DISTANCE_THRESHOLD: int = 100 # How close an item needs to be to move towards it
    PINK_OBJECT_PARRY_DISTANCE_THRESHOLD: int = 60 # Very close for parry timing

    # Assumed player position within the SCREEN_REGION
    # This is a heuristic. A real bot could detect Cuphead's precise location.
    # We assume Cuphead is typically near the bottom-middle of the screen.
    PLAYER_CENTER_X: int = SCREEN_REGION[2] // 2
    PLAYER_CENTER_Y: int = SCREEN_REGION[3] * 3 // 4


# Extend Detection class for bot-specific utility
def _distance_to_player(detection: Detection) -> float:
    """Calculates Euclidean distance from detection to assumed player position."""
    dx = detection.center_x - Config.PLAYER_CENTER_X
    dy = detection.center_y - Config.PLAYER_CENTER_Y
    return (dx**2 + dy**2)**0.5

# Dynamically add the method to the Detection class
setattr(Detection, 'distance_to_player', _distance_to_player)


class GameBot:
    """
    Main class for the Cuphead AI Bot. It orchestrates screen capture, object detection,
    decision-making, and game control execution using the components from Part 1.
    """
    def __init__(self):
        """
        Initializes all necessary components: YOLODetector, ScreenCapture,
        KeyboardController, and EmergencyStop. Sets up the bot's running state.
        """
        self.detector = YOLODetector(Config.MODEL_PATH, Config.CONFIDENCE_THRESHOLD)
        self.screen = ScreenCapture(Config.SCREEN_REGION)
        self.keyboard = KeyboardController()
        self.emergency_stop = EmergencyStop()
        self.running = True

        # State management for continuous actions like shooting
        self._is_shooting_held = False

        print("GameBot initialized and ready.")

    def _stop_callback(self):
        """Callback function invoked by EmergencyStop to signal the bot to stop."""
        print("!!! Emergency stop initiated. Bot is shutting down. !!!")
        self.running = False

    def categorize_detections(self, detections: List[Detection]) -> Dict[str, List[Detection]]:
        """
        Categorizes raw object detections into predefined gameplay-relevant groups.

        Args:
            detections (List[Detection]): A list of raw Detection objects from the YOLO model.

        Returns:
            Dict[str, List[Detection]]: A dictionary where keys are category names (e.g., 'enemies',
                                       'projectiles') and values are lists of relevant Detection objects,
                                       sorted by distance to the player for priority.
        """
        categorized: Dict[str, List[Detection]] = {
            'enemies': [],
            'projectiles': [],
            'items': [],
            'obstacles': [], # Not used in current logic, but good to have
            'pink_objects': [],
            'player': [] # If player detection is ever implemented
        }

        for d in detections:
            if d.label == 'enemy':
                categorized['enemies'].append(d)
            elif d.label == 'projectile':
                categorized['projectiles'].append(d)
            elif d.label == 'item':
                categorized['items'].append(d)
            elif d.label == 'obstacle':
                categorized['obstacles'].append(d)
            elif d.label == 'pink_object':
                categorized['pink_objects'].append(d)
            # Add other labels as needed based on your YOLO model's output

        # Sort important categories by distance to player for easier prioritization
        categorized['projectiles'].sort(key=lambda d: d.distance_to_player())
        categorized['enemies'].sort(key=lambda d: d.distance_to_player())
        categorized['items'].sort(key=lambda d: d.distance_to_player())
        categorized['pink_objects'].sort(key=lambda d: d.distance_to_player())

        return categorized

    def make_decision(self, categorized_detections: Dict[str, List[Detection]]) -> str:
        """
        Makes a high-level action decision based on the current game state (detections).
        Follows a simple priority-based decision tree.

        Args:
            categorized_detections (Dict[str, List[Detection]]): Categorized detected objects.

        Returns:
            str: A string representing the chosen action (e.g., "parry", "dodge",
                 "attack", "collect", "explore").
        """
        # Decision Hierarchy (highest priority first)
        
        # 1. Parry pink objects (requires precise timing and being airborne)
        if categorized_detections['pink_objects']:
            closest_pink = categorized_detections['pink_objects'][0]
            if closest_pink.distance_to_player() < Config.PINK_OBJECT_PARRY_DISTANCE_THRESHOLD:
                # Basic parry attempt. More complex logic would check if Cuphead is airborne.
                return "parry"

        # 2. Dodge incoming projectiles (immediate threat)
        if categorized_detections['projectiles']:
            closest_projectile = categorized_detections['projectiles'][0]
            if closest_projectile.distance_to_player() < Config.DODGE_DISTANCE_THRESHOLD:
                # A more sophisticated bot would determine optimal dodge direction.
                return "dodge"

        # 3. Attack enemies
        if categorized_detections['enemies']:
            # Prioritize attacking if any enemies are within range
            return "attack"

        # 4. Collect items (if no immediate threats)
        if categorized_detections['items']:
            closest_item = categorized_detections['items'][0]
            if closest_item.distance_to_player() < Config.ITEM_COLLECT_DISTANCE_THRESHOLD:
                return "collect"

        # 5. Default action: Explore / Progress right
        return "explore"

    def execute_action(self, action: str, categorized_detections: Dict[str, List[Detection]] = None):
        """
        Executes the chosen action by sending commands to the KeyboardController.
        Manages key presses and releases for various game actions.

        Args:
            action (str): The action to perform (e.g., "parry", "dodge", "attack").
            categorized_detections (Dict[str, List[Detection]], optional): Used for
                                       context-aware actions like moving towards an enemy/item.
        """
        # Release 'J' (shoot) if the current action doesn't require continuous shooting
        if action not in ["attack", "explore"] and self._is_shooting_held:
            self.keyboard.release('J')
            self._is_shooting_held = False
        
        # Always release aim keys unless explicitly used
        if action not in ["attack"]: # Attack may need W/S
            self.keyboard.release('W')
            self.keyboard.release('S')

        # Always release directional movement keys unless explicitly used
        if action not in ["attack", "collect", "explore"]:
            self.keyboard.release('A')
            self.keyboard.release('D')

        if action == "parry":
            self.keyboard.tap('Spacebar', duration=0.05) # First tap to jump
            time.sleep(0.05) # Small delay to ensure airborne state for parry
            self.keyboard.tap('Spacebar', duration=0.05) # Second tap to parry
            print(f"Executing: {action.upper()}")

        elif action == "dodge":
            self.keyboard.tap('L', duration=0.1) # 'L' is Dash
            print(f"Executing: {action.upper()}")

        elif action == "attack":
            # Hold 'J' for continuous shooting
            if not self._is_shooting_held:
                self.keyboard.press('J')
                self._is_shooting_held = True
            
            # Basic movement and aiming towards the closest enemy
            if categorized_detections and categorized_detections['enemies']:
                closest_enemy = categorized_detections['enemies'][0]
                
                # Horizontal movement
                if closest_enemy.center_x < Config.PLAYER_CENTER_X - 50: # Enemy is significantly to the left
                    self.keyboard.press('A')
                    self.keyboard.release('D')
                elif closest_enemy.center_x > Config.PLAYER_CENTER_X + 50: # Enemy is significantly to the right
                    self.keyboard.press('D')
                    self.keyboard.release('A')
                else: # Enemy is relatively centered, try to maintain position
                    self.keyboard.release('A')
                    self.keyboard.release('D')

                # Vertical aiming
                if closest_enemy.center_y < Config.PLAYER_CENTER_Y - 80: # Enemy is significantly above
                    self.keyboard.press('W') # Aim Up
                    self.keyboard.release('S')
                elif closest_enemy.center_y > Config.PLAYER_CENTER_Y + 80: # Enemy is significantly below
                    self.keyboard.press('S') # Aim Down (or crouch)
                    self.keyboard.release('W')
                else: # Enemy is roughly on the same vertical level
                    self.keyboard.release('W')
                    self.keyboard.release('S')
            else:
                # Fallback: if 'attack' is chosen but no enemies detected (e.g., just cleared screen)
                # continue moving right and shooting.
                self.keyboard.press('D')
                self.keyboard.release('A')
                self.keyboard.release('W')
                self.keyboard.release('S') # Release aim if no specific target
            print(f"Executing: {action.upper()}")

        elif action == "collect":
            if categorized_detections and categorized_detections['items']:
                closest_item = categorized_detections['items'][0]
                # Move towards the item
                if closest_item.center_x < Config.PLAYER_CENTER_X - 20: # Item to the left
                    self.keyboard.press('A')
                    self.keyboard.release('D')
                elif closest_item.center_x > Config.PLAYER_CENTER_X + 20: # Item to the right
                    self.keyboard.press('D')
                    self.keyboard.release('A')
                else:
                    self.keyboard.release('A')
                    self.keyboard.release('D')
                
                # If item is high, consider jumping
                if closest_item.center_y < Config.PLAYER_CENTER_Y - 50:
                    self.keyboard.tap('Spacebar') # Jump
            else:
                # Fallback if 'collect' chosen but no items found
                print(f"Executing: {action.upper()} (No item found, defaulting to explore)")
                self.execute_action("explore", categorized_detections) # Recurse for fallback
                return # Exit to prevent double action
            print(f"Executing: {action.upper()}")

        elif action == "explore":
            # Default action: Move right and continuously shoot
            self.keyboard.press('D') # Move right
            self.keyboard.release('A')

            if not self._is_shooting_held:
                self.keyboard.press('J') # Shoot
                self._is_shooting_held = True
            
            # Ensure no aiming up/down when just exploring
            self.keyboard.release('W')
            self.keyboard.release('S')
            print(f"Executing: {action.upper()}")

        time.sleep(0.01) # Small delay after key presses to ensure action registration


    def run(self):
        """
        The main loop of the GameBot. It continuously performs the following steps:
        1. Captures the current game screen.
        2. Detects objects (enemies, projectiles, items, etc.).
        3. Makes a strategic decision based on detections.
        4. Executes the chosen action via keyboard commands.
        5. Limits the loop's FPS to avoid excessive processing and improve stability.
        """
        print("\nStarting GameBot main loop...")
        self.emergency_stop.start_listener(self._stop_callback)

        frame_count = 0
        start_time = time.time()

        while self.running:
            loop_start_time = time.time()

            # 1. Capture screen
            screenshot = self.screen.capture()

            # 2. Detect objects
            detections = self.detector.detect(screenshot)
            categorized_detections = self.categorize_detections(detections)

            # 3. Make decision
            action = self.make_decision(categorized_detections)

            # 4. Execute action
            self.execute_action(action, categorized_detections)

            frame_count += 1
            
            # FPS limiting to prevent CPU overload and ensure consistent timing
            elapsed_time = time.time() - loop_start_time
            sleep_time = (1.0 / Config.FPS) - elapsed_time
            if sleep_time > 0:
                time.sleep(sleep_time)
            
            # Optional: Print actual FPS periodically
            current_time = time.time()
            if current_time - start_time > 1.0:
                actual_fps = frame_count / (current_time - start_time)
                # print(f"Current FPS: {actual_fps:.2f}")
                frame_count = 0
                start_time = current_time

        # Cleanup actions when the bot stops
        print("\nGameBot loop terminated.")
        self.keyboard.release_all() # Ensure all keys are released
        self.emergency_stop.stop_listener() # Stop the emergency listener thread
        print("GameBot shutdown procedures complete.")


game_title = "Cuphead"

if __name__ == "__main__":
    print("=" * 60)
    print(f"{game_title} AI Bot - Part 2: Game Logic and Execution")
    print("=" * 60)
    print("Instructions:")
    print("1. Ensure 'Cuphead' is running and you are in a level.")
    print(f"2. The bot will try to control the game within the region: {Config.SCREEN_REGION}")
    print("3. To stop the bot, press Ctrl+C in this console.")
    print("   (An actual EmergencyStop listener for in-game 'ESC' would be implemented in Part 1).")
    print("\nStarting bot in 3 seconds...")
    time.sleep(3)

    bot = GameBot()

    try:
        bot.run()
    except KeyboardInterrupt:
        print("\nUser manually stopped the bot via KeyboardInterrupt (Ctrl+C).")
        # If run() was interrupted, its internal cleanup might not have completed.
        # Ensure final cleanup here.
        if bot.running: # If the internal loop didn't set running to False yet
            bot._stop_callback()
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        # Ensure cleanup even on unforeseen errors
        if bot.running:
            bot.running = False # Force stop the loop if it's still running
            bot.keyboard.release_all()
            bot.emergency_stop.stop_listener()
    finally:
        print("\nBot execution block finished.")
        sys.exit(0)