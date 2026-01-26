"""
Configuration file for main.py
Paper Engine - Game execution and screenshot capture settings
"""

# Game directory settings
GAME_PATH = "game/"

# Screenshot capture settings
SCREENSHOT_INTERVAL = 5  # Seconds between screenshots during gameplay
GAME_INITIALIZATION_WAIT = 10  # Seconds to wait after launching game before capturing

# Label Studio settings
LABEL_STUDIO_PORT = 8080
LABEL_STUDIO_ENV = {
    "LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED": "true"
}

# User interaction
DEFAULT_LAUNCH_GAME = True  # Set to False to skip directly to annotation mode
PROMPT_USER_FOR_GAME_LAUNCH = True  # Set to False to use DEFAULT_LAUNCH_GAME without prompting

# Directory validation
AUTO_CREATE_DIRECTORIES = True  # Automatically create missing directories
