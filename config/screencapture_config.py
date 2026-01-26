"""
Configuration file for screencapture.py
Paper Engine - Screenshot capture and screen protection settings
"""

# Screenshot directory settings
SCREENSHOTS_DIRECTORY = "screenshots"

# Screenshot file naming
TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"  # Format: YYYYMMDD_HHMMSS
FILENAME_PREFIX = "screenshot_"
FILENAME_EXTENSION = ".png"

# Platform-specific settings for Linux (Wayland)
LINUX_ENV_VARS = {
    "XDG_CURRENT_DESKTOP": "sway",
    "QT_QPA_PLATFORM": "wayland"
}

# Flameshot settings (Linux)
FLAMESHOT_COMMAND = ['flameshot', 'window', '-n', '1', '-r']

# Screen protection settings (for ScreenProtector class)
SCREEN_PROTECTOR_STOP_KEY = "esc"  # Key to stop screen protection
OVERLAY_ALPHA = 0.3  # Transparency level (0.0 to 1.0)
OVERLAY_COLOR = "black"
OVERLAY_WARNING_TEXT = "PROTECTED"
OVERLAY_TEXT_COLOR = "red"
OVERLAY_FONT_SIZE = 48
OVERLAY_FONT_FAMILY = "Arial"

# Screenshot capture behavior
AUTO_CREATE_DIRECTORY = True  # Automatically create screenshots directory if missing
PRINT_CAPTURE_STATUS = True  # Print status messages when capturing screenshots
