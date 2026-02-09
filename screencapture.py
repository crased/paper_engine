from pynput import keyboard
from datetime import datetime
import os
import sys
import subprocess
from conf.config_parser import screencapture_conf as config




# Global flag to control the program
should_stop = False
def create_screenshots_directory():
    """
    Creates a 'screenshots' directory if it doesn't exist.
    Returns the path to the directory.
    """
    directory = "screenshots"
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created '{directory}' directory")
    return directory

def take_screenshot(directory):
    """
    Takes a screenshot and saves it with a timestamp.
    """
    # Generate filename with current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"screenshot_{timestamp}.png"
    filepath = os.path.join(directory, filename)

    if sys.platform == "win32":
       from Pil import imageGrab
       try:
          # Capture the screen
          screenshot = ImageGrab.grab()
          # Save the screenshot
          screenshot.save(filepath)
          print(f"Screenshot saved: {filename}")
       except Exception as e:
          print(f"Error taking screenshot: {e}")
    else:
       try:
          # Use shell redirection for flameshot compatibility
          cmd = ' '.join(config.FLAMESHOT_COMMAND) + f' > "{filepath}"'
          subprocess.run(cmd, check=True, shell=True)
          print(f"✓ Screenshot (flameshot): {filename}")
          return str(filepath)
       except Exception as e:
        print(f"Error taking screenshot: {e}")