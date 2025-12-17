import os
import time
from datetime import datetime
from PIL import ImageGrab
from pynput import keyboard
import threading
import sys
import subprocess
# Global flag to control the program
should_stop = False

def on_key_press(key):
    """
    Callback function that runs when a key is pressed.
    Sets the global flag to True when 'q' is pressed.
    """
    global should_stop
    try:
        # Check if the key pressed is 'q'
        if key.char == 'q':
            print("\n'q' pressed - stopping screenshot capture...")
            should_stop = True
            return False  # Stop the listener
    except AttributeError:
        # Special keys (like ctrl, alt, etc.) don't have a char attribute
        pass

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
          subprocess.run(['grim', str(filepath)], check=True, stderr=subprocess.DEVNULL)
          print(f"✓ Screenshot (grim): {filename}")
          return str(filepath)
       except Exception as e:
          print(f"Error taking screenshot: {e}")
def main():
    """
    Main function that coordinates the screenshot capture process.
    """
    global should_stop
    
    print("=== Screenshot Capture Program ===")
    print("Press 'q' at any time to quit\n")
    
    # set default interval 
    
    interval = 10
            
    
    # Create screenshots directory
    directory = create_screenshots_directory()
    
    # Start keyboard listener in a separate thread
    listener = keyboard.Listener(on_press=on_key_press)
    listener.start()
    
    print(f"\nStarting screenshot capture every {interval} seconds...")
    print("Press 'q' to stop\n")
    
    # Main screenshot loop
    screenshot_count = 0
    while not should_stop:
        take_screenshot(directory)
        screenshot_count += 1
        
        # Wait for the specified interval, but check frequently if we should stop
        # This allows the program to respond quickly to the quit command
        elapsed = 0
        check_interval = 0.1  # Check every 100ms
        while elapsed < interval and not should_stop:
            time.sleep(check_interval)
            elapsed += check_interval
    
    # Clean up
    listener.stop()
    print(f"\nProgram ended. Total screenshots taken: {screenshot_count}")

if __name__ == "__main__":
    main()