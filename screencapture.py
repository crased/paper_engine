from pynput import keyboard
from datetime import datetime
import os
import time
import threading
import sys
import subprocess
import tkinter as tk
import threading
import time
from config import screencapture_config as config




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
    env_vars = os.environ.copy()
    env_vars["XDG_CURRENT_DESKTOP"] = "sway"
    env_vars["QT_QPA_PLATFORM"] = "wayland"
    
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
          subprocess.run(config.FLAMESHOT_COMMAND, check=True, stdout=open(filepath, 'wb'), env=env_vars)
          print(f"✓ Screenshot (flameshot): {filename}")
          return str(filepath)
       except Exception as e:
        print(f"Error taking screenshot: {e}")

class ScreenProtector:
    def __init__(self):
        self.running = False
        self.overlay = None
        self.stop_key = getattr(keyboard.Key, config.SCREEN_PROTECTOR_STOP_KEY)

    def create_overlay(self):
        """Create semi-transparent overlay"""
        self.overlay = tk.Tk()
        self.overlay.attributes('-fullscreen', True)
        self.overlay.attributes('-alpha', config.OVERLAY_ALPHA)
        self.overlay.attributes('-topmost', True)
        self.overlay.configure(bg=config.OVERLAY_COLOR)

        # Make it click-through on Windows
        try:
            self.overlay.wm_attributes('-transparentcolor', config.OVERLAY_COLOR)
        except:
            pass

        # Label for warning
        label = tk.Label(
            self.overlay,
            text=config.OVERLAY_WARNING_TEXT,
            fg=config.OVERLAY_TEXT_COLOR,
            bg=config.OVERLAY_COLOR,
            font=(config.OVERLAY_FONT_FAMILY, config.OVERLAY_FONT_SIZE)
        )
        label.pack(expand=True)
        
    def keyboard_listener(self):
        """Listen for keyboard events"""
        def on_press(key):
            # Stop on ESC
            if key == keyboard.Key.esc:
                self.stop()
                return False
                
            # Block PrintScreen
            if key == keyboard.Key.print_screen:
                return False
                
        with keyboard.Listener(on_press=on_press) as listener:
            listener.join()
    
    def start(self):
        """Start protection"""
        self.running = True
        
        # Start keyboard listener
        kb_thread = threading.Thread(target=self.keyboard_listener)
        kb_thread.daemon = True
        kb_thread.start()
        
        # Create and run overlay
        self.create_overlay()
        self.overlay.mainloop()
    
    def stop(self):
        """Stop protection"""
        self.running = False
        if self.overlay:
            self.overlay.quit()

# Minimal blocker without overlay
class SimpleScreenBlocker:
    def __init__(self):
        self.running = True
        
    def run(self):
        """Just block screenshots"""
        def on_press(key):
            # Block PrintScreen
            if key == keyboard.Key.print_screen:
                print("Screenshot blocked!")
                return False
                
            # Stop on ESC
            if key == keyboard.Key.esc:
                self.running = False
                return False
        
        print("Blocking screenshots. Press ESC to stop.")
        with keyboard.Listener(on_press=on_press) as listener:
            listener.join()

# Usage
if __name__ == "__main__":
    # Option 1: Full overlay
    protector = ScreenProtector()
    protector.start()
    
    # Option 2: Simple blocker
    # blocker = SimpleScreenBlocker()
    # blocker.run()