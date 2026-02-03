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

def find_wine_window():
    """
    Find the Wine game window using swaymsg (for Wayland/sway).
    Returns the window rect coordinates in "WxH+x+y" format or None if not found.
    """
    try:
        result = subprocess.run(
            ['swaymsg', '-t', 'get_tree'],
            capture_output=True,
            text=True,
            check=True
        )
        import json
        tree = json.loads(result.stdout)

        def search_windows(node):
            """Recursively search for Wine windows"""
            if node.get('app_id') == 'wine' or 'wine' in node.get('app_id', '').lower():
                # Found a Wine window, return its geometry in flameshot format
                rect = node.get('rect', {})
                x = rect.get('x', 0)
                y = rect.get('y', 0)
                w = rect.get('width', 1920)
                h = rect.get('height', 1080)
                return f"{w}x{h}+{x}+{y}"

            for child in node.get('nodes', []) + node.get('floating_nodes', []):
                result = search_windows(child)
                if result:
                    return result
            return None

        return search_windows(tree)
    except Exception as e:
        print(f"Warning: Could not find Wine window: {e}")
        return None

def take_screenshot(directory, window_geometry=None):
    """
    Takes a screenshot and saves it with a timestamp.

    Args:
        directory: Directory to save screenshots
        window_geometry: Optional geometry string for flameshot (e.g., "1920x1080+0+0")
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
          # Check if we should capture a specific window
          if config.CAPTURE_WINDOW_ONLY and window_geometry:
              # Use flameshot gui with geometry for window-specific capture
              cmd = ['flameshot', 'gui', '-g', window_geometry, '-p', filepath]
              subprocess.run(cmd, check=True)
              print(f"✓ Screenshot (flameshot, window): {filename}")
          else:
              # Use flameshot for full screen
              cmd = ' '.join(config.FLAMESHOT_COMMAND) + f' > "{filepath}"'
              subprocess.run(cmd, check=True, shell=True)
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