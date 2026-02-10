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

def check_screenshot_tool():
    """
    Check which screenshot tool is available on the system.
    Returns the preferred tool name or None if none are available.
    """
    tools = ['flameshot', 'scrot', 'import']  # Priority order
    for tool in tools:
        try:
            result = subprocess.run(['which', tool], capture_output=True, text=True)
            if result.returncode == 0:
                return tool
        except Exception:
            continue
    return None

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
       from PIL import ImageGrab
       try:
          # Capture the screen
          screenshot = ImageGrab.grab()
          # Save the screenshot
          screenshot.save(filepath)
          print(f"Screenshot saved: {filename}")
       except Exception as e:
          print(f"Error taking screenshot: {e}")
    else:
       # Try flameshot first (default), then fall back to alternatives
       screenshot_taken = False

       # Try flameshot
       try:
          # Check if we should capture a specific window
          if config.CAPTURE_WINDOW_ONLY and window_geometry:
              # Use flameshot gui with geometry for window-specific capture
              cmd = ['flameshot', 'gui', '--region', window_geometry, '-p', filepath]
              subprocess.run(cmd, check=True)
              print(f"✓ Screenshot (flameshot, window): {filename}")
          else:
              # Use flameshot for full screen
              # SECURITY: Use shell=False and redirect stdout via Python file handle
              with open(filepath, 'wb') as f:
                  subprocess.run(config.FLAMESHOT_COMMAND, stdout=f, check=True)
              print(f"✓ Screenshot (flameshot): {filename}")
          screenshot_taken = True
          return str(filepath)
       except (FileNotFoundError, subprocess.CalledProcessError) as e:
          print(f"⚠️  Flameshot failed: {e}")
          print("Trying alternative screenshot tool...")

       # Fall back to scrot if flameshot failed
       if not screenshot_taken:
           try:
               cmd = ['scrot', filepath]
               subprocess.run(cmd, check=True)
               print(f"✓ Screenshot (scrot): {filename}")
               screenshot_taken = True
               return str(filepath)
           except (FileNotFoundError, subprocess.CalledProcessError):
               print("⚠️  scrot not available or failed")

       # Fall back to ImageMagick import if scrot failed
       if not screenshot_taken:
           try:
               cmd = ['import', '-window', 'root', filepath]
               subprocess.run(cmd, check=True)
               print(f"✓ Screenshot (ImageMagick): {filename}")
               screenshot_taken = True
               return str(filepath)
           except (FileNotFoundError, subprocess.CalledProcessError):
               print("⚠️  ImageMagick import not available or failed")

       # If all tools failed
       if not screenshot_taken:
           print("✗ Error: No screenshot tool available (tried: flameshot, scrot, ImageMagick)")
           print("Install one with: sudo apt install flameshot  OR  sudo apt install scrot")
           return None