import tkinter as tk
from pynput import keyboard
import threading
import time

class ScreenProtector:
    def __init__(self):
        self.running = False
        self.overlay = None
        self.stop_key = keyboard.Key.esc  # ESC to stop
        
    def create_overlay(self):
        """Create semi-transparent overlay"""
        self.overlay = tk.Tk()
        self.overlay.attributes('-fullscreen', True)
        self.overlay.attributes('-alpha', 0.3)  # Transparency
        self.overlay.attributes('-topmost', True)
        self.overlay.configure(bg='black')
        
        # Make it click-through on Windows
        try:
            self.overlay.wm_attributes('-transparentcolor', 'black')
        except:
            pass
            
        # Label for warning
        label = tk.Label(
            self.overlay, 
            text="PROTECTED", 
            fg="red", 
            bg="black",
            font=("Arial", 48)
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