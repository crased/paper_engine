#import neccesary files
#from screenshot import take_screenshot, create_screenshots_directory
#from bot_script import game_bot
#import game
#then run bot_script on supported game this is something i want other people to build.
#for showcase though ill do like doom or something.
#while bot_script  == True.
#activate screengrab.
#if screengrab.py == true.
#set delay to 500ms. this is the delay for screen grab activation.
#record time should be set in screengrab.py.
#after screengrab time == 500.
#return screengrab to footage folder. 
from screenshot import take_screenshot, create_screenshots_directory
from pathlib import Path
import subprocess
import os
import time
# path to the game folder
game_path = Path("game/")
def path_finder(game_path):

   if not game_path.exists():
     print(f"Game folder '{game_path}' not found!")
     return None   
   
   exe_files = list(game_path.glob("*.exe"))
   
   if not exe_files:
       print("No .exe files found in game folder.")
       return None
   exe_path = exe_files[0]
   return exe_path # Assuming the first .exe file is the game executables
def main(): 
   exe_path = path_finder(game_path)

   if game_path.exists():
     game_process = subprocess.Popen([str(exe_path)])  # Replace with actual game executable
     #delay for game to load
     time.sleep(15)  # Adjust delay as needed
     
     while game_process.poll() is None:
         #add like a 15 sec delay on start with countdown.
         take_screenshot(create_screenshots_directory())
         time.sleep(10)  # Delay between screenshots
  











































































































































if __name__ == "__main__":
    main()