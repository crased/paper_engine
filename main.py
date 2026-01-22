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
from screencapture import take_screenshot, create_screenshots_directory
from pathlib import Path
import subprocess
import os
import time
import sys
from pynput import keyboard
from game_exe_func import get_title, path_finder

def main(): 
   game_path = "game/"
   exe_path = path_finder(game_path)
   env = os.environ.copy()
   env["LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED"] = "true"
   print(f"To run {str(get_title(game_path))} Y:N")
   if input() == "Y":
     if sys.platform == "linux":
      game_process = subprocess.Popen(["wine", exe_path])
     time.sleep(10)
       # Adjust delay as needed
     while game_process.poll() is None:
         time.sleep(5)
         take_screenshot(create_screenshots_directory())
     if game_process.poll() is not None:
        if sys.platform == "linux":
          command = ["label-studio","start" ,"--port", "8080"]   
          label_process = subprocess.Popen(command,env=env,start_new_session=True)
   elif input() == "N":
     # this is kinda redundant ill probbaly change this system
     if sys.platform == "linux":#in future change env so i can just call "label-studio"
          command = ["label-studio","start" ,"--port", "8080"]   
          label_process = subprocess.Popen(command,env=env,start_new_session=True)
   if game_path is None:
     game_path.mkdir(parents=True,exist_ok=True)
 #annotate data use ai to finishing annotation then setup export so data gets pumped back into complex_bot script
 #need to setup some terminal keys like y and n   
     
    
 








































































































































if __name__ == "__main__":
    main()