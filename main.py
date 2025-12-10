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
import game
import subprocess
import os
# path to the game folder
def main():
   game_path = Path("game/game")   
   if game_path.exists == False:
     print("Game folder not found exists.")
     game_path.mkdir(parents=True, exist_ok=True      )   
   else:
     subprocess.run([r"C:\path\to\your\game.exe", str(game_path)])
     while subprocess.run == True:
         take_screenshot(create_screenshots_directory())















































































































































if __name__ == "__main__":
    main()