import game
import os
from pathlib import Path
def get_title(game_path):
   game_titles = []
   for file in Path(game_path).iterdir():
     if file.name.endswith(".exe"):
       title = file.name.rstrip(".exe").strip()
       game_titles.append(title)
   return game_titles
def path_finder(game_path):
   game_path = Path("game/")
#you may have to change games x permisions level to continue.
   if not game_path.exists():
     print(f"Game folder '{game_path}' not found!")
     return None   
   
   exe_files = list(game_path.glob("*.exe"))
   
   if not exe_files:
       print("No .exe files found in game folder.")
       return None
   exe_path = exe_files[0]
   return exe_path # Assuming the first .exe file is the game executables    
      






