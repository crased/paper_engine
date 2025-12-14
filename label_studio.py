import subprocess
import sys
from label_studio_sdk import LabelStudio
import os
from pathlib import Path

def label_studio():
  if sys.platform == "win32":
    command = [sys.executable, "m", "label-studio", "start", "--port", "8080"]
    label_process = subprocess.Popen(command,creationflags=subprocess.CREATE_NEW_CONSOLE )
  else:
    command = ["label-studio","start" ,"--port", "8080"]   
    label_process = subprocess.Popen(command,start_new_session=True)