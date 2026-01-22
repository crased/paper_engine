from enum import Enum
import pyautogui
DISABLE_FAILSAFE_INTERVAL = True

class Action(str, Enum):
    UP = 'up'
    DOWN = 'down'
    LEFT = 'left'
    RIGHT = 'right'
    SHOOT = 'x'
    JUMP = 'z'
    DASH = 'shift'

class HandleAction:
    @staticmethod
    def execute(action, interval = 0.1):
        pyautogui.press(action, interval)

    @staticmethod
    def hold(action):
        pyautogui.hold(action)

    @staticmethod
    def release(action):
        pyautogui.release(action)

    @staticmethod
    def release_all():
        pyautogui.release([a.value for a in Action])