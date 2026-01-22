import pyautogui
import time

# remove delay after a keypress
pyautogui.PAUSE = 0

class HandleKeypress:
    @staticmethod
    def __handle_multiple__(function, event):
        if isinstance(event, list):
            for event in event: function(event)
        elif isinstance(event, str):
            function(event)
        else:
            raise TypeError("'event' must be a string or a list of strings")

    @classmethod
    def press(cls, event, interval = 0):
        cls.hold(event)
        if interval != 0: time.sleep(interval)
        cls.release(event)

    @classmethod
    def hold(cls, event):
        cls.__handle_multiple__(pyautogui.keyDown, event)

    @classmethod
    def release(cls, event):
        cls.__handle_multiple__(pyautogui.keyUp, event)