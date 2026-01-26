 **Dynamic Game Wallpaper Engine**

A desktop wallpaper application that transforms game executables into live, interactive backgrounds.

**Core Features:**

- **Game Integration**:
    # Features
    # import and run feature 
    # **exe file check** and **file gettr**
    # 
    #
    #

-  **Performance mode**
     just take a recording of you/bot playing a game
- **Smart Replay System**: Automatically "records", ,stores and loops your last 5 minutes of gameplay when the wallpaper isn't in focus
- **Selective Interactivity**: Define specific screen regions or "hotspots" where mouse/keyboard input passes through to the wallpaper game, while other areas remain normal desktop space
- **Performance Mode**: Automatically pauses or reduces framerate when running fullscreen applications

**Interface:**

- **GUI Control Panel**:
    - Game library manager
    - Recording settings (loop duration, quality)
    - Interactive zone editor (draw rectangles to define interactive areas)
    - Performance profiles (active/idle states)
    - Preview window before applying

**Playback Behavior:**

- When desktop is visible: Shows live gameplay or recorded loop
- When you hover/click designated areas: Allows real-time interaction
- When other apps are focused: Cycles through your recorded gameplay segments
- Smooth transitions between live and recorded states

**how to use**
- You can either just run game or you can put game.exe in game.exe folder run and screenshots.py

**Configuration:**

- Configuration files are located in the `conf` directory in INI format for easy editing
- **main_config.ini**: Game execution, screenshot intervals, Label Studio settings
- **screencapture_config.ini**: Screenshot capture and screen protection settings
- **training_config.ini**: YOLO model training parameters

To customize settings, simply edit the `.ini` files in any text editor. Changes take effect on next run.

Example - Change screenshot interval:
```ini
[Screenshot]
screenshot_interval = 10  # Changed from default 5 seconds
```
