# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Paper Engine is a game automation and computer vision project that:
1. Runs Windows game executables (via Wine on Linux)
2. Captures gameplay screenshots automatically during gameplay
3. Annotates screenshots using Label Studio for object detection
4. Trains YOLO models on the annotated dataset for game AI/bot development

The project uses Cuphead as the reference game but is designed to work with any game executable.

## Architecture

### Main Workflow (main.py:23-52)

The main script orchestrates two primary modes:

1. **Game + Screenshot Mode**: Launches a game executable from `game/` directory, automatically captures screenshots every 5 seconds during gameplay, and opens Label Studio for annotation after the game closes
2. **Annotation-Only Mode**: Skips game execution and directly opens Label Studio for annotating existing screenshots

Key components:
- `game_exe_func.py`: Discovers and validates `.exe` files in the `game/` directory
- `screencapture.py`: Handles periodic screenshot capture using flameshot
- `training_model.py`: Placeholder for YOLO model training (not fully implemented)

### Screenshot Capture System (screencapture.py)

Uses platform-specific screenshot methods:
- **Linux (Wayland)**: Uses `flameshot` command-line tool with specific display server configuration
- **Windows**: Falls back to PIL ImageGrab (not actively tested)

Screenshots are saved to `screenshots/` with timestamp filenames (format: `screenshot_YYYYMMDD_HHMMSS.png`).

### Dataset Structure

The `dataset/` directory contains Label Studio annotation exports in JSON format. Each file is a numbered annotation with:
- Bounding box coordinates (x, y, width, height) in percentage values
- Object labels (e.g., "cuphead", "mugman", "startup button")
- Original image dimensions (1920x1080)
- Task metadata and user information

These annotations are meant to be imported into YOLO training format.

## Development Commands

### Environment Setup
```bash
# Activate virtual environment (required for all Python commands)
source env/bin/activate

# Install dependencies (if not already installed)
pip install ultralytics label-studio label-studio-sdk pynput torch torchvision
```

### Running the Application
```bash
# Main workflow: run game and capture screenshots
python main.py
# When prompted "To run [game name] Y:N", type Y to launch game or N to skip to annotation

# Run Label Studio manually (port 8080)
label-studio start --port 8080
# Access at http://localhost:8080
```

### File Requirements
- Place game `.exe` files in `game/` directory
- Ensure Wine is installed for running Windows executables on Linux
- Ensure flameshot is installed on Linux systems for screenshot capture

## Key Environment Variables

The application sets `LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true` to allow Label Studio to access local screenshot files from the `screenshots/` directory.

## Important Implementation Details

### Wine Game Execution
Games are launched via Wine with `subprocess.Popen(['wine', exe_path])` (main.py:31). The script polls the game process every 5 seconds and captures screenshots while the game is running.

### Screenshot Timing
Screenshots are captured every 5 seconds during active gameplay (main.py:36-37). The delay is hardcoded in the main loop.

### Label Studio Integration
After game execution completes, Label Studio is automatically launched in a new session using `start_new_session=True` to prevent it from blocking the terminal (main.py:41).

### YOLO Training
The `training_model.py` file contains a basic YOLO training stub that needs to be properly configured:
- Dataset path should point to properly formatted YOLO dataset (currently hardcoded to "/dataset")
- Epochs set to 5 for quick testing
- Uses YOLOv8n (nano) model as base

## Testing

No formal test suite exists. Manual testing workflow:
1. Place a game executable in `game/`
2. Run `python main.py`
3. Play the game briefly
4. Close the game
5. Verify screenshots appear in `screenshots/`
6. Verify Label Studio opens automatically
7. Import and annotate screenshots in Label Studio
