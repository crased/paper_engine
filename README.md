# Paper Engine

> **⚠️ DISCLAIMER: WORK IN PROGRESS**
>
> This project is currently under active development and is **NOT** production-ready. Expect:
> - Incomplete features and functionality
> - Breaking changes without notice
> - Bugs and stability issues
> - Incomplete documentation
> - Experimental code and workflows
>
> **Use at your own risk. This is a research/educational project.**

A game automation and computer vision framework that combines Wine game execution, screenshot capture, Label Studio annotation, and YOLO object detection to build intelligent game bots.

## What It Does

Paper Engine automates the complete pipeline for creating game bots:

1. **Game Execution**: Runs Windows game executables via Wine (Linux)
2. **Screenshot Capture**: Automatically captures gameplay screenshots every 5 seconds
3. **Annotation**: Label Studio integration for object detection annotation
4. **Model Training**: Converts annotations to YOLO format and trains YOLO11 models
5. **Controls Discovery**: AI-powered web search for game controls
6. **Bot Generation**: Automatically generates Python bot scripts using YOLO + controls

Currently tested with Cuphead, but designed to work with any game executable.

## Requirements

### System Requirements
- **OS**: Linux (tested on Arch Linux with Wayland)
- **Wine**: For running Windows executables
- **flameshot**: For screenshot capture on Linux/Wayland
- **Python**: 3.8+

### Python Dependencies
```bash
pip install ultralytics label-studio label-studio-sdk pynput torch torchvision anthropic python-dotenv
```

## Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd paper_engine
   ```

2. **Create virtual environment**
   ```bash
   python -m venv env
   source env/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install ultralytics label-studio label-studio-sdk pynput torch torchvision anthropic python-dotenv
   ```

4. **Add your game**
   - Place a Windows `.exe` game file in the `game/` directory

5. **Configure API key** (for AI features)
   - Run `python main.py` once to auto-generate `.env` file
   - Edit `.env` and replace `your-api-key-here` with your API key
   - For Anthropic: Get your key at https://console.anthropic.com/settings/keys
   - Configure LLM provider in `conf/main_conf.ini` (default: anthropic)

## Usage

### Complete End-to-End Workflow

```bash
source env/bin/activate
python main.py
```

This runs the complete pipeline:
1. **Launch game** (Y/N) → Captures screenshots during gameplay
2. **Label Studio** (10 sec countdown, press any key to skip) → Annotate screenshots
3. **Train YOLO model** (Y/N) → Convert annotations and train
4. **Search game controls** (Y/N) → AI searches web for controls
5. **Generate bot script** (Y/N) → Creates executable bot script

### Standalone Utilities

**Generate controls and bot script only:**
```bash
python generate_bot_script.py
```

**Train model manually:**
```bash
python training_model.py
```

**Launch Label Studio manually:**
```bash
label-studio start --port 8080
# Access at http://localhost:8080
```

### Run Your Bot

After completing the pipeline:
```bash
python bot_scripts/<game_name>_bot.py
```

Press ESC to stop the bot.

## Project Structure

```
paper_engine/
├── game/                    # Place .exe files here
├── screenshots/             # Auto-captured screenshots
├── dataset/                 # Label Studio annotations (JSON)
├── yolo_dataset/           # YOLO-formatted dataset
├── runs/detect/            # Trained models
├── conf/                   # Configuration files
│   ├── main_conf.ini       # Main settings
│   └── <game>_controls.ini # Game controls (auto-generated)
├── bot_scripts/            # Generated bot scripts
│   └── <game>_bot.py       # Executable bot (auto-generated)
├── main.py                 # Main workflow
├── generate_bot_script.py  # Controls search & bot generation
├── training_model.py       # YOLO training pipeline
├── screencapture.py        # Screenshot capture system
└── game_exe_func.py        # Game executable discovery
```

## Architecture

### Screenshot Capture
- **Linux (Wayland)**: Uses `flameshot` CLI with display server configuration
- **Windows**: Falls back to PIL ImageGrab (not actively tested)
- Screenshots saved every 5 seconds during gameplay to `screenshots/`

### YOLO Training Pipeline
1. **Convert**: Label Studio JSON → YOLO format (normalized bboxes)
2. **Prepare**: Split dataset into train/val (80/20)
3. **Train**: YOLO11n model (50 epochs, batch size 16)
   - YOLO11 benefits: 5x faster training, 36% faster CPU inference vs YOLOv8
4. **Export**: TorchScript and ONNX formats
- Output: `runs/detect/paper_engine_model/weights/best.pt`

### AI Bot Generation
Uses Claude Opus AI to:
1. Search web for game keyboard/mouse controls
2. Generate complete Python bot scripts with:
   - YOLO model inference
   - Real-time screenshot capture
   - Decision-making logic
   - Game control via pynput
   - Emergency stop (ESC)

## Known Issues & Limitations

> **⚠️ UNFINISHED FEATURES**
>
> - **Limited platform testing**: Only tested on Linux/Wayland
> - **Windows support**: Screenshot capture not verified on Windows
> - **Wine compatibility**: Not all games may work via Wine
> - **Bot intelligence**: Generated bots have basic logic, may need manual refinement
> - **Model accuracy**: Requires substantial annotated data for good performance
> - **Error handling**: Minimal error handling in some components
> - **Configuration**: Limited customization options
> - **Testing**: No formal test suite exists

## Security Notes

- API keys stored in `.env` (never committed to git)
- `.env` auto-generated on first run
- Never share or commit your API keys
- Review generated bot scripts before running

## Testing

Manual testing workflow:
1. Place game `.exe` in `game/`
2. Run `python main.py`
3. Play game briefly, then close
4. Verify screenshots in `screenshots/`
5. Annotate in Label Studio (optional)
6. Train model and verify output
7. Generate and test bot script

## Contributing

This is an experimental project. Contributions welcome, but expect breaking changes.

## License

[Add your license here]

## Acknowledgments

- Uses [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) for object detection
- Uses [Label Studio](https://labelstud.io/) for annotation
- Uses [Claude AI](https://claude.ai) for controls search and bot generation
- Tested with Cuphead by Studio MDHR
