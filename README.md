# Paper Engine

> **🚀 Quick Start:** New to Paper Engine? See [QUICKSTART.md](QUICKSTART.md) for a 5-minute setup guide!

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

> **⚠️ PLATFORM SUPPORT:**
> Paper Engine currently **only works on Linux**. The workflow hardcodes Wine for game execution and uses Wayland/Sway-specific window detection.
> **Windows and macOS are NOT supported** and will not work without significant code modifications.

### System Requirements
- **OS**: Linux (tested on Arch Linux with Wayland)
  - **Windows**: ❌ Not supported (requires Wine code removal)
  - **macOS**: ❌ Not supported
- **Wine**: For running Windows game executables on Linux
- **flameshot**: For screenshot capture on Linux/Wayland
- **Python**: 3.8+

### Python Dependencies

Paper Engine will automatically detect and offer to install missing dependencies on first run.

**Core dependencies:**
```bash
pip install pynput Pillow mss python-dotenv pyyaml label-studio
```

**Optional (for model training):**
```bash
pip install ultralytics torch
```

**Optional (for AI bot generation - choose one or more):**
```bash
# Google Gemini (Default - Free tier available)
pip install google-generativeai

# Advanced models (paid API keys required)
pip install anthropic  # For Claude AI
pip install openai     # For GPT models
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

   **Option A: Automatic installation (recommended)**

   Dependencies will be automatically detected on first run. Paper Engine will prompt you to install missing packages.

   **Option B: Manual installation**

   ```bash
   # Full installation (all features):
   pip install -r requirements.txt

   # Or install only what you need:
   # Core only: pip install pynput Pillow mss python-dotenv PyYAML label-studio
   # Add training: pip install ultralytics torch torchvision
   # Add AI (free): pip install google-generativeai
   ```

4. **Add your game**
   - Place a Windows `.exe` game file in the `game/` directory

5. **Configure AI API key** (optional - for bot generation features)

   > **⚠️ API KEY SECURITY WARNING**
   >
   > - **NEVER share your API key** with anyone
   > - **NEVER commit your `.env` file** to git/version control
   > - **NEVER push API keys** to GitHub or any public repository
   > - **Create separate API keys** for each project/environment
   > - **Immediately revoke and regenerate** if your key is exposed
   >
   > Your API key is linked to your billing account. Exposed keys can lead to unauthorized usage and charges.

   **Default: Google Gemini (Free Tier)**

   Paper Engine uses Google Gemini by default, which offers a generous free tier:

   ```bash
   # Get a free API key:
   # 1. Visit https://aistudio.google.com/apikey
   # 2. Click "Create API Key"
   # 3. Copy your key
   ```

   Then configure:
   ```bash
   # Run main.py to auto-generate .env file
   python main.py

   # Edit .env and add your Google API key
   nano .env  # or use any text editor
   # Add: API_KEY=your-google-api-key-here
   ```

   **Optional: Use Advanced Models**

   For more powerful AI capabilities, you can switch to paid models:

   | Provider | Model | Cost | API Key |
   |----------|-------|------|---------|
   | **Google** | Gemini 2.0 Flash | **Free tier** | [Get key](https://aistudio.google.com/apikey) |
   | Anthropic | Claude Opus/Sonnet | Paid | [Get key](https://console.anthropic.com/settings/keys) |
   | OpenAI | GPT-4/GPT-3.5 | Paid | [Get key](https://platform.openai.com/api-keys) |

   To switch providers:
   ```bash
   # Edit conf/main_conf.ini
   nano conf/main_conf.ini

   # Change the [LLM] section:
   [LLM]
   llm_provider = anthropic  # or: openai, google
   llm_model = claude-opus-4-5-20251101  # or: gpt-4, gemini-2.0-flash-exp

   # Then update your API_KEY in .env with the new provider's key
   ```

## Usage

### Quick Start

```bash
./paperengine
```

The script will automatically:
- Create virtual environment if missing
- Activate the environment
- Offer to install dependencies
- Launch Paper Engine

### What Happens Next?

Paper Engine will guide you through a complete 6-step workflow:

1. **Launch game?** → Play for a few minutes to capture screenshots
2. **Label Studio?** → Annotate objects in screenshots (10 sec countdown, press any key to skip)
3. **Train YOLO?** → Train object detection model
4. **Test model?** → Verify detection works
5. **Search controls?** → AI finds keyboard controls (requires API key)
6. **Generate bot?** → AI creates bot script (requires API key)

After completion, your bot is ready to run:
```bash
python bot_scripts/<game_name>_bot.py
```
Press **ESC** to stop the bot.

### Manual Start

```bash
# Activate virtual environment
source env/bin/activate

# Run Paper Engine
python main.py
```

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

Uses AI (default: Google Gemini - free tier) to:
1. Search web for game keyboard/mouse controls
2. Generate complete Python bot scripts with:
   - YOLO model inference
   - Real-time screenshot capture
   - Decision-making logic
   - Game control via pynput
   - Emergency stop (ESC)

Supports multiple LLM providers: Google Gemini (free), Anthropic Claude (paid), OpenAI GPT (paid)

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

### API Key Protection

**CRITICAL: Protect Your API Keys**

- ✅ API keys stored in `.env` file (automatically excluded from git)
- ✅ `.env` auto-generated on first run with placeholder
- ❌ **NEVER share your API keys** with anyone
- ❌ **NEVER commit `.env` to version control** (git, GitHub, etc.)
- ❌ **NEVER push or branch repositories** containing real API keys
- ❌ **NEVER hardcode API keys** in source code

**If your API key is exposed:**
1. Immediately revoke the key in your provider's console
2. Generate a new API key
3. Update your `.env` file with the new key
4. If pushed to git: Use `git filter-branch` or BFG Repo-Cleaner to remove from history

**Best Practices:**
- Use separate API keys for development and production
- Set usage limits in your provider's console
- Monitor your API usage regularly
- Review `.gitignore` to ensure `.env` is excluded
- Rotate API keys periodically

### Other Security

- Review generated bot scripts before running
- Wine executes with your user privileges - only run trusted game executables
- Be cautious with games from unknown sources

## Troubleshooting

### "Permission denied: ./paperengine"
```bash
chmod +x paperengine
./paperengine
```

### "python not found"
Try `python3` instead:
```bash
python3 main.py
```

### Missing system tools (Linux only)

```bash
# Install Wine (for running Windows games)
sudo apt install wine         # Ubuntu/Debian
sudo pacman -S wine          # Arch
sudo dnf install wine        # Fedora

# Install Flameshot (for screenshots)
sudo apt install flameshot    # Ubuntu/Debian
sudo pacman -S flameshot     # Arch
sudo dnf install flameshot   # Fedora
```

### Dependencies not installing

Manual installation:
```bash
source env/bin/activate
pip install -r requirements.txt
```

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
- Uses [Google Gemini AI](https://ai.google.dev/) (default, free tier) for controls search and bot generation
- Optional support for [Anthropic Claude](https://claude.ai) and [OpenAI GPT](https://openai.com) models
- Tested with Cuphead by Studio MDHR
