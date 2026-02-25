# Paper Engine

> **🚀 Quick Start:** New to Paper Engine? See [QUICKSTART.md](information/QUICKSTART.md) for a 5-minute setup guide!

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

A game automation and computer vision framework that combines Wine game execution, screenshot capture, built-in annotation, and YOLO object detection to build intelligent game bots. Designed for bug detection, exploit identification, automated testing, and game development research.

## What It Does

Paper Engine automates the complete pipeline for creating game bots:

1. **Game Execution**: Runs game executables (Windows via Wine, Linux native, scripts)
2. **Screenshot Capture**: Automatically captures gameplay screenshots every 5 seconds
3. **Annotation**: Built-in annotation tool for object detection labelling
4. **Model Training**: Converts annotations to YOLO format and trains YOLO11 models
5. **Controls Discovery**: AI-powered web search for game controls
6. **Bot Generation**: Automatically generates Python bot scripts using YOLO + controls

Supports Windows games (.exe via Wine), Linux native executables, shell scripts (.sh), and Python games (.py).
Currently tested with Cuphead, but designed to work with any game executable.

## Use Cases

Paper Engine is designed for legitimate game testing, research, and development:

### 1. **Bug & Exploit Detection** (Primary Use Case)
- **Automated Testing**: Train bots to repeatedly test game mechanics and boundary conditions
- **Glitch Discovery**: Detect visual anomalies, clipping issues, and rendering bugs through object detection
- **Exploit Identification**: Identify unintended game behaviors and sequence breaks
- **Regression Testing**: Verify bug fixes by automating reproduction steps
- **QA Automation**: Supplement manual testing with automated gameplay coverage

### 2. **Game Development & QA**
- Automated playtesting for indie developers
- Performance testing under various gameplay scenarios
- Balance testing for game mechanics

### 3. **AI & Computer Vision Research**
- Game state recognition and object detection
- Reinforcement learning for game AI
- Educational tool for teaching computer vision concepts

### 4. **Speedrunning & Optimization**
- Route optimization and strategy testing
- Frame-perfect input analysis
- Glitch documentation for speedrunning communities

**Note:** This tool is intended for **single-player games** and **authorized testing only**. Do not use for:
- Multiplayer games or competitive advantage
- Violating game terms of service
- Unauthorized exploitation of online games

## Features

- **Annotation**: Built-in annotation tool with coloured bounding boxes, class labels, and YOLO format export
- **AI-Generated Bot Scripts**: AI searches for game controls and generates Python bot scripts with YOLO inference
- **Configurable Screenshot Capture**: Automated gameplay capture with configurable intervals via flameshot
- **Screenshot Soft Limit**: Automatically pauses at 201 screenshots and prompts to continue or stop

## To Be Implemented

- Admin dashboard (web UI for monitoring bot performance, managing sessions, viewing metrics)
- Data-driven semi self-improving bot scripts using LLM (bots that learn from gameplay data and refine their logic)
- Notification/ping when screenshot soft limit is reached (audio alert or system notification)
- Screen recording of bot actions (record bot gameplay for analysis/demonstration)
- Video training datasets
- Bot script repository for sharing game scripts

## Requirements

> **⚠️ PLATFORM SUPPORT:**
> Paper Engine supports **Linux and macOS**. Windows support is planned for future releases.

### System Requirements
- **OS**: Linux or macOS
  - **Linux**: ✅ Fully supported (tested on Arch Linux with Wayland)
  - **macOS**: ✅ Supported (uses native screencapture, Wine via Homebrew)
  - **Windows**: ❌ Not supported yet (planned for future releases)
- **Wine**: For running Windows game executables
  - **Linux**: Install via package manager (`apt`, `pacman`, `dnf`)
  - **macOS**: Install via Homebrew (`brew install wine-stable`)
- **Screenshot tool**: Platform-specific (automatically detected)
  - **Linux**: `flameshot` (recommended), `scrot` (fallback), or `imagemagick` (fallback)
  - **macOS**: `flameshot` (recommended via Homebrew), native `screencapture` (built-in fallback)
  - **Windows**: PIL/ImageGrab (not yet supported)
- **Python**: 3.8+

### Python Dependencies

Paper Engine will automatically detect and offer to install missing dependencies on first run.

**Core dependencies (required):**
```bash
pip install pynput python-dotenv pyyaml
```

**LLM providers (installed by script when you choose one):**
- The script will detect missing LLM and prompt you to install one
- Google Gemini (free tier) - Recommended
- Anthropic Claude (paid)
- OpenAI GPT (paid)

**Heavy packages (install manually when needed):**
```bash
pip install ultralytics      # For YOLO training (Step 3)
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

   **Option A: Automatic (recommended)**

   ```bash
   # Just run Paper Engine - it will prompt you to install missing packages
   ./paperengine
   ```

   **Option B: Manual**

   ```bash
   # Install core dependencies
   pip install -r information/requirements.txt

   # Install heavy packages manually when needed:
   pip install ultralytics      # For YOLO training

   # LLM provider will be installed by the script when you choose one
   ```

4. **Add your game**

   Paper Engine supports multiple game types. Choose the method that matches your game:

   **For Windows Games (.exe files via Wine):**

   ⚠️ **Important:** You need the entire game folder (with all DLLs), not just the .exe

   **Option A: Copy entire folder**
   ```bash
   # Copy entire game directory
   cp -r /path/to/Cuphead/ game/
   # Result: game/Cuphead/ with all files
   ```

   **Option B: Symlink (recommended, saves space)**
   ```bash
   # Create symlink to game installation
   ln -s /path/to/Cuphead game/Cuphead
   # Result: game/Cuphead/ pointing to original location
   ```

   **For Shell Script Games (.sh files):**
   ```bash
   # Copy or move your shell script
   cp /path/to/mygame.sh game/
   # Make sure it's executable
   chmod +x game/mygame.sh
   ```

   **For Python Games (.py files):**
   ```bash
   # Copy or move your Python game
   cp /path/to/mygame.py game/
   # Optionally mark as executable
   chmod +x game/mygame.py
   ```

   **For Native Linux Games (no extension):**
   ```bash
   # Copy the executable
   cp /path/to/mygame game/
   # Ensure it has execute permissions
   chmod +x game/mygame
   ```

   The script will auto-detect executables in the `game/` directory and let you choose if multiple are found.

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

   **When you first run Paper Engine**, it will prompt you to choose an LLM provider and automatically update `conf/main_conf.ini` with your choice.

   To manually switch providers later:
   ```bash
   # Option 1: Reinstall with a different provider
   ./paperengine  # Choose a different LLM when prompted

   # Option 2: Manually edit conf/main_conf.ini
   nano conf/main_conf.ini

   # Change the [LLM] section:
   [LLM]
   llm_provider = anthropic  # or: openai, google
   llm_model = claude-sonnet-4-5-20250514  # or: gpt-4, gemini-2.0-flash-exp

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

Paper Engine will guide you through a complete 5-step workflow:

1. **Launch game?** → Play for a few minutes to capture screenshots
2. **Annotate?** → Draw bounding boxes on screenshots to label game objects
3. **Train YOLO?** → Train object detection model
4. **Review Results?** → Verify detection works, correct predictions
5. **Generate bot?** → AI creates bot script (automatically searches for controls)

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

### Run Your Bot

After completing the pipeline:
```bash
python bot_scripts/<game_name>_bot.py
```

Press ESC to stop the bot.

## Project Structure

```
paper_engine/
├── game/                    # Place game executables here (.exe, .sh, .py, or native)
├── screenshots/             # Auto-captured screenshots
├── dataset/                 # Legacy annotations (JSON)
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
1. **Convert**: Annotations → YOLO format (normalized bboxes)
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

### Executable Security

**CRITICAL: Code Execution Risks**

Paper Engine executes game files with your user privileges. This includes:
- **Windows .exe files** via Wine
- **Shell scripts (.sh)** via bash
- **Python scripts (.py)** via python interpreter
- **Native Linux executables** with direct execution

⚠️ **Security Warnings:**
- ❌ **NEVER run games/executables from untrusted sources**
- ❌ **NEVER place unknown scripts in the game/ directory**
- ❌ **ALWAYS verify game files before running Paper Engine**
- ✅ **Only use games from legitimate sources** (Steam, GOG, official stores)
- ✅ **Review any .sh or .py files** before placing them in game/
- ✅ **Ensure your game/ directory is secure** and not world-writable

**Potential Risks:**
- Malicious scripts could access your files, install malware, or compromise your system
- Shell scripts and Python scripts execute with full user permissions
- Games can read/write files, make network connections, and execute system commands

**Best Practices:**
- Only download games from trusted, official sources
- Inspect any custom scripts before adding them to game/
- Run Paper Engine in a limited user account or virtual machine for untrusted games
- Keep backups of important data
- Monitor game/ directory for unexpected files
- Review generated bot scripts before running them

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

### Game won't launch: "Library [DLL] not found"
You need the entire game folder, not just the .exe:

```bash
# Wrong: Just copying .exe
cp /path/to/Cuphead/Cuphead.exe game/

# Correct Option 1: Copy entire game folder
cp -r /path/to/Cuphead/ game/

# Correct Option 2: Symlink (saves space)
ln -s /path/to/Cuphead game/Cuphead
```

Games like Cuphead (Unity games) need their DLL files and data folders to run.
The symlink creates a pointer to your game installation without copying.

### Script/executable won't launch: "Permission denied"
Your shell script or native executable needs execute permissions:

```bash
# For shell scripts
chmod +x game/mygame.sh

# For native Linux executables
chmod +x game/mygame

# Check permissions
ls -la game/
# Should show: -rwxr-xr-x (x = executable)
```

### Python game won't launch: "python: command not found"
Install Python or try `python3`:

```bash
# Install Python 3
sudo apt install python3        # Ubuntu/Debian
sudo pacman -S python          # Arch
sudo dnf install python3       # Fedora

# Or edit main.py to use python3 instead of python
```

### No game executables found
Make sure your game is in the `game/` directory:

```bash
# Check what's in the game folder
ls -la game/

# For Windows games: Need entire folder with .exe
# For scripts: Need .sh or .py file
# For native: Need executable file with proper permissions
```

### Missing system tools

**Linux:**
```bash
# Install Wine (for running Windows games)
sudo apt install wine         # Ubuntu/Debian
sudo pacman -S wine          # Arch
sudo dnf install wine        # Fedora

# Install screenshot tool (choose one, flameshot recommended)
# Flameshot (default, best for Wayland)
sudo apt install flameshot    # Ubuntu/Debian
sudo pacman -S flameshot     # Arch
sudo dnf install flameshot   # Fedora

# OR scrot (alternative)
sudo apt install scrot        # Ubuntu/Debian
sudo pacman -S scrot         # Arch
sudo dnf install scrot       # Fedora

# OR ImageMagick (alternative)
sudo apt install imagemagick  # Ubuntu/Debian
sudo pacman -S imagemagick   # Arch
sudo dnf install imagemagick # Fedora
```

**macOS:**
```bash
# Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Wine (for running Windows games)
brew install wine-stable

# Install Flameshot (recommended for consistency with Linux)
brew install flameshot
# Note: Built-in 'screencapture' works as fallback if flameshot not installed
```

### Dependencies not installing

Manual installation:
```bash
source env/bin/activate
pip install -r requirements.txt
```

## Testing

Manual testing workflow:
1. Place game executable in `game/` (.exe, .sh, .py, or native Linux executable)
2. Run `python main.py`
3. Play game briefly, then close
4. Verify screenshots in `screenshots/`
5. Click "Annotate" to label screenshots (optional)
6. Train model and verify output
7. Generate and test bot script

## Contributing

This is an experimental project. Contributions welcome, but expect breaking changes.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Uses [Flameshot](https://flameshot.org/) for screenshot capture on Linux
- Uses [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) for object detection
- Uses a built-in CustomTkinter annotation tool for labelling
- Uses [Google Gemini AI](https://ai.google.dev/) (default, free tier) for controls search and bot generation
- Optional support for [Anthropic Claude](https://claude.ai) and [OpenAI GPT](https://openai.com) models
- Tested with Cuphead by Studio MDHR
