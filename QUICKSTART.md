# Paper Engine - Quick Start Guide

Get up and running in 5 minutes!

## Prerequisites

> **⚠️ LINUX ONLY:** Paper Engine currently only works on Linux. Windows and macOS are not supported.

- **Linux OS** (tested on Arch with Wayland)
- **Python 3.8+** installed
- **Git** installed
- **Wine** installed (for running Windows games)
- **Flameshot** installed (for screenshots)
- **Game .exe file** ready

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/crased/paper_engine.git
cd paper_engine
```

### 2. Add Your Game

```bash
# Copy your game executable to the game/ directory
cp /path/to/your/game.exe game/
```

### 3. Run Paper Engine

```bash
./run.sh
```

That's it! The run script handles everything else automatically.

## What Happens Next?

The run script will:

1. ✅ Create a virtual environment (first run only)
2. ✅ Activate the environment
3. ✅ Offer to install dependencies (first run only)
4. ✅ Launch Paper Engine

Paper Engine will then guide you through:

1. **Launch game?** → Play for a few minutes
2. **Label Studio?** → Annotate objects in screenshots
3. **Train YOLO?** → Train object detection model
4. **Test model?** → Verify detection works
5. **Search controls?** → AI finds keyboard controls (requires API key)
6. **Generate bot?** → AI creates bot script (requires API key)

## API Key Setup (Optional - for AI features)

Steps 5 & 6 require an API key. We recommend **Google Gemini (FREE)**:

1. Visit: https://aistudio.google.com/apikey
2. Click "Create API Key"
3. Copy your key
4. Edit `.env` file and paste your key

```bash
# .env file
API_KEY=your-google-api-key-here
```

## Running Your Bot

After completing the workflow:

```bash
# Your bot is saved in bot_scripts/
python bot_scripts/<game_name>_bot.py
```

Press **ESC** to stop the bot.

## Troubleshooting

### "Permission denied: ./run.sh"
```bash
chmod +x run.sh
./run.sh
```

### "python not found"
Try `python3`:
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

Manual install:
```bash
source env/bin/activate
pip install -r requirements.txt
```

## Need Help?

- **Documentation:** See [README.md](README.md)
- **Issues:** https://github.com/crased/paper_engine/issues
- **Developer Guide:** See [CLAUDE.md](CLAUDE.md)

## Security Warning

⚠️ **NEVER share your API key or commit .env to git!**

See README.md for comprehensive security guidelines.

---

**Paper Engine v1.0 - Pre-Release**  
Create AI-powered game bots in minutes!
