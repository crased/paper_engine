# Paper Engine - Quick Start Guide

Get up and running in 5 minutes!

> **⚠️ LINUX ONLY:** Paper Engine currently only works on Linux. Windows and macOS are not supported.

## Prerequisites

- **Linux OS** (tested on Arch with Wayland)
- **Python 3.8+** installed
- **Git** installed
- **Wine** installed (for running Windows .exe games)
- **Screenshot tool** installed (flameshot recommended, or scrot/imagemagick)
- **Game executable** ready (.exe, .sh, .py, or Linux native)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/crased/paper_engine.git
cd paper_engine
```

### 2. Add Your Game

**Option A: Copy entire game folder**
```bash
cp -r /path/to/Cuphead/ game/
```

**Option B: Symlink (saves space)**
```bash
ln -s /path/to/Cuphead game/Cuphead
```

⚠️ You need the **entire game folder**, not just the .exe (games need their DLL files)

### 3. Run Paper Engine

```bash
./paperengine
```

That's it! The script handles everything else automatically.

## What Happens Next?

Paper Engine will guide you through a complete 6-step workflow:

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

## Need Help?

- **Troubleshooting:** See [README.md - Troubleshooting section](README.md#troubleshooting)
- **Full Documentation:** See [README.md](README.md)
- **Developer Guide:** See [CLAUDE.md](CLAUDE.md)
- **Issues:** https://github.com/crased/paper_engine/issues

## Security Warnings

⚠️ **API Key Security:**
- **NEVER share your API key or commit .env to git!**

⚠️ **Executable Security:**
- **ONLY run games from trusted sources** (Steam, GOG, official stores)
- Paper Engine executes .exe, .sh, .py files with your user privileges
- **NEVER place untrusted scripts in the game/ directory**
- Malicious scripts could compromise your system

See [README.md - Security Notes](README.md#security-notes) for comprehensive security guidelines.

---

**Paper Engine v1.0 - Pre-Release**
Create AI-powered game bots in minutes!
