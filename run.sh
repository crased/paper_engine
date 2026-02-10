#!/bin/bash
# Paper Engine v1.0 - Quick Start Script

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}          Paper Engine v1.0 - Pre-Release${NC}"
echo -e "${BLUE}============================================================${NC}"
echo ""

# Check if virtual environment exists
if [ ! -d "env" ]; then
    echo "Virtual environment not found. Creating one..."
    python3 -m venv env
    echo -e "${GREEN}✓ Virtual environment created${NC}"
    echo ""
fi

# Activate virtual environment
echo "Activating virtual environment..."
source env/bin/activate
echo -e "${GREEN}✓ Virtual environment activated${NC}"
echo ""

# Check if dependencies are installed (check for a core package)
if ! python -c "import pynput" 2>/dev/null; then
    echo "Dependencies not detected."
    echo "Would you like to install them now? (Y/N)"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        echo "Installing dependencies..."
        pip install -r requirements.txt
        echo -e "${GREEN}✓ Dependencies installed${NC}"
        echo ""
    else
        echo "Skipping dependency installation."
        echo "Note: main.py will offer to install missing packages automatically."
        echo ""
    fi
fi

# Run Paper Engine
echo "Starting Paper Engine..."
echo ""
python main.py

# Deactivate on exit
deactivate 2>/dev/null
