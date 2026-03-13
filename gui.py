"""Paper Engine -- GUI entry point.

Usage:
    python gui.py

Delegates to gui_app.PaperEngineApp (modular page-based GUI).
The old monolithic GUI is preserved as gui_old.py for reference.
"""

from gui_app import main


if __name__ == "__main__":
    main()
