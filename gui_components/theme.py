"""Paper Engine -- Design tokens and shared theme constants.

Inspired by Linear, GitHub Desktop, and Figma.
All color tuples are (light_mode, dark_mode).
"""

import customtkinter as ctk

# ======================================================================
# Color Palette
# ======================================================================

# Backgrounds -- layered depth system
BG_BASE = ("#f0f1f3", "#0d1117")  # Deepest background (behind everything)
BG_SIDEBAR = ("#e8eaed", "#010409")  # Sidebar (slightly darker than base)
BG_SURFACE = ("#ffffff", "#161b22")  # Cards, elevated panels
BG_SURFACE_ALT = ("#f6f8fa", "#0d1117")  # Alternate surface (e.g. log area)
BG_HOVER = ("#e1e4e8", "#1c2128")  # Hover states
BG_ACTIVE = ("#d0d7de", "#1f2937")  # Active/selected states
BG_INPUT = ("#ffffff", "#0d1117")  # Input fields

# Borders
BORDER = ("#d0d7de", "#30363d")  # Default border
BORDER_SUBTLE = ("#e1e4e8", "#21262d")  # Subtle border (cards, dividers)
BORDER_ACTIVE = ("#0969da", "#58a6ff")  # Active/focused border

# Text
TEXT_PRIMARY = ("#1f2328", "#e6edf3")  # Primary text
TEXT_SECONDARY = ("#656d76", "#8b949e")  # Secondary / descriptive text
TEXT_MUTED = ("#8b949e", "#484f58")  # Muted / disabled text
TEXT_ON_ACCENT = "#ffffff"  # Text on accent-colored backgrounds

# Accent colors -- semantic
ACCENT = ("#0969da", "#58a6ff")  # Primary accent (links, active items)
SUCCESS = ("#1a7f37", "#3fb950")  # Success / positive
WARNING = ("#9a6700", "#d29922")  # Warning / caution
DANGER = ("#cf222e", "#f85149")  # Error / destructive
INFO = ("#0550ae", "#79c0ff")  # Informational

# Accent colors -- branding / categories
CLR_GREEN = ("#1a7f37", "#3fb950")  # Session / Launch
CLR_BLUE = ("#0969da", "#58a6ff")  # Training / Primary actions
CLR_ORANGE = ("#bc4c00", "#d29922")  # Tools / Pipeline
CLR_PURPLE = ("#8250df", "#bc8cff")  # Settings / Import
CLR_RED = ("#cf222e", "#f85149")  # Stop / Delete
CLR_CYAN = ("#0e7490", "#22d3ee")  # Labels / Info

# Sidebar specific
SIDEBAR_BG = ("#e8eaed", "#010409")
SIDEBAR_ACTIVE_BG = ("#d0d7de", "#1c2128")
SIDEBAR_HOVER_BG = ("#dce0e5", "#161b22")
SIDEBAR_TEXT = ("#57606a", "#8b949e")
SIDEBAR_TEXT_ACTIVE = ("#1f2328", "#e6edf3")
SIDEBAR_BRAND = ("#1f2328", "#e6edf3")
SIDEBAR_VERSION = ("#8b949e", "#484f58")

# ======================================================================
# Typography
# ======================================================================


def font_heading():
    """Page title -- 24px bold."""
    return ctk.CTkFont(size=24, weight="bold")


def font_subheading():
    """Section header -- 16px bold."""
    return ctk.CTkFont(size=16, weight="bold")


def font_body():
    """Body text -- 13px normal."""
    return ctk.CTkFont(size=13)


def font_body_bold():
    """Emphasized body -- 13px bold."""
    return ctk.CTkFont(size=13, weight="bold")


def font_small():
    """Small text / captions -- 11px normal."""
    return ctk.CTkFont(size=11)


def font_small_bold():
    """Small emphasized -- 11px bold."""
    return ctk.CTkFont(size=11, weight="bold")


def font_nav():
    """Sidebar nav item -- 13px normal."""
    return ctk.CTkFont(size=13)


def font_nav_active():
    """Sidebar nav item (active) -- 13px bold."""
    return ctk.CTkFont(size=13, weight="bold")


def font_brand():
    """Brand name in sidebar -- 16px bold."""
    return ctk.CTkFont(size=16, weight="bold")


def font_mono():
    """Monospace (log, command input) -- 12px."""
    return ctk.CTkFont(family="monospace", size=12)


def font_mono_small():
    """Monospace small -- 11px."""
    return ctk.CTkFont(family="monospace", size=11)


# ======================================================================
# Spacing
# ======================================================================

SP_XS = 4
SP_SM = 8
SP_MD = 12
SP_LG = 16
SP_XL = 24
SP_2XL = 32
SP_3XL = 40

# Page-level padding
PAGE_PAD_X = 28
PAGE_PAD_TOP = 24
PAGE_PAD_BOTTOM = 20

# ======================================================================
# Dimensions
# ======================================================================

SIDEBAR_WIDTH = 140
RADIUS_SM = 6
RADIUS_MD = 8
RADIUS_LG = 12
RADIUS_XL = 16

BTN_HEIGHT_SM = 28
BTN_HEIGHT_MD = 34
BTN_HEIGHT_LG = 40

INPUT_HEIGHT = 36
CARD_BORDER_WIDTH = 1
ACCENT_BAR_WIDTH = 3
ACCENT_BAR_TOP = 3

# ======================================================================
# Nav icons (Unicode -- well-supported geometric shapes)
# ======================================================================

ICON_HOME = "\u25c6"  # ◆
ICON_SESSION = "\u25b6"  # ▶
ICON_TOOLS = "\u2726"  # ✦
ICON_SETTINGS = "\u2699"  # ⚙
ICON_DASHBOARD = "\u25a3"  # ▣

# Status indicators
ICON_DOT = "\u25cf"  # ●
ICON_DOT_EMPTY = "\u25cb"  # ○
ICON_CHECK = "\u2713"  # ✓
ICON_CROSS = "\u2717"  # ✗
ICON_ARROW_DOWN = "\u25bc"  # ▼
ICON_ARROW_UP = "\u25b2"  # ▲
ICON_ARROW_RIGHT = "\u25b8"  # ▸

# ======================================================================
# Nav config (used by gui_app.py sidebar)
# ======================================================================

CLR_TEAL = ("#0e7490", "#2dd4bf")  # Dashboard / Metrics

NAV_ITEMS = [
    ("home", "Home", ICON_HOME, CLR_BLUE),
    ("dashboard", "Metrics", ICON_DASHBOARD, CLR_TEAL),
    ("session", "Test", ICON_SESSION, CLR_GREEN),
    ("tools", "Tools", ICON_TOOLS, CLR_ORANGE),
    ("settings", "Settings", ICON_SETTINGS, CLR_PURPLE),
]
