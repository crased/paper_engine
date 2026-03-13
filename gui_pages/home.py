"""Home page -- landing screen with quick-action cards and status bar."""

import customtkinter as ctk
from pathlib import Path

from gui_components.theme import (
    BG_SURFACE,
    BG_HOVER,
    BORDER_SUBTLE,
    BORDER,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    TEXT_MUTED,
    CLR_GREEN,
    CLR_BLUE,
    CLR_PURPLE,
    CLR_ORANGE,
    SUCCESS,
    WARNING,
    DANGER,
    INFO,
    ICON_DOT,
    RADIUS_LG,
    RADIUS_MD,
    CARD_BORDER_WIDTH,
    ACCENT_BAR_TOP,
    SP_SM,
    SP_MD,
    SP_LG,
    SP_XL,
    SP_2XL,
    SP_3XL,
    PAGE_PAD_X,
    PAGE_PAD_TOP,
    PAGE_PAD_BOTTOM,
    BTN_HEIGHT_MD,
    font_heading,
    font_subheading,
    font_body,
    font_body_bold,
    font_small,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class HomePage(ctk.CTkFrame):
    """Landing page: title banner, quick-action cards, project status."""

    def __init__(self, master, app):
        super().__init__(master, fg_color="transparent")
        self._app = app
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)
        self._build()

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def _build(self):
        # -- Title banner --
        banner = ctk.CTkFrame(self, fg_color="transparent")
        banner.grid(row=0, column=0, sticky="ew", padx=PAGE_PAD_X, pady=(SP_3XL, SP_LG))

        ctk.CTkLabel(
            banner,
            text="Paper Engine",
            font=font_heading(),
            text_color=TEXT_PRIMARY,
        ).pack(anchor="w")
        ctk.CTkLabel(
            banner,
            text="Game automation and computer vision framework",
            font=font_body(),
            text_color=TEXT_SECONDARY,
        ).pack(anchor="w", pady=(4, 0))

        # -- Quick-action cards (2x2 grid) --
        cards = ctk.CTkFrame(self, fg_color="transparent")
        cards.grid(
            row=1, column=0, sticky="ew", padx=PAGE_PAD_X - 6, pady=(SP_SM, SP_MD)
        )
        cards.grid_columnconfigure(0, weight=1)
        cards.grid_columnconfigure(1, weight=1)

        _card_defs = [
            {
                "title": "Launch Game",
                "desc": "Start game and capture screenshots",
                "icon": "\u25b6",
                "accent": CLR_GREEN,
                "cmd": lambda: self._app.navigate("session"),
            },
            {
                "title": "Train Model",
                "desc": "Train YOLO model on your dataset",
                "icon": "\u2726",
                "accent": CLR_BLUE,
                "cmd": lambda: self._app.navigate("tools"),
            },
            {
                "title": "Import Model",
                "desc": "Import an existing .pt model",
                "icon": "\u21e3",
                "accent": CLR_PURPLE,
                "cmd": lambda: self._go_tools("import"),
            },
            {
                "title": "Annotate",
                "desc": "Label screenshots for training",
                "icon": "\u25a1",
                "accent": CLR_ORANGE,
                "cmd": lambda: self._go_tools("annotate"),
            },
        ]

        for idx, d in enumerate(_card_defs):
            card = ctk.CTkFrame(
                cards,
                corner_radius=RADIUS_LG,
                border_width=CARD_BORDER_WIDTH,
                border_color=BORDER_SUBTLE,
                fg_color=BG_SURFACE,
            )
            card.grid(row=idx // 2, column=idx % 2, sticky="nsew", padx=6, pady=6)
            card.grid_columnconfigure(0, weight=1)

            # Accent strip at top of card
            ctk.CTkFrame(
                card,
                height=ACCENT_BAR_TOP,
                corner_radius=0,
                fg_color=d["accent"],
            ).grid(row=0, column=0, sticky="ew")

            # Icon + title row
            title_row = ctk.CTkFrame(card, fg_color="transparent")
            title_row.grid(row=1, column=0, sticky="ew", padx=SP_LG, pady=(SP_MD, 2))

            ctk.CTkLabel(
                title_row,
                text=d["icon"],
                font=ctk.CTkFont(size=16),
                text_color=d["accent"],
                width=24,
            ).pack(side="left", padx=(0, SP_SM))

            ctk.CTkLabel(
                title_row,
                text=d["title"],
                font=font_subheading(),
                text_color=TEXT_PRIMARY,
                anchor="w",
            ).pack(side="left")

            ctk.CTkLabel(
                card,
                text=d["desc"],
                font=font_small(),
                text_color=TEXT_SECONDARY,
                anchor="w",
            ).grid(row=2, column=0, sticky="w", padx=SP_LG, pady=(0, SP_SM))

            ctk.CTkButton(
                card,
                text=d["title"],
                width=120,
                height=BTN_HEIGHT_MD,
                corner_radius=RADIUS_MD,
                fg_color=d["accent"],
                hover_color=BG_HOVER,
                command=d["cmd"],
            ).grid(row=3, column=0, sticky="w", padx=SP_LG, pady=(2, SP_LG))

        # -- Status bar --
        status = ctk.CTkFrame(
            self,
            corner_radius=RADIUS_LG,
            border_width=CARD_BORDER_WIDTH,
            border_color=BORDER_SUBTLE,
            fg_color=BG_SURFACE,
        )
        status.grid(
            row=2,
            column=0,
            sticky="new",
            padx=PAGE_PAD_X,
            pady=(SP_SM, PAGE_PAD_BOTTOM),
        )
        status.grid_columnconfigure(0, weight=1)
        status.grid_columnconfigure(1, weight=1)
        status.grid_columnconfigure(2, weight=1)

        # Model status
        self._model_dot, self._model_lbl = self._build_status_item(
            status, 0, "Model: --"
        )

        # Dataset status
        self._dataset_dot, self._dataset_lbl = self._build_status_item(
            status, 1, "Dataset: --"
        )

        # GPU status
        self._gpu_dot, self._gpu_lbl = self._build_status_item(
            status, 2, "GPU: --", sticky="e"
        )

    def _build_status_item(self, parent, col, text, sticky="w"):
        """Build a status row with dot indicator + label. Returns (dot, label)."""
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.grid(row=0, column=col, padx=SP_LG, pady=SP_MD, sticky=sticky)

        dot = ctk.CTkLabel(
            row,
            text=ICON_DOT,
            font=ctk.CTkFont(size=10),
            text_color=TEXT_MUTED,
            width=14,
        )
        dot.pack(side="left", padx=(0, SP_SM))

        lbl = ctk.CTkLabel(
            row,
            text=text,
            font=font_small(),
            text_color=TEXT_SECONDARY,
        )
        lbl.pack(side="left")

        return dot, lbl

    # ------------------------------------------------------------------
    # Navigation helpers
    # ------------------------------------------------------------------

    def _go_tools(self, action: str):
        """Navigate to tools page and optionally trigger an action."""
        self._app.navigate("tools")
        tools = self._app._pages.get("tools")
        if tools and hasattr(tools, "trigger_action"):
            self.after(100, lambda: tools.trigger_action(action))

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def on_show(self):
        self._refresh()

    def _refresh(self):
        _exts = (".png", ".jpg", ".jpeg")
        ds = PROJECT_ROOT / "yolo_dataset"

        # -- Model info --
        model_path = (
            PROJECT_ROOT
            / "bot_logic"
            / "models"
            / "paper_engine_yolo26"
            / "weights"
            / "best.pt"
        )
        if model_path.exists():
            size_mb = model_path.stat().st_size / (1024 * 1024)
            self._model_lbl.configure(
                text=f"Model: YOLO26n ({size_mb:.0f} MB)",
                text_color=TEXT_PRIMARY,
            )
            self._model_dot.configure(text_color=SUCCESS)
        else:
            models_dir = PROJECT_ROOT / "bot_logic" / "models"
            found = (
                sorted(models_dir.glob("*/weights/best.pt"))
                if models_dir.exists()
                else []
            )
            if found:
                name = found[-1].parent.parent.name
                self._model_lbl.configure(
                    text=f"Model: {name}", text_color=TEXT_PRIMARY
                )
                self._model_dot.configure(text_color=WARNING)
            else:
                self._model_lbl.configure(text="Model: none", text_color=TEXT_MUTED)
                self._model_dot.configure(text_color=DANGER)

        # -- Dataset count --
        train_n = val_n = 0
        for split in ("train", "val"):
            d = ds / split / "images"
            if d.exists():
                n = len([f for f in d.iterdir() if f.suffix.lower() in _exts])
                if split == "train":
                    train_n = n
                else:
                    val_n = n
        total = train_n + val_n
        self._dataset_lbl.configure(
            text=f"Dataset: {total:,} images ({train_n:,} train / {val_n:,} val)",
            text_color=TEXT_PRIMARY if total else TEXT_MUTED,
        )
        if total > 1000:
            self._dataset_dot.configure(text_color=SUCCESS)
        elif total > 0:
            self._dataset_dot.configure(text_color=WARNING)
        else:
            self._dataset_dot.configure(text_color=DANGER)

        # -- GPU info (static) --
        self._gpu_lbl.configure(text="GPU: AMD RX 9070 XT", text_color=TEXT_PRIMARY)
        self._gpu_dot.configure(text_color=INFO)
