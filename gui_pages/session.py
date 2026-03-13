"""Session page -- QA test session: game launch, screenshot capture, bot control.

Primary workflow: select game path -> launch -> capture screenshots / run bot.
"""

import sys
import subprocess
import threading
import time
import customtkinter as ctk
from pathlib import Path

from conf.config_parser import main_conf as config
from gui_components import StepLog
from gui_components.theme import (
    BG_SURFACE,
    BG_HOVER,
    BORDER_SUBTLE,
    BORDER,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    TEXT_MUTED,
    CLR_GREEN,
    CLR_RED,
    CLR_BLUE,
    DANGER,
    WARNING,
    SUCCESS,
    ICON_DOT,
    RADIUS_LG,
    RADIUS_MD,
    CARD_BORDER_WIDTH,
    SP_SM,
    SP_MD,
    SP_LG,
    SP_XL,
    SP_2XL,
    PAGE_PAD_X,
    PAGE_PAD_TOP,
    PAGE_PAD_BOTTOM,
    BTN_HEIGHT_MD,
    BTN_HEIGHT_LG,
    INPUT_HEIGHT,
    font_heading,
    font_body,
    font_body_bold,
    font_small,
    font_small_bold,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class SessionPage(ctk.CTkFrame):
    """QA test session -- launch game, capture screenshots, run bot."""

    def __init__(self, master, app):
        super().__init__(master, fg_color="transparent")
        self._app = app

        # State
        self._game_process = None
        self._capturing = False
        self._window_geometry = None
        self._last_path_file = PROJECT_ROOT / ".last_game_path"
        self._game_path_var = ctk.StringVar(value=self._load_last_game_path())
        self._screenshot_count_var = ctk.StringVar(value="0 screenshots")
        self._status_var = ctk.StringVar(value="Ready")

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(4, weight=1)

        self._build()

    @property
    def game_path(self):
        return self._game_path_var.get()

    # ==================================================================
    # Build
    # ==================================================================

    def _build(self):
        # -- Header --
        hdr = ctk.CTkFrame(self, fg_color="transparent")
        hdr.grid(
            row=0, column=0, sticky="ew", padx=PAGE_PAD_X, pady=(PAGE_PAD_TOP, SP_MD)
        )
        hdr.grid_columnconfigure(0, weight=1)

        # Title row with status pill
        title_row = ctk.CTkFrame(hdr, fg_color="transparent")
        title_row.grid(row=0, column=0, sticky="ew")
        title_row.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            title_row,
            text="Test Session",
            font=font_heading(),
            text_color=TEXT_PRIMARY,
        ).grid(row=0, column=0, sticky="w")

        self._status_pill = ctk.CTkLabel(
            title_row,
            text="  Ready  ",
            font=font_small_bold(),
            corner_radius=RADIUS_MD,
            fg_color=BG_SURFACE,
            text_color=TEXT_SECONDARY,
            height=26,
        )
        self._status_pill.grid(row=0, column=1, sticky="e", padx=(SP_SM, 0))

        ctk.CTkLabel(
            hdr,
            text="Launch a game, capture screenshots, or run the bot",
            font=font_body(),
            text_color=TEXT_SECONDARY,
        ).grid(row=1, column=0, sticky="w", pady=(4, 0))

        # -- Game path card --
        path_card = ctk.CTkFrame(
            self,
            corner_radius=RADIUS_LG,
            border_width=CARD_BORDER_WIDTH,
            border_color=BORDER_SUBTLE,
            fg_color=BG_SURFACE,
        )
        path_card.grid(row=1, column=0, sticky="ew", padx=PAGE_PAD_X, pady=(0, SP_MD))
        path_card.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            path_card,
            text="Game path",
            font=font_body_bold(),
            text_color=TEXT_PRIMARY,
        ).grid(row=0, column=0, padx=(SP_LG, SP_SM), pady=(SP_MD, SP_MD), sticky="w")

        ctk.CTkEntry(
            path_card,
            textvariable=self._game_path_var,
            height=INPUT_HEIGHT,
            corner_radius=RADIUS_MD,
            placeholder_text="/path/to/game",
            border_width=1,
            border_color=BORDER,
        ).grid(row=0, column=1, sticky="ew", padx=(0, SP_SM), pady=SP_MD)

        ctk.CTkButton(
            path_card,
            text="Browse",
            width=80,
            height=INPUT_HEIGHT,
            corner_radius=RADIUS_MD,
            fg_color=BG_HOVER,
            text_color=TEXT_PRIMARY,
            hover_color=BORDER,
            command=self._browse_game_dir,
        ).grid(row=0, column=2, padx=(0, SP_LG), pady=SP_MD)

        # -- Action buttons --
        btn_row = ctk.CTkFrame(self, fg_color="transparent")
        btn_row.grid(row=2, column=0, sticky="ew", padx=PAGE_PAD_X, pady=(0, SP_MD))

        self._launch_btn = ctk.CTkButton(
            btn_row,
            text="\u25b6  Launch Game",
            width=170,
            height=BTN_HEIGHT_LG,
            corner_radius=RADIUS_MD,
            fg_color=CLR_GREEN,
            font=font_body_bold(),
            command=self._on_launch_game,
        )
        self._launch_btn.pack(side="left", padx=(0, SP_SM))

        self._stop_btn = ctk.CTkButton(
            btn_row,
            text="Stop",
            width=80,
            height=BTN_HEIGHT_LG,
            corner_radius=RADIUS_MD,
            fg_color=CLR_RED,
            state="disabled",
            command=self._on_stop_capture,
        )
        self._stop_btn.pack(side="left", padx=(0, SP_LG))

        # Screenshot counter badge
        self._ss_badge = ctk.CTkLabel(
            btn_row,
            textvariable=self._screenshot_count_var,
            font=font_small_bold(),
            corner_radius=RADIUS_MD,
            fg_color=BG_SURFACE,
            text_color=CLR_BLUE,
            height=28,
        )
        self._ss_badge.pack(side="right", padx=(0, SP_SM))

        # -- Step log --
        self._step_log = StepLog(self)
        self._step_log.grid(
            row=4,
            column=0,
            sticky="nsew",
            padx=PAGE_PAD_X - 8,
            pady=(SP_SM, PAGE_PAD_BOTTOM),
        )

    # ==================================================================
    # Game path persistence
    # ==================================================================

    def _load_last_game_path(self):
        try:
            if self._last_path_file.exists():
                saved = self._last_path_file.read_text().strip()
                if saved:
                    return saved
        except Exception:
            pass
        return str(config.GAME_PATH)

    def _save_last_game_path(self, path: str):
        try:
            self._last_path_file.write_text(path)
        except Exception:
            pass

    def _browse_game_dir(self):
        from tkinter import filedialog

        d = filedialog.askdirectory(title="Select game directory")
        if d:
            self._game_path_var.set(d)
            self._save_last_game_path(d)

    # ==================================================================
    # Launch / Stop
    # ==================================================================

    def _on_launch_game(self):
        self._run_in_thread(self._launch_game_worker)

    def _launch_game_worker(self):
        from tools.functions import path_finder
        from tools.screencapture import (
            take_screenshot,
            create_screenshots_directory,
            find_wine_window,
        )

        game_path = self._game_path_var.get()
        self._save_last_game_path(game_path)
        s_launch = self._step_log.add("Launching game")
        self._set_status("Launching...", "active")

        exe_path = path_finder(game_path)
        if exe_path is None:
            self._step_log.fail(s_launch, "No executable found")
            self._set_status("Error", "error")
            return

        exe = Path(exe_path)
        self._step_log.complete(s_launch, f"Found {exe.name}")

        s_start = self._step_log.add("Starting game process")
        try:
            suffix = exe.suffix.lower()
            if suffix == ".exe":
                self._game_process = subprocess.Popen(
                    [
                        "wine",
                        "explorer",
                        f"/desktop=game,{config.WINE_DESKTOP_RESOLUTION}",
                        str(exe),
                    ]
                )
            elif suffix == ".sh":
                self._game_process = subprocess.Popen(["bash", str(exe)])
            elif suffix == ".py":
                self._game_process = subprocess.Popen(["python", str(exe)])
            else:
                self._game_process = subprocess.Popen([str(exe)])
            self._step_log.complete(s_start, "Game started")
        except FileNotFoundError as e:
            self._step_log.fail(s_start, f"Launch failed: {e}")
            self._set_status("Error", "error")
            return

        wait = config.GAME_INITIALIZATION_WAIT
        s_wait = self._step_log.add(f"Waiting {wait}s for game to initialise")
        time.sleep(wait)
        self._step_log.complete(s_wait)

        s_window = self._step_log.add("Finding game window")
        self._window_geometry = find_wine_window()
        if self._window_geometry:
            self._step_log.complete(s_window, f"Window: {self._window_geometry}")
        else:
            self._step_log.complete(s_window, "Full-screen capture mode")

        screenshots_dir = create_screenshots_directory()
        self._capturing = True
        self._set_status("Capturing", "active")
        s_capture = self._step_log.add("Capturing screenshots")
        self.after(0, lambda: self._stop_btn.configure(state="normal"))
        self.after(0, lambda: self._launch_btn.configure(state="disabled"))

        count = 0
        while self._capturing and (
            self._game_process is None or self._game_process.poll() is None
        ):
            time.sleep(config.SCREENSHOT_INTERVAL)
            take_screenshot(screenshots_dir, self._window_geometry)
            count += 1
            self.after(
                0, lambda c=count: self._screenshot_count_var.set(f"{c} screenshots")
            )
            self._step_log.update(s_capture, f"Capturing screenshots ({count})")

        self._capturing = False
        self._step_log.complete(s_capture, f"Captured {count} screenshots")
        self.after(0, lambda: self._stop_btn.configure(state="disabled"))
        self.after(0, lambda: self._launch_btn.configure(state="normal"))
        self._set_status("Ready", "idle")

    def _on_stop_capture(self):
        self._capturing = False
        if self._game_process and self._game_process.poll() is None:
            self._game_process.terminate()
        self._set_status("Stopping...", "warning")

    # ==================================================================
    # Helpers
    # ==================================================================

    def _set_status(self, text: str, mode: str = "idle"):
        """Update the status pill. Mode: idle, active, error, warning."""
        styles = {
            "error": (DANGER, ("#3E2723", "#2d1012")),
            "active": (SUCCESS, ("#1B5E20", "#0d2818")),
            "warning": (WARNING, ("#4E342E", "#2d1f0d")),
            "idle": (TEXT_SECONDARY, BG_SURFACE),
        }
        text_clr, bg_clr = styles.get(mode, styles["idle"])

        def _do():
            self._status_var.set(text)
            self._status_pill.configure(
                text=f"  {text}  ",
                fg_color=bg_clr,
                text_color=text_clr,
            )

        if threading.current_thread() is threading.main_thread():
            _do()
        else:
            self.after(0, _do)

    def _run_in_thread(self, fn, *args):
        def _worker():
            old_out, old_err = sys.stdout, sys.stderr
            redir = _LogRedirector(self._app.log)
            sys.stdout = redir
            sys.stderr = redir
            try:
                fn(*args)
            except Exception as exc:
                self._app.log(f"\nERROR: {exc}\n")
            finally:
                sys.stdout, sys.stderr = old_out, old_err

        threading.Thread(target=_worker, daemon=True).start()


class _LogRedirector:
    def __init__(self, cb):
        self._cb = cb

    def write(self, text):
        if text:
            self._cb(text)

    def flush(self):
        pass
