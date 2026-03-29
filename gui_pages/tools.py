"""Tools page -- ML pipeline operations, labels management, step log.

Sub-tabs: Pipeline | Labels
Pipeline: Annotate, Train, Import Model, Verify/Retrain, Generate Bot,
          Game Report (session), Dir Report (multi-source game analysis)
Labels:   table + Add/Delete/Consolidate/Rename
Bottom:   step log, progress bar, expandable raw log + command input
"""

import sys
import threading
import subprocess
import shutil
import configparser
import time
import tkinter as tk
import customtkinter as ctk
from pathlib import Path

from gui_components import StepLog
from gui_components.theme import (
    BG_BASE,
    BG_SURFACE,
    BG_SURFACE_ALT,
    BG_HOVER,
    BG_ACTIVE,
    BG_INPUT,
    BORDER,
    BORDER_SUBTLE,
    BORDER_ACTIVE,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    TEXT_MUTED,
    TEXT_ON_ACCENT,
    ACCENT,
    SUCCESS,
    WARNING,
    DANGER,
    INFO,
    CLR_GREEN,
    CLR_BLUE,
    CLR_ORANGE,
    CLR_PURPLE,
    CLR_RED,
    CLR_CYAN,
    ICON_DOT,
    ICON_ARROW_DOWN,
    ICON_ARROW_UP,
    RADIUS_SM,
    RADIUS_MD,
    RADIUS_LG,
    RADIUS_XL,
    CARD_BORDER_WIDTH,
    ACCENT_BAR_TOP,
    SP_XS,
    SP_SM,
    SP_MD,
    SP_LG,
    SP_XL,
    SP_2XL,
    SP_3XL,
    PAGE_PAD_X,
    PAGE_PAD_TOP,
    PAGE_PAD_BOTTOM,
    BTN_HEIGHT_SM,
    BTN_HEIGHT_MD,
    BTN_HEIGHT_LG,
    INPUT_HEIGHT,
    font_heading,
    font_subheading,
    font_body,
    font_body_bold,
    font_small,
    font_small_bold,
    font_mono,
    font_mono_small,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class ToolsPage(ctk.CTkFrame):
    """Pipeline tools + labels management page."""

    def __init__(self, master, app):
        super().__init__(master, fg_color="transparent")
        self._app = app
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # State
        self._selected_label_id = None
        self._label_row_widgets = {}
        self._log_expanded = False
        self._cmd_history = []
        self._cmd_history_idx = -1
        self._active_subtab = None

        self._build()

    # ==================================================================
    # Build
    # ==================================================================

    def _build(self):
        # -- Sub-tab bar --
        tab_bar = ctk.CTkFrame(self, fg_color="transparent", height=44)
        tab_bar.grid(
            row=0, column=0, sticky="ew", padx=PAGE_PAD_X, pady=(PAGE_PAD_TOP, 0)
        )

        self._subtab_btns = {}
        self._subtab_accents = {}
        _tabs = [
            ("pipeline", "Pipeline", CLR_ORANGE),
            ("labels", "Labels", CLR_CYAN),
        ]

        for key, label, color in _tabs:
            wrap = ctk.CTkFrame(tab_bar, fg_color="transparent")
            wrap.pack(side="left", padx=(0, SP_SM))

            btn = ctk.CTkButton(
                wrap,
                text=label,
                width=100,
                height=32,
                corner_radius=RADIUS_MD,
                font=font_body(),
                fg_color="transparent",
                text_color=TEXT_SECONDARY,
                hover_color=BG_HOVER,
                command=lambda k=key: self._switch_subtab(k),
            )
            btn.pack()

            # Underline accent
            acc = ctk.CTkFrame(wrap, height=2, corner_radius=1, fg_color="transparent")
            acc.pack(fill="x", padx=SP_MD)
            self._subtab_accents[key] = (acc, color)
            self._subtab_btns[key] = btn

        # -- Content area (stacked frames) --
        self._content = ctk.CTkFrame(self, fg_color="transparent")
        self._content.grid(row=1, column=0, sticky="nsew")
        self._content.grid_columnconfigure(0, weight=1)
        self._content.grid_rowconfigure(0, weight=1)

        self._build_pipeline_view()
        self._build_labels_view()

        self._switch_subtab("pipeline")

    # ------------------------------------------------------------------
    # Pipeline view
    # ------------------------------------------------------------------

    def _build_pipeline_view(self):
        self._pipeline_frame = ctk.CTkFrame(self._content, fg_color="transparent")
        self._pipeline_frame.grid(row=0, column=0, sticky="nsew")
        self._pipeline_frame.grid_columnconfigure(0, weight=1)
        # Row 0: action buttons, Row 1: mode selector, Row 2: step log (flex),
        # Row 3: progress, Row 4: toggle, Row 5: raw log, Row 6: cmd bar
        self._pipeline_frame.grid_rowconfigure(2, weight=1)

        # -- Action buttons in a card --
        btn_card = ctk.CTkFrame(
            self._pipeline_frame,
            corner_radius=RADIUS_LG,
            border_width=CARD_BORDER_WIDTH,
            border_color=BORDER_SUBTLE,
            fg_color=BG_SURFACE,
        )
        btn_card.grid(
            row=0, column=0, sticky="ew", padx=PAGE_PAD_X, pady=(SP_MD, SP_SM)
        )

        btn_inner = ctk.CTkFrame(btn_card, fg_color="transparent")
        btn_inner.pack(padx=SP_MD, pady=SP_MD)

        _btns = [
            ("Annotate", CLR_ORANGE, self._on_annotate),
            ("Describe & Review", ("#6A1B9A", "#AB47BC"), self._on_describe_review),
            ("Train Model", CLR_BLUE, self._on_train_model),
            ("Import Model", CLR_PURPLE, self._on_import_model),
            ("Verify / Retrain", CLR_GREEN, self._on_verify_retrain),
            ("Generate Bot", ("#455A64", "#78909C"), self._on_generate_bot),
            ("Game Report", CLR_CYAN, self._on_game_report),
            ("Dir Report", ("#0097A7", "#00BCD4"), self._on_dir_report),
        ]
        for text, fg, cmd in _btns:
            ctk.CTkButton(
                btn_inner,
                text=text,
                width=130,
                height=BTN_HEIGHT_MD,
                corner_radius=RADIUS_MD,
                fg_color=fg,
                command=cmd,
            ).pack(side="left", padx=SP_XS)

        # -- Training mode selector --
        mode_row = ctk.CTkFrame(self._pipeline_frame, fg_color="transparent")
        mode_row.grid(
            row=1, column=0, sticky="ew", padx=PAGE_PAD_X, pady=(SP_XS, SP_SM)
        )

        ctk.CTkLabel(
            mode_row,
            text="Training mode",
            font=font_small(),
            text_color=TEXT_MUTED,
        ).pack(side="left", padx=(0, SP_MD))

        self._train_mode_var = ctk.StringVar(value="local")

        # Segmented-control style buttons
        seg_frame = ctk.CTkFrame(
            mode_row,
            corner_radius=RADIUS_MD,
            border_width=1,
            border_color=BORDER,
            fg_color=BG_SURFACE,
        )
        seg_frame.pack(side="left")

        self._mode_local_btn = ctk.CTkButton(
            seg_frame,
            text="Local GPU",
            width=100,
            height=BTN_HEIGHT_SM,
            corner_radius=RADIUS_SM,
            font=font_small_bold(),
            fg_color=CLR_BLUE,
            text_color=TEXT_ON_ACCENT,
            hover_color=BG_HOVER,
            command=lambda: self._set_train_mode("local"),
        )
        self._mode_local_btn.pack(side="left", padx=2, pady=2)

        self._mode_cloud_btn = ctk.CTkButton(
            seg_frame,
            text="Cloud",
            width=80,
            height=BTN_HEIGHT_SM,
            corner_radius=RADIUS_SM,
            font=font_small(),
            fg_color="transparent",
            text_color=TEXT_SECONDARY,
            hover_color=BG_HOVER,
            command=lambda: self._set_train_mode("cloud"),
        )
        self._mode_cloud_btn.pack(side="left", padx=2, pady=2)

        self._mode_hint = ctk.CTkLabel(
            mode_row,
            text="AMD RX 9070 XT",
            font=font_small(),
            text_color=TEXT_MUTED,
        )
        self._mode_hint.pack(side="left", padx=(SP_MD, 0))

        # -- Step log (always visible) --
        self._step_log = StepLog(self._pipeline_frame)
        self._step_log.grid(
            row=2, column=0, sticky="nsew", padx=PAGE_PAD_X - 4, pady=(SP_SM, 0)
        )

        # -- Progress bar --
        prog_row = ctk.CTkFrame(self._pipeline_frame, fg_color="transparent")
        prog_row.grid(row=3, column=0, sticky="ew", padx=PAGE_PAD_X, pady=(SP_SM, 0))
        prog_row.grid_columnconfigure(1, weight=1)

        self._progress_label = ctk.CTkLabel(
            prog_row,
            text="",
            font=font_small(),
            text_color=TEXT_SECONDARY,
            anchor="w",
            width=50,
        )
        self._progress_label.grid(row=0, column=0, padx=(0, SP_SM), sticky="w")

        self._progress_bar = ctk.CTkProgressBar(
            prog_row,
            height=8,
            corner_radius=4,
        )
        self._progress_bar.grid(row=0, column=1, sticky="ew", padx=(0, SP_SM))
        self._progress_bar.set(0)

        self._progress_detail = ctk.CTkLabel(
            prog_row,
            text="",
            font=font_small(),
            text_color=TEXT_MUTED,
            anchor="e",
            width=100,
        )
        self._progress_detail.grid(row=0, column=2, padx=(0, 0), sticky="e")

        # -- Show/hide toggle --
        toggle_row = ctk.CTkFrame(self._pipeline_frame, fg_color="transparent")
        toggle_row.grid(row=4, column=0, sticky="ew", padx=PAGE_PAD_X, pady=(SP_SM, 0))

        self._log_toggle_btn = ctk.CTkButton(
            toggle_row,
            text=f"Show details  {ICON_ARROW_DOWN}",
            width=130,
            height=BTN_HEIGHT_SM,
            corner_radius=RADIUS_SM,
            font=font_small(),
            fg_color="transparent",
            hover_color=BG_HOVER,
            text_color=TEXT_MUTED,
            command=self._toggle_log_expand,
        )
        self._log_toggle_btn.pack(side="left")

        ctk.CTkButton(
            toggle_row,
            text="Clear",
            width=60,
            height=BTN_HEIGHT_SM,
            corner_radius=RADIUS_SM,
            font=font_small(),
            fg_color="transparent",
            hover_color=BG_HOVER,
            text_color=TEXT_MUTED,
            command=self._clear_all,
        ).pack(side="right")

        # -- Raw log textbox (hidden by default) --
        self._log_textbox = ctk.CTkTextbox(
            self._pipeline_frame,
            wrap="none",
            state="disabled",
            corner_radius=RADIUS_MD,
            font=font_mono_small(),
            fg_color=BG_SURFACE_ALT,
            border_width=1,
            border_color=BORDER_SUBTLE,
        )
        # Starts hidden -- not gridded

        # -- Command input bar (hidden with log) --
        self._cmd_bar = ctk.CTkFrame(self._pipeline_frame, fg_color="transparent")
        self._cmd_bar.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            self._cmd_bar,
            text="\u276f",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=ACCENT,
            width=20,
        ).grid(row=0, column=0, sticky="w", padx=(SP_SM, 0))
        self._cmd_var = ctk.StringVar()
        self._cmd_entry = ctk.CTkEntry(
            self._cmd_bar,
            textvariable=self._cmd_var,
            height=INPUT_HEIGHT,
            corner_radius=RADIUS_MD,
            placeholder_text="Enter command...",
            font=font_mono(),
            border_width=1,
            border_color=BORDER,
        )
        self._cmd_entry.grid(row=0, column=0, sticky="ew", padx=(SP_XL, SP_SM))
        self._cmd_entry.bind("<Return>", self._on_cmd_enter)
        self._cmd_entry.bind("<Up>", self._on_cmd_history_up)
        self._cmd_entry.bind("<Down>", self._on_cmd_history_down)

    # ------------------------------------------------------------------
    # Labels view
    # ------------------------------------------------------------------

    def _build_labels_view(self):
        self._labels_frame = ctk.CTkFrame(self._content, fg_color="transparent")
        self._labels_frame.grid(row=0, column=0, sticky="nsew")
        self._labels_frame.grid_columnconfigure(0, weight=1)
        self._labels_frame.grid_rowconfigure(2, weight=1)

        # Header
        hdr = ctk.CTkFrame(self._labels_frame, fg_color="transparent")
        hdr.grid(row=0, column=0, sticky="ew", padx=PAGE_PAD_X, pady=(SP_MD, SP_SM))
        hdr.grid_columnconfigure(0, weight=1)

        hdr_inner = ctk.CTkFrame(hdr, fg_color="transparent")
        hdr_inner.grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(
            hdr_inner,
            text=ICON_DOT,
            font=ctk.CTkFont(size=10),
            text_color=CLR_CYAN,
            width=16,
        ).pack(side="left", padx=(0, SP_SM))
        ctk.CTkLabel(
            hdr_inner,
            text="Label Management",
            font=font_subheading(),
            text_color=TEXT_PRIMARY,
        ).pack(side="left")
        ctk.CTkButton(
            hdr,
            text="Refresh",
            width=80,
            height=BTN_HEIGHT_SM,
            corner_radius=RADIUS_MD,
            fg_color=BG_HOVER,
            text_color=TEXT_PRIMARY,
            hover_color=BORDER,
            command=self._refresh_labels_view,
        ).grid(row=0, column=1, sticky="e")

        # Column header
        col_hdr = ctk.CTkFrame(
            self._labels_frame,
            fg_color=BG_SURFACE,
            corner_radius=RADIUS_MD,
            border_width=CARD_BORDER_WIDTH,
            border_color=BORDER_SUBTLE,
            height=36,
        )
        col_hdr.grid(row=1, column=0, sticky="ew", padx=PAGE_PAD_X, pady=(0, SP_XS))
        for txt, w in [
            ("ID", 50),
            ("Name", 0),
            ("Train", 70),
            ("Val", 70),
            ("Total", 70),
        ]:
            lbl = ctk.CTkLabel(
                col_hdr,
                text=txt,
                width=w,
                font=font_small_bold(),
                text_color=TEXT_MUTED,
                anchor="center" if txt != "Name" else "w",
            )
            lbl.pack(
                side="left", padx=SP_SM, expand=(w == 0), fill="x" if w == 0 else "none"
            )

        # Table body
        self._labels_table = ctk.CTkScrollableFrame(
            self._labels_frame,
            fg_color="transparent",
        )
        self._labels_table.grid(
            row=2, column=0, sticky="nsew", padx=PAGE_PAD_X, pady=SP_XS
        )

        # Action buttons
        lbl_btns = ctk.CTkFrame(self._labels_frame, fg_color="transparent")
        lbl_btns.grid(
            row=3, column=0, sticky="ew", padx=PAGE_PAD_X, pady=(SP_SM, PAGE_PAD_BOTTOM)
        )
        for txt, color, cmd in [
            ("Add Class", CLR_GREEN, self._on_add_class),
            ("Delete Selected", CLR_RED, self._on_delete_class),
            ("Consolidate", CLR_BLUE, self._on_consolidate),
            ("Rename", CLR_PURPLE, self._on_rename_class),
        ]:
            ctk.CTkButton(
                lbl_btns,
                text=txt,
                width=130,
                height=BTN_HEIGHT_MD,
                corner_radius=RADIUS_MD,
                fg_color=color,
                command=cmd,
            ).pack(side="left", padx=SP_XS)

    # ==================================================================
    # Sub-tab switching
    # ==================================================================

    def _switch_subtab(self, key):
        if self._active_subtab == key:
            return
        self._active_subtab = key
        # Hide all
        self._pipeline_frame.grid_remove()
        self._labels_frame.grid_remove()
        # Show selected
        if key == "pipeline":
            self._pipeline_frame.grid(row=0, column=0, sticky="nsew")
        elif key == "labels":
            self._labels_frame.grid(row=0, column=0, sticky="nsew")
            self._refresh_labels_view()
        # Highlight + accent underline
        for k, btn in self._subtab_btns.items():
            acc, color = self._subtab_accents[k]
            if k == key:
                btn.configure(
                    fg_color=BG_ACTIVE,
                    text_color=TEXT_PRIMARY,
                    font=font_body_bold(),
                )
                acc.configure(fg_color=color)
            else:
                btn.configure(
                    fg_color="transparent",
                    text_color=TEXT_SECONDARY,
                    font=font_body(),
                )
                acc.configure(fg_color="transparent")

    # ==================================================================
    # Public services (called by app shell)
    # ==================================================================

    @property
    def step_log(self):
        return self._step_log

    _log_busy = False  # re-entrancy guard for log()

    def log(self, text: str):
        """Append to raw log textbox (thread-safe).

        Uses a re-entrancy guard so that any stdout/stderr produced by
        the textbox update itself (e.g. CustomTkinter warnings) is
        silently dropped instead of recursing back into this method.
        """

        def _do():
            if self._log_busy:
                return
            self._log_busy = True
            try:
                self._log_textbox.configure(state="normal")
                self._log_textbox.insert("end", text)
                self._log_textbox.see("end")
                self._log_textbox.configure(state="disabled")
            finally:
                self._log_busy = False

        if threading.current_thread() is threading.main_thread():
            _do()
        else:
            self.after(0, _do)

    def set_status(self, text: str):
        """Update app-level status (terminal implementation).

        gui_app.set_status() delegates here, so this method must NOT
        call self._app.set_status() — that would create mutual recursion.
        """
        try:
            suffix = f" — {text}" if text and text != "Idle" else ""
            self.winfo_toplevel().title(f"Paper Engine{suffix}")
        except Exception:
            pass

    @property
    def train_mode(self):
        """Return 'local' or 'cloud'."""
        return self._train_mode_var.get()

    def _set_train_mode(self, mode: str):
        """Switch between local and cloud training mode."""
        self._train_mode_var.set(mode)
        if mode == "local":
            self._mode_local_btn.configure(
                fg_color=CLR_BLUE,
                text_color=TEXT_ON_ACCENT,
                font=font_small_bold(),
            )
            self._mode_cloud_btn.configure(
                fg_color="transparent",
                text_color=TEXT_SECONDARY,
                font=font_small(),
            )
            self._mode_hint.configure(text="AMD RX 9070 XT")
        else:
            self._mode_cloud_btn.configure(
                fg_color=CLR_PURPLE,
                text_color=TEXT_ON_ACCENT,
                font=font_small_bold(),
            )
            self._mode_local_btn.configure(
                fg_color="transparent",
                text_color=TEXT_SECONDARY,
                font=font_small(),
            )
            self._mode_hint.configure(text="Ultralytics HUB / HuggingFace")

    def trigger_action(self, action: str):
        """Trigger a pipeline action by name (used by Home page cards)."""
        actions = {
            "annotate": self._on_annotate,
            "describe_review": self._on_describe_review,
            "train": self._on_train_model,
            "import": self._on_import_model,
            "verify": self._on_verify_retrain,
            "generate": self._on_generate_bot,
            "report": self._on_game_report,
            "dir_report": self._on_dir_report,
        }
        fn = actions.get(action)
        if fn:
            fn()

    # ==================================================================
    # Log / progress helpers (thread-safe)
    # ==================================================================

    def _log_progress(self, current: int, total: int, label: str = ""):
        pct = current / max(total, 1)
        pct_text = f"{pct * 100:.1f}%"
        detail = f"{current}/{total}"
        if label:
            detail += f"  {label}"

        def _do():
            self._progress_bar.set(pct)
            self._progress_label.configure(text=pct_text)
            self._progress_detail.configure(text=detail)
            self._log_textbox.configure(state="normal")
            self._log_textbox.delete("end-2l", "end-1l")
            self._log_textbox.insert("end", f"  [{pct_text}] {detail}\n")
            self._log_textbox.see("end")
            self._log_textbox.configure(state="disabled")

        if threading.current_thread() is threading.main_thread():
            _do()
        else:
            self.after(0, _do)

    def _reset_progress(self):
        def _do():
            self._progress_bar.set(0)
            self._progress_label.configure(text="")
            self._progress_detail.configure(text="")

        if threading.current_thread() is threading.main_thread():
            _do()
        else:
            self.after(0, _do)

    def _toggle_log_expand(self):
        self._log_expanded = not self._log_expanded
        if self._log_expanded:
            self._pipeline_frame.grid_rowconfigure(5, weight=1)
            self._pipeline_frame.grid_rowconfigure(2, weight=0)
            self._log_textbox.grid(
                row=5, column=0, sticky="nsew", padx=PAGE_PAD_X, pady=(SP_SM, 0)
            )
            self._cmd_bar.grid(
                row=6, column=0, sticky="ew", padx=PAGE_PAD_X, pady=(SP_SM, SP_SM)
            )
            self._log_toggle_btn.configure(text=f"Hide details  {ICON_ARROW_UP}")
        else:
            self._pipeline_frame.grid_rowconfigure(5, weight=0)
            self._pipeline_frame.grid_rowconfigure(2, weight=1)
            self._log_textbox.grid_remove()
            self._cmd_bar.grid_remove()
            self._log_toggle_btn.configure(text=f"Show details  {ICON_ARROW_DOWN}")

    def _clear_all(self):
        self._log_textbox.configure(state="normal")
        self._log_textbox.delete("1.0", "end")
        self._log_textbox.configure(state="disabled")
        self._step_log.clear()
        self._reset_progress()

    # ==================================================================
    # Thread runner
    # ==================================================================

    def _run_in_thread(self, fn, *args):
        def _worker():
            old_out, old_err = sys.stdout, sys.stderr
            redir_out = _LogRedirector(self.log, old_out)
            redir_err = _LogRedirector(self.log, old_err)
            sys.stdout = redir_out
            sys.stderr = redir_err
            try:
                fn(*args)
            except Exception as exc:
                import traceback as _tb

                sys.stdout, sys.stderr = old_out, old_err
                tb_text = _tb.format_exc()
                sys.stdout, sys.stderr = redir_out, redir_err
                self.log(f"\nERROR: {exc}\n{tb_text}\n")
            finally:
                sys.stdout, sys.stderr = old_out, old_err

        threading.Thread(target=_worker, daemon=True).start()

    # ==================================================================
    # Command input
    # ==================================================================

    def _on_cmd_enter(self, _event=None):
        cmd = self._cmd_var.get().strip()
        if not cmd:
            return
        self._cmd_var.set("")
        self._cmd_history.append(cmd)
        self._cmd_history_idx = -1
        self.log(f"\n> {cmd}\n")
        self._run_in_thread(lambda: self._execute_cmd(cmd))

    def _execute_cmd(self, cmd):
        try:
            full_cmd = f"source env/bin/activate && {cmd}"
            result = subprocess.run(
                full_cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=300,
                cwd=str(PROJECT_ROOT),
                executable="/bin/bash",
            )
            if result.stdout:
                self.log(result.stdout)
            if result.stderr:
                self.log(result.stderr)
            if result.returncode != 0:
                self.log(f"[exit code: {result.returncode}]\n")
        except subprocess.TimeoutExpired:
            self.log("[command timed out after 300s]\n")
        except Exception as e:
            self.log(f"[error: {e}]\n")

    def _on_cmd_history_up(self, _event=None):
        if not self._cmd_history:
            return
        if self._cmd_history_idx == -1:
            self._cmd_history_idx = len(self._cmd_history) - 1
        elif self._cmd_history_idx > 0:
            self._cmd_history_idx -= 1
        self._cmd_var.set(self._cmd_history[self._cmd_history_idx])

    def _on_cmd_history_down(self, _event=None):
        if not self._cmd_history or self._cmd_history_idx == -1:
            return
        if self._cmd_history_idx < len(self._cmd_history) - 1:
            self._cmd_history_idx += 1
            self._cmd_var.set(self._cmd_history[self._cmd_history_idx])
        else:
            self._cmd_history_idx = -1
            self._cmd_var.set("")

    # ==================================================================
    # Pipeline actions
    # ==================================================================

    # -- Annotate --

    def _on_annotate(self):
        sources = self._gather_annotation_sources()
        if not sources:
            self.log("\nERROR: No image sources found.\n")
            self.log("Capture screenshots (Launch Game) or record sessions first.\n")
            return
        if len(sources) == 1:
            self._open_annotate_for_source(sources[0])
            return
        self._show_annotation_source_dialog(sources)

    def _gather_annotation_sources(self):
        sources = []

        ss_dir = PROJECT_ROOT / "screenshots" / "captures"
        if ss_dir.exists():
            imgs = list(ss_dir.glob("*.png")) + list(ss_dir.glob("*.jpg"))
            if imgs:
                sources.append(
                    {
                        "label": f"Screenshots  ({len(imgs)} images)",
                        "path": str(ss_dir),
                        "count": len(imgs),
                        "kind": "screenshots",
                    }
                )

        sessions_root = PROJECT_ROOT / "recordings" / "sessions"
        if sessions_root.exists():
            for sess in sorted(sessions_root.iterdir(), reverse=True):
                if sess.is_dir() and (sess / "session.json").exists():
                    frames = list(sess.glob("frame_*.png"))
                    if frames:
                        sources.append(
                            {
                                "label": f"Session: {sess.name}  ({len(frames)} frames)",
                                "path": str(sess),
                                "count": len(frames),
                                "kind": "session",
                            }
                        )

        yolo_imgs = PROJECT_ROOT / "yolo_dataset" / "train" / "images"
        if yolo_imgs.exists():
            imgs = list(yolo_imgs.glob("*.png")) + list(yolo_imgs.glob("*.jpg"))
            if imgs:
                sources.append(
                    {
                        "label": f"YOLO Dataset  ({len(imgs)} images)",
                        "path": str(yolo_imgs),
                        "count": len(imgs),
                        "kind": "dataset",
                    }
                )

        return sources

    def _show_annotation_source_dialog(self, sources):
        dialog = ctk.CTkToplevel(self)
        dialog.title("Select Annotation Source")
        dialog.geometry("520x440")
        dialog.resizable(False, True)
        dialog.transient(self.winfo_toplevel())

        ctk.CTkLabel(
            dialog,
            text="Choose an image source to annotate",
            font=font_subheading(),
            text_color=TEXT_PRIMARY,
        ).pack(padx=SP_XL, pady=(SP_XL, SP_MD))

        scroll = ctk.CTkScrollableFrame(dialog, width=460, height=280)
        scroll.pack(padx=SP_XL, pady=(0, SP_MD), fill="both", expand=True)

        selected = tk.IntVar(value=0)
        for idx, src in enumerate(sources):
            ctk.CTkRadioButton(
                scroll,
                text=src["label"],
                variable=selected,
                value=idx,
                font=font_body(),
            ).pack(anchor="w", padx=SP_MD, pady=SP_XS)

        btn_frame = ctk.CTkFrame(dialog, fg_color="transparent")
        btn_frame.pack(padx=SP_XL, pady=(0, SP_XL))

        def _open():
            dialog.destroy()
            self._open_annotate_for_source(sources[selected.get()])

        ctk.CTkButton(
            btn_frame,
            text="Open",
            width=120,
            height=BTN_HEIGHT_MD,
            corner_radius=RADIUS_MD,
            fg_color=CLR_GREEN,
            command=_open,
        ).pack(side="left", padx=SP_SM)
        ctk.CTkButton(
            btn_frame,
            text="Cancel",
            width=90,
            height=BTN_HEIGHT_MD,
            corner_radius=RADIUS_MD,
            fg_color="gray",
            command=dialog.destroy,
        ).pack(side="left", padx=SP_SM)

        dialog.after(100, lambda: dialog.grab_set())

    def _open_annotate_for_source(self, source):
        from pipeline.review_results import AnnotationWindow

        self.log(
            f"\n--- Opening Annotation Tool ({source['kind']}: {Path(source['path']).name}) ---\n"
        )
        self._app.set_status("Annotating...")
        AnnotationWindow(self.winfo_toplevel(), screenshots_dir=source["path"])

    # -- Describe & Review --

    def _on_describe_review(self):
        """Open AnnotationWindow in Describe & Review mode.

        Reuses annotation source picker, then opens the annotation tool
        with the review queue pre-activated. User reviews N images by
        writing descriptions, generating boxes via LLM, and correcting.
        """
        sources = self._gather_annotation_sources()
        if not sources:
            self.log("\nERROR: No image sources found.\n")
            self.log("Capture screenshots or record sessions first.\n")
            return
        if len(sources) == 1:
            self._open_describe_review_for_source(sources[0])
            return

        # Show source picker, then open in review mode
        self._show_describe_review_source_dialog(sources)

    def _show_describe_review_source_dialog(self, sources):
        """Source picker that opens in Describe & Review mode."""
        dialog = ctk.CTkToplevel(self)
        dialog.title("Describe & Review — Select Source")
        dialog.geometry("520x500")
        dialog.resizable(False, True)
        dialog.transient(self.winfo_toplevel())

        ctk.CTkLabel(
            dialog,
            text="Choose an image source to review",
            font=font_subheading(),
            text_color=TEXT_PRIMARY,
        ).pack(padx=SP_XL, pady=(SP_XL, SP_MD))

        scroll = ctk.CTkScrollableFrame(dialog, width=460, height=220)
        scroll.pack(padx=SP_XL, pady=(0, SP_MD), fill="both", expand=True)

        selected = tk.IntVar(value=0)
        for idx, src in enumerate(sources):
            ctk.CTkRadioButton(
                scroll,
                text=src["label"],
                variable=selected,
                value=idx,
                font=font_body(),
            ).pack(anchor="w", padx=SP_MD, pady=SP_XS)

        # Count picker
        count_frame = ctk.CTkFrame(dialog, fg_color="transparent")
        count_frame.pack(padx=SP_XL, pady=(0, SP_MD))
        ctk.CTkLabel(
            count_frame, text="Images to review:", font=font_body(),
        ).pack(side="left", padx=(0, SP_SM))
        count_var = tk.StringVar(value="100")
        ctk.CTkEntry(
            count_frame, textvariable=count_var, width=80, font=font_body(),
        ).pack(side="left")

        btn_frame = ctk.CTkFrame(dialog, fg_color="transparent")
        btn_frame.pack(padx=SP_XL, pady=(0, SP_XL))

        def _open():
            try:
                count = int(count_var.get())
            except ValueError:
                count = 100
            dialog.destroy()
            self._open_describe_review_for_source(sources[selected.get()], count)

        ctk.CTkButton(
            btn_frame, text="Start Review", width=130, height=BTN_HEIGHT_MD,
            corner_radius=RADIUS_MD, fg_color=("#6A1B9A", "#AB47BC"), command=_open,
        ).pack(side="left", padx=SP_SM)
        ctk.CTkButton(
            btn_frame, text="Cancel", width=90, height=BTN_HEIGHT_MD,
            corner_radius=RADIUS_MD, fg_color="gray", command=dialog.destroy,
        ).pack(side="left", padx=SP_SM)

        dialog.after(100, lambda: dialog.grab_set())

    def _open_describe_review_for_source(self, source, count: int = 100):
        """Open AnnotationWindow and immediately start review queue."""
        from pipeline.review_results import AnnotationWindow

        self.log(
            f"\n--- Describe & Review ({source['kind']}: {Path(source['path']).name}, "
            f"{count} images) ---\n"
        )
        self._app.set_status("Describe & Review...")
        window = AnnotationWindow(self.winfo_toplevel(), screenshots_dir=source["path"])
        # Activate review queue after window is ready
        window.after(500, lambda: window._start_review_queue(count=count))

    # -- Train Model --

    def _on_train_model(self):
        from conf.config_parser import training_conf

        output_name = training_conf.MODEL_OUTPUT_NAME or "paper_engine_model"
        models_dir = PROJECT_ROOT / "bot_logic" / "models"

        existing = sorted(
            p
            for p in models_dir.glob(f"{output_name}*")
            if p.is_dir() and (p / "weights").exists()
        )

        if existing:
            self._show_save_previous_dialog(existing, output_name, models_dir)
        else:
            self._run_in_thread(self._train_model_worker)

    def _show_save_previous_dialog(self, existing_dirs, output_name, models_dir):
        dialog = ctk.CTkToplevel(self)
        dialog.title("Previous Model Found")
        dialog.geometry("500x260")
        dialog.resizable(False, False)
        dialog.transient(self.winfo_toplevel())

        dir_names = "\n".join(f"  {p.name}/" for p in existing_dirs)
        ctk.CTkLabel(
            dialog,
            text=f"Found {len(existing_dirs)} existing model(s):\n{dir_names}",
            font=font_body(),
            text_color=TEXT_SECONDARY,
            justify="left",
        ).pack(padx=SP_XL, pady=(SP_XL, SP_SM), anchor="w")
        ctk.CTkLabel(
            dialog,
            text="What do you want to do with them?",
            font=font_body_bold(),
            text_color=TEXT_PRIMARY,
        ).pack(padx=SP_XL, pady=(0, SP_LG))

        bf = ctk.CTkFrame(dialog, fg_color="transparent")
        bf.pack(padx=SP_XL, pady=(0, SP_XL))

        def _save_and_train():
            dialog.destroy()
            self._run_in_thread(
                lambda: self._archive_and_train(existing_dirs, output_name)
            )

        def _delete_and_train():
            dialog.destroy()
            self._run_in_thread(lambda: self._delete_and_train_impl(existing_dirs))

        ctk.CTkButton(
            bf,
            text="Save & Train",
            width=130,
            height=BTN_HEIGHT_MD,
            corner_radius=RADIUS_MD,
            fg_color=CLR_GREEN,
            command=_save_and_train,
        ).pack(side="left", padx=SP_SM)
        ctk.CTkButton(
            bf,
            text="Delete & Train",
            width=130,
            height=BTN_HEIGHT_MD,
            corner_radius=RADIUS_MD,
            fg_color=CLR_RED,
            command=_delete_and_train,
        ).pack(side="left", padx=SP_SM)
        ctk.CTkButton(
            bf,
            text="Cancel",
            width=90,
            height=BTN_HEIGHT_MD,
            corner_radius=RADIUS_MD,
            fg_color="gray",
            command=dialog.destroy,
        ).pack(side="left", padx=SP_SM)

        dialog.after(100, lambda: dialog.grab_set())

    def _archive_and_train(self, existing_dirs, output_name):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        for model_dir in existing_dirs:
            archive_name = f"{model_dir.name}_saved_{timestamp}"
            archive_dir = model_dir.parent / archive_name
            self.log(f"\nArchiving {model_dir.name}/ -> {archive_name}/\n")
            try:
                model_dir.rename(archive_dir)
                self.log("  Archived.\n")
            except Exception as e:
                self.log(f"  WARNING: Could not archive: {e}\n")
        self._train_model_worker()

    def _delete_and_train_impl(self, existing_dirs):
        for model_dir in existing_dirs:
            self.log(f"\nDeleting {model_dir.name}/\n")
            try:
                shutil.rmtree(model_dir)
                self.log("  Deleted.\n")
            except Exception as e:
                self.log(f"  WARNING: Could not delete: {e}\n")
        self._train_model_worker()

    def _train_model_worker(self):
        self._switch_subtab_safe("pipeline")
        mode = self._train_mode_var.get()

        if mode == "cloud":
            self._train_cloud_worker()
            return

        s_train = self._step_log.add("Training YOLO model (local)")
        self.log("\n--- Training YOLO Model (Local GPU) ---\n")
        self._app.set_status("Training model...")
        from pipeline import training_model

        self._log_progress(0, 1, "Starting...")
        _self = self

        def _on_epoch_end(trainer):
            epoch = trainer.epoch + 1
            total = trainer.epochs
            metrics = trainer.metrics or {}
            mAP = metrics.get("metrics/mAP50(B)", 0)
            loss = metrics.get("train/box_loss", 0)
            label = ""
            if mAP:
                label = f"mAP50={mAP:.3f}"
            elif loss:
                label = f"loss={loss:.3f}"
            _self._log_progress(epoch, total, label)
            _self._step_log.update(s_train, f"Training epoch {epoch}/{total}  {label}")
            _self._app.set_status(f"Training epoch {epoch}/{total}")

        training_model.main(on_epoch_end=_on_epoch_end)
        self._step_log.complete(s_train, "Training complete")
        self.log("\nTraining complete.\n")
        self._reset_progress()
        self._app.set_status("Idle")

    def _train_cloud_worker(self):
        """Cloud training stub -- Ultralytics HUB / HuggingFace (not yet implemented)."""
        s = self._step_log.add("Cloud training")
        self.log("\n--- Cloud Training ---\n")
        self.log(
            "Cloud training via Ultralytics HUB or HuggingFace is not yet implemented.\n"
        )
        self.log("To use cloud training:\n")
        self.log("  1. Configure your cloud provider in Settings\n")
        self.log("  2. Upload your dataset via Settings > Cloud Storage\n")
        self.log("  3. Cloud training integration is coming soon\n")
        self._step_log.fail(s, "Cloud training not yet implemented")
        self._app.set_status("Idle")

    # -- Import Model --

    def _on_import_model(self):
        from tkinter import filedialog

        path = filedialog.askopenfilename(
            title="Import YOLO Model (.pt)",
            filetypes=[("PyTorch Model", "*.pt"), ("All Files", "*.*")],
            initialdir=str(Path.home()),
        )
        if not path:
            return
        self._run_in_thread(lambda: self._import_model_worker(Path(path)))

    def _import_model_worker(self, src_path: Path):
        self._switch_subtab_safe("pipeline")
        s1 = self._step_log.add(f"Importing {src_path.name}")
        self._app.set_status("Importing model...")

        models_dir = PROJECT_ROOT / "bot_logic" / "models"
        models_dir.mkdir(parents=True, exist_ok=True)

        stem = src_path.stem
        if stem in ("best", "last"):
            model_name = f"imported_{src_path.parent.name}"
        else:
            model_name = f"imported_{stem}"
        model_name = model_name.replace(" ", "_").replace("/", "_")

        dest_dir = models_dir / model_name / "weights"
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_file = dest_dir / "best.pt"

        s2 = self._step_log.add(f"Copying to {model_name}/weights/best.pt")
        try:
            shutil.copy2(str(src_path), str(dest_file))
            self._step_log.complete(s2, f"Copied to {model_name}/weights/best.pt")
        except Exception as e:
            self._step_log.fail(s2, f"Copy failed: {e}")
            self._step_log.fail(s1, "Import failed")
            self._app.set_status("Error")
            return

        s3 = self._step_log.add("Updating configuration")
        try:
            ini_path = PROJECT_ROOT / "conf" / "training_conf.ini"
            cp = configparser.ConfigParser()
            cp.read(ini_path)
            if not cp.has_section("Output"):
                cp.add_section("Output")
            cp.set("Output", "model_output_name", model_name)
            cp.set(
                "Output", "best_model_path", str(dest_file.relative_to(PROJECT_ROOT))
            )
            with open(ini_path, "w") as f:
                cp.write(f)
            self._step_log.complete(s3, "Configuration updated")
        except Exception as e:
            self._step_log.fail(s3, f"Config update failed: {e}")

        size_mb = dest_file.stat().st_size / (1024 * 1024)
        self._step_log.complete(s1, f"Imported {src_path.name} ({size_mb:.1f} MB)")
        self.log(f"\nModel imported: {dest_file}\n")
        self._app.set_status("Idle")

    # -- Verify / Retrain --

    def _on_verify_retrain(self):
        dialog = ctk.CTkToplevel(self)
        dialog.title("Verify / Retrain")
        dialog.geometry("500x340")
        dialog.resizable(False, False)
        dialog.transient(self.winfo_toplevel())

        ctk.CTkLabel(
            dialog,
            text="Label Verification & Retraining",
            font=font_subheading(),
            text_color=TEXT_PRIMARY,
        ).pack(padx=SP_XL, pady=(SP_XL, SP_MD))

        split_frame = ctk.CTkFrame(dialog, fg_color="transparent")
        split_frame.pack(padx=SP_XL, pady=(0, SP_MD))
        ctk.CTkLabel(split_frame, text="Target split:", font=font_body()).pack(
            side="left", padx=(0, SP_MD)
        )
        split_var = ctk.StringVar(value="both")
        for val, label in [("train", "Train"), ("val", "Val"), ("both", "Both")]:
            ctk.CTkRadioButton(
                split_frame, text=label, variable=split_var, value=val, font=font_body()
            ).pack(side="left", padx=SP_SM)

        max_frame = ctk.CTkFrame(dialog, fg_color="transparent")
        max_frame.pack(padx=SP_XL, pady=(0, SP_MD))
        ctk.CTkLabel(max_frame, text="Max images to audit:", font=font_body()).pack(
            side="left", padx=(0, SP_MD)
        )
        max_var = ctk.StringVar(value="300")
        ctk.CTkEntry(
            max_frame,
            textvariable=max_var,
            width=80,
            height=INPUT_HEIGHT,
            corner_radius=RADIUS_MD,
        ).pack(side="left")

        bf = ctk.CTkFrame(dialog, fg_color="transparent")
        bf.pack(padx=SP_XL, pady=(SP_MD, SP_MD))

        def _run_verify():
            split = split_var.get()
            max_imgs = int(max_var.get())
            dialog.destroy()
            self._run_in_thread(lambda: self._verify_worker(split, max_imgs))

        def _run_verify_and_train():
            split = split_var.get()
            max_imgs = int(max_var.get())
            dialog.destroy()
            self._run_in_thread(lambda: self._verify_and_train_worker(split, max_imgs))

        def _run_inference():
            dialog.destroy()
            self._run_in_thread(self._inference_review_worker)

        ctk.CTkButton(
            bf,
            text="Verify Labels",
            width=130,
            height=BTN_HEIGHT_MD,
            corner_radius=RADIUS_MD,
            fg_color=CLR_BLUE,
            command=_run_verify,
        ).pack(side="left", padx=SP_SM)
        ctk.CTkButton(
            bf,
            text="Verify & Retrain",
            width=140,
            height=BTN_HEIGHT_MD,
            corner_radius=RADIUS_MD,
            fg_color=CLR_GREEN,
            command=_run_verify_and_train,
        ).pack(side="left", padx=SP_SM)
        ctk.CTkButton(
            bf,
            text="Cancel",
            width=90,
            height=BTN_HEIGHT_MD,
            corner_radius=RADIUS_MD,
            fg_color="gray",
            command=dialog.destroy,
        ).pack(side="left", padx=SP_SM)

        ctk.CTkButton(
            dialog,
            text="Run Inference & Review",
            width=200,
            height=BTN_HEIGHT_MD,
            corner_radius=RADIUS_MD,
            fg_color=CLR_PURPLE,
            command=_run_inference,
        ).pack(padx=SP_XL, pady=(SP_SM, SP_XL))

        dialog.after(100, lambda: dialog.grab_set())

    def _verify_worker(self, split, max_images):
        self.log(f"\n--- Verifying Labels ({split}) ---\n")
        self._app.set_status(f"Verifying {split} labels...")

        try:
            from pipeline.audit_labels import run_audit, sample_auto_labeled

            if split == "both":
                train_n = int(max_images * 0.85)
                val_n = max_images - train_n
                train_samples = sample_auto_labeled("train", max_images=train_n)
                val_samples = sample_auto_labeled("val", max_images=val_n)
                total = len(train_samples) + len(val_samples)
                self.log(
                    f"Sampling {len(train_samples)} train + {len(val_samples)} val = {total} images\n"
                )
            else:
                self.log(f"Sampling up to {max_images} images from {split}/\n")

            report = run_audit(max_images=max_images, batch_size=5)
            if report:
                ok = report.get("ok", 0)
                issues = report.get("issues", 0)
                total_imgs = report.get("total_images", 0)
                rate = issues / max(total_imgs, 1) * 100
                self.log(
                    f"\nVerification complete: {ok}/{total_imgs} ok, "
                    f"{issues} issues ({rate:.1f}% error rate)\n"
                )
                self.log("Full report: audit_report.json\n")

            self._app.set_status("Idle")
        except Exception as e:
            self.log(f"\nERROR: Verification failed: {e}\n")
            self._app.set_status("Error")

    def _verify_and_train_worker(self, split, max_images):
        self._verify_worker(split, max_images)
        self.log("\n--- Starting Retraining ---\n")
        self._train_model_worker()

    def _inference_review_worker(self):
        self.log("\n--- Running Model Inference ---\n")
        self._app.set_status("Running inference...")
        from pipeline import test_model as tm

        results = tm.test_model()
        class_names = tm.get_class_names()

        if results is None:
            self.log("\nInference failed or returned no results.\n")
            self._app.set_status("Error")
            return

        self.log("\nInference complete. Opening Review Results...\n")
        self._app.set_status("Idle")
        self.after(0, lambda: self._open_review_window(results, class_names))

    def _open_review_window(self, results, class_names):
        from pipeline.review_results import ReviewResultsWindow

        ReviewResultsWindow(self.winfo_toplevel(), results, class_names=class_names)

    # -- Generate Bot --

    def _on_generate_bot(self):
        self._run_in_thread(self._generate_bot_worker)

    def _generate_bot_worker(self):
        self.log("\n--- Generating Bot Script ---\n")
        self._app.set_status("Generating bot...")

        from pipeline.generate_bot_script import (
            generate_bot_script,
            save_bot_script,
            search_game_controls,
            save_controls_to_config,
            read_controls_from_config,
        )
        from tools.functions import get_title, path_finder

        # Try to get game path from session page
        session = self._app._pages.get("session")
        game_path = ""
        if session and hasattr(session, "game_path"):
            game_path = session.game_path
        if not game_path:
            from conf.config_parser import main_conf as config

            game_path = str(config.GAME_PATH)

        exe_path = path_finder(game_path)
        if not exe_path:
            self.log("ERROR: No game executable found.\n")
            self._app.set_status("Error")
            return

        game_title = get_title(exe_path)
        self.log(f"Game: {game_title}\n")

        existing = read_controls_from_config(game_title)
        if existing:
            self.log("Found existing controls, improving...\n")
            controls = search_game_controls(game_title, existing)
        else:
            self.log("Searching for controls...\n")
            controls = search_game_controls(game_title)

        if not controls:
            self.log("ERROR: Could not retrieve controls.\n")
            self._app.set_status("Error")
            return

        save_controls_to_config(game_title, exe_path, controls)
        self.log("Controls saved.\n")

        self.log("Generating bot script (this may take 1-2 minutes)...\n")
        code = generate_bot_script(game_title, controls)
        if code:
            path = save_bot_script(game_title, code)
            self.log(f"\nBot script saved to: {path}\n")
        else:
            self.log("ERROR: Bot script generation failed.\n")

        self._app.set_status("Idle")

    # -- Game Report --

    def _on_game_report(self):
        """Show session picker then generate LLM-powered game feedback report."""
        sessions_dir = PROJECT_ROOT / "recordings" / "sessions"
        sessions = []
        if sessions_dir.exists():
            for d in sorted(sessions_dir.iterdir(), reverse=True):
                if d.is_dir() and (
                    list(d.glob("frame_*.json")) or list(d.glob("frame_*.png"))
                ):
                    frame_count = len(list(d.glob("frame_*.png")))
                    sessions.append({"name": d.name, "path": d, "frames": frame_count})

        if not sessions:
            self.log("\nNo recorded sessions found.\n")
            self.log("Record a game session first (Test > Launch Game).\n")
            return

        if len(sessions) == 1:
            self._run_in_thread(lambda: self._game_report_worker(sessions[0]["path"]))
            return

        # Session picker dialog
        dlg = ctk.CTkToplevel(self)
        dlg.title("Game Report -- Select Session")
        dlg.geometry("480x380")
        dlg.resizable(False, True)
        dlg.transient(self.winfo_toplevel())

        ctk.CTkLabel(
            dlg,
            text="Select a session to analyze",
            font=font_subheading(),
            text_color=TEXT_PRIMARY,
        ).pack(padx=SP_XL, pady=(SP_XL, SP_MD))

        import tkinter as _tk

        scroll = ctk.CTkScrollableFrame(dlg, height=200)
        scroll.pack(fill="both", expand=True, padx=SP_XL, pady=(0, SP_MD))

        selected = _tk.IntVar(value=0)
        for idx, s in enumerate(sessions):
            ctk.CTkRadioButton(
                scroll,
                text=f"{s['name']}  ({s['frames']} frames)",
                variable=selected,
                value=idx,
                font=font_body(),
            ).pack(anchor="w", padx=SP_MD, pady=SP_XS)

        bf = ctk.CTkFrame(dlg, fg_color="transparent")
        bf.pack(padx=SP_XL, pady=(0, SP_XL))

        def _run():
            path = sessions[selected.get()]["path"]
            dlg.destroy()
            self._run_in_thread(lambda: self._game_report_worker(path))

        ctk.CTkButton(
            bf,
            text="Analyze",
            width=120,
            height=BTN_HEIGHT_MD,
            corner_radius=RADIUS_MD,
            fg_color=CLR_CYAN,
            command=_run,
        ).pack(side="left", padx=SP_SM)
        ctk.CTkButton(
            bf,
            text="Cancel",
            width=90,
            height=BTN_HEIGHT_MD,
            corner_radius=RADIUS_MD,
            fg_color="gray",
            command=dlg.destroy,
        ).pack(side="left", padx=SP_SM)

        dlg.after(100, lambda: dlg.grab_set())

    def _game_report_worker(self, session_path):
        """Run game feedback analysis in background thread."""
        from pipeline.game_feedback import generate_report, export_report

        self._switch_subtab_safe("pipeline")
        s = self._step_log.add("Generating game report")
        self._app.set_status("Analyzing session...")

        report = generate_report(
            session_dir=session_path,
            log_fn=self.log,
        )

        self._step_log.complete(s, f"Report done -- {report.overall_score():.1f}/5")

        # Print the full text report to log
        self.log("\n" + report.to_text() + "\n")

        # Auto-export to file
        export_path = PROJECT_ROOT / "reports"
        export_path.mkdir(exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_file = export_path / f"game_report_{report.session_name}_{ts}.txt"
        export_report(report, out_file)
        self.log(f"\nExported to: {out_file}\n")
        self._app.set_status("Idle")

    # -- Directory Report (multi-source) --

    def _on_dir_report(self):
        """Show directory picker dialog for multi-source game analysis."""
        from tkinter import filedialog as _fd

        dlg = ctk.CTkToplevel(self)
        dlg.title("Dir Report -- Game Directory Analysis")
        dlg.geometry("560x400")
        dlg.resizable(False, False)
        dlg.transient(self.winfo_toplevel())

        ctk.CTkLabel(
            dlg,
            text="Multi-Source Game Analysis",
            font=font_subheading(),
            text_color=TEXT_PRIMARY,
        ).pack(padx=SP_XL, pady=(SP_XL, SP_SM))

        ctk.CTkLabel(
            dlg,
            text="Scans engine configs, detects engine type, fetches ProtonDB\n"
            "data, reads Wine/DXVK prefix, checks known fixes, then\n"
            "sends everything to the LLM for expert-level analysis.",
            font=font_small(),
            text_color=TEXT_MUTED,
            justify="center",
        ).pack(padx=SP_XL, pady=(0, SP_MD))

        # -- Game directory picker --
        dir_frame = ctk.CTkFrame(dlg, fg_color="transparent")
        dir_frame.pack(fill="x", padx=SP_XL, pady=(0, SP_SM))

        ctk.CTkLabel(
            dir_frame,
            text="Game directory",
            font=font_body_bold(),
            text_color=TEXT_SECONDARY,
        ).pack(anchor="w")

        dir_row = ctk.CTkFrame(dir_frame, fg_color="transparent")
        dir_row.pack(fill="x", pady=(SP_XS, 0))

        game_dir_var = tk.StringVar()
        dir_entry = ctk.CTkEntry(
            dir_row,
            textvariable=game_dir_var,
            height=INPUT_HEIGHT,
            font=font_mono_small(),
            corner_radius=RADIUS_SM,
        )
        dir_entry.pack(side="left", fill="x", expand=True, padx=(0, SP_SM))

        def _browse_dir():
            path = _fd.askdirectory(title="Select game install directory")
            if path:
                game_dir_var.set(path)

        ctk.CTkButton(
            dir_row,
            text="Browse",
            width=80,
            height=INPUT_HEIGHT,
            corner_radius=RADIUS_SM,
            fg_color=BG_HOVER,
            text_color=TEXT_PRIMARY,
            command=_browse_dir,
        ).pack(side="right")

        # -- Prefix path (optional) --
        pfx_frame = ctk.CTkFrame(dlg, fg_color="transparent")
        pfx_frame.pack(fill="x", padx=SP_XL, pady=(0, SP_SM))

        ctk.CTkLabel(
            pfx_frame,
            text="Wine prefix (optional, auto-detected)",
            font=font_body_bold(),
            text_color=TEXT_SECONDARY,
        ).pack(anchor="w")

        pfx_row = ctk.CTkFrame(pfx_frame, fg_color="transparent")
        pfx_row.pack(fill="x", pady=(SP_XS, 0))

        prefix_var = tk.StringVar()
        ctk.CTkEntry(
            pfx_row,
            textvariable=prefix_var,
            height=INPUT_HEIGHT,
            font=font_mono_small(),
            corner_radius=RADIUS_SM,
            placeholder_text="Leave blank to auto-detect",
        ).pack(side="left", fill="x", expand=True, padx=(0, SP_SM))

        def _browse_pfx():
            path = _fd.askdirectory(title="Select Wine prefix directory")
            if path:
                prefix_var.set(path)

        ctk.CTkButton(
            pfx_row,
            text="Browse",
            width=80,
            height=INPUT_HEIGHT,
            corner_radius=RADIUS_SM,
            fg_color=BG_HOVER,
            text_color=TEXT_PRIMARY,
            command=_browse_pfx,
        ).pack(side="right")

        # -- Options --
        opt_frame = ctk.CTkFrame(dlg, fg_color="transparent")
        opt_frame.pack(fill="x", padx=SP_XL, pady=(SP_SM, SP_MD))

        cache_var = tk.BooleanVar(value=True)
        ctk.CTkCheckBox(
            opt_frame,
            text="Cache report data (save raw source data for later LLM use)",
            variable=cache_var,
            font=font_body(),
            text_color=TEXT_SECONDARY,
            corner_radius=RADIUS_SM,
            checkbox_width=20,
            checkbox_height=20,
        ).pack(anchor="w")

        # -- Buttons --
        bf = ctk.CTkFrame(dlg, fg_color="transparent")
        bf.pack(padx=SP_XL, pady=(0, SP_XL))

        def _run():
            gdir = game_dir_var.get().strip()
            if not gdir:
                self.log("\nERROR: No game directory selected.\n")
                return
            pfx = prefix_var.get().strip() or None
            cache = cache_var.get()
            dlg.destroy()
            self._run_in_thread(lambda: self._dir_report_worker(gdir, pfx, cache))

        ctk.CTkButton(
            bf,
            text="Analyze",
            width=120,
            height=BTN_HEIGHT_MD,
            corner_radius=RADIUS_MD,
            fg_color=("#0097A7", "#00BCD4"),
            command=_run,
        ).pack(side="left", padx=SP_SM)
        ctk.CTkButton(
            bf,
            text="Cancel",
            width=90,
            height=BTN_HEIGHT_MD,
            corner_radius=RADIUS_MD,
            fg_color="gray",
            command=dlg.destroy,
        ).pack(side="left", padx=SP_SM)

        dlg.after(100, lambda: dlg.grab_set())

    def _dir_report_worker(self, game_dir_str, prefix_str, cache_data):
        """Run multi-source game directory analysis in background thread."""
        from pipeline.game_feedback import (
            generate_report_from_dir,
            export_report,
        )
        import json as _json

        self._switch_subtab_safe("pipeline")
        game_path = Path(game_dir_str)
        s = self._step_log.add(f"Dir report: {game_path.name}")
        self._app.set_status(f"Analyzing {game_path.name}...")

        prefix_path = Path(prefix_str) if prefix_str else None

        report = generate_report_from_dir(
            game_dir=game_path,
            prefix_dir=prefix_path,
            log_fn=self.log,
        )

        self._step_log.complete(s, f"Report done -- {report.overall_score():.1f}/5")

        # Print the full text report to log
        self.log("\n" + report.to_text() + "\n")

        # Auto-export text report
        export_dir = PROJECT_ROOT / "reports"
        export_dir.mkdir(exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        safe_name = game_path.name.replace(" ", "_")
        out_file = export_dir / f"dir_report_{safe_name}_{ts}.txt"
        export_report(report, out_file)
        self.log(f"\nExported to: {out_file}\n")

        # Cache raw source data + report JSON for later LLM use
        if cache_data:
            cache_dir = export_dir / "cache"
            cache_dir.mkdir(exist_ok=True)
            cache_file = cache_dir / f"dir_report_{safe_name}_{ts}.json"
            cache_payload = {
                "game": report.game,
                "generated_at": report.generated_at,
                "session_name": report.session_name,
                "report": report.to_dict(),
                "raw_analysis": report.raw_analysis,
            }
            cache_file.write_text(_json.dumps(cache_payload, indent=2, default=str))
            self.log(f"Cached report data: {cache_file}\n")

        self._app.set_status("Idle")

    # ==================================================================
    # Labels operations
    # ==================================================================

    def _scan_label_counts(self):
        import yaml as _yaml

        ds = PROJECT_ROOT / "yolo_dataset"
        yaml_path = ds / "dataset.yaml"
        names = {}
        if yaml_path.exists():
            with open(yaml_path) as f:
                cfg = _yaml.safe_load(f) or {}
            raw = cfg.get("names", {})
            names = _parse_names(raw)

        counts = {}
        for split in ("train", "val"):
            lbl_dir = ds / split / "labels"
            if not lbl_dir.exists():
                continue
            for lbl_file in lbl_dir.glob("*.txt"):
                for line in lbl_file.read_text().strip().splitlines():
                    parts = line.strip().split()
                    if not parts:
                        continue
                    cid = int(parts[0])
                    if cid not in counts:
                        counts[cid] = {"train": 0, "val": 0}
                    counts[cid][split] += 1

        return names, counts

    def _refresh_labels_view(self):
        for w in self._label_row_widgets.values():
            w.destroy()
        self._label_row_widgets.clear()
        self._selected_label_id = None

        names, counts = self._scan_label_counts()
        all_ids = sorted(set(list(names.keys()) + list(counts.keys())))

        for cls_id in all_ids:
            name = names.get(cls_id, f"class_{cls_id}")
            tr = counts.get(cls_id, {}).get("train", 0)
            vl = counts.get(cls_id, {}).get("val", 0)
            total = tr + vl

            row = ctk.CTkFrame(
                self._labels_table,
                corner_radius=RADIUS_SM,
                fg_color="transparent",
                cursor="hand2",
            )
            row.pack(fill="x", padx=SP_XS, pady=1)

            for txt, w, bold in [
                (str(cls_id), 50, False),
                (name, 0, False),
                (str(tr), 70, False),
                (str(vl), 70, False),
                (str(total), 70, True),
            ]:
                lbl = ctk.CTkLabel(
                    row,
                    text=txt,
                    width=w,
                    font=font_body_bold() if bold else font_body(),
                    text_color=TEXT_PRIMARY if bold else TEXT_SECONDARY,
                    anchor="center" if w else "w",
                )
                lbl.pack(
                    side="left",
                    padx=SP_SM,
                    expand=(w == 0),
                    fill="x" if w == 0 else "none",
                )
                lbl.bind(
                    "<Button-1>", lambda e, cid=cls_id: self._select_label_row(cid)
                )

            row.bind("<Button-1>", lambda e, cid=cls_id: self._select_label_row(cid))
            self._label_row_widgets[cls_id] = row

    def _select_label_row(self, cls_id):
        self._selected_label_id = cls_id
        for cid, row in self._label_row_widgets.items():
            if cid == cls_id:
                row.configure(
                    border_width=1,
                    border_color=BORDER_ACTIVE,
                    fg_color=BG_ACTIVE,
                )
            else:
                row.configure(
                    border_width=0,
                    border_color=BORDER_SUBTLE,
                    fg_color="transparent",
                )

    def _read_dataset_yaml(self):
        import yaml as _yaml

        yaml_path = PROJECT_ROOT / "yolo_dataset" / "dataset.yaml"
        if not yaml_path.exists():
            return {}
        with open(yaml_path) as f:
            return _yaml.safe_load(f) or {}

    def _write_dataset_yaml(self, cfg):
        import yaml as _yaml

        yaml_path = PROJECT_ROOT / "yolo_dataset" / "dataset.yaml"
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        with open(yaml_path, "w") as f:
            _yaml.dump(cfg, f, default_flow_style=False)

    def _rewrite_label_files(self, transform_fn):
        ds = PROJECT_ROOT / "yolo_dataset"
        changed = 0
        for split in ("train", "val"):
            lbl_dir = ds / split / "labels"
            if not lbl_dir.exists():
                continue
            for lbl_file in lbl_dir.glob("*.txt"):
                lines = lbl_file.read_text().strip().splitlines()
                new_lines = []
                for line in lines:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    cid = int(parts[0])
                    new_cid = transform_fn(cid)
                    if new_cid is None:
                        changed += 1
                        continue
                    if new_cid != cid:
                        changed += 1
                    parts[0] = str(new_cid)
                    new_lines.append(" ".join(parts))
                lbl_file.write_text("\n".join(new_lines) + ("\n" if new_lines else ""))
        return changed

    def _on_add_class(self):
        dialog = ctk.CTkInputDialog(text="Enter new class name:", title="Add Class")
        name = dialog.get_input()
        if not name or not name.strip():
            return
        name = name.strip()

        cfg = self._read_dataset_yaml()
        names = _parse_names(cfg.get("names", {}))
        next_id = max(names.keys()) + 1 if names else 0
        names[next_id] = name
        cfg["names"] = names
        self._write_dataset_yaml(cfg)

        self.log(f"\nAdded class {next_id}: '{name}'\n")
        self._refresh_labels_view()

    def _on_delete_class(self):
        if self._selected_label_id is None:
            return
        cid = self._selected_label_id

        cfg = self._read_dataset_yaml()
        names = _parse_names(cfg.get("names", {}))
        cls_name = names.get(cid, f"class_{cid}")

        from gui_components import confirm_dialog

        if not confirm_dialog(
            self.winfo_toplevel(),
            "Confirm Delete",
            f"Delete ALL '{cls_name}' (id={cid}) annotations?",
            "This removes every annotation line with this class\nand reindexes higher IDs down by 1.",
        ):
            return

        def _transform(c):
            if c == cid:
                return None
            return c - 1 if c > cid else c

        n = self._rewrite_label_files(_transform)

        new_names = {}
        for old_id, nm in sorted(names.items()):
            if old_id == cid:
                continue
            new_id = old_id - 1 if old_id > cid else old_id
            new_names[new_id] = nm
        cfg["names"] = new_names
        self._write_dataset_yaml(cfg)

        self.log(f"\nDeleted class '{cls_name}' -- {n} annotations changed/removed\n")
        self._selected_label_id = None
        self._refresh_labels_view()

    def _on_consolidate(self):
        cfg = self._read_dataset_yaml()
        names = _parse_names(cfg.get("names", {}))
        if len(names) < 2:
            self.log("\nNeed at least 2 classes to consolidate.\n")
            return

        _, counts = self._scan_label_counts()

        dlg = ctk.CTkToplevel(self)
        dlg.title("Consolidate Labels")
        dlg.geometry("540x580")
        dlg.resizable(False, True)
        dlg.transient(self.winfo_toplevel())

        _SEL_BG = BG_ACTIVE

        # -- Target section --
        ctk.CTkLabel(
            dlg,
            text="Merge into (target):",
            font=font_subheading(),
            text_color=TEXT_PRIMARY,
        ).pack(anchor="w", padx=SP_XL, pady=(SP_XL, SP_XS))
        ctk.CTkLabel(
            dlg,
            text="The class that will receive all merged annotations",
            font=font_small(),
            text_color=TEXT_MUTED,
        ).pack(anchor="w", padx=SP_XL, pady=(0, SP_SM))

        target_frame = ctk.CTkScrollableFrame(dlg, height=130)
        target_frame.pack(fill="x", padx=SP_XL, pady=(0, SP_MD))

        target_var = ctk.IntVar(value=-1)
        _tgt_rows = {}

        def _on_target_change(*_a):
            tid = target_var.get()
            for c, row in _tgt_rows.items():
                if c == tid:
                    row.configure(
                        border_width=1, border_color=BORDER_ACTIVE, fg_color=_SEL_BG
                    )
                else:
                    row.configure(
                        border_width=0,
                        border_color=BORDER_SUBTLE,
                        fg_color="transparent",
                    )
            _rebuild_sources(tid)

        for c in sorted(names.keys()):
            nm = names[c]
            total = counts.get(c, {}).get("train", 0) + counts.get(c, {}).get("val", 0)
            row = ctk.CTkFrame(target_frame, corner_radius=RADIUS_SM)
            row.pack(fill="x", pady=2)
            ctk.CTkRadioButton(
                row,
                text=f"{c}: {nm}",
                variable=target_var,
                value=c,
                font=font_body(),
                command=lambda: _on_target_change(),
            ).pack(side="left", padx=(SP_SM, SP_XS), pady=SP_XS)
            ctk.CTkLabel(
                row,
                text=f"{total:,}",
                font=font_small(),
                text_color=TEXT_MUTED,
                width=60,
            ).pack(side="right", padx=SP_SM)
            _tgt_rows[c] = row

        # -- Source section --
        ctk.CTkLabel(
            dlg,
            text="Select sources to absorb:",
            font=font_subheading(),
            text_color=TEXT_PRIMARY,
        ).pack(anchor="w", padx=SP_XL, pady=(SP_SM, SP_XS))
        ctk.CTkLabel(
            dlg,
            text="These classes will be merged and removed",
            font=font_small(),
            text_color=TEXT_MUTED,
        ).pack(anchor="w", padx=SP_XL, pady=(0, SP_SM))

        source_frame = ctk.CTkScrollableFrame(dlg, height=160)
        source_frame.pack(fill="x", padx=SP_XL, pady=(0, SP_MD))

        _src_vars = {}
        _src_rows = {}
        _src_labels = {}

        def _on_source_toggle(c):
            row = _src_rows.get(c)
            var = _src_vars.get(c)
            if not row or not var:
                return
            if var.get():
                row.configure(
                    border_width=1, border_color=BORDER_ACTIVE, fg_color=_SEL_BG
                )
                for lbl in _src_labels.get(c, []):
                    lbl.configure(text_color=TEXT_PRIMARY)
            else:
                row.configure(
                    border_width=0, border_color=BORDER_SUBTLE, fg_color="transparent"
                )
                for lbl in _src_labels.get(c, []):
                    lbl.configure(text_color=TEXT_MUTED)

        def _rebuild_sources(exclude_id):
            for w in source_frame.winfo_children():
                w.destroy()
            _src_vars.clear()
            _src_rows.clear()
            _src_labels.clear()

            for c in sorted(names.keys()):
                if c == exclude_id:
                    continue
                nm = names[c]
                total = counts.get(c, {}).get("train", 0) + counts.get(c, {}).get(
                    "val", 0
                )

                var = ctk.BooleanVar(value=False)
                _src_vars[c] = var

                row = ctk.CTkFrame(source_frame, corner_radius=RADIUS_SM)
                row.pack(fill="x", pady=2)

                ctk.CTkCheckBox(
                    row,
                    text="",
                    variable=var,
                    width=24,
                    command=lambda ci=c: _on_source_toggle(ci),
                ).pack(side="left", padx=(SP_SM, 0), pady=SP_XS)

                nlbl = ctk.CTkLabel(
                    row,
                    text=f"{c}: {nm}",
                    font=font_body(),
                    text_color=TEXT_MUTED,
                )
                nlbl.pack(side="left", padx=SP_XS)
                clbl = ctk.CTkLabel(
                    row,
                    text=f"{total:,}",
                    font=font_small(),
                    text_color=TEXT_MUTED,
                    width=60,
                )
                clbl.pack(side="right", padx=SP_SM)
                _src_rows[c] = row
                _src_labels[c] = (nlbl, clbl)

        _rebuild_sources(-1)

        # -- Buttons --
        result = [None]

        def _ok():
            tid = target_var.get()
            srcs = [c for c, v in _src_vars.items() if v.get()]
            if tid < 0 or not srcs:
                return

            from gui_components import confirm_dialog

            if not confirm_dialog(
                dlg,
                "Warning",
                "This action cannot be undone.",
                "All selected labels will be permanently\n"
                "rewritten. You will not be able to revert\n"
                "them back to their previous state.",
            ):
                return

            result[0] = (tid, srcs)
            dlg.destroy()

        bf = ctk.CTkFrame(dlg, fg_color="transparent")
        bf.pack(fill="x", padx=SP_XL, pady=(SP_SM, SP_XL))
        ctk.CTkButton(
            bf,
            text="Consolidate",
            width=140,
            height=BTN_HEIGHT_MD,
            corner_radius=RADIUS_MD,
            fg_color=CLR_BLUE,
            command=_ok,
        ).pack(side="left", padx=SP_XS)
        ctk.CTkButton(
            bf,
            text="Cancel",
            width=100,
            height=BTN_HEIGHT_MD,
            corner_radius=RADIUS_MD,
            fg_color="gray",
            command=dlg.destroy,
        ).pack(side="right", padx=SP_XS)

        dlg.after(100, lambda: dlg.grab_set())
        dlg.wait_window()

        if result[0] is None:
            return

        tgt_id, src_ids = result[0]
        src_set = set(src_ids)
        tgt_name = names[tgt_id]
        src_names = [names.get(s, str(s)) for s in sorted(src_set)]

        final_tgt = tgt_id - sum(1 for s in src_set if s < tgt_id)

        def _transform(c):
            if c in src_set:
                return final_tgt
            shift = sum(1 for s in src_set if s < c)
            return c - shift

        n = self._rewrite_label_files(_transform)

        new_names = {}
        for old_id, nm in sorted(names.items()):
            if old_id in src_set:
                continue
            new_id = old_id - sum(1 for s in src_set if s < old_id)
            new_names[new_id] = nm
        cfg["names"] = new_names
        self._write_dataset_yaml(cfg)

        self.log(
            f"\nConsolidated [{', '.join(src_names)}] -> '{tgt_name}'"
            f" -- {n} annotations changed\n"
        )
        self._selected_label_id = None
        self._refresh_labels_view()

    def _on_rename_class(self):
        if self._selected_label_id is None:
            return
        cid = self._selected_label_id

        cfg = self._read_dataset_yaml()
        names = _parse_names(cfg.get("names", {}))
        old_name = names.get(cid, f"class_{cid}")

        dialog = ctk.CTkInputDialog(
            text=f"Rename '{old_name}' (id={cid}) to:",
            title="Rename Class",
        )
        new_name = dialog.get_input()
        if not new_name or not new_name.strip():
            return
        new_name = new_name.strip()

        names[cid] = new_name
        cfg["names"] = names
        self._write_dataset_yaml(cfg)

        self.log(f"\nRenamed class {cid}: '{old_name}' -> '{new_name}'\n")
        self._refresh_labels_view()

    # ==================================================================
    # Helpers
    # ==================================================================

    def _switch_subtab_safe(self, key):
        if threading.current_thread() is threading.main_thread():
            self._switch_subtab(key)
        else:
            self.after(0, lambda: self._switch_subtab(key))


# ======================================================================
# Module-level helpers
# ======================================================================


def _parse_names(raw):
    """Normalise dataset.yaml 'names' (list or dict) to {int: str}."""
    if isinstance(raw, list):
        return {i: v for i, v in enumerate(raw)}
    if isinstance(raw, dict):
        return {int(k): v for k, v in raw.items()}
    return {}


class _LogRedirector:
    """Redirect stdout/stderr to a callback, but only from the owning thread.

    ``sys.stdout`` is a process-global, so redirecting it in a worker thread
    also affects the main (GUI) thread.  If the main thread's textbox update
    produces any stdout/stderr output (e.g. CustomTkinter warnings), it feeds
    back through the redirector → ``log()`` → synchronous ``_do()`` → textbox
    insert → more stdout → infinite recursion.

    Fix: only route writes that originate from the thread that created this
    redirector.  All other threads write to the original stream.
    """

    def __init__(self, cb, original_stdout):
        self._cb = cb
        self._original = original_stdout
        self._owner = threading.current_thread().ident

    def write(self, text):
        if text:
            if threading.current_thread().ident == self._owner:
                self._cb(text)
            else:
                self._original.write(text)

    def flush(self):
        if hasattr(self._original, "flush"):
            self._original.flush()
