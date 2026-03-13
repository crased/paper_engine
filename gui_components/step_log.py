"""Step Logger widget -- circle indicators for operation progress.

Usage:
    log = StepLog(parent)
    idx = log.add("Starting training")
    log.update(idx, "Training epoch 5/100")
    log.complete(idx, "Training done")
    log.fail(idx, "Training failed")
    log.clear()
"""

import threading
import customtkinter as ctk

from gui_components.theme import (
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    TEXT_MUTED,
    DANGER,
    SUCCESS,
    ICON_DOT,
    ICON_DOT_EMPTY,
    SP_SM,
    SP_MD,
    font_body,
    font_small,
)


class StepLog(ctk.CTkFrame):
    """Scrollable list of operation steps with circle indicators."""

    def __init__(self, master, **kw):
        super().__init__(master, fg_color="transparent", **kw)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self._scroll = ctk.CTkScrollableFrame(self, fg_color="transparent")
        self._scroll.grid(row=0, column=0, sticky="nsew")
        self._scroll.grid_columnconfigure(1, weight=1)
        self._steps: list[dict] = []

    # -- public API --

    def add(self, text: str) -> int:
        """Add a pending step. Returns its index."""
        idx = len(self._steps)
        _safe(self, lambda: self._build_row(idx, text))
        return idx

    def complete(self, idx: int, text: str | None = None):
        """Mark step as done (filled circle)."""
        _safe(self, lambda: self._set(idx, "done", text))

    def update(self, idx: int, text: str):
        """Update text without changing status."""
        _safe(self, lambda: self._set_text(idx, text))

    def fail(self, idx: int, text: str | None = None):
        """Mark step as failed (red circle)."""
        _safe(self, lambda: self._set(idx, "failed", text))

    def clear(self):
        """Remove all steps."""
        _safe(self, self._do_clear)

    # -- internals --

    def _build_row(self, idx: int, text: str):
        row = ctk.CTkFrame(self._scroll, fg_color="transparent", height=30)
        row.grid(row=idx, column=0, sticky="ew", padx=SP_SM, pady=1)
        row.grid_columnconfigure(1, weight=1)

        circle = ctk.CTkLabel(
            row,
            text=ICON_DOT_EMPTY,
            font=ctk.CTkFont(size=12),
            text_color=TEXT_MUTED,
            width=18,
        )
        circle.grid(row=0, column=0, padx=(SP_SM, SP_SM), sticky="w")

        label = ctk.CTkLabel(
            row,
            text=text,
            font=font_small(),
            text_color=TEXT_MUTED,
            anchor="w",
        )
        label.grid(row=0, column=1, sticky="ew")

        self._steps.append(
            {
                "frame": row,
                "circle": circle,
                "label": label,
                "status": "pending",
                "text": text,
            }
        )
        self._autoscroll()

    def _set(self, idx: int, status: str, text: str | None):
        if idx >= len(self._steps):
            return
        s = self._steps[idx]
        s["status"] = status
        if status == "done":
            s["circle"].configure(text=ICON_DOT, text_color=SUCCESS)
            s["label"].configure(text_color=TEXT_PRIMARY)
        elif status == "failed":
            s["circle"].configure(text=ICON_DOT, text_color=DANGER)
            s["label"].configure(text_color=DANGER)
        if text is not None:
            s["text"] = text
            s["label"].configure(text=text)

    def _set_text(self, idx: int, text: str):
        if idx >= len(self._steps):
            return
        s = self._steps[idx]
        s["text"] = text
        s["label"].configure(text=text)

    def _do_clear(self):
        for s in self._steps:
            s["frame"].destroy()
        self._steps.clear()

    def _autoscroll(self):
        try:
            self._scroll._parent_canvas.yview_moveto(1.0)
        except Exception:
            pass


def _safe(widget, fn):
    """Run fn on the main thread."""
    if threading.current_thread() is threading.main_thread():
        fn()
    else:
        widget.after(0, fn)
