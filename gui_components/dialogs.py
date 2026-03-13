"""Shared dialog helpers for the Paper Engine GUI."""

import customtkinter as ctk

from gui_components.theme import (
    BG_SURFACE,
    BORDER_SUBTLE,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    CLR_RED,
    DANGER,
    RADIUS_LG,
    RADIUS_MD,
    SP_SM,
    SP_LG,
    SP_XL,
    BTN_HEIGHT_MD,
    font_subheading,
    font_body,
    font_small,
)


def confirm_dialog(parent, title: str, message: str, detail: str = "") -> bool:
    """Show a yes/no confirmation dialog. Returns True if confirmed."""
    result = [False]

    dlg = ctk.CTkToplevel(parent)
    dlg.title(title)
    dlg.geometry("440x200")
    dlg.resizable(False, False)
    dlg.transient(parent)

    ctk.CTkLabel(
        dlg,
        text=message,
        font=font_subheading(),
        text_color=TEXT_PRIMARY,
    ).pack(padx=SP_XL, pady=(SP_XL, SP_SM))

    if detail:
        ctk.CTkLabel(
            dlg,
            text=detail,
            font=font_small(),
            text_color=TEXT_SECONDARY,
            wraplength=400,
        ).pack(padx=SP_XL, pady=(0, SP_LG))

    bf = ctk.CTkFrame(dlg, fg_color="transparent")
    bf.pack(pady=(0, SP_XL))

    def _yes():
        result[0] = True
        dlg.destroy()

    ctk.CTkButton(
        bf,
        text="Confirm",
        width=110,
        height=BTN_HEIGHT_MD,
        corner_radius=RADIUS_MD,
        fg_color=CLR_RED,
        command=_yes,
    ).pack(side="left", padx=SP_SM)
    ctk.CTkButton(
        bf,
        text="Cancel",
        width=110,
        height=BTN_HEIGHT_MD,
        corner_radius=RADIUS_MD,
        fg_color="gray",
        command=dlg.destroy,
    ).pack(side="left", padx=SP_SM)

    dlg.after(100, lambda: dlg.grab_set())
    dlg.wait_window()
    return result[0]


def input_dialog(parent, title: str, prompt: str) -> str | None:
    """Show a text input dialog. Returns the string or None if cancelled."""
    dlg = ctk.CTkInputDialog(text=prompt, title=title)
    value = dlg.get_input()
    if value and value.strip():
        return value.strip()
    return None
