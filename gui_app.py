"""Paper Engine -- Main application shell.

Provides the window frame, sidebar navigation, page routing,
and shared services (logging, status, step log, threading).
"""

import sys
import shutil
import subprocess
import threading
import customtkinter as ctk
from pathlib import Path

from gui_components.theme import (
    BG_BASE,
    BG_SURFACE,
    SIDEBAR_BG,
    SIDEBAR_ACTIVE_BG,
    SIDEBAR_HOVER_BG,
    SIDEBAR_TEXT,
    SIDEBAR_TEXT_ACTIVE,
    SIDEBAR_BRAND,
    SIDEBAR_VERSION,
    BORDER_SUBTLE,
    TEXT_MUTED,
    SIDEBAR_WIDTH,
    RADIUS_MD,
    ACCENT_BAR_WIDTH,
    SP_SM,
    SP_MD,
    SP_LG,
    SP_XL,
    SP_2XL,
    NAV_ITEMS,
    font_brand,
    font_small,
    font_nav,
    font_nav_active,
)

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


# ======================================================================
# Dependency helpers
# ======================================================================

_ALL_PACKAGES = {
    # import_name -> (pip_name, category)
    "pynput": ("pynput", "core"),
    "PIL": ("Pillow", "core"),
    "mss": ("mss", "core"),
    "dotenv": ("python-dotenv", "core"),
    "yaml": ("pyyaml", "core"),
    "ultralytics": ("ultralytics", "training"),
    "torch": ("torch", "training"),
    "anthropic": ("anthropic", "llm"),
    "openai": ("openai", "llm"),
    "google.genai": ("google-genai", "llm"),
}


def check_packages():
    """Return {pip_name: (installed, category)} for every tracked package."""
    results = {}
    for import_name, (pip_name, cat) in _ALL_PACKAGES.items():
        try:
            __import__(import_name)
            results[pip_name] = (True, cat)
        except ImportError:
            results[pip_name] = (False, cat)
    return results


def check_system_tools():
    """Return {tool: (found, detail)} for platform-required tools."""
    tools = []
    if sys.platform != "win32":
        tools += ["wine", "flameshot"]
    return {
        t: (bool(shutil.which(t)), shutil.which(t) or f"{t} not found") for t in tools
    }


def pip_install(package):
    """Install *package* via pip. Returns True on success."""
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", package],
            capture_output=True,
            text=True,
            check=True,
        )
        return True
    except subprocess.CalledProcessError:
        return False


# ======================================================================
# Stdout/stderr redirect
# ======================================================================


class _LogRedirector:
    def __init__(self, cb):
        self._cb = cb

    def write(self, text):
        if text:
            self._cb(text)

    def flush(self):
        pass


# ======================================================================
# Main Application
# ======================================================================


class PaperEngineApp(ctk.CTk):
    """Root window -- sidebar nav + swappable page area."""

    def __init__(self):
        super().__init__()
        self.title("Paper Engine")
        self.geometry("1040x720")
        self.minsize(860, 560)
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self._active_page = None
        self._pages = {}
        self._nav_btns = {}

        self._build_shell()
        self._register_pages()
        self.navigate("home")

    # ------------------------------------------------------------------
    # Shell: sidebar + content area
    # ------------------------------------------------------------------

    def _build_shell(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # -- Sidebar (pack layout for tight vertical stacking) --
        sb = ctk.CTkFrame(
            self,
            width=SIDEBAR_WIDTH,
            corner_radius=0,
            fg_color=SIDEBAR_BG,
            border_width=0,
        )
        sb.grid(row=0, column=0, sticky="nsew")
        sb.grid_propagate(False)

        # Brand area
        ctk.CTkLabel(
            sb,
            text="Paper Engine",
            font=font_brand(),
            text_color=SIDEBAR_BRAND,
            anchor="w",
        ).pack(padx=SP_XL, pady=(SP_LG, 0), anchor="w")

        ctk.CTkLabel(
            sb,
            text="v1.0",
            font=font_small(),
            text_color=SIDEBAR_VERSION,
            anchor="w",
        ).pack(padx=SP_XL, pady=(0, SP_SM), anchor="w")

        # Subtle divider
        ctk.CTkFrame(
            sb,
            height=1,
            corner_radius=0,
            fg_color=BORDER_SUBTLE,
        ).pack(fill="x", padx=SP_LG, pady=(0, SP_SM))

        # Nav buttons -- packed directly, no wrapper frames
        nav_group = ctk.CTkFrame(sb, fg_color="transparent")
        nav_group.pack(fill="x", anchor="n")
        nav_group.pack_propagate(True)

        self._nav_accents = {}
        for key, label, icon, accent_color in NAV_ITEMS[:-1]:  # Settings is at bottom
            btn = ctk.CTkButton(
                nav_group,
                text=f" {icon}  {label}",
                height=32,
                corner_radius=RADIUS_MD,
                anchor="w",
                font=font_nav(),
                fg_color="transparent",
                text_color=SIDEBAR_TEXT,
                hover_color=SIDEBAR_HOVER_BG,
                border_width=0,
                command=lambda k=key: self.navigate(k),
            )
            btn.pack(fill="x", padx=SP_SM, pady=1)
            self._nav_btns[key] = btn
            # Store accent color for active state highlight
            self._nav_accents[key] = (btn, accent_color)

        # Spacer pushes bottom section down
        spacer = ctk.CTkFrame(sb, fg_color="transparent", height=0)
        spacer.pack(fill="both", expand=True)

        # Bottom section: divider + Settings + Appearance
        ctk.CTkFrame(
            sb,
            height=1,
            corner_radius=0,
            fg_color=BORDER_SUBTLE,
        ).pack(fill="x", padx=SP_LG, pady=(0, SP_SM))

        # Settings button pinned to bottom
        settings_btn = ctk.CTkButton(
            sb,
            text=f" {NAV_ITEMS[-1][2]}  {NAV_ITEMS[-1][1]}",
            height=32,
            corner_radius=RADIUS_MD,
            anchor="w",
            font=font_nav(),
            fg_color="transparent",
            text_color=SIDEBAR_TEXT,
            hover_color=SIDEBAR_HOVER_BG,
            border_width=0,
            command=lambda: self.navigate("settings"),
        )
        settings_btn.pack(fill="x", padx=SP_SM, pady=1)
        self._nav_btns["settings"] = settings_btn
        self._nav_accents["settings"] = (settings_btn, NAV_ITEMS[-1][3])

        ctk.CTkLabel(
            sb,
            text="Appearance",
            font=font_small(),
            text_color=TEXT_MUTED,
            anchor="w",
        ).pack(padx=SP_XL, pady=(SP_SM, 2), anchor="w")

        ctk.CTkOptionMenu(
            sb,
            values=["Dark", "Light", "System"],
            width=SIDEBAR_WIDTH - 48,
            height=28,
            corner_radius=RADIUS_MD,
            font=font_small(),
            command=lambda v: ctk.set_appearance_mode(v.lower()),
        ).pack(padx=SP_XL, pady=(0, SP_LG))

        # -- Content area --
        self._content = ctk.CTkFrame(self, fg_color=BG_BASE, corner_radius=0)
        self._content.grid(row=0, column=1, sticky="nsew")
        self._content.grid_columnconfigure(0, weight=1)
        self._content.grid_rowconfigure(0, weight=1)

    # ------------------------------------------------------------------
    # Page registration & navigation
    # ------------------------------------------------------------------

    def _register_pages(self):
        from gui_pages.home import HomePage
        from gui_pages.dashboard import DashboardPage
        from gui_pages.session import SessionPage
        from gui_pages.tools import ToolsPage
        from gui_pages.settings import SettingsPage

        for key, cls in [
            ("home", HomePage),
            ("dashboard", DashboardPage),
            ("session", SessionPage),
            ("tools", ToolsPage),
            ("settings", SettingsPage),
        ]:
            page = cls(self._content, self)
            page.grid(row=0, column=0, sticky="nsew")
            page.grid_remove()  # start hidden
            self._pages[key] = page

    def navigate(self, key: str):
        """Switch to the named page."""   
        if self._active_page == key:
            return
        # Hide current
        if self._active_page and self._active_page in self._pages:
            self._pages[self._active_page].grid_remove()
            if hasattr(self._pages[self._active_page], "on_hide"):
                self._pages[self._active_page].on_hide()
        # Show new
        self._active_page = key
        if key in self._pages:
            self._pages[key].grid(row=0, column=0, sticky="nsew")
            if hasattr(self._pages[key], "on_show"):
                self._pages[key].on_show()
        # Highlight active nav button
        for k, btn in self._nav_btns.items():
            _btn, acc_color = self._nav_accents[k]
            if k == key:
                btn.configure(
                    fg_color=SIDEBAR_ACTIVE_BG,
                    text_color=SIDEBAR_TEXT_ACTIVE,
                    font=font_nav_active(),
                    border_width=0,
                )
            else:
                btn.configure(
                    fg_color="transparent",
                    text_color=SIDEBAR_TEXT,
                    font=font_nav(),
                    border_width=0,
                )

    # ------------------------------------------------------------------
    # Shared services (used by pages)
    # ------------------------------------------------------------------

    def run_in_thread(self, fn, *args):
        """Run fn in a daemon thread with stdout redirected to log."""

        def _worker():
            old_out, old_err = sys.stdout, sys.stderr
            redir = _LogRedirector(self.log)
            sys.stdout = redir
            sys.stderr = redir
            try:
                fn(*args)
            except Exception as exc:
                self.log(f"\nERROR: {exc}\n")
            finally:
                sys.stdout, sys.stderr = old_out, old_err

        threading.Thread(target=_worker, daemon=True).start()

    def log(self, text: str):
        """Append to the tools page raw log (if it exists)."""
        tools = self._pages.get("tools")
        if tools and hasattr(tools, "log"):
            tools.log(text)

    def set_status(self, text: str):
        """Update the status label on the tools page."""
        tools = self._pages.get("tools")
        if tools and hasattr(tools, "set_status"):
            tools.set_status(text)

    @property
    def step_log(self):
        """Access the step log widget on the tools page."""
        tools = self._pages.get("tools")
        if tools and hasattr(tools, "step_log"):
            return tools.step_log
        return None

    def _run_dependency_check(self):
        """Run at startup -- check packages + system tools, log results."""
        sl = self.step_log
        if sl is None:
            return
        s_dep = sl.add("Checking dependencies")

        pkgs = check_packages()
        tools = check_system_tools()

        missing = {cat: [] for cat in ("core", "training", "llm")}
        installed_llm = []
        for name, (ok, cat) in pkgs.items():
            if not ok:
                missing[cat].append(name)
            elif cat == "llm":
                installed_llm.append(name)

        ok_all = True

        if missing["core"]:
            ok_all = False
            self.log(f"\n  Missing core packages: {', '.join(missing['core'])}\n")
            for pkg in missing["core"]:
                s_pkg = sl.add(f"Installing {pkg}")
                if pip_install(pkg):
                    sl.complete(s_pkg, f"Installed {pkg}")
                else:
                    sl.fail(s_pkg, f"Failed to install {pkg}")

        if missing["training"]:
            sl.add("Training packages not installed (optional)")
            self.log(
                f"\n  Missing training packages (optional): {', '.join(missing['training'])}\n"
            )

        if installed_llm:
            sl.complete(sl.add(f"LLM provider: {', '.join(installed_llm)}"))
        else:
            sl.add("No LLM provider installed")

        missing_tools = [t for t, (ok, _) in tools.items() if not ok]
        if missing_tools:
            ok_all = False
            sl.fail(sl.add(f"Missing tools: {', '.join(missing_tools)}"))

        if ok_all and not missing["training"]:
            sl.complete(s_dep, "All dependencies OK")
        else:
            sl.complete(s_dep, "Dependency check done (warnings above)")

        sl.complete(sl.add("Ready"))


# ======================================================================
# First-run setup
# ======================================================================

_LLM_IMPORTS = [
    ("google.genai", "google-genai"),
    ("anthropic", "anthropic"),
    ("openai", "openai"),
]


def _is_installed(import_name):
    try:
        __import__(import_name)
        return True
    except ImportError:
        return False


def _needs_first_run_setup():
    """Return True if no LLM provider is installed (first run)."""
    return not any(_is_installed(imp) for imp, _ in _LLM_IMPORTS)


def _run_first_run_setup():
    """Show ConfigDialog (from settings page) as a first-run wizard."""
    from gui_pages.settings import _LLM_PROVIDERS, _pip_install, _update_llm_config
    from gui_components.theme import (
        BG_SURFACE,
        TEXT_PRIMARY,
        TEXT_SECONDARY,
        ACCENT,
        RADIUS_LG,
        SP_LG,
        SP_XL,
        font_heading,
        font_body,
        font_small,
    )

    root = ctk.CTk()
    root.withdraw()

    dlg = ctk.CTkToplevel(root)
    dlg.title("Paper Engine - First Run Setup")
    dlg.geometry("520x420")
    dlg.resizable(False, False)

    ctk.CTkLabel(
        dlg,
        text="Welcome to Paper Engine",
        font=font_heading(),
        text_color=TEXT_PRIMARY,
    ).pack(padx=SP_XL, pady=(SP_XL, SP_SM))
    ctk.CTkLabel(
        dlg,
        text="Select an LLM provider to get started.\nGoogle Gemini has a free tier.",
        font=font_body(),
        text_color=TEXT_SECONDARY,
    ).pack(padx=SP_XL, pady=(0, SP_LG))

    llm_var = ctk.StringVar(value="google")
    for prov in _LLM_PROVIDERS:
        ctk.CTkRadioButton(
            dlg,
            text=prov["label"],
            variable=llm_var,
            value=prov["id"],
            font=font_body(),
        ).pack(anchor="w", padx=40, pady=3)

    api_var = ctk.StringVar()
    ctk.CTkLabel(
        dlg,
        text="API Key:",
        font=font_body(),
        anchor="w",
    ).pack(fill="x", padx=40, pady=(SP_LG, 2))
    ctk.CTkEntry(
        dlg,
        textvariable=api_var,
        height=36,
        placeholder_text="Paste your API key here",
        show="*",
    ).pack(fill="x", padx=40, pady=(0, SP_LG))

    msg = ctk.CTkLabel(dlg, text="", font=font_small(), text_color="orange")
    msg.pack(pady=(0, SP_SM))

    def _on_setup():
        prov = next(p for p in _LLM_PROVIDERS if p["id"] == llm_var.get())
        msg.configure(text=f"Installing {prov['pip']}...")
        dlg.update()
        if not _pip_install(prov["pip"]):
            msg.configure(text=f"Failed to install {prov['pip']}")
            return
        _update_llm_config(prov["pip"])
        key = api_var.get().strip()
        if key:
            try:
                from tools.functions import store_api_key

                store_api_key(key)
            except Exception:
                pass
        dlg.destroy()

    def _on_skip():
        dlg.destroy()

    bf = ctk.CTkFrame(dlg, fg_color="transparent")
    bf.pack(pady=(0, SP_LG))
    ctk.CTkButton(
        bf,
        text="Install & Continue",
        width=180,
        height=38,
        corner_radius=RADIUS_LG,
        command=_on_setup,
    ).pack(side="left", padx=SP_SM)
    ctk.CTkButton(
        bf,
        text="Skip",
        width=100,
        height=38,
        corner_radius=RADIUS_LG,
        fg_color="gray",
        command=_on_skip,
    ).pack(side="left", padx=SP_SM)

    dlg.wait_window()
    root.destroy()


# ======================================================================
# Entry
# ======================================================================


def main():
    if _needs_first_run_setup():
        _run_first_run_setup()

    app = PaperEngineApp()
    # Run dependency check after the event loop starts
    app.after(200, lambda: app.run_in_thread(app._run_dependency_check))
    app.mainloop()


if __name__ == "__main__":
    main()
