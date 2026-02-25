"""
Paper Engine - CustomTkinter GUI

Entry point for Paper Engine. Replaces the old CLI (main.py).

Usage:
    python gui.py
"""

import customtkinter as ctk
import threading
import subprocess
import sys
import os
import time
import shutil
import configparser
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from conf.config_parser import main_conf as config


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


def update_llm_config(provider_key):
    """Write the chosen LLM provider/model into conf/main_conf.ini."""
    mapping = {
        "google-genai": ("google", "gemini-2.5-flash"),
        "anthropic": ("anthropic", "claude-sonnet-4-5-20250514"),
        "openai": ("openai", "gpt-4"),
    }
    if provider_key not in mapping:
        return
    provider, model = mapping[provider_key]
    ini = PROJECT_ROOT / "conf" / "main_conf.ini"
    try:
        p = configparser.ConfigParser()
        p.read(ini)
        if not p.has_section("LLM"):
            p.add_section("LLM")
        p.set("LLM", "llm_provider", provider)
        p.set("LLM", "llm_model", model)
        if not p.has_option("LLM", "max_tokens_search"):
            p.set("LLM", "max_tokens_search", "4096")
        with open(ini, "w") as f:
            p.write(f)
    except Exception:
        pass


# ======================================================================
# Stdout/stderr -> GUI log redirect
# ======================================================================


class _LogRedirector:
    def __init__(self, callback):
        self._cb = callback

    def write(self, text):
        if text:
            self._cb(text)

    def flush(self):
        pass


# ======================================================================
# Main window
# ======================================================================


class PaperEngineGUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Paper Engine - Pre-Release v1.0")
        self.geometry("900x700")
        self.minsize(750, 550)
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self._game_process = None
        self._capturing = False
        self._window_geometry = None
        self._last_path_file = PROJECT_ROOT / ".last_game_path"

        self._build_ui()
        self._log("=" * 60 + "\n")
        self._log("          PAPER ENGINE - Pre-Release v1.0\n")
        self._log("=" * 60 + "\n")
        self._run_dependency_check()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # -- Sidebar --
        sb = ctk.CTkFrame(self, width=200, corner_radius=0)
        sb.grid(row=0, column=0, sticky="nsew")
        sb.grid_rowconfigure(10, weight=1)

        ctk.CTkLabel(
            sb, text="Paper Engine", font=ctk.CTkFont(size=20, weight="bold")
        ).grid(row=0, column=0, padx=20, pady=(20, 5))
        ctk.CTkLabel(sb, text="Pre-Release v1.0", font=ctk.CTkFont(size=12)).grid(
            row=1, column=0, padx=20, pady=(0, 20)
        )

        steps = [
            ("1. Launch Game", self._on_launch_game),
            ("   Stop Capture", self._on_stop_capture),
            ("2. Annotate", self._on_annotate),
            ("3. Train Model", self._on_train_model),
            ("4. Review Results", self._on_test_model),
            ("5. Generate Bot", self._on_generate_bot),
        ]
        for i, (text, cmd) in enumerate(steps):
            btn = ctk.CTkButton(sb, text=text, command=cmd, width=170)
            btn.grid(row=i + 2, column=0, padx=15, pady=6)
            if "Stop" in text:
                self._stop_btn = btn
                btn.configure(state="disabled", fg_color="gray")

        ctk.CTkButton(
            sb,
            text="Configuration",
            width=170,
            fg_color="grey30",
            command=self._on_configuration,
        ).grid(row=9, column=0, padx=15, pady=(10, 6))

        ctk.CTkLabel(sb, text="Theme:").grid(row=11, column=0, padx=20, pady=(10, 0))
        ctk.CTkOptionMenu(
            sb,
            values=["Dark", "Light", "System"],
            command=lambda v: ctk.set_appearance_mode(v.lower()),
        ).grid(row=12, column=0, padx=20, pady=(5, 20))

        # -- Main area --
        main = ctk.CTkFrame(self)
        main.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        main.grid_columnconfigure(0, weight=1)
        main.grid_rowconfigure(2, weight=1)

        # Row 0 : game path
        path_frame = ctk.CTkFrame(main)
        path_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 2))
        path_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(path_frame, text="Game path:").grid(
            row=0, column=0, padx=(10, 5), pady=8
        )
        self._game_path_var = ctk.StringVar(value=self._load_last_game_path())
        ctk.CTkEntry(path_frame, textvariable=self._game_path_var, height=34).grid(
            row=0, column=1, padx=5, pady=8, sticky="ew"
        )
        ctk.CTkButton(
            path_frame,
            text="Browse",
            width=80,
            height=34,
            command=self._browse_game_dir,
        ).grid(row=0, column=2, padx=(5, 10), pady=8)

        # Row 1 : status
        status = ctk.CTkFrame(main)
        status.grid(row=1, column=0, sticky="ew", padx=10, pady=(2, 2))
        self._status_var = ctk.StringVar(value="Idle")
        ctk.CTkLabel(
            status,
            textvariable=self._status_var,
            font=ctk.CTkFont(size=13, weight="bold"),
        ).grid(row=0, column=0, padx=10, pady=5)
        self._screenshot_count_var = ctk.StringVar(value="Screenshots: 0")
        ctk.CTkLabel(status, textvariable=self._screenshot_count_var).grid(
            row=0, column=1, padx=20, pady=5
        )

        # Row 2 : log
        self._log_textbox = ctk.CTkTextbox(main, wrap="word", state="disabled")
        self._log_textbox.grid(row=2, column=0, sticky="nsew", padx=10, pady=(2, 5))

        # Row 3 : toolbar
        bar = ctk.CTkFrame(main)
        bar.grid(row=3, column=0, sticky="ew", padx=10, pady=(0, 10))
        ctk.CTkButton(bar, text="Clear Log", width=100, command=self._clear_log).pack(
            side="right", padx=5, pady=5
        )

        # row 4 : bot replay footage

    # ------------------------------------------------------------------
    # Logging / status (thread-safe)
    # ------------------------------------------------------------------

    def _log(self, text: str):
        def _do():
            self._log_textbox.configure(state="normal")
            self._log_textbox.insert("end", text)
            self._log_textbox.see("end")
            self._log_textbox.configure(state="disabled")

        if threading.current_thread() is threading.main_thread():
            _do()
        else:
            self.after(0, _do)

    def _clear_log(self):
        self._log_textbox.configure(state="normal")
        self._log_textbox.delete("1.0", "end")
        self._log_textbox.configure(state="disabled")

    def _set_status(self, text: str):
        if threading.current_thread() is threading.main_thread():
            self._status_var.set(text)
        else:
            self.after(0, lambda: self._status_var.set(text))

    # ------------------------------------------------------------------
    # Dependency check (runs once at startup)
    # ------------------------------------------------------------------

    def _run_dependency_check(self):
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
            self._log(f"\n  Missing core packages: {', '.join(missing['core'])}\n")
            self._log("  Auto-installing...\n")
            for pkg in missing["core"]:
                if pip_install(pkg):
                    self._log(f"    Installed {pkg}\n")
                else:
                    self._log(f"    FAILED {pkg}\n")

        if missing["training"]:
            self._log(
                f"\n  Missing training packages (optional): {', '.join(missing['training'])}\n"
            )
            self._log("    pip install ultralytics torch\n")

        if installed_llm:
            self._log(f"\n  LLM provider: {', '.join(installed_llm)}\n")
        else:
            self._log("\n  No LLM provider installed (needed for bot generation).\n")
            self._log("    pip install google-genai / anthropic / openai\n")

        missing_tools = [t for t, (ok, _) in tools.items() if not ok]
        if missing_tools:
            ok_all = False
            self._log(f"\n  Missing system tools: {', '.join(missing_tools)}\n")

        if ok_all and not missing["training"]:
            self._log("\n  All dependencies OK.\n")

        self._log("\nReady.\n")

    # ------------------------------------------------------------------
    # Background thread runner
    # ------------------------------------------------------------------

    def _run_in_thread(self, fn, *args):
        def _worker():
            old_out, old_err = sys.stdout, sys.stderr
            redir = _LogRedirector(self._log)
            sys.stdout = redir
            sys.stderr = redir
            try:
                fn(*args)
            except Exception as exc:
                self._log(f"\nERROR: {exc}\n")
            finally:
                sys.stdout, sys.stderr = old_out, old_err

        threading.Thread(target=_worker, daemon=True).start()

    # ------------------------------------------------------------------
    # Game path persistence
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # 1. Launch Game + screenshot capture
    # ------------------------------------------------------------------

    def _on_launch_game(self):
        self._run_in_thread(self._launch_game_worker)

    def _launch_game_worker(self):
        from functions import path_finder
        from screencapture import (
            take_screenshot,
            create_screenshots_directory,
            find_wine_window,
        )

        game_path = self._game_path_var.get()
        self._save_last_game_path(game_path)
        self._log(f"\n--- Launching game from: {game_path} ---\n")
        self._set_status("Launching game...")

        exe_path = path_finder(game_path)
        if exe_path is None:
            self._log("ERROR: No executable found in game directory.\n")
            self._set_status("Error")
            return

        exe = Path(exe_path)
        self._log(f"Found executable: {exe}\n")

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
            self._log(f"Game started.\n")
        except FileNotFoundError as e:
            self._log(f"ERROR: {e}\n")
            self._set_status("Error")
            return

        wait = config.GAME_INITIALIZATION_WAIT
        self._log(f"Waiting {wait}s for game to initialise...\n")
        time.sleep(wait)

        self._window_geometry = find_wine_window()
        if self._window_geometry:
            self._log(f"Locked onto game window: {self._window_geometry}\n")
        else:
            self._log("Using full-screen capture mode.\n")

        screenshots_dir = create_screenshots_directory()
        self._capturing = True
        self._set_status("Capturing screenshots...")
        self.after(
            0,
            lambda: self._stop_btn.configure(
                state="normal", fg_color=("#DB3E39", "#C62828")
            ),
        )

        count = 0
        while self._capturing and (
            self._game_process is None or self._game_process.poll() is None
        ):
            time.sleep(config.SCREENSHOT_INTERVAL)
            take_screenshot(screenshots_dir, self._window_geometry)
            count += 1
            self.after(
                0, lambda c=count: self._screenshot_count_var.set(f"Screenshots: {c}")
            )

        self._capturing = False
        self.after(
            0, lambda: self._stop_btn.configure(state="disabled", fg_color="gray")
        )
        self._log(f"\nCapture finished. {count} screenshots taken.\n")
        self._set_status("Idle")

    def _on_stop_capture(self):
        self._capturing = False
        if self._game_process and self._game_process.poll() is None:
            self._game_process.terminate()
            self._log("Game process terminated.\n")
        self._set_status("Stopping...")

    # ------------------------------------------------------------------
    # 2. Annotate (in-house annotation tool)
    # ------------------------------------------------------------------

    def _on_annotate(self):
        from review_results import AnnotationWindow

        screenshots_dir = PROJECT_ROOT / "screenshots"
        if not screenshots_dir.exists() or not any(screenshots_dir.glob("*.png")):
            self._log("\nERROR: No screenshots found in screenshots/\n")
            self._log("Capture screenshots first using Launch Game.\n")
            return

        self._log("\n--- Opening Annotation Tool ---\n")
        self._set_status("Annotating...")
        AnnotationWindow(self, screenshots_dir=str(screenshots_dir))

    # ------------------------------------------------------------------
    # 3. Train Model
    # ------------------------------------------------------------------

    def _on_train_model(self):
        self._run_in_thread(self._train_model_worker)

    def _train_model_worker(self):
        self._log("\n--- Training YOLO Model ---\n")
        self._set_status("Training model...")
        import training_model

        training_model.main()
        self._log("\nTraining complete.\n")
        self._set_status("Idle")

    # ------------------------------------------------------------------
    # 4. Review Results (was Test Model)
    # ------------------------------------------------------------------

    def _on_test_model(self):
        self._run_in_thread(self._test_model_worker)

    def _test_model_worker(self):
        self._log("\n--- Running Model Inference ---\n")
        self._set_status("Running inference...")
        import test_model as tm

        results = tm.test_model()
        class_names = tm.get_class_names()

        if results is None:
            self._log("\nInference failed or returned no results.\n")
            self._set_status("Error")
            return

        self._log(f"\nInference complete. Opening Review Results...\n")
        self._set_status("Idle")

        # Open the review window on the main thread
        self.after(0, lambda: self._open_review_window(results, class_names))

    def _open_review_window(self, results, class_names):
        from review_results import ReviewResultsWindow

        ReviewResultsWindow(self, results, class_names=class_names)

    # ------------------------------------------------------------------
    # 5. Generate Bot Script
    # ------------------------------------------------------------------

    def _on_generate_bot(self):
        self._run_in_thread(self._generate_bot_worker)

    def _generate_bot_worker(self):
        self._log("\n--- Generating Bot Script ---\n")
        self._set_status("Generating bot...")

        from generate_bot_script import (
            generate_bot_script,
            save_bot_script,
            search_game_controls,
            save_controls_to_config,
            read_controls_from_config,
        )
        from functions import get_title, path_finder

        exe_path = path_finder(self._game_path_var.get())
        if not exe_path:
            self._log("ERROR: No game executable found.\n")
            self._set_status("Error")
            return

        game_title = get_title(exe_path)
        self._log(f"Game: {game_title}\n")

        existing = read_controls_from_config(game_title)
        if existing:
            self._log("Found existing controls, improving...\n")
            controls = search_game_controls(game_title, existing)
        else:
            self._log("Searching for controls...\n")
            controls = search_game_controls(game_title)

        if not controls:
            self._log("ERROR: Could not retrieve controls.\n")
            self._set_status("Error")
            return

        save_controls_to_config(game_title, exe_path, controls)
        self._log("Controls saved.\n")

        self._log("Generating bot script (this may take 1-2 minutes)...\n")
        code = generate_bot_script(game_title, controls)
        if code:
            path = save_bot_script(game_title, code)
            self._log(f"\nBot script saved to: {path}\n")
        else:
            self._log("ERROR: Bot script generation failed.\n")

        self._set_status("Idle")

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def _on_configuration(self):
        dialog = ConfigDialog(self)
        result = dialog.wait_for_result()
        if result:
            self._log(f"\nConfiguration updated: {result}\n")


# ======================================================================
# Configuration Dialog (LLM + Training packages)
# ======================================================================

_LLM_PROVIDERS = [
    {
        "id": "google",
        "label": "Google Gemini (Free tier)",
        "pip": "google-genai",
        "import": "google.genai",
        "key_url": "https://aistudio.google.com/apikey",
    },
    {
        "id": "anthropic",
        "label": "Anthropic Claude (Paid)",
        "pip": "anthropic",
        "import": "anthropic",
        "key_url": "https://console.anthropic.com/settings/keys",
    },
    {
        "id": "openai",
        "label": "OpenAI GPT (Paid)",
        "pip": "openai",
        "import": "openai",
        "key_url": "https://platform.openai.com/api-keys",
    },
]

_TRAINING_PACKAGES = [
    {"label": "Ultralytics (YOLO)", "pip": "ultralytics", "import": "ultralytics"},
    {"label": "PyTorch", "pip": "torch", "import": "torch"},
]


class ConfigDialog(ctk.CTkToplevel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.title("Paper Engine - Configuration")
        self.geometry("560x620")
        self.resizable(False, False)
        self.transient(parent)
        self.result = None
        self._llm_labels = {}
        self._training_vars = {}
        self._training_labels = {}
        self._build()
        self._detect_state()
        self.after(100, self._try_grab)

    def _try_grab(self):
        try:
            self.grab_set()
        except Exception:
            self.after(100, self._try_grab)

    def _build(self):
        # -- LLM section --
        ctk.CTkLabel(
            self, text="Configuration", font=ctk.CTkFont(size=22, weight="bold")
        ).pack(pady=(20, 2))

        ctk.CTkLabel(
            self,
            text="LLM Provider  (for bot generation)",
            font=ctk.CTkFont(size=14, weight="bold"),
            anchor="w",
        ).pack(fill="x", padx=30, pady=(16, 4))
        ctk.CTkLabel(
            self,
            text="You only need one. Google Gemini has a free tier.",
            font=ctk.CTkFont(size=12),
            text_color="gray",
            anchor="w",
        ).pack(fill="x", padx=30, pady=(0, 8))

        self._llm_var = ctk.StringVar(value="google")
        for prov in _LLM_PROVIDERS:
            row = ctk.CTkFrame(self, fg_color="transparent")
            row.pack(fill="x", padx=30, pady=1)
            ctk.CTkRadioButton(
                row,
                text=prov["label"],
                variable=self._llm_var,
                value=prov["id"],
                font=ctk.CTkFont(size=13),
            ).pack(side="left")
            lbl = ctk.CTkLabel(row, text="", font=ctk.CTkFont(size=11))
            lbl.pack(side="right", padx=8)
            self._llm_labels[prov["id"]] = lbl

        # API key
        ctk.CTkLabel(self, text="API Key:", font=ctk.CTkFont(size=13), anchor="w").pack(
            fill="x", padx=30, pady=(12, 2)
        )
        self._api_key_var = ctk.StringVar(value=os.environ.get("API_KEY", ""))
        ctk.CTkEntry(
            self,
            textvariable=self._api_key_var,
            height=34,
            placeholder_text="Paste your API key here",
            show="*",
        ).pack(fill="x", padx=30, pady=(0, 2))
        self._key_hint = ctk.CTkLabel(
            self, text="", font=ctk.CTkFont(size=11), text_color="gray", anchor="w"
        )
        self._key_hint.pack(fill="x", padx=30)
        self._llm_var.trace_add("write", lambda *_: self._update_hint())

        # -- Separator --
        ctk.CTkFrame(self, height=2, fg_color="gray40").pack(
            fill="x", padx=30, pady=(14, 10)
        )

        # -- Training section --
        ctk.CTkLabel(
            self,
            text="Training Packages  (for local YOLO training)",
            font=ctk.CTkFont(size=14, weight="bold"),
            anchor="w",
        ).pack(fill="x", padx=30, pady=(0, 4))
        ctk.CTkLabel(
            self,
            text="~6.5 GB total. Only needed if you want to train models locally.",
            font=ctk.CTkFont(size=12),
            text_color="gray",
            anchor="w",
        ).pack(fill="x", padx=30, pady=(0, 8))

        for pkg in _TRAINING_PACKAGES:
            row = ctk.CTkFrame(self, fg_color="transparent")
            row.pack(fill="x", padx=30, pady=1)
            var = ctk.BooleanVar(value=False)
            ctk.CTkCheckBox(
                row, text=pkg["label"], variable=var, font=ctk.CTkFont(size=13)
            ).pack(side="left")
            lbl = ctk.CTkLabel(row, text="", font=ctk.CTkFont(size=11))
            lbl.pack(side="right", padx=8)
            self._training_vars[pkg["pip"]] = var
            self._training_labels[pkg["pip"]] = lbl

        # -- Buttons --
        btns = ctk.CTkFrame(self, fg_color="transparent")
        btns.pack(fill="x", padx=30, pady=(20, 14))
        ctk.CTkButton(
            btns, text="Save & Install", width=180, height=40, command=self._on_confirm
        ).pack(side="left", padx=(0, 10))
        ctk.CTkButton(
            btns,
            text="Skip",
            width=100,
            height=40,
            fg_color="gray",
            command=self._on_skip,
        ).pack(side="right")

        self._msg = ctk.CTkLabel(
            self, text="", font=ctk.CTkFont(size=12), text_color="orange"
        )
        self._msg.pack(pady=(0, 10))

    # -- state detection --

    def _detect_state(self):
        current_llm = getattr(config, "LLM_PROVIDER", "google")
        for prov in _LLM_PROVIDERS:
            lbl = self._llm_labels[prov["id"]]
            try:
                __import__(prov["import"])
                lbl.configure(text="installed", text_color="green")
                prov["_ok"] = True
            except ImportError:
                lbl.configure(text="not installed", text_color="gray")
                prov["_ok"] = False
            if prov["id"] == current_llm:
                self._llm_var.set(prov["id"])

        for pkg in _TRAINING_PACKAGES:
            lbl = self._training_labels[pkg["pip"]]
            try:
                __import__(pkg["import"])
                lbl.configure(text="installed", text_color="green")
                self._training_vars[pkg["pip"]].set(True)
                pkg["_ok"] = True
            except ImportError:
                lbl.configure(text="not installed", text_color="gray")
                pkg["_ok"] = False

        self._update_hint()

    def _update_hint(self):
        sel = self._llm_var.get()
        for prov in _LLM_PROVIDERS:
            if prov["id"] == sel:
                self._key_hint.configure(text=f"Get key: {prov['key_url']}")
                return

    # -- actions --

    def _set_msg(self, text):
        self._msg.configure(text=text)
        self.update()

    def _on_confirm(self):
        # -- Install LLM provider if needed --
        prov = next(p for p in _LLM_PROVIDERS if p["id"] == self._llm_var.get())
        if not prov.get("_ok"):
            self._set_msg(f"Installing {prov['pip']}...")
            if not pip_install(prov["pip"]):
                self._set_msg(f"Failed to install {prov['pip']}")
                return

        # -- Install training packages if checked and not already installed --
        for pkg in _TRAINING_PACKAGES:
            wanted = self._training_vars[pkg["pip"]].get()
            already = pkg.get("_ok", False)
            if wanted and not already:
                self._set_msg(f"Installing {pkg['pip']} (this may take a while)...")
                if not pip_install(pkg["pip"]):
                    self._set_msg(f"Failed to install {pkg['pip']}")
                    return

        # -- Save API key --
        api_key = self._api_key_var.get().strip()
        if api_key:
            env_path = Path(".env")
            lines = env_path.read_text().splitlines(True) if env_path.exists() else []
            replaced = False
            for i, line in enumerate(lines):
                if line.strip().startswith("API_KEY="):
                    lines[i] = f"API_KEY={api_key}\n"
                    replaced = True
                    break
            if not replaced:
                lines.append(f"API_KEY={api_key}\n")
            env_path.write_text("".join(lines))
            os.environ["API_KEY"] = api_key

        # -- Save LLM config --
        update_llm_config(prov["pip"])

        self.result = prov["id"]
        self.grab_release()
        self.destroy()

    def _on_skip(self):
        self.grab_release()
        self.destroy()

    def wait_for_result(self):
        self.wait_window()
        return self.result


# ======================================================================
# Entry point
# ======================================================================


def _is_installed(import_name):
    try:
        __import__(import_name)
        return True
    except ImportError:
        return False


def main():
    # Show config dialog at startup if no LLM provider is installed
    needs_setup = not any(_is_installed(p["import"]) for p in _LLM_PROVIDERS)
    if needs_setup:
        root = ctk.CTk()
        root.withdraw()
        ConfigDialog(root).wait_for_result()
        root.destroy()

    PaperEngineGUI().mainloop()


if __name__ == "__main__":
    main()
