"""
Paper Engine GUI - CustomTkinter Interface

Provides a graphical interface for the Paper Engine workflow:
  1. Launch game + capture screenshots
  2. Launch Label Studio for annotation
  3. Train YOLO model
  4. Test trained model
  5. Generate AI bot script

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

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent))

from conf.config_parser import main_conf as config


# ======================================================================
# Dependency checking (moved from main.py)
# ======================================================================


def check_python_packages():
    """Check if required Python packages are installed.

    Returns:
        dict: {package_name: (is_installed: bool, error_msg: str|None)}
    """
    results = {}

    core_packages = {
        "pynput": "pynput",
        "PIL": "Pillow",
        "mss": "mss",
        "dotenv": "python-dotenv",
        "yaml": "pyyaml",
    }

    training_packages = {
        "ultralytics": "ultralytics",
        "torch": "torch",
    }

    llm_packages = {
        "anthropic": "anthropic",
        "openai": "openai",
        "google.genai": "google-genai",
    }

    for import_name, pip_name in core_packages.items():
        try:
            __import__(import_name)
            results[pip_name] = (True, None)
        except ImportError as e:
            results[pip_name] = (False, str(e))

    for import_name, pip_name in training_packages.items():
        try:
            __import__(import_name)
            results[pip_name] = (True, None)
        except ImportError as e:
            results[pip_name] = (False, str(e))

    for import_name, pip_name in llm_packages.items():
        try:
            __import__(import_name)
            results[pip_name] = (True, None)
        except ImportError as e:
            results[pip_name] = (False, str(e))

    return results


def check_system_tools():
    """Check if required system tools are available in PATH.

    Returns:
        dict: {tool_name: (is_found: bool, path_or_error: str)}
    """
    results = {}

    if sys.platform == "darwin":
        tools = ["wine", "flameshot", "label-studio"]
    elif sys.platform == "win32":
        tools = ["label-studio"]
    else:
        tools = ["wine", "flameshot", "label-studio"]

    for tool in tools:
        path = shutil.which(tool)
        if path:
            results[tool] = (True, path)
        else:
            results[tool] = (False, f"{tool} not found in PATH")

    return results


def install_python_package(package):
    """Install a Python package using pip.

    Args:
        package: Package name to install

    Returns:
        bool: True if installation succeeded, False otherwise
    """
    print(f"  Installing {package}...")
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", package],
            capture_output=True,
            text=True,
            check=True,
        )
        print(f"  Installed {package}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  Failed to install {package}: {e}")
        return False


def update_llm_config(provider):
    """Update conf/main_conf.ini with chosen LLM provider and default model."""
    config_file = Path(__file__).parent / "conf" / "main_conf.ini"

    provider_map = {
        "anthropic": {"provider": "anthropic", "model": "claude-sonnet-4-5-20250514"},
        "openai": {"provider": "openai", "model": "gpt-4"},
        "google-genai": {"provider": "google", "model": "gemini-2.5-flash"},
    }

    if provider not in provider_map:
        return

    try:
        parser = configparser.ConfigParser()
        parser.read(config_file)

        if not parser.has_section("LLM"):
            parser.add_section("LLM")

        parser.set("LLM", "llm_provider", provider_map[provider]["provider"])
        parser.set("LLM", "llm_model", provider_map[provider]["model"])

        if not parser.has_option("LLM", "max_tokens_search"):
            parser.set("LLM", "max_tokens_search", "4096")

        with open(config_file, "w") as f:
            parser.write(f)
    except Exception:
        pass


class LogRedirector:
    """Captures print output and routes it to a callback."""

    def __init__(self, callback):
        self.callback = callback
        self._buffer = ""

    def write(self, text):
        if text:
            self.callback(text)

    def flush(self):
        pass


class PaperEngineGUI(ctk.CTk):
    """Main application window for Paper Engine."""

    def __init__(self):
        super().__init__()

        # Window setup
        self.title("Paper Engine - Pre-Release v1.0")
        self.geometry("900x700")
        self.minsize(750, 550)

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # State
        self._game_process = None
        self._screenshot_thread = None
        self._capturing = False
        self._window_geometry = None
        self._last_path_file = Path(__file__).parent / ".last_game_path"

        self._build_ui()
        self._log("\n" + "=" * 60)
        self._log("\n          PAPER ENGINE - Pre-Release v1.0\n")
        self._log("=" * 60 + "\n")

        # Run dependency check at startup
        self._run_dependency_check() 

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        # Grid layout: sidebar + main area
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # ---- Sidebar ----
        sidebar = ctk.CTkFrame(self, width=200, corner_radius=0)
        sidebar.grid(row=0, column=0, sticky="nsew")
        sidebar.grid_rowconfigure(10, weight=1)  # spacer

        logo = ctk.CTkLabel(
            sidebar,
            text="Paper Engine",
            font=ctk.CTkFont(size=20, weight="bold"),
        )
        logo.grid(row=0, column=0, padx=20, pady=(20, 5))

        version_label = ctk.CTkLabel(
            sidebar, text="Pre-Release v1.0", font=ctk.CTkFont(size=12)
        )
        version_label.grid(row=1, column=0, padx=20, pady=(0, 20))

        # Step buttons
        steps = [
            ("1. Launch Game", self._on_launch_game),
            ("   Stop Capture", self._on_stop_capture),
            ("2. Label Studio", self._on_label_studio),
            ("3. Train Model", self._on_train_model),
            ("4. Test Model", self._on_test_model),
            ("5. Generate Bot", self._on_generate_bot),
        ]

        for i, (text, cmd) in enumerate(steps):
            btn = ctk.CTkButton(sidebar, text=text, command=cmd, width=170)
            btn.grid(row=i + 2, column=0, padx=15, pady=6)
            # Keep a reference to stop-capture so we can disable/enable it
            if "Stop" in text:
                self._stop_btn = btn
                btn.configure(state="disabled", fg_color="gray")

        # LLM settings button
        llm_btn = ctk.CTkButton(
            sidebar,
            text="LLM Settings",
            width=170,
            fg_color="gray30",
            command=self._on_llm_settings,
        )
        llm_btn.grid(row=9, column=0, padx=15, pady=(10, 6))

        # Appearance selector at bottom
        appearance_label = ctk.CTkLabel(sidebar, text="Theme:")
        appearance_label.grid(row=11, column=0, padx=20, pady=(10, 0))

        appearance_menu = ctk.CTkOptionMenu(
            sidebar,
            values=["Dark", "Light", "System"],
            command=lambda v: ctk.set_appearance_mode(v.lower()),
        )
        appearance_menu.grid(row=12, column=0, padx=20, pady=(5, 20))

        # ---- Main content area ----
        main_frame = ctk.CTkFrame(self)
        main_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        main_frame.grid_columnconfigure(0, weight=1)
        # Row 0 = game path, Row 1 = status bar, Row 2 = log (expands), Row 3 = bottom
        main_frame.grid_rowconfigure(2, weight=1)

        # -- Row 0: Game path bar --
        info_frame = ctk.CTkFrame(main_frame)
        info_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 2))
        info_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(info_frame, text="Game path:").grid(
            row=0, column=0, padx=(10, 5), pady=8
        )
        self._game_path_var = ctk.StringVar(value=self._load_last_game_path())
        game_entry = ctk.CTkEntry(
            info_frame, textvariable=self._game_path_var, height=34
        )
        game_entry.grid(row=0, column=1, padx=5, pady=8, sticky="ew")

        browse_btn = ctk.CTkButton(
            info_frame,
            text="Browse",
            width=80,
            height=34,
            command=self._browse_game_dir,
        )
        browse_btn.grid(row=0, column=2, padx=(5, 10), pady=8)

        # -- Row 1: Status indicators (separate row, no overlap) --
        status_frame = ctk.CTkFrame(main_frame)
        status_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=(2, 2))

        self._status_var = ctk.StringVar(value="Idle")
        self._status_label = ctk.CTkLabel(
            status_frame,
            textvariable=self._status_var,
            font=ctk.CTkFont(size=13, weight="bold"),
        )
        self._status_label.grid(row=0, column=0, padx=10, pady=5)

        self._screenshot_count_var = ctk.StringVar(value="Screenshots: 0")
        ctk.CTkLabel(status_frame, textvariable=self._screenshot_count_var).grid(
            row=0, column=1, padx=20, pady=5
        )

        # -- Row 2: Log area (fills remaining space) --
        self._log_textbox = ctk.CTkTextbox(main_frame, wrap="word", state="disabled")
        self._log_textbox.grid(row=2, column=0, sticky="nsew", padx=10, pady=(2, 5))

        # -- Row 3: Bottom controls --
        bottom = ctk.CTkFrame(main_frame)
        bottom.grid(row=3, column=0, sticky="ew", padx=10, pady=(0, 10))

        clear_btn = ctk.CTkButton(
            bottom, text="Clear Log", width=100, command=self._clear_log
        )
        clear_btn.pack(side="right", padx=5, pady=5)

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def _log(self, text: str):
        """Append text to the log textbox (thread-safe)."""

        def _insert():
            self._log_textbox.configure(state="normal")
            self._log_textbox.insert("end", text)
            self._log_textbox.see("end")
            self._log_textbox.configure(state="disabled")

        # Schedule on main thread if called from worker
        if threading.current_thread() is not threading.main_thread():
            self.after(0, _insert)
        else:
            _insert()

    def _clear_log(self):
        self._log_textbox.configure(state="normal")
        self._log_textbox.delete("1.0", "end")
        self._log_textbox.configure(state="disabled")

    def _set_status(self, text: str):
        def _update():
            self._status_var.set(text)

        if threading.current_thread() is not threading.main_thread():
            self.after(0, _update)
        else:
            _update()

    # ------------------------------------------------------------------
    # Dependency checking (mirrors main.py validate_dependencies)
    # ------------------------------------------------------------------

    def _run_dependency_check(self):
        """Run dependency checks at startup and log results."""
        self._log("\nChecking dependencies...\n")

        pkg_results = check_python_packages()
        tool_results = check_system_tools()

        # Categorise
        core_pkgs = ["pynput", "Pillow", "mss", "python-dotenv", "pyyaml"]
        training_pkgs = ["ultralytics", "torch"]
        llm_pkgs = ["anthropic", "openai", "google-genai"]

        missing_core = [
            p for p in core_pkgs if p in pkg_results and not pkg_results[p][0]
        ]
        missing_training = [
            p for p in training_pkgs if p in pkg_results and not pkg_results[p][0]
        ]
        missing_llm = [
            p for p in llm_pkgs if p in pkg_results and not pkg_results[p][0]
        ]
        installed_llm = [p for p in llm_pkgs if p in pkg_results and pkg_results[p][0]]
        missing_tools = [t for t, (found, _) in tool_results.items() if not found]

        all_good = True

        # --- Core packages (auto-install) ---
        if missing_core:
            all_good = False
            self._log(f"\n  Missing CORE packages: {', '.join(missing_core)}\n")
            self._log("  Auto-installing...\n")
            for pkg in missing_core:
                ok = install_python_package(pkg)
                if ok:
                    self._log(f"    Installed {pkg}\n")
                else:
                    self._log(f"    FAILED to install {pkg}\n")

        # --- Training packages ---
        if missing_training:
            self._log(
                f"\n  Missing TRAINING packages (optional): {', '.join(missing_training)}\n"
            )
            self._log("    Install with: pip install ultralytics torch\n")

        # --- LLM packages ---
        if not installed_llm:
            self._log("\n  No LLM provider installed (needed for bot generation).\n")
            self._log(
                "    Install one with: pip install anthropic / openai / google-genai\n"
            )
        else:
            self._log(f"\n  LLM provider: {', '.join(installed_llm)}\n")

        # --- System tools ---
        if missing_tools:
            all_good = False
            self._log(f"\n  Missing system tools: {', '.join(missing_tools)}\n")
            for tool in missing_tools:
                if tool == "wine":
                    self._log("    wine: sudo apt install wine\n")
                elif tool == "flameshot":
                    self._log("    flameshot: sudo apt install flameshot\n")
                elif tool == "label-studio":
                    self._log("    label-studio: pip install label-studio\n")

        if all_good and not missing_training:
            self._log("\n  All dependencies OK.\n")

        self._log("\nReady.  Choose an action from the sidebar.\n")

    # ------------------------------------------------------------------
    # Run a function in a background thread, capturing stdout
    # ------------------------------------------------------------------

    def _run_in_thread(self, target, *args, **kwargs):
        """Run *target* in a daemon thread, redirecting stdout/stderr to the log."""

        def _worker():
            old_stdout, old_stderr = sys.stdout, sys.stderr
            redirector = LogRedirector(self._log)
            sys.stdout = redirector
            sys.stderr = redirector
            try:
                target(*args, **kwargs)
            except Exception as exc:
                self._log(f"\nERROR: {exc}\n")
            finally:
                sys.stdout, sys.stderr = old_stdout, old_stderr

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        return t

    # ------------------------------------------------------------------
    # Game path persistence
    # ------------------------------------------------------------------

    def _load_last_game_path(self):
        """Load the last-used game path, falling back to config default."""
        try:
            if self._last_path_file.exists():
                saved = self._last_path_file.read_text().strip()
                if saved:
                    return saved
        except Exception:
            pass
        return str(config.GAME_PATH)

    def _save_last_game_path(self, path: str):
        """Persist the current game path for next launch."""
        try:
            self._last_path_file.write_text(path)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Browse for game directory
    # ------------------------------------------------------------------

    def _browse_game_dir(self):
        from tkinter import filedialog

        d = filedialog.askdirectory(title="Select game directory")
        if d:
            self._game_path_var.set(d)
            self._save_last_game_path(d)

    # ------------------------------------------------------------------
    # Step 1: Launch Game + Screenshot capture
    # ------------------------------------------------------------------

    def _on_launch_game(self):
        self._run_in_thread(self._launch_game_worker)

    def _launch_game_worker(self):
        from functions import path_finder, get_title
        from screencapture import (
            take_screenshot,
            create_screenshots_directory,
            find_wine_window,
        )

        game_path = self._game_path_var.get()
        self._save_last_game_path(game_path)
        self._log(f"\n--- Launching game from: {game_path} ---\n")
        self._set_status("Launching game...")

        # Find executable
        exe_path = path_finder(game_path)
        if exe_path is None:
            self._log("ERROR: No executable found in game directory.\n")
            self._set_status("Error")
            return

        exe_path_obj = Path(exe_path)
        self._log(f"Found executable: {exe_path}\n")

        # Launch
        try:
            if exe_path_obj.suffix.lower() == ".exe":
                self._game_process = subprocess.Popen(
                    [
                        "wine",
                        "explorer",
                        f"/desktop=game,{config.WINE_DESKTOP_RESOLUTION}",
                        str(exe_path),
                    ]
                )
                self._log(f"Started via Wine: {exe_path}\n")
            elif exe_path_obj.suffix.lower() == ".sh":
                self._game_process = subprocess.Popen(["bash", str(exe_path)])
                self._log(f"Started shell script: {exe_path}\n")
            elif exe_path_obj.suffix.lower() == ".py":
                self._game_process = subprocess.Popen(["python", str(exe_path)])
                self._log(f"Started Python game: {exe_path}\n")
            else:
                self._game_process = subprocess.Popen([str(exe_path)])
                self._log(f"Started native executable: {exe_path}\n")
        except FileNotFoundError as e:
            self._log(f"ERROR: {e}\n")
            self._set_status("Error")
            return

        # Wait for game to initialise
        wait = config.GAME_INITIALIZATION_WAIT
        self._log(f"Waiting {wait}s for game to initialise...\n")
        time.sleep(wait)

        # Detect window geometry
        self._window_geometry = find_wine_window()
        if self._window_geometry:
            self._log(f"Locked onto game window: {self._window_geometry}\n")
        else:
            self._log("Using full-screen capture mode.\n")

        # Begin screenshot loop
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
                0,
                lambda c=count: self._screenshot_count_var.set(f"Screenshots: {c}"),
            )

        self._capturing = False
        self.after(
            0, lambda: self._stop_btn.configure(state="disabled", fg_color="gray")
        )
        self._log(f"\nCapture finished. {count} screenshots taken.\n")
        self._set_status("Idle")

    def _on_stop_capture(self):
        """Stop the running screenshot capture loop and kill the game process."""
        self._capturing = False
        if self._game_process and self._game_process.poll() is None:
            self._game_process.terminate()
            self._log("Game process terminated.\n")
        self._set_status("Stopping...")

    # ------------------------------------------------------------------
    # Step 2: Label Studio
    # ------------------------------------------------------------------

    def _on_label_studio(self):
        self._run_in_thread(self._label_studio_worker)

    def _label_studio_worker(self):
        from functions import launch_label_studio

        self._log("\n--- Starting Label Studio ---\n")
        self._set_status("Label Studio running")
        env = os.environ.copy()
        env.update(config.LABEL_STUDIO_ENV)
        launch_label_studio(env)
        self._log("Label Studio launched on http://localhost:8080\n")

    # ------------------------------------------------------------------
    # Step 3: Train Model
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
    # Step 4: Test Model
    # ------------------------------------------------------------------

    def _on_test_model(self):
        self._run_in_thread(self._test_model_worker)

    def _test_model_worker(self):
        self._log("\n--- Testing Trained Model ---\n")
        self._set_status("Testing model...")
        import test_model

        test_model.test_model()
        self._log("\nModel testing complete.\n")
        self._set_status("Idle")

    # ------------------------------------------------------------------
    # Step 5: Generate Bot Script
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

        game_path = self._game_path_var.get()
        exe_path = path_finder(game_path)
        if not exe_path:
            self._log("ERROR: No game executable found.\n")
            self._set_status("Error")
            return

        game_title = get_title(exe_path)
        self._log(f"Game: {game_title}\n")

        # Controls
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

        save_controls_to_config(game_title, game_path, controls)
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
    # LLM Settings
    # ------------------------------------------------------------------

    def _on_llm_settings(self):
        """Open the LLM provider selection dialog."""
        dialog = LLMSetupDialog(self, force_show=True)
        result = dialog.wait_for_result()
        if result:
            self._log(f"\nLLM provider set to: {result}\n")


# ======================================================================
# LLM Provider Selection Dialog
# ======================================================================


class LLMSetupDialog(ctk.CTkToplevel):
    """Startup dialog for choosing an LLM provider."""

    PROVIDERS = [
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

    def __init__(self, parent=None, force_show=False):
        super().__init__(parent)
        self.title("Paper Engine - LLM Setup")
        self.geometry("520x480")
        self.resizable(False, False)
        self.transient(parent)

        self.result = None  # will hold the chosen provider id
        self._force_show = force_show

        self._build()
        self._detect_installed()

        # Make modal -- must wait until the window is rendered
        self.after(100, self._try_grab)

    def _try_grab(self):
        """Attempt to grab focus; retry if window isn't viewable yet."""
        try:
            self.grab_set()
        except Exception:
            self.after(100, self._try_grab)

    def _build(self):
        ctk.CTkLabel(
            self,
            text="LLM Provider Setup",
            font=ctk.CTkFont(size=22, weight="bold"),
        ).pack(pady=(24, 4))

        ctk.CTkLabel(
            self,
            text="Choose which AI provider to use for bot generation.\n"
            "You only need one. Google Gemini has a free tier.",
            font=ctk.CTkFont(size=13),
            justify="center",
        ).pack(pady=(0, 16))

        self._radio_var = ctk.StringVar(value="google")
        self._radio_buttons = {}

        for prov in self.PROVIDERS:
            frame = ctk.CTkFrame(self, fg_color="transparent")
            frame.pack(fill="x", padx=30, pady=2)

            rb = ctk.CTkRadioButton(
                frame,
                text=prov["label"],
                variable=self._radio_var,
                value=prov["id"],
                font=ctk.CTkFont(size=14),
            )
            rb.pack(side="left")

            self._radio_buttons[prov["id"]] = rb

            # Show installed badge
            status_label = ctk.CTkLabel(frame, text="", font=ctk.CTkFont(size=12))
            status_label.pack(side="right", padx=10)
            prov["_status_label"] = status_label

        # API key entry
        ctk.CTkLabel(self, text="API Key:", font=ctk.CTkFont(size=13), anchor="w").pack(
            fill="x", padx=30, pady=(20, 2)
        )

        self._api_key_var = ctk.StringVar(value=os.environ.get("API_KEY", ""))
        self._key_entry = ctk.CTkEntry(
            self,
            textvariable=self._api_key_var,
            height=36,
            placeholder_text="Paste your API key here",
            show="*",
        )
        self._key_entry.pack(fill="x", padx=30, pady=(0, 4))

        self._key_hint = ctk.CTkLabel(
            self,
            text="",
            font=ctk.CTkFont(size=11),
            text_color="gray",
            anchor="w",
        )
        self._key_hint.pack(fill="x", padx=30)

        # Update hint when provider changes
        self._radio_var.trace_add("write", lambda *_: self._update_hint())
        self._update_hint()

        # Buttons
        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.pack(fill="x", padx=30, pady=(20, 20))

        self._confirm_btn = ctk.CTkButton(
            btn_frame,
            text="Confirm & Install",
            width=180,
            height=40,
            command=self._on_confirm,
        )
        self._confirm_btn.pack(side="left", padx=(0, 10))

        ctk.CTkButton(
            btn_frame,
            text="Skip",
            width=100,
            height=40,
            fg_color="gray",
            command=self._on_skip,
        ).pack(side="right")

        self._status_text = ctk.CTkLabel(
            self, text="", font=ctk.CTkFont(size=12), text_color="orange"
        )
        self._status_text.pack(pady=(0, 10))

    def _detect_installed(self):
        """Check which providers are already installed and mark them."""
        current_provider = getattr(config, "LLM_PROVIDER", "google")
        for prov in self.PROVIDERS:
            try:
                __import__(prov["import"])
                prov["_status_label"].configure(text="(installed)", text_color="green")
                prov["_installed"] = True
            except ImportError:
                prov["_status_label"].configure(
                    text="(not installed)", text_color="gray"
                )
                prov["_installed"] = False

            # Pre-select whatever is currently configured
            if prov["id"] == current_provider:
                self._radio_var.set(prov["id"])

    def _update_hint(self):
        selected = self._radio_var.get()
        for prov in self.PROVIDERS:
            if prov["id"] == selected:
                self._key_hint.configure(text=f"Get key: {prov['key_url']}")
                break

    def _on_confirm(self):
        selected_id = self._radio_var.get()
        prov = next(p for p in self.PROVIDERS if p["id"] == selected_id)

        # Install if needed
        if not prov.get("_installed"):
            self._status_text.configure(text=f"Installing {prov['pip']}...")
            self.update()
            ok = install_python_package(prov["pip"])
            if not ok:
                self._status_text.configure(text=f"Failed to install {prov['pip']}")
                return
            self._status_text.configure(text=f"Installed {prov['pip']}")

        # Save API key to .env
        api_key = self._api_key_var.get().strip()
        if api_key:
            env_path = Path(".env")
            # Read existing content or create new
            lines = []
            if env_path.exists():
                with open(env_path, "r") as f:
                    lines = f.readlines()

            # Replace or add API_KEY line
            found = False
            for i, line in enumerate(lines):
                if line.strip().startswith("API_KEY="):
                    lines[i] = f"API_KEY={api_key}\n"
                    found = True
                    break
            if not found:
                lines.append(f"API_KEY={api_key}\n")

            with open(env_path, "w") as f:
                f.writelines(lines)

            # Also set in current environment so it takes effect immediately
            os.environ["API_KEY"] = api_key

        # Update config
        update_llm_config(prov["pip"] if prov["id"] != "google" else "google-genai")

        self.result = selected_id
        self.grab_release()
        self.destroy()

    def _on_skip(self):
        self.result = None
        self.grab_release()
        self.destroy()

    def wait_for_result(self):
        """Block until the dialog is closed."""
        self.wait_window()
        return self.result


# ======================================================================
# Entry point
# ======================================================================


def main():
    # Detect if any LLM provider is installed
    has_llm = False
    for import_name in ["anthropic", "openai", "google.genai"]:
        try:
            __import__(import_name)
            has_llm = True
            break
        except ImportError:
            pass

    # Show LLM setup dialog if none installed
    if not has_llm:
        # Need a temporary root for the dialog
        root = ctk.CTk()
        root.withdraw()  # hide the main window
        dialog = LLMSetupDialog(root)
        dialog.wait_for_result()
        root.destroy()

    app = PaperEngineGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
