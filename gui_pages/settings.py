"""Settings page -- LLM provider, API keys, training packages, cloud storage.

Ported from the old ConfigDialog in gui.py, now displayed as a full page
rather than a modal dialog.
"""

import os
import subprocess
import configparser
import customtkinter as ctk
from pathlib import Path

from gui_components.theme import (
    BG_SURFACE,
    BG_HOVER,
    BORDER,
    BORDER_SUBTLE,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    TEXT_MUTED,
    TEXT_ON_ACCENT,
    ACCENT,
    SUCCESS,
    WARNING,
    DANGER,
    INFO,
    CLR_BLUE,
    CLR_GREEN,
    CLR_ORANGE,
    CLR_PURPLE,
    ICON_DOT,
    RADIUS_SM,
    RADIUS_MD,
    RADIUS_LG,
    CARD_BORDER_WIDTH,
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
    BTN_HEIGHT_MD,
    BTN_HEIGHT_LG,
    INPUT_HEIGHT,
    font_heading,
    font_subheading,
    font_body,
    font_body_bold,
    font_small,
    font_small_bold,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ======================================================================
# Data definitions
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

_CLOUD_PROVIDERS = [
    {"id": "none", "label": "None (disabled)", "fields": []},
    {
        "id": "gdrive",
        "label": "Google Drive",
        "pip": "google-api-python-client",
        "import": "googleapiclient",
        "fields": [
            ("gdrive_folder_id", "Folder ID"),
            ("gdrive_credentials", "Credentials JSON path"),
        ],
    },
    {
        "id": "s3",
        "label": "AWS S3",
        "pip": "boto3",
        "import": "boto3",
        "fields": [
            ("s3_bucket", "Bucket"),
            ("s3_region", "Region"),
            ("s3_access_key", "Access Key"),
            ("s3_secret_key", "Secret Key"),
        ],
    },
    {
        "id": "dropbox",
        "label": "Dropbox",
        "pip": "dropbox",
        "import": "dropbox",
        "fields": [
            ("dropbox_token", "Access Token"),
            ("dropbox_path", "Remote Path"),
        ],
    },
    {
        "id": "azure",
        "label": "Azure Blob Storage",
        "pip": "azure-storage-blob",
        "import": "azure.storage.blob",
        "fields": [
            ("azure_connection_string", "Connection String"),
            ("azure_container", "Container Name"),
        ],
    },
    {
        "id": "hf",
        "label": "Hugging Face Hub",
        "pip": "huggingface_hub",
        "import": "huggingface_hub",
        "fields": [
            ("hf_repo_id", "Repository ID"),
            ("hf_token", "Access Token"),
        ],
    },
]


# ======================================================================
# Helpers
# ======================================================================


def _pip_install(package: str) -> bool:
    try:
        subprocess.check_call(
            ["pip", "install", package],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except Exception:
        return False


def _update_llm_config(pip_name: str):
    """Write LLM provider + model name to main_conf.ini."""
    provider_map = {
        "google-genai": ("google", "gemini-2.5-flash"),
        "anthropic": ("anthropic", "claude-sonnet-4-5-20250514"),
        "openai": ("openai", "gpt-4"),
    }
    provider, model = provider_map.get(pip_name, ("google", "gemini-2.5-flash"))

    ini = PROJECT_ROOT / "conf" / "main_conf.ini"
    cp = configparser.ConfigParser()
    cp.read(ini)
    if not cp.has_section("LLM"):
        cp.add_section("LLM")
    cp.set("LLM", "provider", provider)
    cp.set("LLM", "model_name", model)
    with open(ini, "w") as f:
        cp.write(f)


# ======================================================================
# Settings Page
# ======================================================================


class SettingsPage(ctk.CTkFrame):
    """Configuration page: LLM provider, training packages, cloud storage."""

    def __init__(self, master, app):
        super().__init__(master, fg_color="transparent")
        self._app = app
        self._llm_labels = {}
        self._training_vars = {}
        self._training_labels = {}
        self._cloud_var = ctk.StringVar(value="None (disabled)")
        self._cloud_field_vars = {}
        self._cloud_fields_frame = None

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self._build()

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def _build(self):
        scroll = ctk.CTkScrollableFrame(self, fg_color="transparent")
        scroll.grid(row=0, column=0, sticky="nsew")
        scroll.grid_columnconfigure(0, weight=1)
        self._scroll = scroll

        # Title
        ctk.CTkLabel(
            scroll,
            text="Settings",
            font=font_heading(),
            text_color=TEXT_PRIMARY,
        ).pack(anchor="w", padx=PAGE_PAD_X, pady=(PAGE_PAD_TOP, SP_SM))

        # ============ LLM Provider Card ============
        llm_card = self._section_card(scroll)

        self._section_header(
            llm_card,
            "LLM Provider",
            "For bot generation and annotation",
            CLR_BLUE,
        )
        ctk.CTkLabel(
            llm_card,
            text="You only need one. Google Gemini has a free tier.",
            font=font_small(),
            text_color=TEXT_MUTED,
            anchor="w",
        ).pack(fill="x", padx=SP_LG, pady=(0, SP_MD))

        self._llm_var = ctk.StringVar(value="google")
        for prov in _LLM_PROVIDERS:
            row = ctk.CTkFrame(llm_card, fg_color="transparent")
            row.pack(fill="x", padx=SP_LG, pady=2)
            ctk.CTkRadioButton(
                row,
                text=prov["label"],
                variable=self._llm_var,
                value=prov["id"],
                font=font_body(),
            ).pack(side="left")
            lbl = ctk.CTkLabel(row, text="", font=font_small())
            lbl.pack(side="right", padx=SP_SM)
            self._llm_labels[prov["id"]] = lbl

        # API key
        ctk.CTkLabel(
            llm_card,
            text="API Key",
            font=font_body_bold(),
            text_color=TEXT_PRIMARY,
            anchor="w",
        ).pack(fill="x", padx=SP_LG, pady=(SP_LG, SP_XS))

        self._api_key_var = ctk.StringVar(value="")
        ctk.CTkEntry(
            llm_card,
            textvariable=self._api_key_var,
            height=INPUT_HEIGHT,
            corner_radius=RADIUS_MD,
            placeholder_text="Paste your API key here",
            show="*",
            border_width=1,
            border_color=BORDER,
        ).pack(fill="x", padx=SP_LG, pady=(0, SP_XS))

        self._key_hint = ctk.CTkLabel(
            llm_card,
            text="",
            font=font_small(),
            text_color=TEXT_MUTED,
            anchor="w",
        )
        self._key_hint.pack(fill="x", padx=SP_LG, pady=(0, SP_LG))
        self._llm_var.trace_add("write", lambda *_: self._update_hint())

        # ============ Training Packages Card ============
        train_card = self._section_card(scroll)

        self._section_header(
            train_card,
            "Training Packages",
            "For local YOLO model training",
            CLR_GREEN,
        )
        ctk.CTkLabel(
            train_card,
            text="~6.5 GB total. Only needed if you want to train models locally.",
            font=font_small(),
            text_color=TEXT_MUTED,
            anchor="w",
        ).pack(fill="x", padx=SP_LG, pady=(0, SP_MD))

        for pkg in _TRAINING_PACKAGES:
            row = ctk.CTkFrame(train_card, fg_color="transparent")
            row.pack(fill="x", padx=SP_LG, pady=2)
            var = ctk.BooleanVar(value=False)
            ctk.CTkCheckBox(
                row,
                text=pkg["label"],
                variable=var,
                font=font_body(),
            ).pack(side="left")
            lbl = ctk.CTkLabel(row, text="", font=font_small())
            lbl.pack(side="right", padx=SP_SM)
            self._training_vars[pkg["pip"]] = var
            self._training_labels[pkg["pip"]] = lbl

        # Bottom padding inside card
        ctk.CTkFrame(train_card, fg_color="transparent", height=SP_LG).pack()

        # ============ Cloud Storage Card ============
        cloud_card = self._section_card(scroll)

        self._section_header(
            cloud_card,
            "Cloud Storage",
            "Dataset sync and backup",
            CLR_ORANGE,
        )
        ctk.CTkLabel(
            cloud_card,
            text="Optional. Link a cloud provider to back up datasets and models.",
            font=font_small(),
            text_color=TEXT_MUTED,
            anchor="w",
        ).pack(fill="x", padx=SP_LG, pady=(0, SP_MD))

        cloud_row = ctk.CTkFrame(cloud_card, fg_color="transparent")
        cloud_row.pack(fill="x", padx=SP_LG, pady=(0, SP_SM))
        ctk.CTkLabel(
            cloud_row,
            text="Provider",
            font=font_body_bold(),
            text_color=TEXT_PRIMARY,
        ).pack(side="left", padx=(0, SP_MD))
        ctk.CTkOptionMenu(
            cloud_row,
            variable=self._cloud_var,
            values=[p["label"] for p in _CLOUD_PROVIDERS],
            command=self._on_cloud_change,
            width=220,
            height=INPUT_HEIGHT,
            corner_radius=RADIUS_MD,
        ).pack(side="left")

        self._cloud_fields_frame = ctk.CTkFrame(cloud_card, fg_color="transparent")
        self._cloud_fields_frame.pack(fill="x", padx=SP_LG, pady=(0, SP_LG))

        # ============ Memory Reading Card ============
        mem_card = self._section_card(scroll)

        self._section_header(
            mem_card,
            "Memory Reading",
            "Live game state via process memory",
            CLR_PURPLE,
        )
        ctk.CTkLabel(
            mem_card,
            text=(
                "The memory reader currently supports Mono-based Unity games only "
                "(e.g. Cuphead). Other engines (IL2CPP Unity, Unreal, Godot) are "
                "not supported yet.\n\n"
                "Memory reading is used by the bot for live game state (HP, scene, "
                "super meter) and by the session recorder for training data. "
                "It requires launching the game as a child process on Linux."
            ),
            font=font_small(),
            text_color=TEXT_SECONDARY,
            anchor="w",
            wraplength=500,
            justify="left",
        ).pack(fill="x", padx=SP_LG, pady=(0, SP_LG))

        # ============ Save Button ============
        save_row = ctk.CTkFrame(scroll, fg_color="transparent")
        save_row.pack(fill="x", padx=PAGE_PAD_X, pady=(SP_XL, SP_MD))

        ctk.CTkButton(
            save_row,
            text="Save & Install",
            width=180,
            height=BTN_HEIGHT_LG,
            corner_radius=RADIUS_MD,
            fg_color=CLR_BLUE,
            font=font_body_bold(),
            command=self._on_save,
        ).pack(side="left")

        self._msg = ctk.CTkLabel(
            save_row,
            text="",
            font=font_small(),
            text_color=WARNING,
        )
        self._msg.pack(side="left", padx=(SP_LG, 0))

        # Bottom spacer
        ctk.CTkFrame(scroll, fg_color="transparent", height=SP_2XL).pack()

    # ------------------------------------------------------------------
    # Card / section helpers
    # ------------------------------------------------------------------

    def _section_card(self, parent) -> ctk.CTkFrame:
        """Create a card frame for a settings section."""
        card = ctk.CTkFrame(
            parent,
            corner_radius=RADIUS_LG,
            border_width=CARD_BORDER_WIDTH,
            border_color=BORDER_SUBTLE,
            fg_color=BG_SURFACE,
        )
        card.pack(fill="x", padx=PAGE_PAD_X, pady=(0, SP_MD))
        return card

    def _section_header(self, parent, title: str, subtitle: str, accent_color):
        """Add a section header with colored dot to a card."""
        hdr = ctk.CTkFrame(parent, fg_color="transparent")
        hdr.pack(fill="x", padx=SP_LG, pady=(SP_LG, SP_XS))

        ctk.CTkLabel(
            hdr,
            text=ICON_DOT,
            font=ctk.CTkFont(size=10),
            text_color=accent_color,
            width=16,
        ).pack(side="left", padx=(0, SP_SM))
        ctk.CTkLabel(
            hdr,
            text=title,
            font=font_subheading(),
            text_color=TEXT_PRIMARY,
            anchor="w",
        ).pack(side="left")

        ctk.CTkLabel(
            parent,
            text=subtitle,
            font=font_small(),
            text_color=TEXT_SECONDARY,
            anchor="w",
        ).pack(fill="x", padx=SP_LG, pady=(0, SP_SM))

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def on_show(self):
        self._detect_state()

    # ------------------------------------------------------------------
    # State detection
    # ------------------------------------------------------------------

    def _detect_state(self):
        from conf.config_parser import main_conf as config

        current_llm = getattr(config, "LLM_PROVIDER", "google")
        for prov in _LLM_PROVIDERS:
            lbl = self._llm_labels[prov["id"]]
            try:
                __import__(prov["import"])
                lbl.configure(text="installed", text_color=SUCCESS)
                prov["_ok"] = True
            except ImportError:
                lbl.configure(text="not installed", text_color=TEXT_MUTED)
                prov["_ok"] = False
            if prov["id"] == current_llm:
                self._llm_var.set(prov["id"])

        for pkg in _TRAINING_PACKAGES:
            lbl = self._training_labels[pkg["pip"]]
            try:
                __import__(pkg["import"])
                lbl.configure(text="installed", text_color=SUCCESS)
                self._training_vars[pkg["pip"]].set(True)
                pkg["_ok"] = True
            except ImportError:
                lbl.configure(text="not installed", text_color=TEXT_MUTED)
                pkg["_ok"] = False

        self._update_hint()

        # Pre-fill API key
        try:
            from tools.functions import get_api_key

            key = get_api_key() or ""
            self._api_key_var.set(key)
        except Exception:
            pass

        # Detect cloud provider
        ini = PROJECT_ROOT / "conf" / "main_conf.ini"
        p = configparser.ConfigParser()
        p.read(ini)
        saved_id = p.get("CloudStorage", "provider", fallback="none")
        for cp in _CLOUD_PROVIDERS:
            if cp["id"] == saved_id:
                self._on_cloud_change(cp["label"])
                break

    def _update_hint(self):
        sel = self._llm_var.get()
        for prov in _LLM_PROVIDERS:
            if prov["id"] == sel:
                self._key_hint.configure(text=f"Get key: {prov['key_url']}")
                return

    def _on_cloud_change(self, label_value):
        prov = next(
            (p for p in _CLOUD_PROVIDERS if p["label"] == label_value),
            _CLOUD_PROVIDERS[0],
        )
        self._cloud_var.set(prov["label"])

        for w in self._cloud_fields_frame.winfo_children():
            w.destroy()
        self._cloud_field_vars.clear()

        if not prov.get("fields"):
            return

        for key, label in prov["fields"]:
            row = ctk.CTkFrame(self._cloud_fields_frame, fg_color="transparent")
            row.pack(fill="x", pady=2)
            ctk.CTkLabel(
                row,
                text=f"{label}",
                font=font_body(),
                text_color=TEXT_SECONDARY,
                width=160,
                anchor="w",
            ).pack(side="left")
            var = ctk.StringVar(value=self._load_cloud_field(key))
            show = "*" if "key" in key or "secret" in key or "token" in key else ""
            ctk.CTkEntry(
                row,
                textvariable=var,
                height=INPUT_HEIGHT,
                corner_radius=RADIUS_MD,
                show=show,
                border_width=1,
                border_color=BORDER,
            ).pack(
                side="left",
                fill="x",
                expand=True,
                padx=(SP_SM, 0),
            )
            self._cloud_field_vars[key] = var

    def _load_cloud_field(self, key):
        try:
            from tools.functions import get_secret, is_sensitive_field

            if is_sensitive_field(key):
                return get_secret(key)
            ini = PROJECT_ROOT / "conf" / "main_conf.ini"
            p = configparser.ConfigParser()
            p.read(ini)
            return p.get("CloudStorage", key, fallback="")
        except Exception:
            return ""

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def _set_msg(self, text):
        self._msg.configure(text=text)
        self.update()

    def _on_save(self):
        # -- Install LLM provider if needed --
        prov = next(p for p in _LLM_PROVIDERS if p["id"] == self._llm_var.get())
        if not prov.get("_ok"):
            self._set_msg(f"Installing {prov['pip']}...")
            if not _pip_install(prov["pip"]):
                self._set_msg(f"Failed to install {prov['pip']}")
                return

        # -- Install training packages if checked --
        for pkg in _TRAINING_PACKAGES:
            wanted = self._training_vars[pkg["pip"]].get()
            already = pkg.get("_ok", False)
            if wanted and not already:
                self._set_msg(f"Installing {pkg['pip']} (this may take a while)...")
                if not _pip_install(pkg["pip"]):
                    self._set_msg(f"Failed to install {pkg['pip']}")
                    return

        # -- Save API key --
        api_key = self._api_key_var.get().strip()
        if api_key:
            try:
                from tools.functions import store_api_key

                if not store_api_key(api_key):
                    # Keyring unavailable -- fall back to .env
                    env_path = Path(".env")
                    lines = (
                        env_path.read_text().splitlines(True)
                        if env_path.exists()
                        else []
                    )
                    replaced = False
                    for i, line in enumerate(lines):
                        if line.strip().startswith("API_KEY="):
                            lines[i] = f"API_KEY={api_key}\n"
                            replaced = True
                            break
                    if not replaced:
                        lines.append(f"API_KEY={api_key}\n")
                    env_path.write_text("".join(lines))
                    os.chmod(env_path, 0o600)
            except Exception as e:
                self._set_msg(f"Failed to store API key: {e}")

        # -- Save LLM config --
        _update_llm_config(prov["pip"])

        # -- Save cloud storage config --
        try:
            from tools.functions import is_sensitive_field, store_secret
        except ImportError:
            is_sensitive_field = lambda k: False
            store_secret = lambda k, v: None

        cloud_label = self._cloud_var.get()
        cloud_prov = next(
            (cp for cp in _CLOUD_PROVIDERS if cp["label"] == cloud_label),
            _CLOUD_PROVIDERS[0],
        )

        ini = PROJECT_ROOT / "conf" / "main_conf.ini"
        cp = configparser.ConfigParser()
        cp.read(ini)
        if not cp.has_section("CloudStorage"):
            cp.add_section("CloudStorage")
        cp.set("CloudStorage", "provider", cloud_prov["id"])
        for key, var in self._cloud_field_vars.items():
            val = var.get()
            if is_sensitive_field(key):
                store_secret(key, val)
            else:
                cp.set("CloudStorage", key, val)
        with open(ini, "w") as f:
            cp.write(f)

        # -- Auto-install cloud SDK if needed --
        if cloud_prov.get("pip"):
            try:
                __import__(
                    cloud_prov.get("import", cloud_prov["pip"].replace("-", "_"))
                )
            except ImportError:
                self._set_msg(f"Installing {cloud_prov['pip']}...")
                if not _pip_install(cloud_prov["pip"]):
                    self._set_msg(f"Failed to install {cloud_prov['pip']}")
                    return

        self._set_msg("Settings saved successfully.")
        self._app.log("Configuration updated.\n")
