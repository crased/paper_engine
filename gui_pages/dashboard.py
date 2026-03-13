"""Dashboard page -- training metrics, dataset stats, model comparison,
session history, and system status at a glance.
"""

import csv
import json
import shutil
import subprocess
import customtkinter as ctk
from pathlib import Path
from datetime import datetime

from gui_components.theme import (
    BG_BASE,
    BG_SURFACE,
    BG_SURFACE_ALT,
    BG_HOVER,
    BG_ACTIVE,
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
    CLR_GREEN,
    CLR_BLUE,
    CLR_ORANGE,
    CLR_PURPLE,
    CLR_RED,
    CLR_CYAN,
    ICON_DOT,
    ICON_DOT_EMPTY,
    ICON_CHECK,
    ICON_CROSS,
    RADIUS_SM,
    RADIUS_MD,
    RADIUS_LG,
    RADIUS_XL,
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
    font_heading,
    font_subheading,
    font_body,
    font_body_bold,
    font_small,
    font_small_bold,
    font_mono_small,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Colors for class distribution bars (cycle through these)
_CLASS_COLORS = [
    "#58a6ff",
    "#3fb950",
    "#d29922",
    "#f85149",
    "#bc8cff",
    "#2dd4bf",
    "#f97583",
    "#79c0ff",
    "#b392f0",
    "#85e89d",
    "#ffab70",
    "#f692ce",
]


class DashboardPage(ctk.CTkFrame):
    """Metrics dashboard -- overview, training, dataset, models, sessions."""

    def __init__(self, master, app):
        super().__init__(master, fg_color="transparent")
        self._app = app
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self._build()

    def _build(self):
        scroll = ctk.CTkScrollableFrame(self, fg_color="transparent")
        scroll.grid(row=0, column=0, sticky="nsew")
        scroll.grid_columnconfigure(0, weight=1)
        scroll.grid_columnconfigure(1, weight=1)
        self._scroll = scroll

        # Title
        ctk.CTkLabel(
            scroll,
            text="Metrics",
            font=font_heading(),
            text_color=TEXT_PRIMARY,
        ).grid(
            row=0,
            column=0,
            columnspan=2,
            sticky="w",
            padx=PAGE_PAD_X,
            pady=(PAGE_PAD_TOP, SP_MD),
        )

        # ── Row 1: Overview cards (4 across in 2 columns) ──
        overview = ctk.CTkFrame(scroll, fg_color="transparent")
        overview.grid(
            row=1,
            column=0,
            columnspan=2,
            sticky="ew",
            padx=PAGE_PAD_X - 4,
            pady=(0, SP_SM),
        )
        overview.grid_columnconfigure(0, weight=1)
        overview.grid_columnconfigure(1, weight=1)
        overview.grid_columnconfigure(2, weight=1)
        overview.grid_columnconfigure(3, weight=1)

        self._ov_model = self._overview_card(
            overview, 0, "Active Model", "--", CLR_BLUE
        )
        self._ov_dataset = self._overview_card(overview, 1, "Dataset", "--", CLR_GREEN)
        self._ov_training = self._overview_card(
            overview, 2, "Last Training", "--", CLR_ORANGE
        )
        self._ov_gpu = self._overview_card(overview, 3, "GPU", "--", INFO)

        # ── Row 2: Training Metrics + Dataset Stats ──
        self._metrics_card = self._section_card(scroll, 2, 0, "Training Metrics")
        self._metrics_body = ctk.CTkFrame(self._metrics_card, fg_color="transparent")
        self._metrics_body.pack(fill="both", expand=True, padx=SP_LG, pady=(0, SP_LG))

        self._dataset_card = self._section_card(scroll, 2, 1, "Dataset Stats")
        self._dataset_body = ctk.CTkFrame(self._dataset_card, fg_color="transparent")
        self._dataset_body.pack(fill="both", expand=True, padx=SP_LG, pady=(0, SP_LG))

        # ── Row 3: Model Comparison + Session History ──
        self._models_card = self._section_card(scroll, 3, 0, "Models")
        self._models_body = ctk.CTkFrame(self._models_card, fg_color="transparent")
        self._models_body.pack(fill="both", expand=True, padx=SP_LG, pady=(0, SP_LG))

        self._sessions_card = self._section_card(scroll, 3, 1, "Session History")
        self._sessions_body = ctk.CTkFrame(self._sessions_card, fg_color="transparent")
        self._sessions_body.pack(fill="both", expand=True, padx=SP_LG, pady=(0, SP_LG))

        # ── Row 4: System Status (full width) ──
        self._system_card = self._section_card(scroll, 4, 0, "System", colspan=2)
        self._system_body = ctk.CTkFrame(self._system_card, fg_color="transparent")
        self._system_body.pack(fill="both", expand=True, padx=SP_LG, pady=(0, SP_LG))

    # ==================================================================
    # Card builders
    # ==================================================================

    def _overview_card(self, parent, col, title, value, color):
        """Small KPI card. Returns dict with label refs for updating."""
        card = ctk.CTkFrame(
            parent,
            corner_radius=RADIUS_LG,
            border_width=CARD_BORDER_WIDTH,
            border_color=BORDER_SUBTLE,
            fg_color=BG_SURFACE,
        )
        card.grid(row=0, column=col, sticky="nsew", padx=4, pady=4)

        # Color accent bar at top
        ctk.CTkFrame(
            card,
            height=3,
            corner_radius=0,
            fg_color=color,
        ).pack(fill="x")

        ctk.CTkLabel(
            card,
            text=title,
            font=font_small(),
            text_color=TEXT_MUTED,
            anchor="w",
        ).pack(fill="x", padx=SP_MD, pady=(SP_SM, 0))

        val_lbl = ctk.CTkLabel(
            card,
            text=value,
            font=font_body_bold(),
            text_color=TEXT_PRIMARY,
            anchor="w",
        )
        val_lbl.pack(fill="x", padx=SP_MD, pady=(2, 0))

        detail_lbl = ctk.CTkLabel(
            card,
            text="",
            font=font_small(),
            text_color=TEXT_SECONDARY,
            anchor="w",
        )
        detail_lbl.pack(fill="x", padx=SP_MD, pady=(0, SP_SM))

        status_lbl = ctk.CTkLabel(
            card,
            text="",
            font=font_small(),
            text_color=TEXT_MUTED,
            anchor="w",
        )
        status_lbl.pack(fill="x", padx=SP_MD, pady=(0, SP_SM))

        return {"value": val_lbl, "detail": detail_lbl, "status": status_lbl}

    def _section_card(self, parent, row, col, title, colspan=1):
        """Section card with header. Returns the card frame."""
        card = ctk.CTkFrame(
            parent,
            corner_radius=RADIUS_LG,
            border_width=CARD_BORDER_WIDTH,
            border_color=BORDER_SUBTLE,
            fg_color=BG_SURFACE,
        )
        card.grid(
            row=row, column=col, columnspan=colspan, sticky="nsew", padx=4, pady=4
        )

        ctk.CTkLabel(
            card,
            text=title,
            font=font_subheading(),
            text_color=TEXT_PRIMARY,
            anchor="w",
        ).pack(fill="x", padx=SP_LG, pady=(SP_LG, SP_SM))

        return card

    # ==================================================================
    # Data loading
    # ==================================================================

    def on_show(self):
        self._refresh()

    def _refresh(self):
        self._refresh_overview()
        self._refresh_training_metrics()
        self._refresh_dataset_stats()
        self._refresh_models()
        self._refresh_sessions()
        self._refresh_system()

    # -- Overview cards --

    def _refresh_overview(self):
        models_dir = PROJECT_ROOT / "bot_logic" / "models"
        ds = PROJECT_ROOT / "yolo_dataset"

        # Model
        best = self._find_latest_model()
        if best:
            name = best["name"]
            size = best["size_mb"]
            self._ov_model["value"].configure(text=name)
            self._ov_model["detail"].configure(text=f"{size:.1f} MB")
            mAP = best.get("mAP50", None)
            if mAP is not None:
                self._ov_model["status"].configure(
                    text=f"mAP50: {mAP:.3f}", text_color=SUCCESS
                )
            else:
                self._ov_model["status"].configure(text="", text_color=TEXT_MUTED)
        else:
            self._ov_model["value"].configure(text="No model")
            self._ov_model["detail"].configure(text="Train or import one")
            self._ov_model["status"].configure(text=ICON_DOT_EMPTY, text_color=DANGER)

        # Dataset
        _exts = (".png", ".jpg", ".jpeg")
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
        self._ov_dataset["value"].configure(text=f"{total:,} images")
        self._ov_dataset["detail"].configure(text=f"{train_n:,} train / {val_n:,} val")
        if total > 1000:
            self._ov_dataset["status"].configure(text="Healthy", text_color=SUCCESS)
        elif total > 0:
            self._ov_dataset["status"].configure(
                text="Small dataset", text_color=WARNING
            )
        else:
            self._ov_dataset["status"].configure(text="Empty", text_color=DANGER)

        # Last training
        latest_csv = self._find_latest_results_csv()
        if latest_csv:
            rows = self._read_results_csv(latest_csv)
            if rows:
                last = rows[-1]
                epochs = len(rows)
                mAP = last.get("metrics/mAP50(B)", 0)
                self._ov_training["value"].configure(text=f"{epochs} epochs")
                self._ov_training["detail"].configure(text=f"mAP50: {mAP:.3f}")
                # Modification time as date
                mtime = datetime.fromtimestamp(latest_csv.stat().st_mtime)
                self._ov_training["status"].configure(
                    text=mtime.strftime("%b %d, %H:%M"), text_color=TEXT_MUTED
                )
        else:
            self._ov_training["value"].configure(text="No runs")
            self._ov_training["detail"].configure(text="")
            self._ov_training["status"].configure(text="")

        # GPU
        self._ov_gpu["value"].configure(text="RX 9070 XT")
        self._ov_gpu["detail"].configure(text="16 GB VRAM")
        try:
            import torch

            if torch.cuda.is_available():
                self._ov_gpu["status"].configure(text="ROCm ready", text_color=SUCCESS)
            else:
                self._ov_gpu["status"].configure(text="CPU only", text_color=WARNING)
        except ImportError:
            self._ov_gpu["status"].configure(
                text="PyTorch not installed", text_color=TEXT_MUTED
            )

    # -- Training metrics --

    def _refresh_training_metrics(self):
        for w in self._metrics_body.winfo_children():
            w.destroy()

        latest_csv = self._find_latest_results_csv()
        if not latest_csv:
            ctk.CTkLabel(
                self._metrics_body,
                text="No training data yet",
                font=font_body(),
                text_color=TEXT_MUTED,
            ).pack(pady=SP_LG)
            return

        rows = self._read_results_csv(latest_csv)
        if not rows:
            return

        last = rows[-1]
        best_mAP = max(r.get("metrics/mAP50(B)", 0) for r in rows)
        best_epoch = next(
            i + 1
            for i, r in enumerate(rows)
            if r.get("metrics/mAP50(B)", 0) == best_mAP
        )

        # Key metrics summary
        metrics_grid = ctk.CTkFrame(self._metrics_body, fg_color="transparent")
        metrics_grid.pack(fill="x", pady=(0, SP_MD))
        metrics_grid.grid_columnconfigure(0, weight=1)
        metrics_grid.grid_columnconfigure(1, weight=1)
        metrics_grid.grid_columnconfigure(2, weight=1)

        for col, (label, value, color) in enumerate(
            [
                ("Best mAP50", f"{best_mAP:.3f}", SUCCESS),
                ("Best Epoch", f"{best_epoch}/{len(rows)}", ACCENT),
                ("Final Loss", f"{last.get('train/box_loss', 0):.3f}", WARNING),
            ]
        ):
            f = ctk.CTkFrame(metrics_grid, fg_color="transparent")
            f.grid(row=0, column=col, sticky="ew", padx=SP_XS)
            ctk.CTkLabel(f, text=label, font=font_small(), text_color=TEXT_MUTED).pack(
                anchor="w"
            )
            ctk.CTkLabel(f, text=value, font=font_body_bold(), text_color=color).pack(
                anchor="w"
            )

        # Epoch-by-epoch table (last 10)
        ctk.CTkLabel(
            self._metrics_body,
            text="Recent Epochs",
            font=font_small_bold(),
            text_color=TEXT_SECONDARY,
            anchor="w",
        ).pack(fill="x", pady=(SP_SM, SP_XS))

        # Header
        hdr = ctk.CTkFrame(
            self._metrics_body, fg_color=BG_SURFACE_ALT, corner_radius=RADIUS_SM
        )
        hdr.pack(fill="x")
        for txt, w in [
            ("Epoch", 50),
            ("Box Loss", 70),
            ("Cls Loss", 70),
            ("mAP50", 70),
            ("mAP50-95", 70),
            ("Precision", 70),
        ]:
            ctk.CTkLabel(
                hdr,
                text=txt,
                width=w,
                font=font_small_bold(),
                text_color=TEXT_MUTED,
                anchor="center",
            ).pack(side="left", padx=2, pady=SP_XS)

        # Data rows (last 10, newest first)
        display = list(reversed(rows[-10:]))
        for r in display:
            row = ctk.CTkFrame(self._metrics_body, fg_color="transparent")
            row.pack(fill="x")
            epoch = int(r.get("epoch", 0))
            is_best = r.get("metrics/mAP50(B)", 0) == best_mAP
            color = SUCCESS if is_best else TEXT_SECONDARY
            for val, w in [
                (str(epoch), 50),
                (f"{r.get('train/box_loss', 0):.3f}", 70),
                (f"{r.get('train/cls_loss', 0):.3f}", 70),
                (f"{r.get('metrics/mAP50(B)', 0):.3f}", 70),
                (f"{r.get('metrics/mAP50-95(B)', 0):.3f}", 70),
                (f"{r.get('metrics/precision(B)', 0):.3f}", 70),
            ]:
                ctk.CTkLabel(
                    row,
                    text=val,
                    width=w,
                    font=font_mono_small(),
                    text_color=color,
                    anchor="center",
                ).pack(side="left", padx=2, pady=1)

        # mAP50 progress visualization (sparkline-style bars)
        ctk.CTkLabel(
            self._metrics_body,
            text="mAP50 Progress",
            font=font_small_bold(),
            text_color=TEXT_SECONDARY,
            anchor="w",
        ).pack(fill="x", pady=(SP_MD, SP_XS))

        bar_frame = ctk.CTkFrame(self._metrics_body, fg_color="transparent")
        bar_frame.pack(fill="x", pady=(0, SP_SM))
        bar_frame.grid_columnconfigure(0, weight=1)

        max_mAP = max(r.get("metrics/mAP50(B)", 0.001) for r in rows)
        for i, r in enumerate(rows):
            mAP_val = r.get("metrics/mAP50(B)", 0)
            pct = mAP_val / max(max_mAP, 0.001)

            row_f = ctk.CTkFrame(bar_frame, fg_color="transparent", height=6)
            row_f.pack(fill="x", pady=0)
            row_f.pack_propagate(False)

            is_best_bar = mAP_val == best_mAP
            bar_color = SUCCESS if is_best_bar else CLR_BLUE

            bar = ctk.CTkFrame(
                row_f,
                fg_color=bar_color,
                corner_radius=2,
                height=4,
            )
            bar.place(relx=0, rely=0.15, relwidth=max(pct, 0.01), relheight=0.7)

    # -- Dataset stats --

    def _refresh_dataset_stats(self):
        for w in self._dataset_body.winfo_children():
            w.destroy()

        ds = PROJECT_ROOT / "yolo_dataset"
        names = self._read_class_names()
        counts = {}  # {class_id: {"train": n, "val": n}}

        for split in ("train", "val"):
            lbl_dir = ds / split / "labels"
            if not lbl_dir.exists():
                continue
            for lbl_file in lbl_dir.glob("*.txt"):
                for line in lbl_file.read_text().strip().splitlines():
                    parts = line.strip().split()
                    if not parts:
                        continue
                    try:
                        cid = int(float(parts[0]))
                    except ValueError:
                        continue
                    if cid not in counts:
                        counts[cid] = {"train": 0, "val": 0}
                    counts[cid][split] += 1

        if not counts:
            ctk.CTkLabel(
                self._dataset_body,
                text="No annotations found",
                font=font_body(),
                text_color=TEXT_MUTED,
            ).pack(pady=SP_LG)
            return

        # Overall stats
        total_ann = sum(c["train"] + c["val"] for c in counts.values())
        ctk.CTkLabel(
            self._dataset_body,
            text=f"{total_ann:,} total annotations across {len(counts)} classes",
            font=font_small(),
            text_color=TEXT_SECONDARY,
            anchor="w",
        ).pack(fill="x", pady=(0, SP_MD))

        # Class distribution bars
        max_total = max((c["train"] + c["val"]) for c in counts.values())

        for cid in sorted(counts.keys()):
            name = names.get(cid, f"class_{cid}")
            train_n = counts[cid]["train"]
            val_n = counts[cid]["val"]
            total = train_n + val_n
            pct = total / max(max_total, 1)
            color = _CLASS_COLORS[cid % len(_CLASS_COLORS)]

            row = ctk.CTkFrame(self._dataset_body, fg_color="transparent")
            row.pack(fill="x", pady=2)
            row.grid_columnconfigure(1, weight=1)

            # Label
            ctk.CTkLabel(
                row,
                text=f"{cid}: {name}",
                font=font_small(),
                text_color=TEXT_PRIMARY,
                width=110,
                anchor="w",
            ).grid(row=0, column=0, sticky="w")

            # Bar
            bar_bg = ctk.CTkFrame(
                row,
                height=14,
                fg_color=BG_SURFACE_ALT,
                corner_radius=3,
            )
            bar_bg.grid(row=0, column=1, sticky="ew", padx=(SP_SM, SP_SM))
            bar_bg.pack_propagate(False)

            bar = ctk.CTkFrame(
                bar_bg,
                fg_color=color,
                corner_radius=3,
                height=14,
            )
            bar.place(relx=0, rely=0, relwidth=max(pct, 0.02), relheight=1.0)

            # Count
            ctk.CTkLabel(
                row,
                text=f"{total:,}",
                font=font_small_bold(),
                text_color=TEXT_SECONDARY,
                width=60,
                anchor="e",
            ).grid(row=0, column=2, sticky="e")

        # Train/Val split bar
        total_train = sum(c["train"] for c in counts.values())
        total_val = sum(c["val"] for c in counts.values())
        if total_ann > 0:
            ctk.CTkLabel(
                self._dataset_body,
                text="Train / Val Split",
                font=font_small_bold(),
                text_color=TEXT_SECONDARY,
                anchor="w",
            ).pack(fill="x", pady=(SP_MD, SP_XS))

            split_bg = ctk.CTkFrame(
                self._dataset_body,
                height=20,
                fg_color=BG_SURFACE_ALT,
                corner_radius=RADIUS_SM,
            )
            split_bg.pack(fill="x")
            split_bg.pack_propagate(False)

            train_pct = total_train / total_ann
            ctk.CTkFrame(
                split_bg,
                fg_color=CLR_BLUE,
                corner_radius=RADIUS_SM,
                height=20,
            ).place(relx=0, rely=0, relwidth=train_pct, relheight=1.0)

            ctk.CTkLabel(
                self._dataset_body,
                text=f"Train: {total_train:,} ({train_pct * 100:.0f}%)  "
                f"Val: {total_val:,} ({(1 - train_pct) * 100:.0f}%)",
                font=font_small(),
                text_color=TEXT_MUTED,
                anchor="w",
            ).pack(fill="x", pady=(SP_XS, 0))

    # -- Model comparison --

    def _refresh_models(self):
        for w in self._models_body.winfo_children():
            w.destroy()

        models_dir = PROJECT_ROOT / "bot_logic" / "models"
        models = []
        if models_dir.exists():
            for d in sorted(models_dir.iterdir()):
                weights = d / "weights" / "best.pt"
                if not weights.exists():
                    continue
                info = {
                    "name": d.name,
                    "size_mb": weights.stat().st_size / (1024 * 1024),
                    "date": datetime.fromtimestamp(weights.stat().st_mtime),
                }
                # Try reading mAP from results.csv
                csv_path = d / "results.csv"
                if csv_path.exists():
                    rows = self._read_results_csv(csv_path)
                    if rows:
                        info["epochs"] = len(rows)
                        info["mAP50"] = max(r.get("metrics/mAP50(B)", 0) for r in rows)
                models.append(info)

        if not models:
            ctk.CTkLabel(
                self._models_body,
                text="No trained models found",
                font=font_body(),
                text_color=TEXT_MUTED,
            ).pack(pady=SP_LG)
            return

        # Header
        hdr = ctk.CTkFrame(
            self._models_body, fg_color=BG_SURFACE_ALT, corner_radius=RADIUS_SM
        )
        hdr.pack(fill="x")
        for txt, w in [
            ("Model", 0),
            ("Size", 65),
            ("Epochs", 55),
            ("mAP50", 65),
            ("Date", 90),
        ]:
            ctk.CTkLabel(
                hdr,
                text=txt,
                width=w,
                font=font_small_bold(),
                text_color=TEXT_MUTED,
                anchor="w" if txt == "Model" else "center",
            ).pack(
                side="left",
                padx=SP_XS,
                pady=SP_XS,
                expand=(w == 0),
                fill="x" if w == 0 else "none",
            )

        # Rows (newest first)
        for m in reversed(models):
            is_latest = m == models[-1]
            row = ctk.CTkFrame(
                self._models_body,
                fg_color="transparent",
                corner_radius=RADIUS_SM,
            )
            row.pack(fill="x", pady=1)
            if is_latest:
                row.configure(fg_color=BG_ACTIVE)

            name_text = m["name"]
            if is_latest:
                name_text += "  (active)"

            for val, w, bold, color in [
                (
                    name_text,
                    0,
                    is_latest,
                    TEXT_PRIMARY if is_latest else TEXT_SECONDARY,
                ),
                (f"{m['size_mb']:.1f} MB", 65, False, TEXT_SECONDARY),
                (str(m.get("epochs", "-")), 55, False, TEXT_SECONDARY),
                (
                    f"{m.get('mAP50', 0):.3f}" if "mAP50" in m else "-",
                    65,
                    is_latest,
                    SUCCESS if is_latest else TEXT_SECONDARY,
                ),
                (m["date"].strftime("%b %d"), 90, False, TEXT_MUTED),
            ]:
                ctk.CTkLabel(
                    row,
                    text=val,
                    width=w,
                    font=font_body_bold() if bold else font_small(),
                    text_color=color,
                    anchor="w" if w == 0 else "center",
                ).pack(
                    side="left",
                    padx=SP_XS,
                    pady=SP_XS,
                    expand=(w == 0),
                    fill="x" if w == 0 else "none",
                )

    # -- Session history --

    def _refresh_sessions(self):
        for w in self._sessions_body.winfo_children():
            w.destroy()

        sessions_dir = PROJECT_ROOT / "recordings" / "sessions"
        sessions = []
        if sessions_dir.exists():
            for d in sorted(sessions_dir.iterdir(), reverse=True):
                manifest = d / "session.json"
                if not manifest.exists():
                    continue
                info = {"name": d.name, "path": d}
                try:
                    meta = json.loads(manifest.read_text())
                    info["game"] = meta.get("game", "Unknown")
                    info["resolution"] = meta.get("resolution", "")
                except Exception:
                    info["game"] = "Unknown"
                    info["resolution"] = ""

                frames = list(d.glob("frame_*.png"))
                info["frames"] = len(frames)
                info["date"] = datetime.fromtimestamp(d.stat().st_mtime)
                sessions.append(info)

        if not sessions:
            ctk.CTkLabel(
                self._sessions_body,
                text="No recorded sessions",
                font=font_body(),
                text_color=TEXT_MUTED,
            ).pack(pady=SP_LG)
            return

        # Header
        hdr = ctk.CTkFrame(
            self._sessions_body, fg_color=BG_SURFACE_ALT, corner_radius=RADIUS_SM
        )
        hdr.pack(fill="x")
        for txt, w in [("Session", 0), ("Game", 80), ("Frames", 55), ("Date", 90)]:
            ctk.CTkLabel(
                hdr,
                text=txt,
                width=w,
                font=font_small_bold(),
                text_color=TEXT_MUTED,
                anchor="w" if txt == "Session" else "center",
            ).pack(
                side="left",
                padx=SP_XS,
                pady=SP_XS,
                expand=(w == 0),
                fill="x" if w == 0 else "none",
            )

        # Rows (limit to 10)
        for s in sessions[:10]:
            row = ctk.CTkFrame(
                self._sessions_body, fg_color="transparent", corner_radius=RADIUS_SM
            )
            row.pack(fill="x", pady=1)

            for val, w, color in [
                (s["name"], 0, TEXT_SECONDARY),
                (s["game"], 80, TEXT_SECONDARY),
                (str(s["frames"]), 55, CLR_GREEN if s["frames"] > 50 else TEXT_MUTED),
                (s["date"].strftime("%b %d, %H:%M"), 90, TEXT_MUTED),
            ]:
                ctk.CTkLabel(
                    row,
                    text=val,
                    width=w,
                    font=font_small(),
                    text_color=color,
                    anchor="w" if w == 0 else "center",
                ).pack(
                    side="left",
                    padx=SP_XS,
                    pady=SP_XS,
                    expand=(w == 0),
                    fill="x" if w == 0 else "none",
                )

    # -- System status --

    def _refresh_system(self):
        for w in self._system_body.winfo_children():
            w.destroy()

        self._system_body.grid_columnconfigure(0, weight=1)
        self._system_body.grid_columnconfigure(1, weight=1)
        self._system_body.grid_columnconfigure(2, weight=1)

        # Packages
        pkg_frame = ctk.CTkFrame(self._system_body, fg_color="transparent")
        pkg_frame.grid(row=0, column=0, sticky="nsew", padx=SP_XS)

        ctk.CTkLabel(
            pkg_frame,
            text="Packages",
            font=font_small_bold(),
            text_color=TEXT_SECONDARY,
            anchor="w",
        ).pack(fill="x", pady=(0, SP_XS))

        packages = [
            ("customtkinter", "customtkinter"),
            ("ultralytics", "ultralytics"),
            ("torch", "torch"),
            ("google-genai", "google.genai"),
            ("anthropic", "anthropic"),
            ("pynput", "pynput"),
            ("mss", "mss"),
            ("pyyaml", "yaml"),
        ]
        for pip_name, import_name in packages:
            try:
                __import__(import_name)
                icon, color = ICON_CHECK, SUCCESS
            except ImportError:
                icon, color = ICON_CROSS, TEXT_MUTED
            ctk.CTkLabel(
                pkg_frame,
                text=f" {icon}  {pip_name}",
                font=font_small(),
                text_color=color,
                anchor="w",
            ).pack(fill="x", padx=SP_XS, pady=0)

        # Disk usage
        disk_frame = ctk.CTkFrame(self._system_body, fg_color="transparent")
        disk_frame.grid(row=0, column=1, sticky="nsew", padx=SP_XS)

        ctk.CTkLabel(
            disk_frame,
            text="Disk Usage",
            font=font_small_bold(),
            text_color=TEXT_SECONDARY,
            anchor="w",
        ).pack(fill="x", pady=(0, SP_XS))

        dirs_to_check = [
            ("Dataset", PROJECT_ROOT / "yolo_dataset"),
            ("Models", PROJECT_ROOT / "bot_logic" / "models"),
            ("Screenshots", PROJECT_ROOT / "screenshots"),
            ("Sessions", PROJECT_ROOT / "recordings" / "sessions"),
        ]
        for label, path in dirs_to_check:
            size = self._dir_size(path)
            ctk.CTkLabel(
                disk_frame,
                text=f" {label}: {self._fmt_size(size)}",
                font=font_small(),
                text_color=TEXT_SECONDARY,
                anchor="w",
            ).pack(fill="x", padx=SP_XS, pady=0)

        # System tools
        tools_frame = ctk.CTkFrame(self._system_body, fg_color="transparent")
        tools_frame.grid(row=0, column=2, sticky="nsew", padx=SP_XS)

        ctk.CTkLabel(
            tools_frame,
            text="System Tools",
            font=font_small_bold(),
            text_color=TEXT_SECONDARY,
            anchor="w",
        ).pack(fill="x", pady=(0, SP_XS))

        for tool in ["wine", "flameshot", "gcc", "git", "pip"]:
            found = shutil.which(tool)
            icon, color = (ICON_CHECK, SUCCESS) if found else (ICON_CROSS, TEXT_MUTED)
            ctk.CTkLabel(
                tools_frame,
                text=f" {icon}  {tool}",
                font=font_small(),
                text_color=color,
                anchor="w",
            ).pack(fill="x", padx=SP_XS, pady=0)

    # ==================================================================
    # Helpers
    # ==================================================================

    def _find_latest_model(self):
        """Find the most recently modified model with best.pt."""
        models_dir = PROJECT_ROOT / "bot_logic" / "models"
        if not models_dir.exists():
            return None
        best = None
        for d in models_dir.iterdir():
            w = d / "weights" / "best.pt"
            if w.exists():
                info = {
                    "name": d.name,
                    "path": w,
                    "size_mb": w.stat().st_size / (1024 * 1024),
                    "mtime": w.stat().st_mtime,
                }
                csv_path = d / "results.csv"
                if csv_path.exists():
                    rows = self._read_results_csv(csv_path)
                    if rows:
                        info["mAP50"] = max(r.get("metrics/mAP50(B)", 0) for r in rows)
                if best is None or info["mtime"] > best["mtime"]:
                    best = info
        return best

    def _find_latest_results_csv(self):
        """Find the most recent results.csv across all model dirs."""
        models_dir = PROJECT_ROOT / "bot_logic" / "models"
        if not models_dir.exists():
            return None
        csvs = sorted(models_dir.glob("*/results.csv"), key=lambda p: p.stat().st_mtime)
        return csvs[-1] if csvs else None

    def _read_results_csv(self, path):
        """Read Ultralytics results.csv into list of dicts with float values."""
        rows = []
        try:
            with open(path) as f:
                reader = csv.DictReader(f)
                for r in reader:
                    cleaned = {}
                    for k, v in r.items():
                        k = k.strip()
                        try:
                            cleaned[k] = float(v.strip())
                        except (ValueError, AttributeError):
                            cleaned[k] = v
                    rows.append(cleaned)
        except Exception:
            pass
        return rows

    def _read_class_names(self):
        """Read class names from dataset.yaml."""
        import yaml as _yaml

        yaml_path = PROJECT_ROOT / "yolo_dataset" / "dataset.yaml"
        if not yaml_path.exists():
            return {}
        try:
            with open(yaml_path) as f:
                cfg = _yaml.safe_load(f) or {}
            raw = cfg.get("names", {})
            if isinstance(raw, list):
                return {i: v for i, v in enumerate(raw)}
            if isinstance(raw, dict):
                return {int(k): v for k, v in raw.items()}
        except Exception:
            pass
        return {}

    def _dir_size(self, path):
        """Get total size of a directory in bytes."""
        if not path.exists():
            return 0
        total = 0
        try:
            for f in path.rglob("*"):
                if f.is_file():
                    total += f.stat().st_size
        except Exception:
            pass
        return total

    def _fmt_size(self, size_bytes):
        """Format bytes to human-readable string."""
        if size_bytes == 0:
            return "0 B"
        for unit in ("B", "KB", "MB", "GB"):
            if size_bytes < 1024:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f} TB"
