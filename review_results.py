"""
Review Results & Annotation Tool

Provides two Toplevel windows:

1. AnnotationWindow  -- opened from the "Annotate" sidebar button.
   Loads screenshots directly from screenshots/ and any existing YOLO
   labels from yolo_dataset/train/labels/.  Lets the user create, edit,
   and delete bounding-box annotations, then save them to YOLO format.

2. ReviewResultsWindow -- opened from the "Review Results" sidebar button
   after model inference.  Shows model predictions with confidence scores
   and lets the user correct them before saving.

Both share the same core drawing, editing, and saving logic.

Keyboard shortcuts (when the window is focused):
   Left / Right   -- previous / next image
   Delete          -- delete selected annotation
   Escape          -- cancel draw mode
   Ctrl+S          -- save corrections to dataset
"""

import customtkinter as ctk
import tkinter as tk
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageTk
import copy
import yaml
from functools import lru_cache

PROJECT_ROOT = Path(__file__).parent

# Distinct colours for class labels (up to 20, then cycles)
_PALETTE = [
    "#FF3838",
    "#FF9D97",
    "#FF701F",
    "#FFB21D",
    "#CFD231",
    "#48F90A",
    "#92CC17",
    "#3DDB86",
    "#1A9334",
    "#00D4BB",
    "#2C99A8",
    "#00C2FF",
    "#344593",
    "#6473FF",
    "#0018EC",
    "#8438FF",
    "#520085",
    "#CB38FF",
    "#FF95C8",
    "#FF37C7",
]


def _color_for_class(class_id):
    return _PALETTE[class_id % len(_PALETTE)]


def _hex_to_rgba(hex_color, alpha=60):
    """Convert '#RRGGBB' to (R, G, B, A) tuple."""
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return (r, g, b, alpha)


# ------------------------------------------------------------------
# Module-level font cache (loaded once, reused everywhere)
# ------------------------------------------------------------------
_FONT_CACHE = {}


def _get_font(size=15):
    """Return a cached font at the given size."""
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf",
    ]
    font = None
    for path in candidates:
        try:
            font = ImageFont.truetype(path, size)
            break
        except (OSError, IOError):
            continue
    if font is None:
        font = ImageFont.load_default()
    _FONT_CACHE[size] = font
    return font


def _draw_boxes_on_image(pil_img, detections, selected_idx=None, show_conf=True):
    """Return a new RGBA PIL image with coloured bounding boxes, labels, and
    semi-transparent fills drawn on top."""
    base = pil_img.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw_overlay = ImageDraw.Draw(overlay)
    draw_base = ImageDraw.Draw(base)

    font = _get_font(15)
    font_small = _get_font(12)

    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det["bbox"]
        color_hex = _color_for_class(det["class_id"])
        is_selected = i == selected_idx
        width = 3 if is_selected else 2

        # Semi-transparent fill
        fill_alpha = 45 if is_selected else 25
        fill_rgba = _hex_to_rgba(color_hex, fill_alpha)
        draw_overlay.rectangle([x1, y1, x2, y2], fill=fill_rgba)

        # Solid border
        draw_base.rectangle([x1, y1, x2, y2], outline=color_hex, width=width)

        # Selection corners
        if is_selected:
            cs = 6
            for cx, cy in [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]:
                draw_base.rectangle(
                    [cx - cs, cy - cs, cx + cs, cy + cs], fill=color_hex
                )

        # Label text
        conf_str = ""
        if show_conf and "confidence" in det and det["confidence"] < 1.0:
            conf_str = f" {det['confidence']:.0%}"
        label = f"{det['class_name']}{conf_str}"

        use_font = font if is_selected else font_small
        bbox_text = draw_base.textbbox((0, 0), label, font=use_font)
        tw = bbox_text[2] - bbox_text[0]
        th = bbox_text[3] - bbox_text[1]

        pad_x, pad_y = 6, 3
        lx = x1
        ly = max(y1 - th - pad_y * 2 - 2, 0)
        draw_base.rectangle(
            [lx, ly, lx + tw + pad_x * 2, ly + th + pad_y * 2],
            fill=color_hex,
        )
        draw_base.text((lx + pad_x, ly + pad_y), label, fill="white", font=use_font)

    result = Image.alpha_composite(base, overlay)
    return result.convert("RGB")


def _load_class_names_from_yaml(yolo_dataset_path):
    """Read class names from dataset.yaml.  Returns dict {id: name}."""
    yaml_path = Path(yolo_dataset_path) / "dataset.yaml"
    if not yaml_path.exists():
        return {}
    try:
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)
        names = data.get("names", {})
        return {int(k): v for k, v in names.items()}
    except Exception:
        return {}


def _load_yolo_labels(label_file, img_w, img_h, class_names):
    """Parse a YOLO label .txt file and return list of detection dicts."""
    detections = []
    if not Path(label_file).exists():
        return detections
    try:
        text = Path(label_file).read_text().strip()
        if not text:
            return detections
        for line in text.splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            class_id = int(parts[0])
            xc, yc, w, h = (
                float(parts[1]),
                float(parts[2]),
                float(parts[3]),
                float(parts[4]),
            )
            x1 = (xc - w / 2) * img_w
            y1 = (yc - h / 2) * img_h
            x2 = (xc + w / 2) * img_w
            y2 = (yc + h / 2) * img_h
            detections.append(
                {
                    "class_id": class_id,
                    "class_name": class_names.get(class_id, f"class_{class_id}"),
                    "confidence": 1.0,
                    "bbox": (x1, y1, x2, y2),
                    "bbox_norm": (xc, yc, w, h),
                }
            )
    except Exception:
        pass
    return detections


def _build_entries_from_screenshots(screenshots_dir, yolo_dataset_path):
    """Scan screenshots/ and load any existing YOLO labels.  Returns a
    list of entry dicts compatible with the editor.
    """
    screenshots_path = Path(screenshots_dir)
    yolo_path = Path(yolo_dataset_path)
    labels_dir = yolo_path / "train" / "labels"

    class_names = _load_class_names_from_yaml(yolo_path)

    images = sorted(
        list(screenshots_path.glob("*.png")) + list(screenshots_path.glob("*.jpg"))
    )

    entries = []
    for img_path in images:
        try:
            with Image.open(img_path) as im:
                img_w, img_h = im.size
        except Exception:
            continue

        label_file = labels_dir / (img_path.stem + ".txt")
        dets = _load_yolo_labels(label_file, img_w, img_h, class_names)

        entries.append(
            {
                "image_path": str(img_path.resolve()),
                "image_name": img_path.name,
                "image_width": img_w,
                "image_height": img_h,
                "detections": dets,
            }
        )

    return entries, class_names


# ======================================================================
# Base Editor Window (shared by Annotate and Review Results)
# ======================================================================


class _BaseEditorWindow(ctk.CTkToplevel):
    """Core annotation/review editor.  Subclassed by AnnotationWindow
    and ReviewResultsWindow."""

    def __init__(
        self,
        parent,
        entries,
        class_names,
        yolo_dataset_path,
        title="Annotation Editor",
        show_conf=True,
    ):
        super().__init__(parent)
        self.title(title)
        self.geometry("1200x800")
        self.minsize(900, 600)

        self._entries = entries or []
        self._class_names = class_names or {}
        self._yolo_dataset_path = Path(yolo_dataset_path)
        self._show_conf = show_conf

        # Working copy
        self._edited = [copy.deepcopy(e) for e in self._entries]
        self._dirty = set()

        self._current_idx = None
        self._prev_selected_idx = None  # for targeted card updates
        self._selected_det = None
        self._tk_images = {}

        # Image caches
        self._pil_cache = {}  # idx -> PIL.Image (full size, max 5)
        self._thumb_cache = {}  # idx -> ImageTk.PhotoImage
        self._PIL_CACHE_MAX = 5

        # Draw state
        self._drawing_box = False
        self._draw_start = None
        self._display_scale = 1.0
        self._display_offset = (0, 0)

        # Resize debounce
        self._resize_job = None

        # Track sidebar card widgets for targeted updates
        self._card_widgets = {}  # idx -> card frame widget

        self._build_ui()
        self._build_sidebar_list()
        self._bind_shortcuts()

        # Auto-select first image
        if self._edited:
            self.after(200, lambda: self._select_image(0))

    # ------------------------------------------------------------------
    # Keyboard shortcuts
    # ------------------------------------------------------------------

    def _bind_shortcuts(self):
        self.bind("<Left>", lambda e: self._prev_image())
        self.bind("<Right>", lambda e: self._next_image())
        self.bind("<Delete>", lambda e: self._delete_selected())
        self.bind("<Escape>", lambda e: self._cancel_draw())
        self.bind("<Control-s>", lambda e: self._save_corrections())

    def _cancel_draw(self):
        if self._drawing_box:
            self._drawing_box = False
            self._draw_start = None
            self._canvas.configure(cursor="")
            self._render_detail()

    # ------------------------------------------------------------------
    # UI layout
    # ------------------------------------------------------------------

    def _build_ui(self):
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # -- Left: thumbnail sidebar --
        left = ctk.CTkFrame(self, width=250)
        left.grid(row=0, column=0, sticky="nsew", padx=(6, 3), pady=6)
        left.grid_rowconfigure(1, weight=1)

        header = ctk.CTkFrame(left, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=4, pady=(4, 2))
        ctk.CTkLabel(
            header, text="Images", font=ctk.CTkFont(size=15, weight="bold")
        ).pack(side="left", padx=4)
        self._img_count_lbl = ctk.CTkLabel(
            header,
            text=f"({len(self._entries)})",
            font=ctk.CTkFont(size=12),
            text_color="gray",
        )
        self._img_count_lbl.pack(side="left")

        self._thumb_scroll = ctk.CTkScrollableFrame(left, width=230)
        self._thumb_scroll.grid(row=1, column=0, sticky="nsew", padx=2, pady=2)

        # -- Right: detail + controls --
        right = ctk.CTkFrame(self)
        right.grid(row=0, column=1, sticky="nsew", padx=(3, 6), pady=6)
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(1, weight=1)

        # Toolbar
        toolbar = ctk.CTkFrame(right, fg_color="transparent")
        toolbar.grid(row=0, column=0, sticky="ew", padx=8, pady=(6, 2))

        self._detail_title = ctk.CTkLabel(
            toolbar,
            text="Select an image to begin",
            font=ctk.CTkFont(size=14, weight="bold"),
        )
        self._detail_title.pack(side="left")

        nav = ctk.CTkFrame(toolbar, fg_color="transparent")
        nav.pack(side="right")
        ctk.CTkButton(nav, text="< Prev", width=70, command=self._prev_image).pack(
            side="left", padx=2
        )
        ctk.CTkButton(nav, text="Next >", width=70, command=self._next_image).pack(
            side="left", padx=2
        )

        # Canvas
        self._canvas_frame = ctk.CTkFrame(right)
        self._canvas_frame.grid(row=1, column=0, sticky="nsew", padx=8, pady=4)
        self._canvas_frame.grid_columnconfigure(0, weight=1)
        self._canvas_frame.grid_rowconfigure(0, weight=1)

        self._canvas = tk.Canvas(self._canvas_frame, bg="#1a1a1a", highlightthickness=0)
        self._canvas.grid(row=0, column=0, sticky="nsew")
        self._canvas.bind("<Button-1>", self._on_canvas_click)
        self._canvas.bind("<B1-Motion>", self._on_canvas_drag)
        self._canvas.bind("<ButtonRelease-1>", self._on_canvas_release)
        self._canvas.bind("<Configure>", self._on_canvas_resize)

        # -- Bottom panel --
        bottom = ctk.CTkFrame(right)
        bottom.grid(row=2, column=0, sticky="ew", padx=8, pady=(2, 6))
        bottom.grid_columnconfigure(0, weight=1)

        det_header = ctk.CTkFrame(bottom, fg_color="transparent")
        det_header.grid(row=0, column=0, sticky="ew")
        ctk.CTkLabel(
            det_header, text="Annotations", font=ctk.CTkFont(size=13, weight="bold")
        ).pack(side="left", padx=4)
        self._det_count_lbl = ctk.CTkLabel(
            det_header, text="", font=ctk.CTkFont(size=11), text_color="gray"
        )
        self._det_count_lbl.pack(side="left")

        self._det_scroll = ctk.CTkScrollableFrame(bottom, height=120)
        self._det_scroll.grid(row=1, column=0, sticky="ew", pady=(2, 4))

        btn_row = ctk.CTkFrame(bottom, fg_color="transparent")
        btn_row.grid(row=2, column=0, sticky="ew", pady=(0, 4))

        ctk.CTkButton(
            btn_row,
            text="Delete Selected",
            width=130,
            fg_color="#C62828",
            hover_color="#B71C1C",
            command=self._delete_selected,
        ).pack(side="left", padx=4)

        ctk.CTkButton(
            btn_row,
            text="Edit Label...",
            width=110,
            command=self._edit_label,
        ).pack(side="left", padx=4)

        ctk.CTkButton(
            btn_row,
            text="Draw Box",
            width=120,
            fg_color="#2E7D32",
            hover_color="#1B5E20",
            command=self._start_add_box,
        ).pack(side="left", padx=4)

        self._mode_lbl = ctk.CTkLabel(
            btn_row, text="", font=ctk.CTkFont(size=11), text_color="gray"
        )
        self._mode_lbl.pack(side="right", padx=(4, 8))

        save_frame = ctk.CTkFrame(bottom, fg_color="transparent")
        save_frame.grid(row=3, column=0, sticky="ew", pady=(2, 0))

        self._save_btn = ctk.CTkButton(
            save_frame,
            text="Save to Dataset  (Ctrl+S)",
            width=220,
            fg_color="#1565C0",
            hover_color="#0D47A1",
            command=self._save_corrections,
        )
        self._save_btn.pack(side="left", padx=4)

        self._save_status = ctk.CTkLabel(
            save_frame, text="", font=ctk.CTkFont(size=11), text_color="gray"
        )
        self._save_status.pack(side="left", padx=8)

    # ------------------------------------------------------------------
    # Sidebar: lightweight text rows (built once, updated in-place)
    # ------------------------------------------------------------------

    def _build_sidebar_list(self):
        """Create one lightweight row per image.  No thumbnails loaded
        up-front — the selected image's thumbnail is shown in the detail
        canvas instead."""
        self._card_widgets = {}
        for i, entry in enumerate(self._edited):
            self._create_sidebar_row(i, entry)

    def _create_sidebar_row(self, idx, entry):
        """Create a single compact sidebar row (text only, no image load)."""
        row = ctk.CTkFrame(self._thumb_scroll, cursor="hand2", height=30)
        row.pack(fill="x", padx=2, pady=1)
        row.pack_propagate(False)

        # Detection count badge
        n_det = len(entry["detections"])
        badge_color = "#4CAF50" if n_det > 0 else "gray50"
        badge = ctk.CTkLabel(
            row,
            text=str(n_det),
            font=ctk.CTkFont(size=10, weight="bold"),
            fg_color=badge_color,
            corner_radius=6,
            width=26,
            height=20,
        )
        badge.pack(side="right", padx=(2, 6), pady=4)

        # Dirty indicator
        dirty_lbl = ctk.CTkLabel(
            row,
            text="*" if idx in self._dirty else "",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color="orange",
            width=12,
        )
        dirty_lbl.pack(side="right", padx=0, pady=4)

        # Filename
        name = entry["image_name"]
        if len(name) > 32:
            name = name[:29] + "..."
        name_lbl = ctk.CTkLabel(row, text=name, font=ctk.CTkFont(size=11), anchor="w")
        name_lbl.pack(side="left", padx=(6, 4), pady=4, fill="x", expand=True)

        # Click binding on all children
        for widget in (row, badge, dirty_lbl, name_lbl):
            widget.bind("<Button-1>", lambda e, i=idx: self._select_image(i))

        # Store references for targeted updates
        self._card_widgets[idx] = {
            "row": row,
            "badge": badge,
            "dirty": dirty_lbl,
            "name": name_lbl,
        }
        self._update_card_highlight(idx)

    def _update_card_highlight(self, idx):
        """Update a single card's active/inactive appearance."""
        if idx not in self._card_widgets:
            return
        card = self._card_widgets[idx]
        is_active = idx == self._current_idx
        if is_active:
            card["row"].configure(fg_color=("#3B8ED0", "#2A6496"), border_width=0)
            card["name"].configure(text_color="white")
        else:
            card["row"].configure(fg_color="transparent", border_width=0)
            card["name"].configure(text_color=("gray10", "gray90"))

    def _update_card_badge(self, idx):
        """Update badge count and dirty indicator for one card."""
        if idx not in self._card_widgets:
            return
        card = self._card_widgets[idx]
        entry = self._edited[idx]
        n_det = len(entry["detections"])
        badge_color = "#4CAF50" if n_det > 0 else "gray50"
        card["badge"].configure(text=str(n_det), fg_color=badge_color)
        card["dirty"].configure(text="*" if idx in self._dirty else "")

    # ------------------------------------------------------------------
    # PIL image cache (LRU, max 5 full-size images in memory)
    # ------------------------------------------------------------------

    def _get_pil_image(self, idx):
        """Return cached PIL image for entry at idx, or load from disk."""
        if idx in self._pil_cache:
            return self._pil_cache[idx]

        # Evict oldest if cache is full
        if len(self._pil_cache) >= self._PIL_CACHE_MAX:
            oldest = next(iter(self._pil_cache))
            del self._pil_cache[oldest]

        try:
            img = Image.open(self._edited[idx]["image_path"])
            img.load()  # force read from disk now, so file handle is released
            self._pil_cache[idx] = img
            return img
        except Exception:
            return None

    def _invalidate_pil_cache(self, idx):
        """Remove a specific entry from the PIL cache."""
        self._pil_cache.pop(idx, None)

    # ------------------------------------------------------------------
    # Image detail
    # ------------------------------------------------------------------

    def _select_image(self, idx):
        prev = self._current_idx
        self._current_idx = idx
        self._selected_det = None
        self._drawing_box = False
        self._draw_start = None
        self._canvas.configure(cursor="")

        # Update only the two affected sidebar cards (old and new)
        if prev is not None:
            self._update_card_highlight(prev)
        self._update_card_highlight(idx)

        self._render_detail()
        self._render_detection_list()

    def _render_detail(self):
        if self._current_idx is None:
            return

        entry = self._edited[self._current_idx]
        self._detail_title.configure(
            text=f"{entry['image_name']}  ({entry['image_width']}x{entry['image_height']})"
        )

        pil_img = self._get_pil_image(self._current_idx)
        if pil_img is None:
            self._canvas.delete("all")
            self._canvas.create_text(
                200,
                100,
                text="Failed to load image",
                fill="red",
                font=("sans-serif", 14),
            )
            return

        annotated = _draw_boxes_on_image(
            pil_img, entry["detections"], self._selected_det, self._show_conf
        )

        # Scale to canvas
        cw = self._canvas.winfo_width() or 800
        ch = self._canvas.winfo_height() or 500
        scale = min(cw / annotated.width, ch / annotated.height, 1.0)
        new_w = max(int(annotated.width * scale), 1)
        new_h = max(int(annotated.height * scale), 1)

        self._display_scale = scale
        self._display_offset = ((cw - new_w) // 2, (ch - new_h) // 2)

        resized = annotated.resize((new_w, new_h), Image.LANCZOS)
        tk_img = ImageTk.PhotoImage(resized)
        self._tk_images["detail"] = tk_img

        self._canvas.delete("all")
        ox, oy = self._display_offset
        self._canvas.create_image(ox, oy, image=tk_img, anchor="nw")

        if self._drawing_box:
            self._canvas.create_text(
                cw // 2,
                18,
                text="DRAW MODE  --  click and drag to create a box  (Esc to cancel)",
                fill="#4CAF50",
                font=("sans-serif", 12, "bold"),
            )

        # Update mode label
        if self._selected_det is not None and self._selected_det < len(
            entry["detections"]
        ):
            det = entry["detections"][self._selected_det]
            self._mode_lbl.configure(
                text=f"Selected: {det['class_name']}  |  "
                f"[{det['bbox'][0]:.0f},{det['bbox'][1]:.0f} - "
                f"{det['bbox'][2]:.0f},{det['bbox'][3]:.0f}]"
            )
        else:
            self._mode_lbl.configure(text="")

    def _on_canvas_resize(self, event):
        """Debounced resize — waits 150ms after last resize event."""
        if self._resize_job is not None:
            self.after_cancel(self._resize_job)
        self._resize_job = self.after(150, self._do_resize_render)

    def _do_resize_render(self):
        self._resize_job = None
        if self._current_idx is not None:
            self._render_detail()

    # ------------------------------------------------------------------
    # Detection list
    # ------------------------------------------------------------------

    def _render_detection_list(self):
        for w in self._det_scroll.winfo_children():
            w.destroy()

        if self._current_idx is None:
            return

        dets = self._edited[self._current_idx]["detections"]
        self._det_count_lbl.configure(text=f"({len(dets)})")

        for i, det in enumerate(dets):
            row = ctk.CTkFrame(self._det_scroll, fg_color="transparent")
            row.pack(fill="x", padx=2, pady=1)

            color = _color_for_class(det["class_id"])
            swatch = ctk.CTkLabel(
                row, text="", width=14, height=14, fg_color=color, corner_radius=3
            )
            swatch.pack(side="left", padx=(4, 6))

            conf_str = ""
            if self._show_conf and det.get("confidence", 1.0) < 1.0:
                conf_str = f"  ({det['confidence']:.0%})"
            text = f"{det['class_name']}{conf_str}"
            lbl = ctk.CTkLabel(row, text=text, font=ctk.CTkFont(size=12))
            lbl.pack(side="left")

            for widget in (row, swatch, lbl):
                widget.bind("<Button-1>", lambda e, idx=i: self._select_detection(idx))

            if i == self._selected_det:
                row.configure(fg_color=("gray80", "gray30"))

            x1, y1, x2, y2 = det["bbox"]
            coord = f"[{x1:.0f},{y1:.0f} - {x2:.0f},{y2:.0f}]"
            ctk.CTkLabel(
                row, text=coord, font=ctk.CTkFont(size=10), text_color="gray"
            ).pack(side="right", padx=4)

    def _select_detection(self, idx):
        self._selected_det = idx
        self._render_detail()
        self._render_detection_list()

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def _prev_image(self):
        if not self._edited:
            return
        if self._current_idx is None:
            self._select_image(0)
        else:
            self._select_image((self._current_idx - 1) % len(self._edited))

    def _next_image(self):
        if not self._edited:
            return
        if self._current_idx is None:
            self._select_image(0)
        else:
            self._select_image((self._current_idx + 1) % len(self._edited))

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def _delete_selected(self):
        if self._current_idx is None or self._selected_det is None:
            return
        dets = self._edited[self._current_idx]["detections"]
        if 0 <= self._selected_det < len(dets):
            dets.pop(self._selected_det)
            self._dirty.add(self._current_idx)
            self._selected_det = None
            self._render_detail()
            self._render_detection_list()
            self._update_card_badge(self._current_idx)

    # ------------------------------------------------------------------
    # Edit label
    # ------------------------------------------------------------------

    def _edit_label(self):
        if self._current_idx is None or self._selected_det is None:
            return
        det = self._edited[self._current_idx]["detections"][self._selected_det]
        EditLabelDialog(self, det, self._class_names, callback=self._on_label_edited)

    def _on_label_edited(self, det, new_class_id, new_class_name):
        det["class_id"] = new_class_id
        det["class_name"] = new_class_name
        self._dirty.add(self._current_idx)
        self._render_detail()
        self._render_detection_list()
        self._update_card_badge(self._current_idx)

    # ------------------------------------------------------------------
    # Draw new box
    # ------------------------------------------------------------------

    def _start_add_box(self):
        if self._current_idx is None:
            return
        self._drawing_box = True
        self._draw_start = None
        self._canvas.configure(cursor="crosshair")
        self._render_detail()

    def _canvas_to_image_coords(self, cx, cy):
        ox, oy = self._display_offset
        scale = self._display_scale
        if scale == 0:
            return cx, cy
        return (cx - ox) / scale, (cy - oy) / scale

    def _on_canvas_click(self, event):
        if not self._drawing_box or self._current_idx is None:
            return
        self._draw_start = (event.x, event.y)

    def _on_canvas_drag(self, event):
        if not self._drawing_box or self._draw_start is None:
            return
        self._canvas.delete("rubberband")
        x0, y0 = self._draw_start
        self._canvas.create_rectangle(
            x0,
            y0,
            event.x,
            event.y,
            outline="#4CAF50",
            width=2,
            dash=(5, 3),
            tags="rubberband",
        )

    def _on_canvas_release(self, event):
        if (
            not self._drawing_box
            or self._draw_start is None
            or self._current_idx is None
        ):
            return

        x0, y0 = self._draw_start
        ix0, iy0 = self._canvas_to_image_coords(x0, y0)
        ix1, iy1 = self._canvas_to_image_coords(event.x, event.y)

        bx1, bx2 = min(ix0, ix1), max(ix0, ix1)
        by1, by2 = min(iy0, iy1), max(iy0, iy1)

        if (bx2 - bx1) < 5 or (by2 - by1) < 5:
            self._drawing_box = False
            self._draw_start = None
            self._canvas.configure(cursor="")
            self._render_detail()
            return

        entry = self._edited[self._current_idx]
        bx1 = max(0.0, bx1)
        by1 = max(0.0, by1)
        bx2 = min(float(entry["image_width"]), bx2)
        by2 = min(float(entry["image_height"]), by2)

        self._drawing_box = False
        self._draw_start = None
        self._canvas.configure(cursor="")

        NewBoxDialog(
            self,
            self._class_names,
            callback=lambda cid, cname: self._finish_add_box(
                bx1, by1, bx2, by2, cid, cname
            ),
        )

    def _finish_add_box(self, x1, y1, x2, y2, class_id, class_name):
        entry = self._edited[self._current_idx]
        img_w = entry["image_width"]
        img_h = entry["image_height"]

        xc = ((x1 + x2) / 2) / img_w
        yc = ((y1 + y2) / 2) / img_h
        w = (x2 - x1) / img_w
        h = (y2 - y1) / img_h

        new_det = {
            "class_id": class_id,
            "class_name": class_name,
            "confidence": 1.0,
            "bbox": (x1, y1, x2, y2),
            "bbox_norm": (xc, yc, w, h),
        }
        entry["detections"].append(new_det)
        self._dirty.add(self._current_idx)
        self._selected_det = len(entry["detections"]) - 1
        self._render_detail()
        self._render_detection_list()
        self._update_card_badge(self._current_idx)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def _save_corrections(self):
        if not self._dirty:
            self._save_status.configure(text="No changes to save.", text_color="gray")
            return

        labels_dir = self._yolo_dataset_path / "train" / "labels"
        labels_dir.mkdir(parents=True, exist_ok=True)

        images_dir = self._yolo_dataset_path / "train" / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        saved_count = 0
        saved_indices = list(self._dirty)
        for idx in sorted(saved_indices):
            entry = self._edited[idx]
            stem = Path(entry["image_name"]).stem
            label_file = labels_dir / f"{stem}.txt"

            img_w = entry["image_width"]
            img_h = entry["image_height"]

            lines = []
            for det in entry["detections"]:
                x1, y1, x2, y2 = det["bbox"]
                xc = ((x1 + x2) / 2) / img_w
                yc = ((y1 + y2) / 2) / img_h
                w = (x2 - x1) / img_w
                h = (y2 - y1) / img_h
                lines.append(f"{det['class_id']} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

            label_file.write_text("\n".join(lines) + ("\n" if lines else ""))

            dest_img = images_dir / entry["image_name"]
            if not dest_img.exists():
                src = Path(entry["image_path"])
                if src.exists():
                    import shutil

                    shutil.copy2(src, dest_img)

            saved_count += 1

        self._update_dataset_yaml()

        self._dirty.clear()
        self._save_status.configure(
            text=f"Saved {saved_count} label(s) to {labels_dir}", text_color="#4CAF50"
        )
        # Update only the cards that were dirty
        for idx in saved_indices:
            self._update_card_badge(idx)

    def _update_dataset_yaml(self):
        """Write/update dataset.yaml with current class names."""
        yaml_path = self._yolo_dataset_path / "dataset.yaml"
        yaml_content = {
            "names": {int(k): v for k, v in self._class_names.items()},
            "path": str(self._yolo_dataset_path.resolve()),
            "train": str((self._yolo_dataset_path / "train" / "images").resolve()),
            "val": str((self._yolo_dataset_path / "val" / "images").resolve()),
        }
        try:
            with open(yaml_path, "w") as f:
                yaml.dump(yaml_content, f, default_flow_style=False)
        except Exception:
            pass


# ======================================================================
# AnnotationWindow -- for creating annotations from scratch
# ======================================================================


class AnnotationWindow(_BaseEditorWindow):
    """Opened from the 'Annotate' sidebar button.  Loads screenshots from
    screenshots/ and any existing labels from yolo_dataset/train/labels/.
    No model required.
    """

    def __init__(self, parent, screenshots_dir=None, yolo_dataset_path=None):
        ss_dir = screenshots_dir or str(PROJECT_ROOT / "screenshots")
        yolo_dir = yolo_dataset_path or str(PROJECT_ROOT / "yolo_dataset")

        entries, class_names = _build_entries_from_screenshots(ss_dir, yolo_dir)

        super().__init__(
            parent,
            entries=entries,
            class_names=class_names,
            yolo_dataset_path=yolo_dir,
            title="Annotation Tool - Create & Edit Labels",
            show_conf=False,
        )


# ======================================================================
# ReviewResultsWindow -- for reviewing model predictions
# ======================================================================


class ReviewResultsWindow(_BaseEditorWindow):
    """Opened after model inference.  Shows predictions with confidence
    and lets the user correct them.
    """

    def __init__(self, parent, results, class_names=None, yolo_dataset_path=None):
        yolo_dir = yolo_dataset_path or str(PROJECT_ROOT / "yolo_dataset")

        super().__init__(
            parent,
            entries=results or [],
            class_names=class_names or {},
            yolo_dataset_path=yolo_dir,
            title="Review Results - Model Predictions",
            show_conf=True,
        )


# ======================================================================
# Edit Label Dialog
# ======================================================================


class EditLabelDialog(ctk.CTkToplevel):
    """Dialog to pick a new class for a detection."""

    def __init__(self, parent, detection, class_names, callback):
        super().__init__(parent)
        self.title("Edit Label")
        self.geometry("340x420")
        self.resizable(False, False)
        self.transient(parent)

        self._detection = detection
        self._class_names = class_names
        self._callback = callback

        self._build()
        self.after(100, self._try_grab)

    def _try_grab(self):
        try:
            self.grab_set()
        except Exception:
            self.after(100, self._try_grab)

    def _build(self):
        ctk.CTkLabel(
            self, text="Select new class:", font=ctk.CTkFont(size=14, weight="bold")
        ).pack(padx=16, pady=(16, 8))

        ctk.CTkLabel(
            self,
            text=f"Current: {self._detection['class_name']}  (id {self._detection['class_id']})",
            font=ctk.CTkFont(size=12),
            text_color="gray",
        ).pack(padx=16, pady=(0, 8))

        custom_frame = ctk.CTkFrame(self, fg_color="transparent")
        custom_frame.pack(fill="x", padx=16, pady=(0, 4))
        ctk.CTkLabel(custom_frame, text="Or type new name:").pack(side="left")
        self._custom_var = ctk.StringVar()
        ctk.CTkEntry(custom_frame, textvariable=self._custom_var, width=140).pack(
            side="right"
        )

        scroll = ctk.CTkScrollableFrame(self, height=220)
        scroll.pack(fill="both", expand=True, padx=16, pady=4)

        self._selected_id = tk.IntVar(value=self._detection["class_id"])

        if self._class_names:
            for cid in sorted(self._class_names.keys()):
                cname = self._class_names[cid]
                color = _color_for_class(cid)
                row = ctk.CTkFrame(scroll, fg_color="transparent")
                row.pack(fill="x", pady=1)
                ctk.CTkRadioButton(
                    row,
                    text=f"{cid}: {cname}",
                    variable=self._selected_id,
                    value=cid,
                    font=ctk.CTkFont(size=12),
                ).pack(side="left")
                ctk.CTkLabel(
                    row, text="", width=12, height=12, fg_color=color, corner_radius=2
                ).pack(side="right", padx=4)
        else:
            ctk.CTkLabel(
                scroll,
                text="No classes defined yet.\nType a name above.",
                text_color="gray",
            ).pack(pady=10)

        btns = ctk.CTkFrame(self, fg_color="transparent")
        btns.pack(fill="x", padx=16, pady=(8, 12))
        ctk.CTkButton(btns, text="Apply", width=100, command=self._apply).pack(
            side="left", padx=4
        )
        ctk.CTkButton(
            btns, text="Cancel", width=80, fg_color="gray", command=self.destroy
        ).pack(side="right", padx=4)

    def _apply(self):
        custom = self._custom_var.get().strip()
        if custom:
            for cid, cname in self._class_names.items():
                if cname.lower() == custom.lower():
                    self._callback(self._detection, cid, cname)
                    self.destroy()
                    return
            max_id = max(self._class_names.keys()) if self._class_names else -1
            new_id = max_id + 1
            self._class_names[new_id] = custom
            self._callback(self._detection, new_id, custom)
        else:
            cid = self._selected_id.get()
            cname = self._class_names.get(cid, f"class_{cid}")
            self._callback(self._detection, cid, cname)
        self.destroy()


# ======================================================================
# New Box Dialog
# ======================================================================


class NewBoxDialog(ctk.CTkToplevel):
    """Pick class for a newly drawn bounding box."""

    def __init__(self, parent, class_names, callback):
        super().__init__(parent)
        self.title("New Annotation - Select Class")
        self.geometry("320x400")
        self.resizable(False, False)
        self.transient(parent)

        self._class_names = class_names
        self._callback = callback

        self._build()
        self.after(100, self._try_grab)

    def _try_grab(self):
        try:
            self.grab_set()
        except Exception:
            self.after(100, self._try_grab)

    def _build(self):
        ctk.CTkLabel(
            self,
            text="Select class for new box:",
            font=ctk.CTkFont(size=14, weight="bold"),
        ).pack(padx=16, pady=(16, 8))

        custom_frame = ctk.CTkFrame(self, fg_color="transparent")
        custom_frame.pack(fill="x", padx=16, pady=(0, 6))
        ctk.CTkLabel(custom_frame, text="Or type name:").pack(side="left")
        self._custom_var = ctk.StringVar()
        ctk.CTkEntry(custom_frame, textvariable=self._custom_var, width=150).pack(
            side="right"
        )

        scroll = ctk.CTkScrollableFrame(self, height=210)
        scroll.pack(fill="both", expand=True, padx=16, pady=4)

        self._selected_id = tk.IntVar(value=0)

        if self._class_names:
            for cid in sorted(self._class_names.keys()):
                cname = self._class_names[cid]
                color = _color_for_class(cid)
                row = ctk.CTkFrame(scroll, fg_color="transparent")
                row.pack(fill="x", pady=1)
                ctk.CTkRadioButton(
                    row,
                    text=f"{cid}: {cname}",
                    variable=self._selected_id,
                    value=cid,
                    font=ctk.CTkFont(size=12),
                ).pack(side="left")
                ctk.CTkLabel(
                    row, text="", width=12, height=12, fg_color=color, corner_radius=2
                ).pack(side="right", padx=4)
        else:
            ctk.CTkLabel(
                scroll,
                text="No classes defined yet.\nType a name above.",
                text_color="gray",
            ).pack(pady=10)

        btns = ctk.CTkFrame(self, fg_color="transparent")
        btns.pack(fill="x", padx=16, pady=(8, 12))
        ctk.CTkButton(
            btns, text="Add", width=100, fg_color="#2E7D32", command=self._apply
        ).pack(side="left", padx=4)
        ctk.CTkButton(
            btns, text="Cancel", width=80, fg_color="gray", command=self.destroy
        ).pack(side="right", padx=4)

    def _apply(self):
        custom = self._custom_var.get().strip()
        if custom:
            for cid, cname in self._class_names.items():
                if cname.lower() == custom.lower():
                    self._callback(cid, cname)
                    self.destroy()
                    return
            max_id = max(self._class_names.keys()) if self._class_names else -1
            new_id = max_id + 1
            self._class_names[new_id] = custom
            self._callback(new_id, custom)
        else:
            cid = self._selected_id.get()
            cname = self._class_names.get(cid, f"class_{cid}")
            self._callback(cid, cname)
        self.destroy()
