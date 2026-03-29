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
import json
import logging
import shutil
import time as _time
import yaml

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

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
    except Exception as e:
        print(f"Warning: could not parse {yaml_path}: {e}")
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
    except Exception as e:
        print(f"Warning: could not parse label file {label_file}: {e}")
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
        self._saved_to_train = set()  # indices saved to train/ (persists across saves)

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

        # Annotation speed tracking
        self._image_opened_at = None  # time.monotonic() when current image selected
        self._session_time = 0.0  # seconds spent this session
        self._session_images = 0  # images visited this session
        self._stats_file = PROJECT_ROOT / ".annotation_stats.json"
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # -- Review queue state (Describe & Review mode) --
        self._review_mode = False
        self._review_queue: list = []  # indices into self._edited
        self._review_pos = 0  # current position in queue
        self._review_session = None  # ReviewSession being built
        self._review_image_opened_at = None  # timing per review image
        self._generated_detections_snapshot: list = []  # LLM output before corrections
        self._human_detections_snapshot: list = []  # human boxes before LLM

        # -- Batch state --
        self._review_phase = "describe"  # "describe" → "review"
        self._batch_descriptions: dict = {}  # idx → description text
        self._batch_human_dets: dict = {}  # idx → human detections snapshot
        self._batch_results: dict = {}  # idx → DescribeResult (after batch LLM call)
        self._batch_size = 10  # images per LLM call

        # -- Parent template state --
        self._parent_idx = None  # index of the parent/template image
        self._parent_mode = False  # when True, clicking sidebar applies parent boxes

        # -- Correction tracking (snapshot of detections at load time) --
        # Keyed by idx. Stores a deep copy of detections when the image was
        # first loaded, so _save_corrections() can diff against it and record
        # review feedback even outside of describe-and-review mode.
        self._load_snapshot: dict = {}  # idx → List[Dict] (detections at load)
        for idx, entry in enumerate(self._edited):
            self._load_snapshot[idx] = copy.deepcopy(entry.get("detections", []))

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

        # Virtualized sidebar — canvas + scrollbar, no per-row widgets
        sidebar_container = ctk.CTkFrame(left)
        sidebar_container.grid(row=1, column=0, sticky="nsew", padx=2, pady=2)
        sidebar_container.grid_columnconfigure(0, weight=1)
        sidebar_container.grid_rowconfigure(0, weight=1)

        self._sidebar_canvas = tk.Canvas(
            sidebar_container, bg="#1a1a1a", highlightthickness=0, borderwidth=0,
        )
        self._sidebar_canvas.grid(row=0, column=0, sticky="nsew")

        self._sidebar_sb = ctk.CTkScrollbar(
            sidebar_container, command=self._sidebar_canvas.yview,
        )
        self._sidebar_sb.grid(row=0, column=1, sticky="ns")
        self._sidebar_canvas.configure(yscrollcommand=self._sidebar_sb.set)

        self._ROW_HEIGHT = 28
        self._sidebar_canvas.bind("<Button-1>", self._on_sidebar_click)
        self._sidebar_canvas.bind("<Configure>", lambda e: self._redraw_sidebar())
        self._sidebar_canvas.bind("<MouseWheel>", self._on_sidebar_mousewheel)
        self._sidebar_canvas.bind("<Button-4>", self._on_sidebar_mousewheel)
        self._sidebar_canvas.bind("<Button-5>", self._on_sidebar_mousewheel)

        # Keep legacy name for compatibility (unused but prevents AttributeError)
        self._thumb_scroll = sidebar_container

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

        # Description mode toggle (hidden by default, enabled in AnnotationWindow)
        self._desc_mode = False
        self._desc_toggle_btn = ctk.CTkButton(
            nav, text="Description Mode", width=130,
            fg_color="#6A1B9A", hover_color="#4A148C",
            font=ctk.CTkFont(size=11),
            command=self._toggle_description_mode,
        )
        # Not packed yet — subclass calls _enable_description_mode() to show it

        # Parent template buttons
        self._parent_btn = ctk.CTkButton(
            nav, text="Set Parent", width=100,
            fg_color="#0E7490", hover_color="#0C6380",
            font=ctk.CTkFont(size=11),
            command=self._toggle_parent_mode,
        )
        self._parent_btn.pack(side="left", padx=(0, 4))

        self._apply_parent_btn = ctk.CTkButton(
            nav, text="Apply Parent", width=100,
            fg_color="#E65100", hover_color="#BF360C",
            font=ctk.CTkFont(size=11),
            command=self._apply_parent_to_current,
        )
        # Hidden until parent mode is active

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

        self._export_val_btn = ctk.CTkButton(
            save_frame,
            text="Export to Valid",
            width=150,
            fg_color="#6A1B9A",
            hover_color="#4A148C",
            command=self._export_to_valid,
        )
        self._export_val_btn.pack(side="left", padx=4)

        self._save_status = ctk.CTkLabel(
            save_frame, text="", font=ctk.CTkFont(size=11), text_color="gray"
        )
        self._save_status.pack(side="left", padx=8)

        # Store reference to bbox bottom panel for mode toggling
        self._bbox_panel = bottom

        # -- Description mode panel (hidden by default, same grid cell) --
        self._desc_panel = ctk.CTkFrame(right)
        self._desc_panel.grid_columnconfigure(0, weight=1)
        self._desc_panel.grid_rowconfigure(1, weight=1)
        # Not gridded yet — _toggle_description_mode() swaps it in

        desc_header = ctk.CTkFrame(self._desc_panel, fg_color="transparent")
        desc_header.grid(row=0, column=0, sticky="ew")
        ctk.CTkLabel(
            desc_header, text="Describe what you see",
            font=ctk.CTkFont(size=13, weight="bold"),
        ).pack(side="left", padx=4)

        self._desc_status_lbl = ctk.CTkLabel(
            desc_header, text="experimental",
            font=ctk.CTkFont(size=10), text_color="#AB47BC",
        )
        self._desc_status_lbl.pack(side="left", padx=8)

        self._extract_btn = ctk.CTkButton(
            desc_header, text="Extract  (Ctrl+Enter)", width=160, height=28,
            fg_color="#7B1FA2", hover_color="#6A1B9A",
            font=ctk.CTkFont(size=11),
            command=self._extract_description,
        )
        self._extract_btn.pack(side="right", padx=4)

        # Generate Boxes button (describe-to-bbox mode) — same row, left of Extract
        self._gen_boxes_btn = ctk.CTkButton(
            desc_header, text="Generate Boxes", width=140, height=28,
            fg_color="#0E7490", hover_color="#0C6380",
            font=ctk.CTkFont(size=11),
            command=self._generate_boxes_from_description,
        )
        self._gen_boxes_btn.pack(side="right", padx=4)

        self._desc_text = ctk.CTkTextbox(self._desc_panel, height=80, wrap="word")
        self._desc_text.grid(row=1, column=0, sticky="nsew", padx=2, pady=(2, 4))

        # Placeholder text
        self._desc_placeholder = "Describe scene here..."
        self._desc_has_placeholder = True
        self._desc_text.insert("0.0", self._desc_placeholder)
        self._desc_text.configure(text_color="gray")
        self._desc_text.bind("<FocusIn>", self._desc_on_focus_in)
        self._desc_text.bind("<FocusOut>", self._desc_on_focus_out)

        # Structured output preview (read-only, shows LLM extraction result)
        self._desc_result_frame = ctk.CTkFrame(self._desc_panel)
        self._desc_result_frame.grid(row=2, column=0, sticky="ew", padx=2, pady=(0, 2))
        self._desc_result_frame.grid_columnconfigure(0, weight=1)

        self._desc_result_text = ctk.CTkTextbox(
            self._desc_result_frame, height=60, wrap="word", state="disabled",
        )
        self._desc_result_text.grid(row=0, column=0, sticky="ew")

        desc_save_row = ctk.CTkFrame(self._desc_panel, fg_color="transparent")
        desc_save_row.grid(row=3, column=0, sticky="ew", pady=(2, 0))

        self._desc_save_btn = ctk.CTkButton(
            desc_save_row, text="Save Scene Context", width=180, height=28,
            fg_color="#1565C0", hover_color="#0D47A1",
            command=self._save_scene_context,
        )
        self._desc_save_btn.pack(side="left", padx=4)

        self._desc_save_status = ctk.CTkLabel(
            desc_save_row, text="", font=ctk.CTkFont(size=11), text_color="gray",
        )
        self._desc_save_status.pack(side="left", padx=8)

        # -- Review queue controls (row 4, hidden until review mode activated) --
        self._review_controls = ctk.CTkFrame(self._desc_panel, fg_color="transparent")
        # Not gridded yet — _start_review_queue() shows it

        self._review_progress_lbl = ctk.CTkLabel(
            self._review_controls, text="0 / 0",
            font=ctk.CTkFont(size=13, weight="bold"), text_color="#AB47BC",
        )
        self._review_progress_lbl.pack(side="left", padx=8)

        self._accept_btn = ctk.CTkButton(
            self._review_controls, text="Accept & Next", width=130, height=28,
            fg_color="#2E7D32", hover_color="#1B5E20",
            font=ctk.CTkFont(size=11),
            command=self._accept_review,
        )
        self._accept_btn.pack(side="left", padx=4)

        self._skip_btn = ctk.CTkButton(
            self._review_controls, text="Skip", width=70, height=28,
            fg_color="#455A64", hover_color="#37474F",
            font=ctk.CTkFont(size=11),
            command=self._skip_review,
        )
        self._skip_btn.pack(side="left", padx=4)

        # Process Batch button — sends accumulated descriptions to LLM
        self._batch_btn = ctk.CTkButton(
            self._review_controls, text="Process Batch", width=130, height=28,
            fg_color="#E65100", hover_color="#BF360C",
            font=ctk.CTkFont(size=11, weight="bold"),
            command=self._process_batch,
        )
        self._batch_btn.pack(side="left", padx=8)

        self._review_status_lbl = ctk.CTkLabel(
            self._review_controls, text="", font=ctk.CTkFont(size=11), text_color="gray",
        )
        self._review_status_lbl.pack(side="left", padx=8)

    # ------------------------------------------------------------------
    # Description text — placeholder behavior
    # ------------------------------------------------------------------

    def _desc_on_focus_in(self, event=None):
        """Clear placeholder when user clicks into the text box."""
        if self._desc_has_placeholder:
            self._desc_text.delete("0.0", "end")
            self._desc_text.configure(text_color="white")
            self._desc_has_placeholder = False

    def _desc_on_focus_out(self, event=None):
        """Restore placeholder if text box is empty."""
        text = self._desc_text.get("0.0", "end").strip()
        if not text:
            self._desc_text.insert("0.0", self._desc_placeholder)
            self._desc_text.configure(text_color="gray")
            self._desc_has_placeholder = True

    def _desc_get_text(self) -> str:
        """Get description text, ignoring placeholder."""
        if self._desc_has_placeholder:
            return ""
        return self._desc_text.get("0.0", "end").strip()

    def _desc_set_text(self, text: str):
        """Set description text, clearing placeholder."""
        self._desc_text.delete("0.0", "end")
        if text:
            self._desc_text.insert("0.0", text)
            self._desc_text.configure(text_color="white")
            self._desc_has_placeholder = False
        else:
            self._desc_text.insert("0.0", self._desc_placeholder)
            self._desc_text.configure(text_color="gray")
            self._desc_has_placeholder = True

    # ------------------------------------------------------------------
    # Parent template — set a parent image, click sidebar to apply boxes
    # ------------------------------------------------------------------

    def _toggle_parent_mode(self):
        """Toggle parent template mode.

        First click: sets current image as parent, shows Apply button.
        User browses images freely, clicks Apply Parent to stamp boxes.
        Click Set Parent again to deactivate.
        """
        if not self._parent_mode:
            if self._current_idx is None:
                return
            self._parent_idx = self._current_idx
            self._parent_mode = True
            parent_name = self._edited[self._parent_idx]["image_name"]
            n_dets = len(self._edited[self._parent_idx].get("detections", []))
            self._parent_btn.configure(
                text="Parent: ON", fg_color="#2E7D32", hover_color="#1B5E20",
            )
            self._apply_parent_btn.pack(side="left", padx=(0, 8))
            self._detail_title.configure(
                text=f"PARENT: {parent_name} ({n_dets} boxes) — browse & Apply"
            )
        else:
            self._parent_mode = False
            self._parent_idx = None
            self._parent_btn.configure(
                text="Set Parent", fg_color="#0E7490", hover_color="#0C6380",
            )
            self._apply_parent_btn.pack_forget()
            if self._current_idx is not None:
                entry = self._edited[self._current_idx]
                self._detail_title.configure(
                    text=f"{entry['image_name']}  ({entry['image_width']}x{entry['image_height']})"
                )

    def _apply_parent_to_current(self):
        """Apply parent's annotations to the currently viewed image."""
        if self._parent_idx is None or self._current_idx is None:
            return
        if self._current_idx == self._parent_idx:
            return

        self._apply_parent_to(self._current_idx)

        # Re-render to show the applied boxes
        self._render_detail()
        self._render_detection_list()

    def _apply_parent_to(self, target_idx):
        """Copy the parent image's annotations to the target image.

        Straight copy. User can then nudge boxes manually.
        """
        if self._parent_idx is None:
            return
        if target_idx == self._parent_idx:
            return

        parent_entry = self._edited[self._parent_idx]
        target_entry = self._edited[target_idx]

        target_entry["detections"] = copy.deepcopy(parent_entry["detections"])
        self._dirty.add(target_idx)

        # Also copy description if in review mode
        if self._review_mode:
            parent_desc = self._batch_descriptions.get(self._parent_idx, "")
            if parent_desc:
                self._batch_descriptions[target_idx] = parent_desc

    # ------------------------------------------------------------------
    # Description mode — toggle, extract, save
    # ------------------------------------------------------------------

    def _enable_description_mode_toggle(self):
        """Show the description mode toggle button in the toolbar.
        Called by AnnotationWindow after super().__init__."""
        self._desc_toggle_btn.pack(side="left", padx=(0, 8))

    def _toggle_description_mode(self):
        """Swap between bbox annotation and description annotation panels."""
        self._desc_mode = not self._desc_mode
        if self._desc_mode:
            self._bbox_panel.grid_remove()
            self._desc_panel.grid(row=2, column=0, sticky="nsew", padx=8, pady=(2, 6))
            self._desc_toggle_btn.configure(
                text="Bbox Mode", fg_color="#2E7D32", hover_color="#1B5E20",
            )
            # Load existing description for current image's scene
            self._load_scene_description()
        else:
            self._desc_panel.grid_remove()
            self._bbox_panel.grid(row=2, column=0, sticky="ew", padx=8, pady=(2, 6))
            self._desc_toggle_btn.configure(
                text="Description Mode", fg_color="#6A1B9A", hover_color="#4A148C",
            )

    def _get_current_scene_name(self) -> str:
        """Derive a scene identifier for the current image.
        Uses session frame metadata if available, otherwise falls back to filename."""
        if self._current_idx is None:
            return ""
        entry = self._edited[self._current_idx]
        img_path = Path(entry["image_path"])

        # Check for companion .json (session frames have frame_NNNNNN.json)
        json_path = img_path.with_suffix(".json")
        if json_path.exists():
            try:
                import json as _json
                data = _json.loads(json_path.read_text())
                scene = data.get("state", {}).get("scene_name", "")
                if scene:
                    return scene
            except Exception:
                pass

        # Fallback: use parent directory name as scene proxy
        return img_path.parent.name

    def _load_scene_description(self):
        """Load existing scene context into the description text box."""
        # Don't auto-load scene descriptions during review mode
        if self._review_mode:
            return
        scene = self._get_current_scene_name()
        if not scene:
            return
        try:
            from pipeline.scene_context import SceneContextStore
            store = SceneContextStore("Cuphead")
            ctx = store.get(scene)
            if ctx and ctx.raw_description:
                self._desc_set_text(ctx.raw_description)
                # Show structured preview
                self._show_extraction_result(ctx.to_prompt_block())
                self._desc_save_status.configure(text=f"Scene: {scene}", text_color="gray")
            else:
                self._desc_save_status.configure(text=f"Scene: {scene} (no context yet)", text_color="gray")
        except Exception:
            pass

    def _extract_description(self):
        """Send description text + current image to LLM for structured extraction."""
        if self._current_idx is None:
            return
        raw_text = self._desc_get_text()
        if not raw_text:
            self._desc_status_lbl.configure(text="write a description first", text_color="#f85149")
            return

        entry = self._edited[self._current_idx]
        image_path = entry["image_path"]
        scene = self._get_current_scene_name()

        self._extract_btn.configure(state="disabled", text="Extracting...")
        self._desc_status_lbl.configure(text="sending to LLM...", text_color="#AB47BC")

        import threading

        def _worker():
            try:
                from pipeline.scene_context import extract_scene_context
                ctx = extract_scene_context(raw_text, image_path, scene)
                self.after(0, lambda: self._on_extraction_done(ctx))
            except NotImplementedError:
                self.after(0, lambda: self._on_extraction_error(
                    "extract_scene_context() not yet implemented — see pipeline/scene_context.py"
                ))
            except Exception as exc:
                self.after(0, lambda: self._on_extraction_error(str(exc)))

        threading.Thread(target=_worker, daemon=True).start()

    def _on_extraction_done(self, ctx):
        """Handle successful LLM extraction."""
        self._extract_btn.configure(state="normal", text="Extract  (Ctrl+Enter)")
        self._desc_status_lbl.configure(text="extracted", text_color="#3fb950")
        self._show_extraction_result(ctx.to_prompt_block())
        # Stash the extracted context for saving
        self._extracted_ctx = ctx

    def _on_extraction_error(self, msg):
        """Handle failed LLM extraction."""
        self._extract_btn.configure(state="normal", text="Extract  (Ctrl+Enter)")
        self._desc_status_lbl.configure(text=f"error: {msg[:60]}", text_color="#f85149")

    def _show_extraction_result(self, text: str):
        """Display structured extraction in the preview textbox."""
        self._desc_result_text.configure(state="normal")
        self._desc_result_text.delete("0.0", "end")
        self._desc_result_text.insert("0.0", text)
        self._desc_result_text.configure(state="disabled")

    def _save_scene_context(self):
        """Save extracted context to the scene context store."""
        ctx = getattr(self, "_extracted_ctx", None)
        if ctx is None:
            self._desc_save_status.configure(text="extract first", text_color="#f85149")
            return
        # Preserve the raw description from the text box
        ctx.raw_description = self._desc_get_text()
        # Set sample image
        if self._current_idx is not None:
            ctx.sample_image = self._edited[self._current_idx]["image_path"]

        try:
            from pipeline.scene_context import SceneContextStore
            store = SceneContextStore("Cuphead")
            store.set(ctx)
            store.save()
            scene = ctx.scene_name or self._get_current_scene_name()
            self._desc_save_status.configure(
                text=f"Saved context for {scene}", text_color="#3fb950",
            )
        except Exception as exc:
            self._desc_save_status.configure(
                text=f"save failed: {str(exc)[:50]}", text_color="#f85149",
            )

    # ------------------------------------------------------------------
    # Describe & Review — review queue + describe-to-bbox
    # ------------------------------------------------------------------

    def _start_review_queue(self, count: int = 100, shuffle: bool = True):
        """Initialize a review queue of N images.

        Two-phase flow:
        1. DESCRIBE phase: user goes through images, writes descriptions +
           draws rough boxes, clicks "Accept & Next" to store and advance.
           No LLM calls — just accumulating descriptions.
        2. After describing a batch (or all), click "Process Batch" to send
           accumulated images + descriptions to LLM in one call.
        3. REVIEW phase: user reviews side-by-side comparisons (human vs LLM),
           accepts refined annotations or corrects them.
        """
        import random

        from pipeline.review_feedback import ReviewSession

        indices = list(range(len(self._edited)))
        if shuffle:
            random.shuffle(indices)
        self._review_queue = indices[:count]
        self._review_pos = 0
        self._review_mode = True
        self._review_phase = "describe"  # start in describe phase

        self._batch_descriptions = {}
        self._batch_human_dets = {}
        self._batch_results = {}

        self._review_session = ReviewSession.create(
            total_images=len(self._review_queue),
            source_path=str(self._edited[0]["image_path"]) if self._edited else "",
        )

        # Activate description mode if not already active
        if not self._desc_mode:
            self._toggle_description_mode()

        # Show review controls, hide scene-context save row
        self._review_controls.grid(row=4, column=0, sticky="ew", pady=(4, 0))
        self._desc_save_btn.pack_forget()
        self._desc_save_status.pack_forget()

        # Hide Extract and Generate Boxes in describe phase (no LLM calls)
        self._extract_btn.pack_forget()
        self._gen_boxes_btn.pack_forget()

        self._update_review_progress()

        # Navigate to first queue image
        if self._review_queue:
            self._select_image(self._review_queue[0])
            self._review_image_opened_at = _time.monotonic()

    def _update_review_progress(self):
        """Update the review progress label."""
        pos = self._review_pos + 1
        total = len(self._review_queue)
        described = len(self._batch_descriptions)

        if self._review_phase == "describe":
            self._review_progress_lbl.configure(
                text=f"Describe: {pos} / {total}  ({described} queued)"
            )
        else:
            self._review_progress_lbl.configure(
                text=f"Review: {pos} / {total}"
            )

    def _generate_boxes_from_description(self):
        """Send description + current image to LLM for independent annotation.

        Phase 1: LLM generates its own bboxes from the description + image.
        The human's existing boxes are captured first so both can be compared
        side by side. If human has boxes, also runs phase 2 (critique).
        """
        if self._current_idx is None:
            return
        raw_text = self._desc_get_text()
        if not raw_text:
            self._desc_status_lbl.configure(
                text="write a description first", text_color="#f85149"
            )
            return

        entry = self._edited[self._current_idx]
        image_path = entry["image_path"]

        # Capture current human annotations before LLM generates
        self._human_detections_snapshot = copy.deepcopy(entry.get("detections", []))

        self._gen_boxes_btn.configure(state="disabled", text="Generating...")
        self._desc_status_lbl.configure(text="Phase 1: LLM annotating...", text_color="#0E7490")

        import threading

        def _worker():
            try:
                from pipeline.describe_annotator import (
                    critique_annotations,
                    describe_to_bboxes,
                    render_comparison,
                )

                # Phase 1: LLM generates its own boxes
                result = describe_to_bboxes(
                    description=raw_text,
                    image_path=image_path,
                    class_names=self._class_names,
                )

                # Render side-by-side comparison
                human_dets = self._human_detections_snapshot
                comparison_img = render_comparison(
                    image_path, human_dets, result.generated_detections, self._class_names
                )

                # Phase 2: Critique (if human has annotations)
                critique = None
                if human_dets:
                    self.after(0, lambda: self._desc_status_lbl.configure(
                        text="Phase 2: critiquing...", text_color="#AB47BC"
                    ))
                    critique = critique_annotations(
                        image_path=image_path,
                        description=raw_text,
                        human_detections=human_dets,
                        llm_detections=result.generated_detections,
                        class_names=self._class_names,
                    )

                self.after(0, lambda: self._on_describe_boxes_done(
                    result, comparison_img, critique
                ))
            except Exception as exc:
                self.after(0, lambda: self._on_describe_boxes_error(str(exc)))

        threading.Thread(target=_worker, daemon=True).start()

    def _on_describe_boxes_done(self, result, comparison_img, critique):
        """Handle completed annotation + critique pipeline.

        Shows side-by-side comparison image and critique text.
        If critique produced refined detections, uses those. Otherwise
        uses the LLM's phase 1 detections.
        """
        self._gen_boxes_btn.configure(state="normal", text="Generate Boxes")

        n_llm = len(result.generated_detections)
        n_human = len(self._human_detections_snapshot)

        # Save comparison image to temp and show in preview
        import tempfile
        comp_path = Path(tempfile.gettempdir()) / "annotation_comparison.jpg"
        comparison_img.save(str(comp_path), quality=95)
        self._comparison_image_path = str(comp_path)

        # Build status + preview text
        if critique and critique.critique_text:
            status = (
                f"Human: {n_human} | LLM: {n_llm} | "
                f"Refined: {len(critique.refined_detections)} "
                f"({result.generation_time_sec + critique.generation_time_sec:.1f}s)"
            )
            preview = f"CRITIQUE:\n{critique.critique_text}\n\nComparison saved: {comp_path}"
            final_dets = critique.refined_detections
        else:
            status = (
                f"Human: {n_human} | LLM: {n_llm} "
                f"({result.generation_time_sec:.1f}s)"
            )
            preview = f"LLM generated {n_llm} boxes.\nComparison saved: {comp_path}"
            final_dets = result.generated_detections

        self._desc_status_lbl.configure(text=status, text_color="#3fb950")
        self._show_extraction_result(preview)

        # Snapshot LLM output for feedback tracking
        self._generated_detections_snapshot = copy.deepcopy(result.generated_detections)

        # Inject refined/LLM detections into current entry
        if self._current_idx is not None:
            entry = self._edited[self._current_idx]
            entry["detections"] = copy.deepcopy(final_dets)
            self._dirty.add(self._current_idx)
            self._render_detail()
            self._render_detection_list()

    def _on_describe_boxes_error(self, msg):
        """Handle failed describe-to-bbox generation."""
        self._gen_boxes_btn.configure(state="normal", text="Generate Boxes")
        self._desc_status_lbl.configure(text=f"error: {msg[:60]}", text_color="#f85149")

    def _accept_review(self):
        """Accept current image and advance to next.

        Behavior depends on phase:
        - DESCRIBE phase: Store description + human boxes, advance to next image.
          No LLM call. Accumulates batch for later processing.
        - REVIEW phase: Store final (possibly corrected) annotations, record
          ImageReview feedback, auto-save YOLO labels, advance to next.
        """
        if not self._review_mode or self._current_idx is None:
            return

        idx = self._current_idx
        entry = self._edited[idx]
        description = self._desc_get_text()

        if self._review_phase == "describe":
            # Store description + human annotations for batch processing
            if description:
                self._batch_descriptions[idx] = description
                self._batch_human_dets[idx] = copy.deepcopy(entry.get("detections", []))

            self._review_pos += 1
            if self._review_pos < len(self._review_queue):
                self._show_extraction_result("")
                n = len(self._batch_descriptions)
                self._desc_status_lbl.configure(
                    text=f"{n} described — click Process Batch when ready",
                    text_color="gray",
                )
                self._update_review_progress()
                next_idx = self._review_queue[self._review_pos]
                self._select_image(next_idx)
                self._review_image_opened_at = _time.monotonic()
            else:
                # Reached end — auto-trigger batch if there are descriptions
                if self._batch_descriptions:
                    self._desc_status_lbl.configure(
                        text=f"All described ({len(self._batch_descriptions)}) — processing...",
                        text_color="#E65100",
                    )
                    self.after(200, self._process_batch)
                else:
                    self._finish_review_queue()

        elif self._review_phase == "review":
            from datetime import datetime

            from pipeline.review_feedback import ImageReview

            final_dets = copy.deepcopy(entry.get("detections", []))
            generated_dets = copy.deepcopy(
                self._batch_results.get(idx, {}).get("detections", [])
                if isinstance(self._batch_results.get(idx), dict)
                else (self._batch_results[idx].generated_detections
                      if idx in self._batch_results else [])
            )

            # Determine if corrections were made
            corrections_made = len(final_dets) != len(generated_dets)
            if not corrections_made and final_dets:
                for fd, gd in zip(final_dets, generated_dets):
                    if fd.get("bbox") != gd.get("bbox") or fd.get("class_id") != gd.get("class_id"):
                        corrections_made = True
                        break

            review_time = 0.0
            if self._review_image_opened_at is not None:
                review_time = _time.monotonic() - self._review_image_opened_at

            review = ImageReview(
                image_path=entry["image_path"],
                image_name=entry["image_name"],
                description=self._batch_descriptions.get(idx, ""),
                generated_detections=generated_dets,
                final_detections=final_dets,
                corrections_made=corrections_made,
                timestamp=datetime.now().isoformat(),
                review_time_sec=review_time,
            )

            if self._review_session:
                self._review_session.add_review(review)

            # Auto-save labels
            if idx in self._dirty:
                self._save_corrections()

            self._review_pos += 1
            self._generated_detections_snapshot = []

            if self._review_pos < len(self._review_queue):
                self._update_review_progress()
                next_idx = self._review_queue[self._review_pos]
                self._select_image(next_idx)
                self._review_image_opened_at = _time.monotonic()
                # Load LLM results for this image if available
                self._load_batch_result_for_current()
            else:
                self._finish_review_queue()

    def _skip_review(self):
        """Skip current image without saving, advance to next."""
        if not self._review_mode:
            return

        self._review_pos += 1
        self._generated_detections_snapshot = []

        if self._review_pos < len(self._review_queue):
            self._show_extraction_result("")
            self._desc_status_lbl.configure(text="", text_color="gray")
            self._update_review_progress()
            next_idx = self._review_queue[self._review_pos]
            self._select_image(next_idx)
            self._review_image_opened_at = _time.monotonic()
            if self._review_phase == "review":
                self._load_batch_result_for_current()
        else:
            if self._review_phase == "describe" and self._batch_descriptions:
                self.after(200, self._process_batch)
            else:
                self._finish_review_queue()

    def _process_batch(self):
        """Send all accumulated descriptions + images to LLM in one batch call.

        Transitions from describe phase → review phase. One API call for
        the entire batch instead of one per image.
        """
        if not self._batch_descriptions:
            self._review_status_lbl.configure(
                text="No descriptions to process", text_color="#f85149"
            )
            return

        self._batch_btn.configure(state="disabled", text="Processing...")
        self._accept_btn.configure(state="disabled")
        n = len(self._batch_descriptions)
        self._desc_status_lbl.configure(
            text=f"Batch: sending {n} images to LLM...", text_color="#E65100"
        )

        import threading

        def _worker():
            try:
                from pipeline.describe_annotator import (
                    BatchItem,
                    batch_describe_to_bboxes,
                    render_comparison,
                    should_critique,
                    critique_annotations,
                )

                # Build batch items (only images that have descriptions)
                described_indices = [
                    idx for idx in self._review_queue if idx in self._batch_descriptions
                ]
                items = []
                for idx in described_indices:
                    entry = self._edited[idx]
                    items.append(BatchItem(
                        image_path=entry["image_path"],
                        description=self._batch_descriptions[idx],
                        human_detections=self._batch_human_dets.get(idx, []),
                    ))

                def _log(msg):
                    self.after(0, lambda: self._desc_status_lbl.configure(
                        text=msg, text_color="#E65100"
                    ))

                # Phase 1: Batch LLM call
                results = batch_describe_to_bboxes(
                    items=items,
                    class_names=self._class_names,
                    log_fn=_log,
                )

                # Store results by index
                batch_results = {}
                critiques_needed = 0
                for idx, result in zip(described_indices, results):
                    batch_results[idx] = result
                    human_dets = self._batch_human_dets.get(idx, [])
                    if human_dets and should_critique(human_dets, result.generated_detections):
                        critiques_needed += 1

                # Phase 2: Critique only where needed (skip if boxes agree)
                if critiques_needed > 0:
                    _log(f"Critiquing {critiques_needed} / {len(results)} images...")
                    for idx, result in zip(described_indices, results):
                        human_dets = self._batch_human_dets.get(idx, [])
                        if human_dets and should_critique(human_dets, result.generated_detections):
                            try:
                                critique = critique_annotations(
                                    image_path=result.image_path,
                                    description=result.description,
                                    human_detections=human_dets,
                                    llm_detections=result.generated_detections,
                                    class_names=self._class_names,
                                )
                                # Replace with refined detections
                                if critique.refined_detections:
                                    result.generated_detections = critique.refined_detections
                            except Exception as exc:
                                logger.warning("Critique failed for %s: %s", result.image_path, exc)

                self.after(0, lambda: self._on_batch_done(batch_results, described_indices))
            except Exception as exc:
                self.after(0, lambda: self._on_batch_error(str(exc)))

        threading.Thread(target=_worker, daemon=True).start()

    def _on_batch_done(self, batch_results, described_indices):
        """Handle completed batch processing. Switch to review phase."""
        self._batch_results = batch_results
        self._batch_btn.configure(state="normal", text="Process Batch")
        self._accept_btn.configure(state="normal")

        # Show Generate Boxes button for per-image re-generation if needed
        self._gen_boxes_btn.pack(side="right", padx=4)

        n = len(batch_results)
        self._desc_status_lbl.configure(
            text=f"Batch complete: {n} images annotated", text_color="#3fb950"
        )

        # Switch to review phase — go back to first described image
        self._review_phase = "review"
        self._review_queue = [idx for idx in self._review_queue if idx in described_indices]
        self._review_pos = 0

        self._update_review_progress()

        if self._review_queue:
            first_idx = self._review_queue[0]
            self._select_image(first_idx)
            self._review_image_opened_at = _time.monotonic()
            self._load_batch_result_for_current()

    def _on_batch_error(self, msg):
        """Handle failed batch processing."""
        self._batch_btn.configure(state="normal", text="Process Batch")
        self._accept_btn.configure(state="normal")
        self._desc_status_lbl.configure(text=f"Batch error: {msg[:60]}", text_color="#f85149")

    def _load_batch_result_for_current(self):
        """Load LLM batch result for the current image and show comparison."""
        if self._current_idx is None:
            return

        idx = self._current_idx
        result = self._batch_results.get(idx)
        if not result:
            self._desc_set_text("")
            self._show_extraction_result("(no LLM result for this image)")
            return

        # Show description
        desc = self._batch_descriptions.get(idx, "")
        self._desc_set_text(desc)

        human_dets = self._batch_human_dets.get(idx, [])
        llm_dets = result.generated_detections
        n_human = len(human_dets)
        n_llm = len(llm_dets)

        # Render side-by-side comparison
        try:
            from pipeline.describe_annotator import render_comparison
            import tempfile

            comparison_img = render_comparison(
                result.image_path, human_dets, llm_dets, self._class_names
            )
            comp_path = Path(tempfile.gettempdir()) / f"comparison_{idx}.jpg"
            comparison_img.save(str(comp_path), quality=95)

            preview = (
                f"Human: {n_human} boxes | LLM: {n_llm} boxes\n"
                f"Comparison: {comp_path}"
            )
        except Exception:
            preview = f"Human: {n_human} boxes | LLM: {n_llm} boxes"

        self._show_extraction_result(preview)
        self._desc_status_lbl.configure(
            text=f"H:{n_human} | L:{n_llm} — review & correct",
            text_color="#3fb950",
        )

        # Inject LLM detections into entry for editing
        entry = self._edited[idx]
        self._generated_detections_snapshot = copy.deepcopy(llm_dets)
        entry["detections"] = copy.deepcopy(llm_dets)
        self._dirty.add(idx)
        self._render_detail()
        self._render_detection_list()

    def _finish_review_queue(self):
        """Complete the review session, save ReviewSession to disk, show summary."""
        if self._review_session:
            self._review_session.finish()
            save_path = self._review_session.save()
            summary = self._review_session.summary()
            self._review_status_lbl.configure(
                text=f"Done! Saved to {save_path.name}", text_color="#3fb950",
            )
            self._show_extraction_result(summary)

        self._review_progress_lbl.configure(
            text=f"{len(self._review_queue)} / {len(self._review_queue)}"
        )
        self._accept_btn.configure(state="disabled")
        self._skip_btn.configure(state="disabled")
        self._batch_btn.configure(state="disabled")

    # ------------------------------------------------------------------
    # Virtualized sidebar — draw rows as canvas items, no per-row widgets
    # ------------------------------------------------------------------

    def _build_sidebar_list(self):
        """Configure canvas scroll region for all entries (no widgets created)."""
        self._card_widgets = {}  # kept empty — guard clauses handle this
        total_h = len(self._edited) * self._ROW_HEIGHT
        self._sidebar_canvas.configure(scrollregion=(0, 0, 230, total_h))
        self._redraw_sidebar()

    def _redraw_sidebar(self):
        """Redraw only the visible rows on the sidebar canvas."""
        c = self._sidebar_canvas
        c.delete("all")
        if not self._edited:
            return

        cw = c.winfo_width() or 230
        rh = self._ROW_HEIGHT
        total = len(self._edited)

        # Determine visible range from scroll position
        try:
            top_frac = c.yview()[0]
        except Exception:
            top_frac = 0.0
        top_px = int(top_frac * total * rh)
        vis_h = c.winfo_height() or 400

        first = max(top_px // rh, 0)
        last = min((top_px + vis_h) // rh + 1, total)

        for i in range(first, last):
            y = i * rh
            entry = self._edited[i]
            is_active = i == self._current_idx
            n_det = len(entry["detections"])
            is_dirty = i in self._dirty

            # Background
            bg = "#2A6496" if is_active else "#1a1a1a"
            c.create_rectangle(0, y, cw, y + rh, fill=bg, outline="")

            # Filename
            name = entry["image_name"]
            if len(name) > 30:
                name = name[:27] + "..."
            text_col = "white" if is_active else "#c8c8c8"
            c.create_text(8, y + rh // 2, text=name, fill=text_col,
                          anchor="w", font=("sans-serif", 10))

            # Detection count badge
            badge_col = "#4CAF50" if n_det > 0 else "#666666"
            bx = cw - 22
            by = y + rh // 2
            c.create_oval(bx - 10, by - 9, bx + 10, by + 9, fill=badge_col, outline="")
            c.create_text(bx, by, text=str(n_det), fill="white",
                          font=("sans-serif", 9, "bold"))

            # Dirty indicator
            if is_dirty:
                c.create_text(cw - 40, y + rh // 2, text="*", fill="orange",
                              font=("sans-serif", 12, "bold"))

    def _on_sidebar_click(self, event):
        """Translate canvas click to image index and select it."""
        if not self._edited:
            return
        # Convert canvas event y to scroll-adjusted y
        top_frac = self._sidebar_canvas.yview()[0]
        total_h = len(self._edited) * self._ROW_HEIGHT
        abs_y = event.y + int(top_frac * total_h)
        idx = abs_y // self._ROW_HEIGHT
        if 0 <= idx < len(self._edited):
            self._select_image(idx)

    def _on_sidebar_mousewheel(self, event):
        """Handle mouse wheel scrolling on the sidebar canvas."""
        if event.num == 4:  # Linux scroll up
            self._sidebar_canvas.yview_scroll(-3, "units")
        elif event.num == 5:  # Linux scroll down
            self._sidebar_canvas.yview_scroll(3, "units")
        else:  # Windows/macOS
            self._sidebar_canvas.yview_scroll(-1 * (event.delta // 120), "units")
        self._redraw_sidebar()

    def _update_card_highlight(self, idx):
        """Redraw sidebar to reflect highlight change (cheap — canvas items only)."""
        self._redraw_sidebar()

    def _update_card_badge(self, idx):
        """Redraw sidebar to reflect badge/dirty change."""
        self._redraw_sidebar()

    def _scroll_sidebar_to(self, idx):
        """Ensure the row for idx is visible in the sidebar canvas."""
        if not self._edited:
            return
        total = len(self._edited)
        rh = self._ROW_HEIGHT
        vis_h = self._sidebar_canvas.winfo_height() or 400
        row_top = idx * rh
        row_bot = row_top + rh
        # Current viewport
        top_frac = self._sidebar_canvas.yview()[0]
        top_px = int(top_frac * total * rh)
        bot_px = top_px + vis_h
        if row_top < top_px:
            self._sidebar_canvas.yview_moveto(row_top / (total * rh))
        elif row_bot > bot_px:
            self._sidebar_canvas.yview_moveto((row_bot - vis_h) / (total * rh))

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
        # Auto-save description for the image we're leaving
        if self._review_mode and self._current_idx is not None:
            text = self._desc_get_text()
            if text:
                self._batch_descriptions[self._current_idx] = text

        # Record time spent on previous image
        if self._image_opened_at is not None:
            elapsed = _time.monotonic() - self._image_opened_at
            if elapsed < 600:  # ignore >10min (user was AFK)
                self._session_time += elapsed
                self._session_images += 1
        self._image_opened_at = _time.monotonic()

        prev = self._current_idx
        self._current_idx = idx
        self._selected_det = None
        self._drawing_box = False
        self._draw_start = None
        self._canvas.configure(cursor="")

        # (Parent mode: user clicks "Apply Parent" button manually after previewing)

        # Scroll sidebar to keep selected row visible, then redraw
        self._scroll_sidebar_to(idx)
        self._redraw_sidebar()

        self._render_detail()
        self._render_detection_list()

        # Auto-load description for the image we're arriving at
        if self._review_mode and self._desc_mode:
            saved_desc = self._batch_descriptions.get(idx, "")
            self._desc_set_text(saved_desc)

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
    # Annotation speed tracking
    # ------------------------------------------------------------------

    def _on_close(self):
        """Flush timing stats to disk, then destroy."""
        # Record final image
        if self._image_opened_at is not None:
            elapsed = _time.monotonic() - self._image_opened_at
            if elapsed < 600:
                self._session_time += elapsed
                self._session_images += 1
        self._save_annotation_stats()
        self.destroy()

    def _save_annotation_stats(self):
        """Append session totals to the cumulative stats file."""
        if self._session_images == 0:
            return
        try:
            stats = (
                json.loads(self._stats_file.read_text())
                if self._stats_file.exists()
                else {}
            )
        except Exception:
            stats = {}
        stats["total_time_sec"] = stats.get("total_time_sec", 0.0) + self._session_time
        stats["images_viewed"] = stats.get("images_viewed", 0) + self._session_images
        try:
            self._stats_file.write_text(json.dumps(stats, indent=2))
        except Exception:
            pass

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
        corrected_reviews = []  # ImageReview records for feedback persistence
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
                    shutil.copy2(src, dest_img)

            saved_count += 1

            # Record correction feedback (diff against load-time snapshot)
            original_dets = self._load_snapshot.get(idx, [])
            final_dets = copy.deepcopy(entry.get("detections", []))
            corrections_made = len(final_dets) != len(original_dets)
            if not corrections_made and final_dets:
                for fd, od in zip(final_dets, original_dets):
                    if (fd.get("bbox") != od.get("bbox")
                            or fd.get("class_id") != od.get("class_id")):
                        corrections_made = True
                        break

            if corrections_made:
                from datetime import datetime
                from pipeline.review_feedback import ImageReview
                review_time = 0.0
                if self._image_opened_at is not None:
                    review_time = _time.monotonic() - self._image_opened_at
                corrected_reviews.append(ImageReview(
                    image_path=entry["image_path"],
                    image_name=entry["image_name"],
                    description="",
                    generated_detections=original_dets,
                    final_detections=final_dets,
                    corrections_made=True,
                    timestamp=datetime.now().isoformat(),
                    review_time_sec=review_time,
                ))

        # Persist correction feedback (even outside describe-and-review mode)
        if corrected_reviews and not self._review_mode:
            from pipeline.review_feedback import ReviewSession
            session = ReviewSession.create(
                total_images=len(corrected_reviews),
                source_path="annotation_editor",
            )
            for r in corrected_reviews:
                session.add_review(r)
            session.finish()
            session.save()
            logger.info(
                "Saved %d correction(s) to review feedback", len(corrected_reviews)
            )

        self._update_dataset_yaml()

        # Update load snapshots to current state (so re-saving doesn't re-record)
        for idx in saved_indices:
            self._load_snapshot[idx] = copy.deepcopy(
                self._edited[idx].get("detections", [])
            )

        self._saved_to_train.update(saved_indices)
        self._dirty.clear()
        self._save_status.configure(
            text=f"Saved {saved_count} label(s) to {labels_dir}", text_color="#4CAF50"
        )
        # Update only the cards that were dirty
        for idx in saved_indices:
            self._update_card_badge(idx)

    def _export_to_valid(self):
        """Move saved annotations with real labels from train/ to val/."""
        if not self._saved_to_train:
            self._save_status.configure(
                text="Nothing to export — save first.", text_color="gray"
            )
            return

        train_labels = self._yolo_dataset_path / "train" / "labels"
        train_images = self._yolo_dataset_path / "train" / "images"
        val_labels = self._yolo_dataset_path / "val" / "labels"
        val_images = self._yolo_dataset_path / "val" / "images"
        val_labels.mkdir(parents=True, exist_ok=True)
        val_images.mkdir(parents=True, exist_ok=True)

        exported = 0
        for idx in sorted(self._saved_to_train):
            entry = self._edited[idx]
            # Only export entries that have detections with real labels
            if not entry["detections"]:
                continue

            stem = Path(entry["image_name"]).stem

            # Move label file from train to val
            src_label = train_labels / f"{stem}.txt"
            if not src_label.exists():
                continue
            shutil.move(str(src_label), str(val_labels / src_label.name))

            # Move image from train to val
            src_img = train_images / entry["image_name"]
            if src_img.exists():
                shutil.move(str(src_img), str(val_images / src_img.name))

            exported += 1

        self._saved_to_train.clear()
        self._save_status.configure(
            text=f"Exported {exported} to val/", text_color="#6A1B9A"
        )

    def _update_dataset_yaml(self):
        """Write/update dataset.yaml with current class names."""
        yaml_path = self._yolo_dataset_path / "dataset.yaml"
        yaml_content = {
            "names": {int(k): v for k, v in self._class_names.items()},
            "path": ".",
            "train": "train/images",
            "val": "val/images",
        }
        try:
            with open(yaml_path, "w") as f:
                yaml.dump(yaml_content, f, default_flow_style=False)
        except Exception as e:
            print(f"Warning: could not write {yaml_path}: {e}")


# ======================================================================
# AnnotationWindow -- for creating annotations from scratch
# ======================================================================


class AnnotationWindow(_BaseEditorWindow):
    """Opened from the 'Annotate' sidebar button.  Loads screenshots from
    screenshots/ and any existing labels from yolo_dataset/train/labels/.
    No model required.

    Includes experimental Description Mode — toggle to annotate via natural
    language descriptions instead of bounding boxes.  LLM extracts structured
    visual context per scene for use in batch annotation.
    """

    def __init__(self, parent, screenshots_dir=None, yolo_dataset_path=None):
        ss_dir = screenshots_dir or str(PROJECT_ROOT / "screenshots" / "captures")
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

        # Enable description mode toggle
        self._enable_description_mode_toggle()
        self.bind("<Control-Return>", lambda e: (
            self._generate_boxes_from_description() if self._review_mode
            else self._extract_description()
        ))


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
