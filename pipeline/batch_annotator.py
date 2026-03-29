"""
Batch Auto-Annotator — generates YOLO annotations from recorded gameplay sessions.

Takes a session directory (created by GameplayRecorder) containing:
  - frame_NNNNNN.png   (screenshots)
  - frame_NNNNNN.json  (memory state + timestamp)
  - session.json       (manifest)

Sends frames + memory context to Gemini for bounding box annotation.
Outputs YOLO-format label files alongside the images, ready for training.

Key design decisions:
  - Batch processing (post-session, not real-time)
  - Rate limiting for Gemini free tier (configurable RPM)
  - Memory state provides context to help LLM identify objects correctly
  - Supports resume: skips frames that already have labels
  - Outputs directly into yolo_dataset/ structure (train/images + train/labels)

Usage:
    python -m pipeline.batch_annotator --session recordings/sessions/Cuphead_20260305_120000
    python -m pipeline.batch_annotator --session recordings/sessions/Cuphead_20260305_120000 --dry-run
    python -m pipeline.batch_annotator --session recordings/sessions/Cuphead_20260305_120000 --rpm 10
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image
import yaml

logger = logging.getLogger(__name__)

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from conf.config_parser import main_conf as config


# ---------------------------------------------------------------------------
# LLM helpers (reuse from auto_review.py)
# ---------------------------------------------------------------------------


def _get_llm_client():
    """Get configured LLM client, provider, and model."""
    from pipeline.auto_review import _get_llm_client as _get_client

    return _get_client()


def _call_llm_vision(client, provider, model, prompt, pil_images, max_tokens=8192):
    """Send images + text to vision LLM."""
    from pipeline.auto_review import _call_llm_vision as _call_vision

    return _call_vision(client, provider, model, prompt, pil_images, max_tokens)


# ---------------------------------------------------------------------------
# YOLO label I/O (reuse from auto_review.py)
# ---------------------------------------------------------------------------


def _load_class_names(yolo_dataset_path: str | Path) -> Dict[int, str]:
    """Load class names from dataset.yaml."""
    from pipeline.auto_review import _load_class_names as _load

    return _load(str(yolo_dataset_path))


def _write_yolo_labels(
    label_file: str | Path, detections: List[Dict], img_w: int, img_h: int
):
    """Write detections to YOLO label file (normalized xywh format)."""
    from pipeline.auto_review import _write_yolo_labels as _write

    _write(str(label_file), detections, img_w, img_h)


def _save_class_names(yolo_dataset_path: str | Path, class_names: Dict[int, str]):
    """Save class names to dataset.yaml."""
    from pipeline.auto_review import _save_class_names as _save

    _save(str(yolo_dataset_path), class_names)


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------


class RateLimiter:
    """Simple token-bucket rate limiter for API calls."""

    def __init__(self, requests_per_minute: int = 10):
        self._rpm = max(1, requests_per_minute)
        self._min_interval = 60.0 / self._rpm
        self._last_request_time = 0.0
        self._request_times: List[float] = []

    def wait(self):
        """Block until we can make the next request."""
        now = time.time()

        # Simple approach: enforce minimum interval between requests
        elapsed = now - self._last_request_time
        if elapsed < self._min_interval:
            sleep_time = self._min_interval - elapsed
            logger.debug("Rate limit: sleeping %.1fs", sleep_time)
            time.sleep(sleep_time)

        self._last_request_time = time.time()

    def record_error(self, is_rate_limit: bool = False):
        """Back off on errors, especially rate limit errors."""
        if is_rate_limit:
            backoff = 60.0  # Wait a full minute on rate limit
            logger.warning("Rate limit hit, backing off %.0fs", backoff)
            time.sleep(backoff)
        else:
            time.sleep(5.0)  # Brief pause on other errors


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

# Legacy class lists (kept for reference, no longer used in prompts)
GAMEPLAY_CLASSES = [
    "cuphead", "enemy", "acorn", "coin", "life_counter",
    "mushroom_projectile", "rose_projectile", "mini_cuphead",
]
MENU_CLASSES = [
    "back_button", "confirm_button", "dlc_menu", "end_game",
    "exit_button", "option_menu", "save_game1", "save_game2",
    "save_game3", "shop", "start_button", "start_game",
]


def _get_golden_rules_block() -> str:
    """Load golden rules for prompt injection."""
    try:
        from pipeline.golden_rules import RuleStore
        store = RuleStore()
        block = store.to_prompt_block()
        return f"\n\n{block}" if block else ""
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Few-shot example support
# ---------------------------------------------------------------------------

# Module-level bank instance (lazy-loaded)
_example_bank = None


def _get_example_bank():
    """Get or create the ExampleBank singleton."""
    global _example_bank
    if _example_bank is None:
        from pipeline.example_bank import ExampleBank
        _example_bank = ExampleBank()
        _example_bank.load()
    return _example_bank


def _get_fewshot_examples(
    memory_state: Dict[str, Any],
    n: int = 4,
) -> Tuple[List[Any], List["Image.Image"]]:
    """Select and render few-shot reference examples for the current frame.

    Returns:
        (examples, rendered_images) — examples for metadata, PIL images for LLM call.
        Returns ([], []) if no examples available (falls back to text-only prompt).
    """
    # TODO: call _get_example_bank().select() with scene hint from memory_state
    # TODO: call _get_example_bank().render() to get annotated PIL images
    # TODO: return (examples, rendered_images)
    # TODO: handle gracefully if no examples exist yet (empty dataset)
    try:
        bank = _get_example_bank()
        scene = memory_state.get("scene_name", "")
        examples = bank.select(n=n, scene_hint=scene)
        if not examples:
            return [], []
        rendered = bank.render(examples)
        return examples, rendered
    except Exception as exc:
        logger.debug("Example bank unavailable: %s", exc)
        return [], []


def _build_annotation_prompt(
    class_names: Dict[int, str],
    memory_state: Dict[str, Any],
    img_w: int,
    img_h: int,
    n_examples: int = 0,
) -> str:
    """Build the annotation prompt. Uses few-shot examples when available.

    When n_examples > 0, the prompt expects N annotated reference images
    followed by 1 target image in the multi-image LLM call. The references
    teach the LLM what correct annotations look like — no manual class
    descriptions needed.

    When n_examples == 0 (no examples available yet), falls back to a
    minimal text-only prompt with just the class list and memory context.
    """
    from pipeline.example_bank import build_fewshot_prompt

    return build_fewshot_prompt(
        class_names=class_names,
        memory_state=memory_state,
        img_w=img_w,
        img_h=img_h,
        n_examples=n_examples,
    )


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def _parse_annotation_response(
    response_text: str, img_w: int, img_h: int, class_names: Dict[int, str]
) -> List[Dict]:
    """Parse LLM response into detection dicts with validated bboxes."""
    # Strip markdown code fences if present
    text = response_text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    if not text or text == "[]":
        return []

    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        logger.warning("Failed to parse LLM JSON response: %s", exc)
        logger.debug("Raw response: %s", text[:500])
        return []

    if not isinstance(raw, list):
        logger.warning("LLM response is not a list: %s", type(raw))
        return []

    # Build reverse name->id map
    name_to_id = {name: cid for cid, name in class_names.items()}

    detections = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            logger.debug("Skipping non-dict item at index %d", i)
            continue

        # Get class_id — prefer explicit id, fall back to name lookup
        class_id = item.get("class_id")
        class_name = item.get("class_name", "")

        if class_id is None and class_name:
            class_id = name_to_id.get(class_name)
        if class_id is None:
            logger.debug("Skipping detection with unknown class: %s", item)
            continue

        class_id = int(class_id)
        if class_id not in class_names:
            logger.debug("Skipping detection with invalid class_id %d", class_id)
            continue

        # Parse bbox
        bbox = item.get("bbox")
        if not bbox or not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            logger.debug("Skipping detection with invalid bbox: %s", bbox)
            continue

        try:
            x1, y1, x2, y2 = (
                float(bbox[0]),
                float(bbox[1]),
                float(bbox[2]),
                float(bbox[3]),
            )
        except (ValueError, TypeError):
            logger.debug("Skipping detection with non-numeric bbox: %s", bbox)
            continue

        # Clamp to image bounds
        x1 = max(0.0, min(x1, img_w))
        y1 = max(0.0, min(y1, img_h))
        x2 = max(0.0, min(x2, img_w))
        y2 = max(0.0, min(y2, img_h))

        # Ensure x1 < x2 and y1 < y2
        if x1 >= x2 or y1 >= y2:
            logger.debug(
                "Skipping zero-area bbox: [%.0f, %.0f, %.0f, %.0f]", x1, y1, x2, y2
            )
            continue

        # Minimum size filter (at least 5x5 pixels)
        if (x2 - x1) < 5 or (y2 - y1) < 5:
            logger.debug("Skipping tiny bbox: [%.0f, %.0f, %.0f, %.0f]", x1, y1, x2, y2)
            continue

        detections.append(
            {
                "class_id": class_id,
                "class_name": class_names[class_id],
                "bbox": (x1, y1, x2, y2),
            }
        )

    return detections


# ---------------------------------------------------------------------------
# Session loading
# ---------------------------------------------------------------------------


@dataclass
class SessionFrame:
    """One frame from a recorded session."""

    index: int
    image_path: Path
    meta_path: Path
    state: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0


def load_session(session_dir: str | Path) -> Tuple[Dict, List[SessionFrame]]:
    """Load a recorded session.  Returns (session_info, frames)."""
    session_dir = Path(session_dir)

    manifest_path = session_dir / "session.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No session.json in {session_dir}")

    with open(manifest_path) as f:
        session_info = json.load(f)

    # Find all frame pairs
    frames = []
    for png_path in sorted(session_dir.glob("frame_*.png")):
        stem = png_path.stem  # e.g. "frame_000042"
        meta_path = session_dir / f"{stem}.json"

        frame = SessionFrame(
            index=int(stem.split("_")[1]),
            image_path=png_path,
            meta_path=meta_path,
        )

        # Load metadata if available
        if meta_path.exists():
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                frame.state = meta.get("state", {})
                frame.timestamp = meta.get("timestamp", 0.0)
            except (json.JSONDecodeError, KeyError) as exc:
                logger.warning("Failed to load metadata for %s: %s", stem, exc)

        frames.append(frame)

    logger.info("Loaded session: %d frames from %s", len(frames), session_dir)
    return session_info, frames


# ---------------------------------------------------------------------------
# Annotation stats
# ---------------------------------------------------------------------------


@dataclass
class AnnotationStats:
    """Track annotation progress and quality."""

    total_frames: int = 0
    annotated: int = 0
    skipped_existing: int = 0
    skipped_loading: int = 0
    empty_annotations: int = 0
    total_detections: int = 0
    errors: int = 0
    class_counts: Dict[str, int] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"Annotation Summary:",
            f"  Total frames:     {self.total_frames}",
            f"  Annotated:        {self.annotated}",
            f"  Empty (no objs):  {self.empty_annotations}",
            f"  Skipped existing: {self.skipped_existing}",
            f"  Skipped loading:  {self.skipped_loading}",
            f"  Errors:           {self.errors}",
            f"  Total detections: {self.total_detections}",
        ]
        if self.class_counts:
            lines.append(f"  Class distribution:")
            for name, count in sorted(self.class_counts.items(), key=lambda x: -x[1]):
                lines.append(f"    {name}: {count}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main annotator
# ---------------------------------------------------------------------------


def annotate_session(
    session_dir: str | Path,
    yolo_dataset_path: Optional[str | Path] = None,
    rpm: int = 10,
    skip_existing: bool = True,
    skip_loading: bool = True,
    dry_run: bool = False,
    copy_images: bool = True,
    max_frames: int = 0,
) -> AnnotationStats:
    """
    Annotate a recorded session and output YOLO labels.

    Args:
        session_dir: Path to session directory from GameplayRecorder
        yolo_dataset_path: Where to output YOLO data (default: yolo_dataset/)
        rpm: Max API requests per minute
        skip_existing: Skip frames that already have label files
        skip_loading: Skip frames where game was loading
        dry_run: Print what would be done without calling the API
        copy_images: Copy frame PNGs into yolo_dataset/train/images/
        max_frames: Stop after N frames (0 = all)

    Returns:
        AnnotationStats with results
    """
    session_dir = Path(session_dir)
    if yolo_dataset_path is None:
        yolo_dataset_path = _PROJECT_ROOT / "yolo_dataset"
    yolo_dataset_path = Path(yolo_dataset_path)

    # Load session
    session_info, frames = load_session(session_dir)
    if not frames:
        logger.error("No frames found in session")
        return AnnotationStats()

    # Load class definitions
    class_names = _load_class_names(yolo_dataset_path)
    if not class_names:
        logger.error("No class names found in %s/dataset.yaml", yolo_dataset_path)
        return AnnotationStats()

    logger.info(
        "Classes loaded: %d (%s)", len(class_names), ", ".join(class_names.values())
    )

    # Set up output directories
    images_dir = yolo_dataset_path / "train" / "images"
    labels_dir = yolo_dataset_path / "train" / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    # Set up LLM client (unless dry run)
    client = provider = model = None
    if not dry_run:
        try:
            from pipeline.auto_review import _ensure_env_loaded

            _ensure_env_loaded()
            client, provider, model = _get_llm_client()
            logger.info("LLM client ready: %s / %s", provider, model)
        except Exception as exc:
            logger.error("Failed to initialize LLM client: %s", exc)
            return AnnotationStats()

    rate_limiter = RateLimiter(rpm)
    stats = AnnotationStats(total_frames=len(frames))

    if max_frames > 0:
        frames = frames[:max_frames]
        stats.total_frames = len(frames)

    # Pre-load few-shot reference examples (shared across all frames in session)
    # These are rendered annotated images from the user's corrected dataset.
    # Loaded once per session, not per frame — same examples, different targets.
    ref_examples, ref_images = [], []
    if not dry_run:
        # Use first frame's state for scene hint (session is usually one scene)
        hint_state = frames[0].state if frames else {}
        ref_examples, ref_images = _get_fewshot_examples(hint_state, n=4)
        if ref_images:
            logger.info(
                "Loaded %d reference examples for few-shot annotation", len(ref_images)
            )
        else:
            logger.info("No reference examples available — using text-only prompt")

    for i, frame in enumerate(frames):
        frame_label = f"[{i + 1}/{stats.total_frames}] {frame.image_path.name}"

        # Check: skip loading screens
        if skip_loading and frame.state.get("is_loading", False):
            logger.debug("Skipping loading frame: %s", frame_label)
            stats.skipped_loading += 1
            continue

        # Output filenames use the session + frame index for uniqueness
        session_name = session_dir.name
        out_stem = f"{session_name}_{frame.image_path.stem}"
        out_label_path = labels_dir / f"{out_stem}.txt"
        out_image_path = images_dir / f"{out_stem}.png"

        # Check: skip existing
        if skip_existing and out_label_path.exists():
            logger.debug("Skipping existing label: %s", frame_label)
            stats.skipped_existing += 1
            continue

        # Load image
        try:
            img = Image.open(frame.image_path).convert("RGB")
            img_w, img_h = img.size
        except Exception as exc:
            logger.warning("Failed to load image %s: %s", frame.image_path, exc)
            stats.errors += 1
            continue

        if dry_run:
            scene = frame.state.get("scene_name", "?")
            hp = frame.state.get("hp", "?")
            print(f"  {frame_label}  scene={scene}  hp={hp}  → {out_label_path.name}")
            stats.annotated += 1
            continue

        # Build prompt — uses few-shot examples when available
        prompt = _build_annotation_prompt(
            class_names, frame.state, img_w, img_h,
            n_examples=len(ref_images),
        )

        # Build image list: reference examples first, then target frame
        # The prompt tells the LLM "the last image is the one to annotate"
        llm_images = ref_images + [img]

        # Call LLM with rate limiting
        rate_limiter.wait()
        try:
            response = _call_llm_vision(
                client, provider, model, prompt, llm_images, max_tokens=4096
            )
        except Exception as exc:
            exc_str = str(exc).lower()
            is_rate_limit = "rate" in exc_str or "429" in exc_str or "quota" in exc_str
            rate_limiter.record_error(is_rate_limit)
            logger.warning("LLM call failed for %s: %s", frame_label, exc)
            stats.errors += 1
            continue

        # Parse response
        detections = _parse_annotation_response(response, img_w, img_h, class_names)

        if not detections:
            stats.empty_annotations += 1
            logger.debug("No detections for %s", frame_label)
        else:
            stats.total_detections += len(detections)
            for det in detections:
                name = det["class_name"]
                stats.class_counts[name] = stats.class_counts.get(name, 0) + 1

        # Write label file
        _write_yolo_labels(out_label_path, detections, img_w, img_h)

        # Copy image to dataset
        if copy_images and not out_image_path.exists():
            shutil.copy2(frame.image_path, out_image_path)

        stats.annotated += 1
        det_summary = (
            ", ".join(f"{d['class_name']}" for d in detections)
            if detections
            else "(empty)"
        )
        logger.info("%s → %d detections: %s", frame_label, len(detections), det_summary)

    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Batch-annotate recorded gameplay sessions"
    )
    parser.add_argument("--session", required=True, help="Path to session directory")
    parser.add_argument(
        "--dataset", default=None, help="YOLO dataset path (default: yolo_dataset/)"
    )
    parser.add_argument(
        "--rpm", type=int, default=10, help="Max API requests per minute (default: 10)"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Preview without calling API"
    )
    parser.add_argument(
        "--no-skip", action="store_true", help="Re-annotate frames with existing labels"
    )
    parser.add_argument(
        "--no-copy", action="store_true", help="Don't copy images to dataset dir"
    )
    parser.add_argument(
        "--max-frames", type=int, default=0, help="Limit number of frames to annotate"
    )
    parser.add_argument(
        "--no-examples", action="store_true",
        help="Disable few-shot examples (use text-only prompt)",
    )
    parser.add_argument("--verbose", action="store_true", help="Debug logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    stats = annotate_session(
        session_dir=args.session,
        yolo_dataset_path=args.dataset,
        rpm=args.rpm,
        skip_existing=not args.no_skip,
        dry_run=args.dry_run,
        copy_images=not args.no_copy,
        max_frames=args.max_frames,
    )

    print(f"\n{stats.summary()}")


if __name__ == "__main__":
    main()
