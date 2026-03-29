"""
Example Bank — few-shot visual reference for LLM annotation.

Instead of manually describing what each class looks like, this module
selects corrected/annotated images from the dataset and renders them
as visual references. The LLM sees "here are N correctly annotated
examples — annotate the new image like these."

The bank improves over time: every time the user corrects annotations,
the corrected images become higher-priority references.

Flow:
    1. Scan yolo_dataset/train/ + val/ for image+label pairs
    2. Parse YOLO labels → bounding boxes with pixel coords
    3. Score & select diverse examples (class coverage, detection count, recency)
    4. Render annotated reference images (draw boxes + class labels)
    5. Return PIL images ready for multi-image LLM call

Integration:
    - batch_annotator sends these as context images alongside target frame
    - describe_annotator can use them for initial annotation passes
    - After each review session, the bank refreshes on next load()

Usage:
    from pipeline.example_bank import ExampleBank

    bank = ExampleBank()
    bank.load()
    examples = bank.select(n=4, scene_hint="scene_level_frogs")
    reference_images = bank.render(examples)
    # → List[PIL.Image] ready to send alongside the target frame

CLI:
    python -m pipeline.example_bank --list
    python -m pipeline.example_bank --select 4
    python -m pipeline.example_bank --render --out /tmp/refs
    python -m pipeline.example_bank --stats
"""

from __future__ import annotations

import json
import logging
import os
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class AnnotatedExample:
    """One image + its verified YOLO annotations."""

    image_path: Path
    label_path: Path
    detections: List[Dict]  # [{class_id, class_name, bbox: (x1,y1,x2,y2)}]
    img_w: int = 0
    img_h: int = 0

    # Metadata for selection scoring
    class_ids_present: Set[int] = field(default_factory=set)
    source: str = ""  # "manual", "reviewed", "auto"
    scene_name: str = ""  # from session metadata if available
    mtime: float = 0.0  # label file modification time (recency)
    is_augmented: bool = False

    @property
    def detection_count(self) -> int:
        return len(self.detections)

    @property
    def stem(self) -> str:
        return self.image_path.stem


# ---------------------------------------------------------------------------
# YOLO label parsing
# ---------------------------------------------------------------------------


def _load_class_names(dataset_path: Path) -> Dict[int, str]:
    """Load class names from dataset.yaml."""
    import yaml

    yaml_path = dataset_path / "dataset.yaml"
    if not yaml_path.exists():
        return {}
    data = yaml.safe_load(yaml_path.read_text())
    names = data.get("names", {})
    return {int(k): v for k, v in names.items()}


def _parse_yolo_label(
    label_path: Path, img_w: int, img_h: int, class_names: Dict[int, str]
) -> List[Dict]:
    """Parse a YOLO label file into detection dicts with pixel bboxes.

    YOLO format per line: class_id cx cy w h (all normalized 0-1)
    Returns: [{class_id, class_name, bbox: (x1, y1, x2, y2)}]
    """
    detections = []
    try:
        text = label_path.read_text().strip()
    except Exception:
        return []

    if not text:
        return []

    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            class_id = int(float(parts[0]))
            cx = float(parts[1])
            cy = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])
        except (ValueError, IndexError):
            continue

        if class_id not in class_names:
            continue

        # Normalized xywh → pixel x1y1x2y2
        x1 = (cx - w / 2) * img_w
        y1 = (cy - h / 2) * img_h
        x2 = (cx + w / 2) * img_w
        y2 = (cy + h / 2) * img_h

        # Clamp
        x1 = max(0.0, min(x1, img_w))
        y1 = max(0.0, min(y1, img_h))
        x2 = max(0.0, min(x2, img_w))
        y2 = max(0.0, min(y2, img_h))

        if x2 - x1 < 2 or y2 - y1 < 2:
            continue

        detections.append({
            "class_id": class_id,
            "class_name": class_names[class_id],
            "bbox": (x1, y1, x2, y2),
        })

    return detections


# ---------------------------------------------------------------------------
# Drawing — render annotations onto reference images
# ---------------------------------------------------------------------------

_PALETTE = [
    "#FF3838", "#FF9D97", "#FF701F", "#FFB21D", "#CFD231",
    "#48F90A", "#92CC17", "#3DDB86", "#1A9334", "#00D4BB",
    "#2C99A8", "#00C2FF", "#344593", "#6473FF", "#0018EC",
    "#8438FF", "#520085", "#CB38FF", "#FF95C8", "#FF37C7",
]


def _get_font(size: int = 14) -> ImageFont.FreeTypeFont:
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def render_annotated(example: AnnotatedExample, thumb_size: int = 640) -> Image.Image:
    """Render an annotated reference image with bounding boxes and class labels.

    Returns a PIL Image resized so longest edge = thumb_size.
    """
    img = Image.open(example.image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    font = _get_font(16)
    small_font = _get_font(12)

    for det in example.detections:
        x1, y1, x2, y2 = det["bbox"]
        color = _PALETTE[det["class_id"] % len(_PALETTE)]

        # Bounding box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        # Label background + text
        label = f"{det['class_id']}:{det['class_name']}"
        bbox_text = draw.textbbox((0, 0), label, font=small_font)
        tw = bbox_text[2] - bbox_text[0]
        th = bbox_text[3] - bbox_text[1]
        ly = max(y1 - th - 6, 0)
        draw.rectangle([x1, ly, x1 + tw + 8, ly + th + 4], fill=color)
        draw.text((x1 + 4, ly + 2), label, fill="white", font=small_font)

    # Resize to thumb_size (longest edge)
    w, h = img.size
    ratio = thumb_size / max(w, h)
    if ratio < 1.0:
        new_size = (int(w * ratio), int(h * ratio))
        img = img.resize(new_size, Image.LANCZOS)

    return img


# ---------------------------------------------------------------------------
# Example selection scoring
# ---------------------------------------------------------------------------


def _score_example(
    example: AnnotatedExample,
    target_scene: str = "",
    target_classes: Optional[Set[int]] = None,
    covered_classes: Optional[Set[int]] = None,
) -> float:
    """Score an example for selection. Higher = better reference.

    Factors:
    - New class coverage: huge bonus for classes not yet covered
    - Detection count: prefer 2-6 detections (good variety, not cluttered)
    - Source quality: reviewed > manual (all current data is manual)
    - Recency: small bonus for newer labels
    - Scene match: bonus if scene hint matches
    """
    score = 0.0

    # Class coverage — main driver
    if covered_classes is not None:
        new_classes = example.class_ids_present - covered_classes
        score += len(new_classes) * 10.0
        # After all classes covered, prefer more classes per image (diverse frames)
        if not new_classes:
            score += len(example.class_ids_present) * 1.5
    else:
        score += len(example.class_ids_present) * 5.0

    # Target class relevance
    if target_classes:
        overlap = example.class_ids_present & target_classes
        score += len(overlap) * 3.0

    # Detection count sweet spot: prefer 3-6 (varied, not cluttered)
    n = example.detection_count
    if 3 <= n <= 6:
        score += 6.0
    elif n == 2:
        score += 4.0
    elif n == 1:
        score += 1.0
    elif n > 6:
        score += 2.0

    # Source quality
    source_bonus = {"reviewed": 4.0, "manual": 2.0, "auto": 0.0}
    score += source_bonus.get(example.source, 1.0)

    # Scene match
    if target_scene and example.scene_name:
        if example.scene_name == target_scene:
            score += 6.0
        elif _scene_type(example.scene_name) == _scene_type(target_scene):
            score += 3.0

    # Recency bonus (newer corrections are more relevant)
    if example.mtime > 0:
        score += 0.5

    # Penalize augmented — prefer originals
    if example.is_augmented:
        score -= 8.0

    return score


def _scene_type(scene_name: str) -> str:
    """Extract scene type from scene name for loose matching."""
    lower = scene_name.lower()
    if "level" in lower:
        return "level"
    if "menu" in lower or "title" in lower:
        return "menu"
    if "map" in lower or "world" in lower:
        return "map"
    return "other"


# ---------------------------------------------------------------------------
# Review feedback integration
# ---------------------------------------------------------------------------


def _get_reviewed_stems(feedback_dir: Optional[Path] = None) -> Set[str]:
    """Load stems of images that the user has reviewed/corrected.

    These are higher quality references since a human verified them.
    """
    if feedback_dir is None:
        feedback_dir = _PROJECT_ROOT / "reports" / "review_feedback"

    stems: Set[str] = set()
    if not feedback_dir.exists():
        return stems

    for session_file in feedback_dir.glob("review_*.json"):
        try:
            data = json.loads(session_file.read_text())
            for review in data.get("reviews", []):
                if review.get("corrections_made", False):
                    name = review.get("image_name", "")
                    if name:
                        stems.add(Path(name).stem)
        except Exception:
            continue

    return stems


# ---------------------------------------------------------------------------
# Main ExampleBank class
# ---------------------------------------------------------------------------


class ExampleBank:
    """Manages selection and rendering of few-shot reference examples.

    Scans the YOLO dataset for image+label pairs, scores them for
    diversity and quality, and renders annotated reference images
    for injection into LLM annotation prompts.
    """

    def __init__(
        self,
        dataset_path: Optional[Path] = None,
        feedback_dir: Optional[Path] = None,
    ):
        self._dataset_path = dataset_path or (_PROJECT_ROOT / "yolo_dataset")
        self._feedback_dir = feedback_dir or (
            _PROJECT_ROOT / "reports" / "review_feedback"
        )
        self._class_names: Dict[int, str] = {}
        self._examples: List[AnnotatedExample] = []
        self._reviewed_stems: Set[str] = set()
        self._loaded = False

    def load(self) -> None:
        """Scan dataset and build the example index."""
        self._class_names = _load_class_names(self._dataset_path)
        if not self._class_names:
            logger.warning("No class names found in %s", self._dataset_path)
            self._loaded = True
            return

        self._reviewed_stems = _get_reviewed_stems(self._feedback_dir)

        self._examples = []

        # Scan both train/ and val/ for examples
        for split in ("train", "val"):
            images_dir = self._dataset_path / split / "images"
            labels_dir = self._dataset_path / split / "labels"
            if not images_dir.exists() or not labels_dir.exists():
                continue

            for img_path in sorted(images_dir.glob("*.png")):
                label_path = labels_dir / f"{img_path.stem}.txt"
                if not label_path.exists():
                    continue

                # Quick check: skip empty label files
                if label_path.stat().st_size == 0:
                    continue

                # Get image dimensions without fully loading the image
                try:
                    with Image.open(img_path) as im:
                        img_w, img_h = im.size
                except Exception:
                    continue

                detections = _parse_yolo_label(
                    label_path, img_w, img_h, self._class_names
                )
                if not detections:
                    continue

                is_aug = "_aug" in img_path.stem
                stem_base = img_path.stem.split("_aug")[0] if is_aug else img_path.stem

                # Determine source quality
                if stem_base in self._reviewed_stems:
                    source = "reviewed"
                else:
                    source = "manual"  # all current labels are human-made

                example = AnnotatedExample(
                    image_path=img_path,
                    label_path=label_path,
                    detections=detections,
                    img_w=img_w,
                    img_h=img_h,
                    class_ids_present={d["class_id"] for d in detections},
                    source=source,
                    mtime=label_path.stat().st_mtime,
                    is_augmented=is_aug,
                )

                self._examples.append(example)

        logger.info(
            "Example bank loaded: %d examples (%d originals, %d augmented)",
            len(self._examples),
            sum(1 for e in self._examples if not e.is_augmented),
            sum(1 for e in self._examples if e.is_augmented),
        )
        self._loaded = True

    def select(
        self,
        n: int = 4,
        scene_hint: str = "",
        target_classes: Optional[Set[int]] = None,
    ) -> List[AnnotatedExample]:
        """Select N diverse, high-quality reference examples.

        Uses greedy selection: pick highest scorer, update covered classes, repeat.
        Skips augmented variants of already-selected base images.
        """
        if not self._loaded:
            self.load()

        if not self._examples:
            return []

        # Filter to candidates (prefer originals, but allow augmented if needed)
        originals = [e for e in self._examples if not e.is_augmented]
        candidates = originals if len(originals) >= n else self._examples

        selected: List[AnnotatedExample] = []
        covered_classes: Set[int] = set()
        used_base_stems: Set[str] = set()

        for _ in range(min(n, len(candidates))):
            best_score = -1.0
            best_idx = -1

            for i, ex in enumerate(candidates):
                # Skip if we already selected this base image
                base = ex.stem.split("_aug")[0]
                if base in used_base_stems:
                    continue

                score = _score_example(
                    ex,
                    target_scene=scene_hint,
                    target_classes=target_classes,
                    covered_classes=covered_classes,
                )
                if score > best_score:
                    best_score = score
                    best_idx = i

            if best_idx < 0:
                break

            pick = candidates[best_idx]
            selected.append(pick)
            covered_classes.update(pick.class_ids_present)
            used_base_stems.add(pick.stem.split("_aug")[0])

        return selected

    def render(
        self, examples: List[AnnotatedExample], thumb_size: int = 640
    ) -> List[Image.Image]:
        """Render selected examples as annotated PIL images."""
        rendered = []
        for ex in examples:
            try:
                img = render_annotated(ex, thumb_size=thumb_size)
                rendered.append(img)
            except Exception as exc:
                logger.warning("Failed to render %s: %s", ex.image_path.name, exc)
        return rendered

    def stats(self) -> str:
        """Human-readable stats about the example bank."""
        if not self._loaded:
            self.load()

        if not self._examples:
            return "Example bank: empty (no annotated images found)"

        total = len(self._examples)
        originals = sum(1 for e in self._examples if not e.is_augmented)
        augmented = total - originals
        reviewed = sum(1 for e in self._examples if e.source == "reviewed")

        # Class distribution
        class_counter: Counter = Counter()
        for ex in self._examples:
            if not ex.is_augmented:  # count originals only
                for d in ex.detections:
                    class_counter[d["class_name"]] += 1

        # Detection count distribution
        det_counts = [e.detection_count for e in self._examples if not e.is_augmented]
        avg_dets = sum(det_counts) / len(det_counts) if det_counts else 0

        lines = [
            f"Example Bank Stats:",
            f"  Total examples:  {total} ({originals} original, {augmented} augmented)",
            f"  Reviewed/corrected: {reviewed}",
            f"  Classes: {len(self._class_names)}",
            f"  Avg detections per image: {avg_dets:.1f}",
            f"",
            f"  Class distribution (originals only):",
        ]
        for name, count in class_counter.most_common():
            lines.append(f"    {name}: {count}")

        # Detection count histogram
        lines.append(f"")
        lines.append(f"  Detections per image:")
        hist: Counter = Counter()
        for n in det_counts:
            if n <= 1:
                hist["1"] += 1
            elif n <= 3:
                hist["2-3"] += 1
            elif n <= 6:
                hist["4-6"] += 1
            else:
                hist["7+"] += 1
        for bucket in ("1", "2-3", "4-6", "7+"):
            lines.append(f"    {bucket}: {hist.get(bucket, 0)}")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Prompt builder — replaces manual descriptions with visual examples
# ---------------------------------------------------------------------------


def build_fewshot_prompt(
    class_names: Dict[int, str],
    memory_state: Dict[str, Any],
    img_w: int,
    img_h: int,
    n_examples: int = 0,
) -> str:
    """Build an annotation prompt that relies on visual examples instead of
    manual class descriptions.

    The caller sends this prompt + N reference images + 1 target image
    to the LLM. The reference images have bounding boxes drawn on them
    showing what correct annotations look like.
    """
    # Memory context (objective metadata — keep)
    context_lines = []
    scene = memory_state.get("scene_name", "")
    if scene:
        context_lines.append(f"Current scene: {scene}")
    if memory_state.get("in_game"):
        context_lines.append(
            f"Player HP: {memory_state.get('hp', '?')}/{memory_state.get('hp_max', '?')}"
        )
    if memory_state.get("is_loading"):
        context_lines.append("Game is loading (may be black/transition)")

    context_block = (
        "\n".join(f"  - {l}" for l in context_lines)
        if context_lines
        else "  (no memory state)"
    )

    class_list = "\n".join(
        f"  {cid}: {name}" for cid, name in sorted(class_names.items())
    )

    # Golden rules (quality standards from corrections — keep)
    golden_rules = ""
    try:
        from pipeline.golden_rules import RuleStore
        store = RuleStore()
        block = store.to_prompt_block()
        if block:
            golden_rules = f"\n\n{block}"
    except Exception:
        pass

    if n_examples > 0:
        prompt = f"""You are annotating a game screenshot for YOLO object detection training.

The first {n_examples} image(s) are CORRECTLY ANNOTATED reference examples.
Each has colored bounding boxes with "class_id:class_name" labels drawn on it.
Study them carefully — they show exactly how objects should be annotated.

The LAST image is the TARGET you must annotate.

TARGET IMAGE DIMENSIONS: {img_w} x {img_h} pixels

GAME STATE (from memory reading):
{context_block}

YOLO CLASSES (id: name):
{class_list}

TASK:
1. Study the reference examples — note the box tightness, class assignments, and what gets annotated.
2. Annotate the TARGET (last) image following the same style.
3. Output ONLY a JSON array of detections:
   - "class_id": integer from the class list
   - "class_name": string (must match exactly)
   - "bbox": [x1, y1, x2, y2] in PIXEL coordinates (top-left to bottom-right)
4. If loading/black/no objects, return: []

OUTPUT (JSON only, no markdown):
[
  {{"class_id": 0, "class_name": "player", "bbox": [x1, y1, x2, y2]}},
  ...
]{golden_rules}"""
    else:
        # Fallback: no examples available — minimal text prompt
        prompt = f"""You are annotating a game screenshot for YOLO object detection training.

IMAGE DIMENSIONS: {img_w} x {img_h} pixels

GAME STATE (from memory reading):
{context_block}

YOLO CLASSES (id: name):
{class_list}

TASK:
1. Identify all visible game objects matching the classes above.
2. For each, provide a tight bounding box.
3. Output ONLY a JSON array:
   - "class_id": integer from class list
   - "class_name": string (must match exactly)
   - "bbox": [x1, y1, x2, y2] in PIXEL coordinates
4. If loading/black/no objects, return: []

OUTPUT (JSON only, no markdown):
[
  {{"class_id": 0, "class_name": "player", "bbox": [x1, y1, x2, y2]}},
  ...
]{golden_rules}"""

    return prompt


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Manage few-shot annotation examples"
    )
    parser.add_argument("--list", action="store_true", help="List available examples")
    parser.add_argument(
        "--select", type=int, default=0, metavar="N",
        help="Select N diverse examples"
    )
    parser.add_argument(
        "--render", action="store_true",
        help="Render selected examples to images"
    )
    parser.add_argument(
        "--out", default="/tmp/example_bank",
        help="Output directory for rendered examples"
    )
    parser.add_argument(
        "--scene", default="", help="Scene hint for selection"
    )
    parser.add_argument("--stats", action="store_true", help="Show example bank stats")
    parser.add_argument("--dataset", default=None, help="Path to yolo_dataset/")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    dataset_path = Path(args.dataset) if args.dataset else None
    bank = ExampleBank(dataset_path=dataset_path)
    bank.load()

    if args.stats:
        print(bank.stats())
    elif args.select > 0 or args.render:
        n = args.select or 4
        examples = bank.select(n=n, scene_hint=args.scene)
        print(f"Selected {len(examples)} examples:")
        for ex in examples:
            classes = ", ".join(sorted({d["class_name"] for d in ex.detections}))
            print(f"  {ex.image_path.name}  ({ex.detection_count} dets: {classes})")

        if args.render:
            out_dir = Path(args.out)
            out_dir.mkdir(parents=True, exist_ok=True)
            rendered = bank.render(examples)
            for i, img in enumerate(rendered):
                path = out_dir / f"ref_{i}.png"
                img.save(path)
                print(f"  → {path}")
    elif args.list:
        if not bank._examples:
            print("No examples found in dataset.")
        else:
            for ex in bank._examples[:50]:
                src = f"[{ex.source}]" if ex.source else ""
                classes = ", ".join(sorted({d["class_name"] for d in ex.detections}))
                print(
                    f"  {ex.image_path.name}: {ex.detection_count} dets ({classes}) {src}"
                )
            if len(bank._examples) > 50:
                print(f"  ... and {len(bank._examples) - 50} more")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
