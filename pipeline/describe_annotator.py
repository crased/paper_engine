"""
Describe & Critique Annotator — human + LLM side-by-side annotation pipeline.

Two-phase flow:
  1. Human draws bounding boxes on an image + writes a text description
  2. LLM receives the original image + human description → generates its own
     bounding boxes independently
  3. Both sets are shown side by side for comparison
  4. LLM critiques the differences and produces refined annotations
  5. Refined annotations are saved as YOLO training data

This is distinct from:
  - batch_annotator.py — auto-annotates from memory state, no human input
  - scene_context.py  — extracts structured ObjectDescriptors, not bboxes

Usage:
    python -m pipeline.describe_annotator --image path.png --description "cuphead in the middle"
    python -m pipeline.describe_annotator --image path.png --description "text" --human-labels path.txt
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path shim — allow running as `python pipeline/describe_annotator.py`
# ---------------------------------------------------------------------------

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# ---------------------------------------------------------------------------
# Default dataset path
# ---------------------------------------------------------------------------

_DEFAULT_DATASET = Path(__file__).resolve().parent.parent / "yolo_dataset"


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class DescribeResult:
    """Result of the LLM's independent annotation pass."""

    image_path: str
    description: str
    generated_detections: List[Dict]  # [{class_id, class_name, confidence, bbox, bbox_norm}]
    raw_llm_response: str
    timestamp: float
    generation_time_sec: float


@dataclass
class CritiqueResult:
    """Result of LLM comparing human vs its own annotations."""

    critique_text: str  # free-form assessment of differences
    refined_detections: List[Dict]  # final merged/corrected annotations
    raw_llm_response: str
    generation_time_sec: float


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def _get_golden_rules_block() -> str:
    """Load golden rules and format for prompt injection."""
    try:
        from pipeline.golden_rules import RuleStore
        store = RuleStore()
        block = store.to_prompt_block()
        return f"\n\n{block}" if block else ""
    except Exception:
        return ""


def _build_describe_prompt(
    description: str,
    class_names: Dict[int, str],
    img_w: int,
    img_h: int,
) -> str:
    """Build prompt for LLM to independently generate bounding boxes.

    The LLM sees the clean image + the human's text description and produces
    its own annotations. These get compared side-by-side with the human's boxes.
    """
    class_list = "\n".join(
        f"  {cid}: {name}" for cid, name in sorted(class_names.items())
    )
    golden_rules = _get_golden_rules_block()

    prompt = f"""You are annotating a Cuphead game screenshot for YOLO object detection training.
A human reviewer has described what they see. Generate your own bounding boxes.

IMAGE DIMENSIONS: {img_w} x {img_h} pixels

HUMAN DESCRIPTION:
{description}

AVAILABLE YOLO CLASSES (id: name):
{class_list}

INSTRUCTIONS:
1. Look at the image and use the human description to understand the scene.
2. For EACH object the human mentioned, find it in the image and draw a tight bounding box.
3. Map described objects to the nearest YOLO class:
   - "cuphead", "player", "character" → class 0 (player)
   - "enemy", "boss", "NPC", "ghost", "frog", hostile characters → class 1 (enemy)
   - "bullet", "projectile", "fireball", "shot" → class 2 (projectile)
   - "platform", "ledge", "floor" → class 3 (platform)
   - "node", "mission", "level select" → class 11 (Misson node)
   - "coin", "collectable", "pickup" → class 10 (collectable)
   - Menu elements → classes 4-9 as appropriate
4. Also annotate any visible objects the human may have missed, if clearly identifiable.
5. Output ONLY a JSON array. Each detection:
   - "class_id": integer from class list above
   - "class_name": string (must match class list exactly)
   - "bbox": [x1, y1, x2, y2] in PIXEL coordinates (top-left to bottom-right)
6. Be precise — bounding boxes should tightly enclose each object.
7. If you cannot confidently identify an object, skip it.

OUTPUT FORMAT (JSON only, no markdown, no explanation):
[
  {{"class_id": 0, "class_name": "player", "bbox": [100, 200, 180, 350]}},
  {{"class_id": 1, "class_name": "enemy", "bbox": [400, 150, 500, 300]}}
]

If nothing to annotate, return: []{golden_rules}"""

    return prompt


def _build_critique_prompt(
    description: str,
    human_detections: List[Dict],
    llm_detections: List[Dict],
    class_names: Dict[int, str],
    img_w: int,
    img_h: int,
) -> str:
    """Build prompt for LLM to critique both annotation sets and produce refined output.

    The LLM sees the image with BOTH sets of boxes (human in blue, LLM in green)
    and decides what the final training annotations should be.
    """

    def _fmt_dets(dets: List[Dict]) -> str:
        if not dets:
            return "  (none)"
        lines = []
        for d in dets:
            x1, y1, x2, y2 = d["bbox"]
            lines.append(
                f"  [{d['class_id']}] {d['class_name']}: ({x1:.0f},{y1:.0f})-({x2:.0f},{y2:.0f})"
            )
        return "\n".join(lines)

    class_list = "\n".join(
        f"  {cid}: {name}" for cid, name in sorted(class_names.items())
    )

    prompt = f"""You are reviewing YOLO annotations for a Cuphead game screenshot.
Two sets of bounding boxes have been drawn — one by a HUMAN and one by an AI.
The image shows both: BLUE boxes are HUMAN annotations, GREEN boxes are AI annotations.

IMAGE DIMENSIONS: {img_w} x {img_h} pixels

HUMAN DESCRIPTION OF THE SCENE:
{description}

HUMAN ANNOTATIONS (blue boxes):
{_fmt_dets(human_detections)}

AI ANNOTATIONS (green boxes):
{_fmt_dets(llm_detections)}

AVAILABLE YOLO CLASSES (id: name):
{class_list}

TASK:
1. Compare both annotation sets against what you see in the image.
2. Write a brief CRITIQUE: which boxes are more accurate? Any missed objects?
   Any misclassifications? Any boxes that are too loose or too tight?
3. Produce the FINAL refined annotations — pick the best box for each object,
   adjust positions if needed, fix any classification errors.

OUTPUT FORMAT (JSON object with two fields):
{{
  "critique": "Brief text comparing human vs AI annotations and explaining your choices",
  "detections": [
    {{"class_id": 0, "class_name": "player", "bbox": [x1, y1, x2, y2]}},
    ...
  ]
}}"""

    return prompt


# ---------------------------------------------------------------------------
# Side-by-side comparison rendering
# ---------------------------------------------------------------------------


def render_comparison(
    image_path: str,
    human_detections: List[Dict],
    llm_detections: List[Dict],
    class_names: Dict[int, str],
) -> "Image.Image":
    """Render a side-by-side comparison image.

    Left: original image with human annotations (blue boxes).
    Right: original image with LLM annotations (green boxes).

    Returns a PIL Image (stitched side by side with labels).
    """
    from PIL import Image, ImageDraw, ImageFont

    img = Image.open(image_path).convert("RGB")
    w, h = img.size

    human_img = img.copy()
    llm_img = img.copy()

    _draw_labeled_boxes(human_img, human_detections, color="#4A9EFF", label_prefix="H")
    _draw_labeled_boxes(llm_img, llm_detections, color="#3FB950", label_prefix="L")

    # Stitch side by side with a divider and labels
    gap = 4
    header_h = 28
    combined = Image.new("RGB", (w * 2 + gap, h + header_h), "#1a1a1a")

    # Draw headers
    draw = ImageDraw.Draw(combined)
    try:
        font = ImageFont.truetype("/usr/share/fonts/TTF/DejaVuSans-Bold.ttf", 16)
    except (OSError, IOError):
        font = ImageFont.load_default()

    draw.text((w // 2 - 60, 4), "HUMAN (blue)", fill="#4A9EFF", font=font)
    draw.text((w + gap + w // 2 - 40, 4), "LLM (green)", fill="#3FB950", font=font)

    combined.paste(human_img, (0, header_h))
    combined.paste(llm_img, (w + gap, header_h))

    return combined


def _draw_labeled_boxes(
    pil_img: "Image.Image",
    detections: List[Dict],
    color: str = "#FF3838",
    label_prefix: str = "",
):
    """Draw bounding boxes with class labels onto a PIL image (in-place)."""
    from PIL import ImageDraw, ImageFont

    draw = ImageDraw.Draw(pil_img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/TTF/DejaVuSans.ttf", 12)
    except (OSError, IOError):
        font = ImageFont.load_default()

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        label = det.get("class_name", str(det.get("class_id", "?")))
        if label_prefix:
            label = f"{label_prefix}: {label}"

        bbox_text = draw.textbbox((0, 0), label, font=font)
        tw = bbox_text[2] - bbox_text[0]
        th = bbox_text[3] - bbox_text[1]
        ly = max(y1 - th - 6, 0)
        draw.rectangle([x1, ly, x1 + tw + 8, ly + th + 4], fill=color)
        draw.text((x1 + 4, ly + 2), label, fill="white", font=font)


# ---------------------------------------------------------------------------
# IoU — decide whether critique is needed
# ---------------------------------------------------------------------------


def compute_iou(box_a: Tuple, box_b: Tuple) -> float:
    """Compute intersection-over-union between two (x1,y1,x2,y2) boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


def should_critique(
    human_dets: List[Dict],
    llm_dets: List[Dict],
    iou_threshold: float = 0.5,
) -> bool:
    """Decide if phase 2 (critique) is worth the API call.

    Skips critique when human and LLM annotations mostly agree:
    - Same number of detections
    - Greedy IoU matching: average IoU above threshold
    - Same class assignments

    Returns True if critique is needed (significant disagreement).
    """
    if len(human_dets) != len(llm_dets):
        return True
    if not human_dets:
        return False  # both empty — nothing to critique

    # Greedy match: for each human box, find best IoU with any LLM box
    used = set()
    ious = []
    class_match = True

    for h in human_dets:
        best_iou = 0.0
        best_j = -1
        for j, l in enumerate(llm_dets):
            if j in used:
                continue
            iou = compute_iou(h["bbox"], l["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_j = j
        if best_j >= 0:
            used.add(best_j)
            ious.append(best_iou)
            if h.get("class_id") != llm_dets[best_j].get("class_id"):
                class_match = False
        else:
            return True  # unmatched human box

    avg_iou = sum(ious) / len(ious) if ious else 0.0
    if avg_iou < iou_threshold:
        return True
    if not class_match:
        return True

    return False


# ---------------------------------------------------------------------------
# Description cache — avoid re-interpreting identical descriptions
# ---------------------------------------------------------------------------

# Module-level cache: normalized_description → list of (class_id, class_name)
# Only caches the class mapping, not positions (those are image-specific).
_description_cache: Dict[str, List[Tuple[int, str]]] = {}


def _normalize_description(text: str) -> str:
    """Normalize description for cache lookup."""
    return " ".join(text.lower().split())


def _get_cached_classes(description: str) -> Optional[List[Tuple[int, str]]]:
    """Check if we've seen this description before and know the class mappings."""
    key = _normalize_description(description)
    return _description_cache.get(key)


def _cache_classes(description: str, detections: List[Dict]) -> None:
    """Cache the class mappings from a successful LLM result."""
    key = _normalize_description(description)
    classes = list({(d["class_id"], d["class_name"]) for d in detections})
    _description_cache[key] = classes


# ---------------------------------------------------------------------------
# Batch API — process N images in one LLM call
# ---------------------------------------------------------------------------


@dataclass
class BatchItem:
    """One image in a batch request."""
    image_path: str
    description: str
    human_detections: List[Dict]  # existing human boxes (may be empty)


def _build_batch_prompt(
    items: List[BatchItem],
    class_names: Dict[int, str],
    image_sizes: List[Tuple[int, int]],
) -> str:
    """Build a single prompt for annotating multiple images at once.

    Images are numbered 1..N. The LLM returns a JSON array with one entry
    per image, in order.
    """
    class_list = "\n".join(
        f"  {cid}: {name}" for cid, name in sorted(class_names.items())
    )

    image_blocks = []
    for i, (item, (w, h)) in enumerate(zip(items, image_sizes), 1):
        image_blocks.append(
            f"IMAGE {i} ({w}x{h}px): {item.description}"
        )

    prompt = f"""You are annotating Cuphead game screenshots for YOLO object detection training.
You will see {len(items)} images. For each image, a human has described what they see.
Generate tight bounding boxes for every described object.

AVAILABLE YOLO CLASSES (id: name):
{class_list}

CLASS MAPPING RULES:
- "cuphead", "player", "character" → 0 (player)
- "enemy", "boss", "NPC", "ghost", hostile characters → 1 (enemy)
- "bullet", "projectile", "fireball" → 2 (projectile)
- "platform", "ledge" → 3 (platform)
- "node", "mission" → 11 (Misson node)
- "coin", "collectable" → 10 (collectable)
- Menu elements → classes 4-9

IMAGES AND DESCRIPTIONS:
{chr(10).join(image_blocks)}

OUTPUT FORMAT — JSON array with one entry per image, in order:
[
  {{
    "image": 1,
    "detections": [
      {{"class_id": 0, "class_name": "player", "bbox": [x1, y1, x2, y2]}},
      ...
    ]
  }},
  {{
    "image": 2,
    "detections": [...]
  }}
]

Rules:
- bbox values are PIXEL coordinates [x1, y1, x2, y2] for each image's own dimensions.
- Tight bounding boxes only. Skip objects you cannot confidently identify.
- Also annotate clearly visible objects the human may have missed.
- If an image has nothing to annotate, use an empty detections array.
- Output JSON only, no markdown, no explanation."""

    return prompt


def batch_describe_to_bboxes(
    items: List[BatchItem],
    class_names: Optional[Dict[int, str]] = None,
    yolo_dataset_path: Optional[str | Path] = None,
    log_fn=None,
) -> List[DescribeResult]:
    """Annotate multiple images in a single LLM call.

    Sends up to N images + descriptions to the LLM at once. Returns one
    DescribeResult per item, in the same order.

    Saves ~(N-1) API calls compared to calling describe_to_bboxes() N times.
    Gemini supports multi-image input natively.
    """
    from PIL import Image

    from pipeline.auto_review import (
        _call_llm_vision,
        _ensure_env_loaded,
        _get_llm_client,
    )
    from pipeline.batch_annotator import _load_class_names, _parse_annotation_response

    if class_names is None:
        dataset_path = Path(yolo_dataset_path) if yolo_dataset_path else _DEFAULT_DATASET
        class_names = _load_class_names(dataset_path)

    if log_fn:
        log_fn(f"Loading {len(items)} images...")

    # Load all images
    pil_images = []
    image_sizes = []
    for item in items:
        img = Image.open(item.image_path)
        pil_images.append(img)
        image_sizes.append(img.size)

    # Build combined prompt
    prompt = _build_batch_prompt(items, class_names, image_sizes)

    if log_fn:
        log_fn(f"Sending batch of {len(items)} images to LLM...")

    _ensure_env_loaded()
    client, provider, model = _get_llm_client()

    t0 = time.time()
    # Scale max_tokens with batch size
    max_tokens = min(4096 * len(items), 16384)
    raw_response = _call_llm_vision(
        client, provider, model, prompt, pil_images, max_tokens=max_tokens
    )
    generation_time = time.time() - t0

    if log_fn:
        log_fn(f"LLM responded in {generation_time:.1f}s, parsing...")

    # Parse batch response
    results = _parse_batch_response(raw_response, items, image_sizes, class_names)

    # Fill in metadata for each result
    per_image_time = generation_time / max(len(items), 1)
    final_results = []
    for i, (item, dets) in enumerate(zip(items, results)):
        img_w, img_h = image_sizes[i]
        for det in dets:
            x1, y1, x2, y2 = det["bbox"]
            det["confidence"] = 1.0
            det["bbox_norm"] = (
                ((x1 + x2) / 2) / img_w,
                ((y1 + y2) / 2) / img_h,
                (x2 - x1) / img_w,
                (y2 - y1) / img_h,
            )

        # Cache description → class mapping
        if dets:
            _cache_classes(item.description, dets)

        final_results.append(DescribeResult(
            image_path=item.image_path,
            description=item.description,
            generated_detections=dets,
            raw_llm_response=raw_response if i == 0 else "(see batch response on item 0)",
            timestamp=time.time(),
            generation_time_sec=per_image_time,
        ))

    return final_results


def _parse_batch_response(
    response_text: str,
    items: List[BatchItem],
    image_sizes: List[Tuple[int, int]],
    class_names: Dict[int, str],
) -> List[List[Dict]]:
    """Parse a batch LLM response into per-image detection lists."""
    import re

    from pipeline.batch_annotator import _parse_annotation_response

    text = response_text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    # Default: empty detections for each image
    all_dets = [[] for _ in items]

    try:
        parsed = json.loads(text.strip())
    except json.JSONDecodeError as exc:
        logger.warning("Failed to parse batch response: %s", exc)
        return all_dets

    if not isinstance(parsed, list):
        logger.warning("Batch response is not a list")
        return all_dets

    for entry in parsed:
        if not isinstance(entry, dict):
            continue
        img_idx = entry.get("image", 0) - 1  # 1-indexed → 0-indexed
        if img_idx < 0 or img_idx >= len(items):
            continue

        raw_dets = entry.get("detections", [])
        img_w, img_h = image_sizes[img_idx]
        # Re-use the standard single-image parser for validation
        validated = _parse_annotation_response(
            json.dumps(raw_dets), img_w, img_h, class_names
        )
        all_dets[img_idx] = validated

    return all_dets


# ---------------------------------------------------------------------------
# Core API — Phase 1 (single image): LLM generates its own boxes
# ---------------------------------------------------------------------------


def describe_to_bboxes(
    description: str,
    image_path: str,
    class_names: Optional[Dict[int, str]] = None,
    yolo_dataset_path: Optional[str | Path] = None,
) -> DescribeResult:
    """Send user description + image to LLM, return LLM's own bounding boxes.

    This is phase 1 — the LLM generates independently. The result is shown
    side-by-side with the human's annotations for comparison.
    """
    from PIL import Image

    from pipeline.auto_review import (
        _call_llm_vision,
        _ensure_env_loaded,
        _get_llm_client,
    )
    from pipeline.batch_annotator import _load_class_names, _parse_annotation_response

    if class_names is None:
        dataset_path = Path(yolo_dataset_path) if yolo_dataset_path else _DEFAULT_DATASET
        class_names = _load_class_names(dataset_path)

    pil_img = Image.open(image_path)
    img_w, img_h = pil_img.size

    prompt = _build_describe_prompt(description, class_names, img_w, img_h)

    _ensure_env_loaded()
    client, provider, model = _get_llm_client()

    t0 = time.time()
    raw_response = _call_llm_vision(
        client, provider, model, prompt, pil_img, max_tokens=4096
    )
    generation_time = time.time() - t0

    detections = _parse_annotation_response(raw_response, img_w, img_h, class_names)

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        det["confidence"] = 1.0
        det["bbox_norm"] = (
            ((x1 + x2) / 2) / img_w,
            ((y1 + y2) / 2) / img_h,
            (x2 - x1) / img_w,
            (y2 - y1) / img_h,
        )

    return DescribeResult(
        image_path=str(image_path),
        description=description,
        generated_detections=detections,
        raw_llm_response=raw_response,
        timestamp=time.time(),
        generation_time_sec=generation_time,
    )


# ---------------------------------------------------------------------------
# Core API — Phase 2: LLM critiques both sets, produces refined annotations
# ---------------------------------------------------------------------------


def critique_annotations(
    image_path: str,
    description: str,
    human_detections: List[Dict],
    llm_detections: List[Dict],
    class_names: Optional[Dict[int, str]] = None,
    yolo_dataset_path: Optional[str | Path] = None,
) -> CritiqueResult:
    """Send side-by-side annotated image to LLM for critique and refinement.

    The LLM sees the image with both human (blue) and LLM (green) boxes overlaid,
    compares them, and outputs refined final annotations for training.
    """
    from PIL import Image

    from pipeline.auto_review import (
        _call_llm_vision,
        _ensure_env_loaded,
        _get_llm_client,
    )
    from pipeline.batch_annotator import _load_class_names, _parse_annotation_response

    if class_names is None:
        dataset_path = Path(yolo_dataset_path) if yolo_dataset_path else _DEFAULT_DATASET
        class_names = _load_class_names(dataset_path)

    pil_img = Image.open(image_path)
    img_w, img_h = pil_img.size

    # Draw both annotation sets on the image for the LLM to see
    comparison_img = pil_img.copy()
    _draw_labeled_boxes(comparison_img, human_detections, color="#4A9EFF", label_prefix="H")
    _draw_labeled_boxes(comparison_img, llm_detections, color="#3FB950", label_prefix="L")

    prompt = _build_critique_prompt(
        description, human_detections, llm_detections, class_names, img_w, img_h
    )

    _ensure_env_loaded()
    client, provider, model = _get_llm_client()

    t0 = time.time()
    raw_response = _call_llm_vision(
        client, provider, model, prompt, comparison_img, max_tokens=4096
    )
    generation_time = time.time() - t0

    # Parse the critique response (JSON with "critique" + "detections")
    critique_text = ""
    refined_dets = []

    try:
        text = raw_response.strip()
        # Strip markdown fences
        import re

        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        parsed = json.loads(text.strip())

        if isinstance(parsed, dict):
            critique_text = parsed.get("critique", "")
            raw_dets = parsed.get("detections", [])
            # Validate through the standard parser by re-encoding
            refined_dets = _parse_annotation_response(
                json.dumps(raw_dets), img_w, img_h, class_names
            )
    except (json.JSONDecodeError, Exception) as exc:
        logger.warning("Failed to parse critique response: %s", exc)
        critique_text = raw_response  # Show raw response as the critique

    # Add confidence + bbox_norm
    for det in refined_dets:
        x1, y1, x2, y2 = det["bbox"]
        det["confidence"] = 1.0
        det["bbox_norm"] = (
            ((x1 + x2) / 2) / img_w,
            ((y1 + y2) / 2) / img_h,
            (x2 - x1) / img_w,
            (y2 - y1) / img_h,
        )

    return CritiqueResult(
        critique_text=critique_text,
        refined_detections=refined_dets,
        raw_llm_response=raw_response,
        generation_time_sec=generation_time,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Generate + critique YOLO bounding boxes from description + image"
    )
    parser.add_argument("--image", required=True, help="Path to image file")
    parser.add_argument("--description", required=True, help="What you see in the image")
    parser.add_argument(
        "--human-labels", default=None,
        help="Optional YOLO label file with human annotations (for critique mode)",
    )
    parser.add_argument(
        "--dataset", default=str(_DEFAULT_DATASET), help="Path to yolo_dataset/"
    )
    parser.add_argument(
        "--save-comparison", default=None,
        help="Save side-by-side comparison image to this path",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    from pipeline.batch_annotator import _load_class_names

    class_names = _load_class_names(args.dataset)

    # Phase 1: LLM generates its own boxes
    print("Phase 1: LLM generating annotations...")
    result = describe_to_bboxes(
        description=args.description,
        image_path=args.image,
        class_names=class_names,
    )

    print(f"\nLLM annotations ({len(result.generated_detections)}, {result.generation_time_sec:.1f}s):")
    for det in result.generated_detections:
        x1, y1, x2, y2 = det["bbox"]
        print(f"  [{det['class_id']}] {det['class_name']}: ({x1:.0f},{y1:.0f})-({x2:.0f},{y2:.0f})")

    # Load human labels if provided
    human_dets = []
    if args.human_labels:
        from PIL import Image as _Img

        with _Img.open(args.image) as im:
            iw, ih = im.size
        from pipeline.review_results import _load_yolo_labels

        human_dets = _load_yolo_labels(args.human_labels, iw, ih, class_names)
        print(f"\nHuman annotations ({len(human_dets)}):")
        for det in human_dets:
            x1, y1, x2, y2 = det["bbox"]
            print(f"  [{det['class_id']}] {det['class_name']}: ({x1:.0f},{y1:.0f})-({x2:.0f},{y2:.0f})")

    # Save comparison image
    if args.save_comparison or human_dets:
        comparison = render_comparison(
            args.image, human_dets, result.generated_detections, class_names
        )
        out_path = args.save_comparison or "/tmp/annotation_comparison.jpg"
        comparison.save(out_path, quality=95)
        print(f"\nSide-by-side comparison saved to: {out_path}")

    # Phase 2: Critique (only if human labels provided)
    if human_dets:
        print("\nPhase 2: LLM critiquing both sets...")
        critique = critique_annotations(
            image_path=args.image,
            description=args.description,
            human_detections=human_dets,
            llm_detections=result.generated_detections,
            class_names=class_names,
        )
        print(f"\nCritique ({critique.generation_time_sec:.1f}s):")
        print(f"  {critique.critique_text}")
        print(f"\nRefined annotations ({len(critique.refined_detections)}):")
        for det in critique.refined_detections:
            x1, y1, x2, y2 = det["bbox"]
            print(f"  [{det['class_id']}] {det['class_name']}: ({x1:.0f},{y1:.0f})-({x2:.0f},{y2:.0f})")


if __name__ == "__main__":
    main()
