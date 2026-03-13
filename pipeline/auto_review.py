"""
Auto Review -- LLM-powered annotation correction (single-prompt batch mode)

Collects all annotation data into one JSON payload and sends it to a
vision-capable LLM in a SINGLE API call, along with a grid of sample
images for visual reference.  The LLM returns corrections for the entire
dataset at once, avoiding per-image API calls and rate-limit issues.

Before calling the LLM, local pre-checks run first to catch issues that
don't need AI (orphaned class IDs, missing dataset.yaml entries, etc.).

Usage (standalone):
    python pipeline/auto_review.py

Or triggered from the GUI via the "Auto Review" button after running
model inference.
"""

import json
import os
import io
import sys
import base64
import shutil
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image, ImageDraw, ImageFont
import yaml

from conf.config_parser import main_conf as config

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Lazy env / LLM helpers (mirrors generate_bot_script.py)
# ---------------------------------------------------------------------------


def _ensure_env_loaded():
    from pipeline.generate_bot_script import create_env_file_if_missing

    create_env_file_if_missing()
    from dotenv import load_dotenv

    load_dotenv()


def _get_llm_client():
    from pipeline.generate_bot_script import get_llm_client
    from tools.functions import get_api_key

    provider = config.LLM_PROVIDER
    model = config.LLM_MODEL
    api_key = get_api_key()
    if not api_key:
        raise RuntimeError(
            "API key not configured. Store with: "
            "secret-tool store --label='Paper Engine' service paper_engine user api_key"
        )
    client = get_llm_client(provider, api_key)
    return client, provider, model


# ---------------------------------------------------------------------------
# Vision call (multi-modal: images + text)
# ---------------------------------------------------------------------------


def _call_llm_vision(client, provider, model, prompt, pil_images, max_tokens=8192):
    """Send one or more images + text prompt to the LLM."""
    if isinstance(pil_images, Image.Image):
        pil_images = [pil_images]

    def _to_png_bytes(img):
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    if provider.lower() == "google":
        contents = list(pil_images) + [prompt]
        response = client.models.generate_content(model=model, contents=contents)
        return response.text

    elif provider.lower() == "anthropic":
        content_blocks = []
        for img in pil_images:
            b64 = base64.b64encode(_to_png_bytes(img)).decode("utf-8")
            content_blocks.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": b64,
                    },
                }
            )
        content_blocks.append({"type": "text", "text": prompt})
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": content_blocks}],
        )
        return response.content[0].text

    elif provider.lower() == "openai":
        content_blocks = []
        for img in pil_images:
            b64 = base64.b64encode(_to_png_bytes(img)).decode("utf-8")
            content_blocks.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                }
            )
        content_blocks.append({"type": "text", "text": prompt})
        response = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": content_blocks}],
        )
        return response.choices[0].message.content

    else:
        raise ValueError(f"Unsupported provider for vision: {provider}")


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

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


def _get_font(size=14):
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


def _draw_numbered_boxes(pil_img, detections):
    """Draw bounding boxes with index numbers on the image."""
    base = pil_img.convert("RGB").copy()
    draw = ImageDraw.Draw(base)
    font = _get_font(14)

    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det["bbox"]
        color = _PALETTE[det["class_id"] % len(_PALETTE)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label = f"[{i}] {det['class_name']}"
        bbox_text = draw.textbbox((0, 0), label, font=font)
        tw, th = bbox_text[2] - bbox_text[0], bbox_text[3] - bbox_text[1]
        lx, ly = x1, max(y1 - th - 6, 0)
        draw.rectangle([lx, ly, lx + tw + 8, ly + th + 4], fill=color)
        draw.text((lx + 4, ly + 2), label, fill="white", font=font)

    return base


def _build_sample_grid(entries, max_samples=9, thumb_size=640):
    """Build a grid of sample annotated screenshots for visual context.

    Picks diverse images: high annotation count, menu screens, gameplay.
    Uses larger thumbnails so boxes are clearly visible to the LLM.
    """
    candidates = [
        e
        for e in entries
        if e.get("detections") and e.get("review_status") != "human_reviewed_empty"
    ]
    if not candidates:
        return None

    # Sort by annotation count, pick evenly spread samples
    candidates.sort(key=lambda e: len(e["detections"]), reverse=True)
    if len(candidates) > max_samples:
        step = len(candidates) / max_samples
        candidates = [candidates[int(i * step)] for i in range(max_samples)]

    thumbs = []
    font = _get_font(16)
    for entry in candidates:
        try:
            img = Image.open(entry["image_path"])
            img.load()
        except Exception:
            continue

        annotated = _draw_numbered_boxes(img, entry["detections"])

        # Add filename banner at top
        draw = ImageDraw.Draw(annotated)
        banner_h = 24
        draw.rectangle([0, 0, annotated.width, banner_h], fill="black")
        draw.text((6, 4), entry["image_name"], fill="yellow", font=font)

        # Resize to thumbnail
        ratio = thumb_size / max(annotated.width, annotated.height)
        new_size = (int(annotated.width * ratio), int(annotated.height * ratio))
        annotated = annotated.resize(new_size, Image.LANCZOS)
        thumbs.append(annotated)

    if not thumbs:
        return None

    cols = min(3, len(thumbs))
    rows = math.ceil(len(thumbs) / cols)
    cell_w = max(t.width for t in thumbs)
    cell_h = max(t.height for t in thumbs)
    grid = Image.new("RGB", (cols * cell_w, rows * cell_h), (30, 30, 30))

    for idx, thumb in enumerate(thumbs):
        r, c = idx // cols, idx % cols
        grid.paste(thumb, (c * cell_w, r * cell_h))

    return grid


# ---------------------------------------------------------------------------
# YOLO label I/O
# ---------------------------------------------------------------------------


def _load_class_names(yolo_dataset_path):
    yaml_path = Path(yolo_dataset_path) / "dataset.yaml"
    if not yaml_path.exists():
        return {}
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    names = data.get("names", {})
    return {int(k): v for k, v in names.items()}


def _save_class_names(yolo_dataset_path, class_names):
    yaml_path = Path(yolo_dataset_path) / "dataset.yaml"
    yaml_content = {
        "names": {int(k): v for k, v in sorted(class_names.items())},
        "path": ".",
        "train": "train/images",
        "val": "val/images",
    }
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_content, f, default_flow_style=False)


def _load_yolo_labels(label_file, img_w, img_h, class_names):
    detections = []
    if not Path(label_file).exists():
        return detections
    text = Path(label_file).read_text().strip()
    if not text:
        return detections
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cid = int(parts[0])
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
                "class_id": cid,
                "class_name": class_names.get(cid, f"class_{cid}"),
                "confidence": 1.0,
                "bbox": (x1, y1, x2, y2),
            }
        )
    return detections


def _write_yolo_labels(label_file, detections, img_w, img_h):
    lines = []
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        xc = ((x1 + x2) / 2) / img_w
        yc = ((y1 + y2) / 2) / img_h
        w = (x2 - x1) / img_w
        h = (y2 - y1) / img_h
        lines.append(f"{det['class_id']} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
    Path(label_file).write_text("\n".join(lines) + ("\n" if lines else ""))


# ---------------------------------------------------------------------------
# Local pre-checks (run before LLM, fix what we can without AI)
# ---------------------------------------------------------------------------


def _find_orphaned_class_ids(entries, class_names):
    """Find class IDs used in labels but missing from dataset.yaml."""
    orphaned = {}
    for entry in entries:
        for det in entry.get("detections", []):
            cid = det["class_id"]
            if cid not in class_names:
                orphaned.setdefault(cid, 0)
                orphaned[cid] += 1
    return orphaned


def _run_local_prechecks(entries, class_names, yolo_path, dry_run=False):
    """Run local validation before calling the LLM.

    Returns (updated_class_names, local_fixes_count).
    """
    fixes = 0
    labels_dir = yolo_path / "train" / "labels"

    # Check 1: Orphaned class IDs
    orphaned = _find_orphaned_class_ids(entries, class_names)
    if orphaned:
        print(
            f"\n  LOCAL FIX: Found {len(orphaned)} class IDs in labels missing from dataset.yaml:"
        )
        for cid, count in sorted(orphaned.items()):
            placeholder = f"class_{cid}"
            print(f"    id={cid}: used {count} times -> adding as '{placeholder}'")
            class_names[cid] = placeholder
            fixes += count

        if not dry_run:
            _save_class_names(yolo_path, class_names)
            print("    Updated dataset.yaml with new class IDs")

        # Update detection names in entries
        for entry in entries:
            for det in entry.get("detections", []):
                if det["class_name"].startswith("class_"):
                    det["class_name"] = class_names.get(
                        det["class_id"], det["class_name"]
                    )

    # Check 2: Class distribution sanity
    class_counts = {}
    for entry in entries:
        for det in entry.get("detections", []):
            cname = det["class_name"]
            class_counts[cname] = class_counts.get(cname, 0) + 1
    if class_counts:
        print(f"\n  Class distribution:")
        for cname, count in sorted(class_counts.items(), key=lambda x: -x[1]):
            print(f"    {cname}: {count}")

    return class_names, fixes


# ---------------------------------------------------------------------------
# Build the single batch prompt (optimized for smaller models)
# ---------------------------------------------------------------------------

_BATCH_PROMPT = """You review YOLO annotations for a Cuphead game AI.

CLASSES:
{class_list}

IMAGES ({reviewable_count} to review, {empty_count} intentionally empty -- skip those):

{entries_text}

TASK: Find annotation errors. Return JSON only.

RULES:
1. Only fix what you are confident about.
2. Never touch "empty" status images -- user left them blank on purpose.
3. For "reviewed" images: fix wrong labels and bad boxes. Only add missing objects if obvious.
4. For "unreviewed" images: fix anything wrong.
5. class names must be from the CLASSES list above.
6. bbox = [x1, y1, x2, y2] in pixels (top-left to bottom-right).
7. "index" refers to the annotation index in that image's list.

OUTPUT FORMAT:
```json
{{"corrections":{{"filename.png":{{"changes":[{{"action":"relabel","index":0,"new_class":"name","reason":"why"}},{{"action":"remove","index":1,"reason":"why"}},{{"action":"add","class":"name","bbox":[x1,y1,x2,y2],"reason":"why"}}]}}}},"patterns":["pattern1"],"summary":"one line"}}
```

Only include files that need changes. Return ONLY JSON."""


def _build_batch_payload(entries, class_names):
    """Build compact text payload. No nested JSON -- use plain text table."""
    class_list = ", ".join(
        f"{cid}={cname}" for cid, cname in sorted(class_names.items())
    )

    lines = []
    reviewable_count = 0
    empty_count = 0

    for entry in entries:
        status = entry.get("review_status", "unreviewed")
        if status == "human_reviewed_empty":
            empty_count += 1
            continue
        reviewable_count += 1

        # Compact format: filename | WxH | status | annotations
        fname = entry["image_name"]
        dims = f"{entry['image_width']}x{entry['image_height']}"
        short_status = "reviewed" if status == "human_reviewed" else "unreviewed"

        anns = []
        for i, det in enumerate(entry["detections"]):
            x1, y1, x2, y2 = det["bbox"]
            conf = ""
            if det.get("confidence", 1.0) < 1.0:
                conf = f" {det['confidence']:.0%}"
            anns.append(
                f"[{i}]{det['class_name']}({round(x1)},{round(y1)},{round(x2)},{round(y2)}){conf}"
            )

        ann_str = " ".join(anns) if anns else "(none)"
        lines.append(f"{fname} | {dims} | {short_status} | {ann_str}")

    entries_text = "\n".join(lines)

    prompt = _BATCH_PROMPT.format(
        class_list=class_list,
        reviewable_count=reviewable_count,
        empty_count=empty_count,
        entries_text=entries_text,
    )

    return prompt


# ---------------------------------------------------------------------------
# Parse batch LLM response
# ---------------------------------------------------------------------------


def _parse_response(response_text):
    """Extract the JSON object from the LLM response."""
    text = response_text.strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
    return None


# ---------------------------------------------------------------------------
# Apply corrections to a single image's detections
# ---------------------------------------------------------------------------


def _name_to_id(class_name, class_names):
    for cid, cname in class_names.items():
        if cname.lower() == class_name.lower():
            return cid
    return None


def apply_corrections(detections, changes, class_names, img_w, img_h):
    """Apply a list of change dicts to detections. Returns (new_dets, descriptions)."""
    if not changes:
        return detections, []

    new_dets = [dict(d) for d in detections]
    descriptions = []

    remove_indices = set()
    for change in changes:
        if change.get("action", "").lower() == "remove":
            idx = change.get("index")
            if idx is not None and 0 <= idx < len(new_dets):
                descriptions.append(
                    f"  REMOVE [{idx}] {new_dets[idx]['class_name']}: {change.get('reason', '')}"
                )
                remove_indices.add(idx)

    for change in changes:
        action = change.get("action", "").lower()
        idx = change.get("index")

        if action == "relabel" and idx is not None and 0 <= idx < len(new_dets):
            if idx in remove_indices:
                continue
            new_class = change.get("new_class", "")
            cid = _name_to_id(new_class, class_names)
            if cid is None:
                descriptions.append(
                    f"  SKIP relabel [{idx}]: unknown class '{new_class}'"
                )
                continue
            old_name = new_dets[idx]["class_name"]
            new_dets[idx]["class_id"] = cid
            new_dets[idx]["class_name"] = class_names[cid]
            descriptions.append(
                f"  RELABEL [{idx}] {old_name} -> {class_names[cid]}: {change.get('reason', '')}"
            )

        elif action == "adjust" and idx is not None and 0 <= idx < len(new_dets):
            if idx in remove_indices:
                continue
            bbox = change.get("bbox")
            if bbox and len(bbox) == 4:
                new_dets[idx]["bbox"] = (
                    max(0, min(bbox[0], img_w)),
                    max(0, min(bbox[1], img_h)),
                    max(0, min(bbox[2], img_w)),
                    max(0, min(bbox[3], img_h)),
                )
                descriptions.append(
                    f"  ADJUST [{idx}] {new_dets[idx]['class_name']}: {change.get('reason', '')}"
                )

    for idx in sorted(remove_indices, reverse=True):
        new_dets.pop(idx)

    for change in changes:
        if change.get("action", "").lower() == "add":
            class_name = change.get("class", "")
            cid = _name_to_id(class_name, class_names)
            if cid is None:
                descriptions.append(f"  SKIP add: unknown class '{class_name}'")
                continue
            bbox = change.get("bbox")
            if not bbox or len(bbox) != 4:
                descriptions.append(f"  SKIP add {class_name}: invalid bbox")
                continue
            x1 = max(0, min(bbox[0], img_w))
            y1 = max(0, min(bbox[1], img_h))
            x2 = max(0, min(bbox[2], img_w))
            y2 = max(0, min(bbox[3], img_h))
            new_dets.append(
                {
                    "class_id": cid,
                    "class_name": class_names[cid],
                    "confidence": 1.0,
                    "bbox": (x1, y1, x2, y2),
                }
            )
            descriptions.append(
                f"  ADD {class_names[cid]} at [{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}]: {change.get('reason', '')}"
            )

    return new_dets, descriptions


# ---------------------------------------------------------------------------
# Main auto-review pipeline
# ---------------------------------------------------------------------------


def auto_review(
    results=None,
    yolo_dataset_path=None,
    screenshots_dir=None,
    max_images=None,
    dry_run=False,
    progress_callback=None,
):
    """Run LLM-powered auto-review on the entire dataset in a single API call."""
    yolo_path = Path(yolo_dataset_path or (PROJECT_ROOT / "yolo_dataset"))
    ss_dir = Path(screenshots_dir or (PROJECT_ROOT / "screenshots" / "captures"))
    labels_dir = yolo_path / "train" / "labels"
    images_dir = yolo_path / "train" / "images"

    labels_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    class_names = _load_class_names(yolo_path)
    if not class_names:
        print("ERROR: No class names found in dataset.yaml")
        return {"error": "No class names found"}

    # Build set of stems that have label files (human-reviewed)
    trained_stems = set()
    for lbl in labels_dir.glob("*.txt"):
        trained_stems.add(lbl.stem)

    # Build entries
    if progress_callback:
        progress_callback(0, 1, "Loading annotations...")

    entries = []
    if results:
        for entry in results:
            entry = dict(entry)
            stem = Path(entry["image_name"]).stem
            if stem in trained_stems:
                lbl = labels_dir / f"{stem}.txt"
                label_text = lbl.read_text().strip() if lbl.exists() else ""
                if not label_text and not entry.get("detections"):
                    entry["review_status"] = "human_reviewed_empty"
                else:
                    entry["review_status"] = "human_reviewed"
            else:
                entry["review_status"] = "unreviewed"
            entries.append(entry)
    else:
        images = sorted(list(ss_dir.glob("*.png")) + list(ss_dir.glob("*.jpg")))
        for img_path in images:
            try:
                with Image.open(img_path) as im:
                    img_w, img_h = im.size
            except Exception:
                continue
            label_file = labels_dir / (img_path.stem + ".txt")
            dets = _load_yolo_labels(label_file, img_w, img_h, class_names)

            if img_path.stem in trained_stems:
                label_text = (
                    label_file.read_text().strip() if label_file.exists() else ""
                )
                review_status = (
                    "human_reviewed_empty" if not label_text else "human_reviewed"
                )
            else:
                review_status = "unreviewed"

            entries.append(
                {
                    "image_path": str(img_path.resolve()),
                    "image_name": img_path.name,
                    "image_width": img_w,
                    "image_height": img_h,
                    "detections": dets,
                    "review_status": review_status,
                }
            )

    if max_images:
        entries = entries[:max_images]

    empty_count = sum(
        1 for e in entries if e.get("review_status") == "human_reviewed_empty"
    )
    reviewable = [
        e for e in entries if e.get("review_status") != "human_reviewed_empty"
    ]

    print(f"\nDataset: {len(entries)} images total")
    print(f"  Reviewable: {len(reviewable)}")
    print(f"  Intentionally empty (skipped): {empty_count}")

    if not reviewable:
        print("Nothing to review.")
        return {
            "total_images": len(entries),
            "images_changed": 0,
            "skipped": len(entries),
        }

    # Phase 1: Local pre-checks (no LLM needed)
    if progress_callback:
        progress_callback(0, 1, "Running local checks...")
    print("\n--- Phase 1: Local Pre-checks ---")
    class_names, local_fixes = _run_local_prechecks(
        entries, class_names, yolo_path, dry_run
    )

    # Phase 2: LLM review
    print("\n--- Phase 2: LLM Review ---")

    client, provider, model = _get_llm_client()
    print(f"Using {provider}/{model} (single call)")

    if progress_callback:
        progress_callback(0, 1, "Building sample grid...")
    print("Building sample grid...")
    grid_img = _build_sample_grid(entries)

    if progress_callback:
        progress_callback(0, 1, "Building prompt...")
    print("Building prompt...")
    prompt = _build_batch_payload(entries, class_names)
    prompt_kb = len(prompt.encode("utf-8")) / 1024
    print(f"Prompt size: {prompt_kb:.0f} KB")

    if progress_callback:
        progress_callback(0, 1, "Sending to LLM (this may take a minute)...")
    print(f"Sending to {provider}/{model}...")

    try:
        images_to_send = [grid_img] if grid_img else []
        response = _call_llm_vision(
            client, provider, model, prompt, images_to_send, max_tokens=8192
        )
    except Exception as e:
        print(f"ERROR: LLM call failed: {e}")
        return {"error": str(e)}

    if progress_callback:
        progress_callback(0, 1, "Parsing response...")
    print("Parsing response...")

    result = _parse_response(response)
    if result is None:
        print("ERROR: Could not parse LLM response")
        print(f"Raw (first 500): {response[:500]}")
        return {"error": "Could not parse response"}

    file_corrections = result.get("corrections", {})
    patterns = result.get("patterns", result.get("patterns_found", []))
    summary = result.get("summary", "")

    print(f"\nLLM Summary: {summary}")
    if patterns:
        print("Patterns found:")
        for p in patterns:
            print(f"  - {p}")

    print(f"Files with corrections: {len(file_corrections)}")

    # Apply corrections
    entry_map = {e["image_name"]: e for e in entries}

    stats = {
        "total_images": len(entries),
        "images_changed": 0,
        "total_corrections": local_fixes,
        "removals": 0,
        "relabels": 0,
        "additions": 0,
        "adjustments": 0,
        "errors": 0,
        "skipped": len(entries) - len(file_corrections),
    }

    for filename, file_data in file_corrections.items():
        entry = entry_map.get(filename)
        if not entry:
            print(f"\n  WARNING: {filename} not found in dataset, skipping")
            stats["errors"] += 1
            continue

        if entry.get("review_status") == "human_reviewed_empty":
            print(f"\n  SKIP {filename}: intentionally empty")
            continue

        changes = file_data.get("changes", [])
        if not changes:
            continue

        img_w = entry["image_width"]
        img_h = entry["image_height"]
        detections = entry["detections"]

        print(f"\n  {filename}: {len(changes)} correction(s)")

        new_dets, descriptions = apply_corrections(
            detections, changes, class_names, img_w, img_h
        )
        for desc in descriptions:
            print(desc)

        for change in changes:
            action = change.get("action", "").lower()
            if action == "remove":
                stats["removals"] += 1
            elif action == "relabel":
                stats["relabels"] += 1
            elif action == "add":
                stats["additions"] += 1
            elif action == "adjust":
                stats["adjustments"] += 1
        stats["total_corrections"] += len(changes)
        stats["images_changed"] += 1

        if dry_run:
            print("    (dry run)")
            continue

        stem = Path(filename).stem
        label_file = labels_dir / f"{stem}.txt"
        _write_yolo_labels(label_file, new_dets, img_w, img_h)

        dest_img = images_dir / filename
        if not dest_img.exists():
            src = Path(entry["image_path"])
            if src.exists():
                shutil.copy2(src, dest_img)

        print(f"    Saved {label_file.name}")

    if not dry_run and stats["images_changed"] > 0:
        _save_class_names(yolo_path, class_names)

    # Summary
    print(f"\n{'=' * 60}")
    print("Auto-Review Summary")
    print(f"{'=' * 60}")
    print(f"  Images in dataset: {stats['total_images']}")
    print(f"  Images changed:    {stats['images_changed']}")
    print(f"  Total corrections: {stats['total_corrections']}")
    print(f"    Removals:        {stats['removals']}")
    print(f"    Relabels:        {stats['relabels']}")
    print(f"    Additions:       {stats['additions']}")
    print(f"    Adjustments:     {stats['adjustments']}")
    print(f"  Errors:            {stats['errors']}")
    if dry_run:
        print(f"\n  (DRY RUN -- no files modified)")
    print(f"{'=' * 60}")

    return stats


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="LLM-powered annotation auto-review (batch)"
    )
    parser.add_argument("--yolo-dataset", type=str, default=None)
    parser.add_argument("--screenshots", type=str, default=None)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--use-predictions", action="store_true")
    args = parser.parse_args()

    results = None
    if args.use_predictions:
        print("Running model inference first...")
        from pipeline.test_model import test_model

        results = test_model()
        if results is None:
            print("ERROR: Inference failed.")
            sys.exit(1)

    stats = auto_review(
        results=results,
        yolo_dataset_path=args.yolo_dataset,
        screenshots_dir=args.screenshots,
        max_images=args.max_images,
        dry_run=args.dry_run,
    )
    if stats.get("error"):
        sys.exit(1)


if __name__ == "__main__":
    main()
