"""
Label Quality Audit — Claude vision evaluates dataset annotations before training.

Golden Rule #4: Always have Claude evaluate <=300 images before training.

Samples up to 300 auto-labeled images, draws bounding boxes on them,
sends batches to Claude vision for quality assessment. Reports issues
found and optionally fixes them.

Requires ANTHROPIC_API_KEY in .env or system keyring.

Usage:
    python pipeline/audit_labels.py [--max-images 300] [--batch-size 5] [--fix]
    python pipeline/audit_labels.py --dry-run   # just show what would be audited
"""

import argparse
import base64
import io
import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image, ImageDraw

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "yolo_dataset"

# Colors per class (RGB)
CLASS_COLORS = {
    0: (0, 255, 0),  # player — green
    1: (255, 0, 0),  # enemy — red
    2: (255, 255, 0),  # projectile — yellow
    3: (0, 255, 255),  # platform — cyan
}

CLASS_NAMES = ["player", "enemy", "projectile", "platform"]

AUDIT_PROMPT = """\
You are auditing YOLO bounding box annotations for Cuphead gameplay frames.
The game is a 2D side-scrolling shooter with hand-drawn cartoon art.

CLASSES:
  0=player (green box): Cuphead (red/white cup character) or Mugman (blue/white cup character)
  1=enemy (red box): All hostile entities — bosses, minions, obstacles
  2=projectile (yellow box): Bullets, fireballs, shots, environmental hazards
  3=platform (cyan box): Platforms the player can stand on

GOLDEN RULES (violations are ERRORS):
  - Max 1 player per image
  - Max 1 healthbar per image
  - Player is ALWAYS Cuphead (red/white cup head) or Mugman (blue/white cup head)

For each image, evaluate the annotations and report:
  - "ok" if all labels are correct
  - List specific issues if any labels are wrong

Common problems to check:
  - Non-player objects labeled as player (enemies, NPCs, decorations)
  - Player not detected at all when clearly visible
  - Enemy boxes that are too large or too small
  - Healthbar labeled when no HP badge is visible (non-gameplay frames)
  - Multiple player or healthbar labels

Respond in JSON format:
{
  "results": [
    {
      "image": "filename.jpg",
      "verdict": "ok" | "issues",
      "issues": ["description of issue 1", ...],
      "suggested_fixes": ["remove player box at [x1,y1,x2,y2]", ...]
    },
    ...
  ],
  "summary": {
    "total": N,
    "ok": N,
    "issues": N,
    "common_problems": ["...", ...]
  }
}
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_anthropic_client():
    """Get Anthropic client using project-wide API key lookup."""
    from tools.functions import get_api_key

    api_key = get_api_key()
    if not api_key:
        raise RuntimeError(
            "API key not found. Store via keyring or set API_KEY in .env"
        )

    import anthropic

    return anthropic.Anthropic(api_key=api_key)


def draw_annotations(img_path, label_path):
    """Draw bounding boxes on image, return annotated PIL Image."""
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size

    if not os.path.exists(label_path):
        return img

    with open(label_path) as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls = int(parts[0])
        cx, cy, bw, bh = (
            float(parts[1]),
            float(parts[2]),
            float(parts[3]),
            float(parts[4]),
        )
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)

        color = CLASS_COLORS.get(cls, (255, 255, 255))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        label = f"{CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else cls}"
        tw = len(label) * 6 + 4
        draw.rectangle([x1, y1 - 12, x1 + tw, y1], fill=color)
        draw.text((x1 + 2, y1 - 11), label, fill=(0, 0, 0))

    return img


def _img_to_b64(img, max_size=800):
    """Convert PIL Image to base64 PNG, resize if needed for API efficiency."""
    w, h = img.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def sample_auto_labeled(split="train", max_images=300, seed=42):
    """Sample auto-labeled (non-screenshot) images from a dataset split."""
    img_dir = DATASET_DIR / split / "images"
    label_dir = DATASET_DIR / split / "labels"

    candidates = []
    for f in sorted(os.listdir(img_dir)):
        if f.startswith("screenshot_"):
            continue  # skip manually annotated
        if not f.endswith((".jpg", ".png")):
            continue
        base = Path(f).stem
        label_path = label_dir / f"{base}.txt"
        if label_path.exists():
            candidates.append((img_dir / f, label_path))

    random.seed(seed)
    if len(candidates) > max_images:
        candidates = random.sample(candidates, max_images)

    return candidates


def audit_batch(client, batch, model="claude-sonnet-4-20250514"):
    """Send a batch of annotated images to Claude for evaluation.

    Args:
        client: Anthropic client
        batch: list of (img_path, label_path) tuples
        model: Claude model to use

    Returns:
        Parsed JSON response from Claude
    """
    content_blocks = []

    # Add each annotated image
    for img_path, label_path in batch:
        annotated = draw_annotations(img_path, label_path)
        b64 = _img_to_b64(annotated)

        content_blocks.append(
            {
                "type": "text",
                "text": f"Image: {Path(img_path).name}",
            }
        )
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

    content_blocks.append(
        {
            "type": "text",
            "text": AUDIT_PROMPT,
        }
    )

    response = client.messages.create(
        model=model,
        max_tokens=4096,
        messages=[{"role": "user", "content": content_blocks}],
    )

    raw = response.content[0].text

    # Extract JSON from response
    try:
        # Try direct parse
        return json.loads(raw)
    except json.JSONDecodeError:
        # Try extracting from markdown code block
        if "```json" in raw:
            json_str = raw.split("```json")[1].split("```")[0].strip()
            return json.loads(json_str)
        elif "```" in raw:
            json_str = raw.split("```")[1].split("```")[0].strip()
            return json.loads(json_str)
        else:
            print(f"WARNING: Could not parse JSON from response:\n{raw[:500]}")
            return {"results": [], "summary": {"error": "parse_failed"}, "raw": raw}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_audit(
    max_images=300, batch_size=5, model="claude-sonnet-4-20250514", dry_run=False
):
    """Run the full label quality audit.

    Args:
        max_images: Max images to audit (<=300 per golden rule)
        batch_size: Images per API call (5 is good balance of context vs cost)
        model: Claude model for vision
        dry_run: If True, just show what would be audited
    """
    max_images = min(max_images, 300)  # Golden rule: <=300

    # Sample from both splits
    train_samples = sample_auto_labeled("train", max_images=int(max_images * 0.85))
    val_samples = sample_auto_labeled("val", max_images=max_images - len(train_samples))
    all_samples = train_samples + val_samples

    print(
        f"Audit: {len(all_samples)} images ({len(train_samples)} train, {len(val_samples)} val)"
    )

    if dry_run:
        for img_path, label_path in all_samples[:20]:
            with open(label_path) as f:
                n_labels = len(f.readlines())
            print(f"  {Path(img_path).name}: {n_labels} annotations")
        if len(all_samples) > 20:
            print(f"  ... and {len(all_samples) - 20} more")
        return

    client = _get_anthropic_client()

    all_results = []
    total_ok = 0
    total_issues = 0
    batches = [
        all_samples[i : i + batch_size] for i in range(0, len(all_samples), batch_size)
    ]

    print(f"Sending {len(batches)} batches of ~{batch_size} images to {model}...")
    print()

    for batch_idx, batch in enumerate(batches):
        print(
            f"  Batch {batch_idx + 1}/{len(batches)} ({len(batch)} images)...",
            end=" ",
            flush=True,
        )

        try:
            result = audit_batch(client, batch, model=model)
            results = result.get("results", [])
            all_results.extend(results)

            ok = sum(1 for r in results if r.get("verdict") == "ok")
            bad = len(results) - ok
            total_ok += ok
            total_issues += bad
            print(f"{ok} ok, {bad} issues")

            # Print issues inline
            for r in results:
                if r.get("verdict") != "ok" and r.get("issues"):
                    print(f"    {r['image']}: {'; '.join(r['issues'])}")

        except Exception as e:
            print(f"ERROR: {e}")

        # Rate limiting
        if batch_idx < len(batches) - 1:
            time.sleep(1)

    # Summary
    print()
    print("=" * 60)
    print(f"AUDIT COMPLETE: {len(all_results)} images evaluated")
    print(f"  OK:     {total_ok}")
    print(f"  Issues: {total_issues}")
    if total_issues > 0:
        print(f"  Error rate: {total_issues / len(all_results) * 100:.1f}%")
    print("=" * 60)

    # Save report
    report_path = PROJECT_ROOT / "audit_report.json"
    report = {
        "total_images": len(all_results),
        "ok": total_ok,
        "issues": total_issues,
        "error_rate": total_issues / max(len(all_results), 1),
        "results": all_results,
        "model": model,
    }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nFull report saved to: {report_path}")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Audit dataset labels with Claude vision"
    )
    parser.add_argument(
        "--max-images", type=int, default=300, help="Max images to audit (<=300)"
    )
    parser.add_argument("--batch-size", type=int, default=5, help="Images per API call")
    parser.add_argument(
        "--model", default="claude-sonnet-4-20250514", help="Claude model"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be audited"
    )
    parser.add_argument(
        "--fix", action="store_true", help="Apply suggested fixes (not yet implemented)"
    )
    args = parser.parse_args()

    run_audit(
        max_images=args.max_images,
        batch_size=args.batch_size,
        model=args.model,
        dry_run=args.dry_run,
    )
