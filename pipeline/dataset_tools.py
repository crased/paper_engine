"""
Dataset management tools for Paper Engine.

Provides cleanup, merge, augmentation, and summary utilities for YOLO datasets.

Usage:
    python -m pipeline.dataset_tools --summary
    python -m pipeline.dataset_tools --cleanup [--dry-run]
    python -m pipeline.dataset_tools --merge /path/to/dataset [--auto-remap] [--dry-run]
    python -m pipeline.dataset_tools --augment [--aug-count N] [--dry-run]
"""

import argparse
import hashlib
import random
import shutil
import sys
import yaml
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

GAMEPLAY_CLASSES = {0: "player", 1: "enemy", 2: "projectile", 3: "platform"}

# Heuristic keywords for auto-remapping external class names to our 4-class schema
# Priority-ordered: projectile checked before enemy so compound names like
# "carrot_projectile" or "onion_boss_projectile" map to projectile, not enemy.
_REMAP_RULES = [
    (2, {"projectile", "bullet", "shot", "missile", "fireball", "seed",
         "attack", "beam", "bomb", "hazard", "orb", "spark", "ring",
         "blaster"}),
    (0, {"player", "cuphead", "mugman", "chalice"}),
    (1, {"enemy", "boss", "minion", "foe", "mob", "npc", "hostile",
         "runner", "carrot", "potato", "onion", "candy", "dragon", "genie",
         "devil", "flower", "slime", "bat", "lobster", "pirate", "robot",
         "train", "phantom", "bee", "queen", "king", "knight", "djimmi",
         "baroness", "beppi", "cagney", "cala", "captain", "dice", "grim",
         "hilda", "mortimer", "rumor", "sally", "wally", "werner", "worm",
         "parry"}),
    (3, {"platform", "ground", "floor", "ledge", "wall", "block", "cloud_platform"}),
]


@dataclass
class CleanupStats:
    total_images: int = 0
    total_labels: int = 0
    orphan_labels_removed: int = 0
    empty_labels_removed: int = 0
    empty_labels_kept: int = 0
    cache_files_removed: int = 0


@dataclass
class MergeStats:
    source_images: int = 0
    images_added_train: int = 0
    images_added_val: int = 0
    images_skipped_duplicate: int = 0
    images_skipped_no_gameplay: int = 0
    labels_remapped: int = 0
    class_counts_added: dict = field(default_factory=lambda: Counter())
    remap_table: dict = field(default_factory=dict)
    unmapped_classes: list = field(default_factory=list)


def _image_stems(directory):
    """Return set of stems for all image files in a directory."""
    if not directory.exists():
        return set()
    return {f.stem for f in directory.iterdir() if f.suffix.lower() in IMAGE_EXTS}


def _label_stems(directory):
    """Return set of stems for all .txt label files in a directory."""
    if not directory.exists():
        return set()
    return {f.stem for f in directory.iterdir() if f.suffix == ".txt"}


def _count_instances(label_dir, stems=None):
    """Count class instances across label files. Returns Counter {class_id: count}."""
    counts = Counter()
    if not label_dir.exists():
        return counts
    for f in label_dir.iterdir():
        if f.suffix != ".txt":
            continue
        if stems is not None and f.stem not in stems:
            continue
        text = f.read_text().strip()
        if not text:
            continue
        for line in text.split("\n"):
            parts = line.strip().split()
            if parts:
                try:
                    counts[int(parts[0])] += 1
                except (ValueError, IndexError):
                    pass
    return counts


def _is_empty_label(label_path):
    """Check if a label file has no bounding boxes."""
    text = label_path.read_text().strip()
    return len(text) == 0


def _file_hash(path):
    """Compute SHA-256 hash of a file for deduplication."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def dataset_summary(dataset_dir=None):
    """Print formatted dataset statistics."""
    dataset_dir = Path(dataset_dir) if dataset_dir else PROJECT_ROOT / "yolo_dataset"

    for split in ["train", "val"]:
        img_dir = dataset_dir / split / "images"
        lbl_dir = dataset_dir / split / "labels"
        imgs = _image_stems(img_dir)
        lbls = _label_stems(lbl_dir)

        orphan_labels = lbls - imgs
        orphan_images = imgs - lbls
        matched = imgs & lbls

        empty = 0
        populated = 0
        for stem in matched:
            if _is_empty_label(lbl_dir / f"{stem}.txt"):
                empty += 1
            else:
                populated += 1

        counts = _count_instances(lbl_dir, matched)

        print(f"\n{'=' * 50}")
        print(f"  {split.upper()} SPLIT")
        print(f"{'=' * 50}")
        print(f"  Images:           {len(imgs)}")
        print(f"  Labels:           {len(lbls)}")
        print(f"  Matched pairs:    {len(matched)}")
        print(f"  Orphan labels:    {len(orphan_labels)}")
        print(f"  Orphan images:    {len(orphan_images)}")
        print(f"  Empty labels:     {empty}")
        print(f"  Populated labels: {populated}")
        print(f"\n  Class distribution:")
        for cid in sorted(counts.keys()):
            name = GAMEPLAY_CLASSES.get(cid, f"class_{cid}")
            print(f"    {cid} ({name}): {counts[cid]}")
        total = sum(counts.values())
        print(f"    Total instances: {total}")

    print()


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

def cleanup_dataset(dataset_dir=None, dry_run=False, max_negative_pct=0.10):
    """Remove orphan labels and excess empty labels. Returns CleanupStats."""
    dataset_dir = Path(dataset_dir) if dataset_dir else PROJECT_ROOT / "yolo_dataset"
    stats = CleanupStats()

    for split in ["train", "val"]:
        img_dir = dataset_dir / split / "images"
        lbl_dir = dataset_dir / split / "labels"
        if not lbl_dir.exists():
            continue

        imgs = _image_stems(img_dir)
        lbls = _label_stems(lbl_dir)
        stats.total_images += len(imgs)
        stats.total_labels += len(lbls)

        # 1. Remove orphan labels
        orphans = lbls - imgs
        for stem in sorted(orphans):
            path = lbl_dir / f"{stem}.txt"
            print(f"  {'[DRY] ' if dry_run else ''}Remove orphan label: {split}/labels/{stem}.txt")
            if not dry_run:
                path.unlink()
            stats.orphan_labels_removed += 1

        # 2. Handle empty labels
        matched = imgs & lbls
        empty_stems = [s for s in sorted(matched) if _is_empty_label(lbl_dir / f"{s}.txt")]
        populated_count = len(matched) - len(empty_stems)
        total_after_orphan_removal = populated_count + len(empty_stems)

        # Keep up to max_negative_pct as background negatives
        max_keep = max(1, int(total_after_orphan_removal * max_negative_pct))
        keep_count = min(len(empty_stems), max_keep)

        # Keep a random sample, remove the rest
        random.seed(42)
        if len(empty_stems) > keep_count:
            keep_set = set(random.sample(empty_stems, keep_count))
        else:
            keep_set = set(empty_stems)

        for stem in empty_stems:
            if stem in keep_set:
                stats.empty_labels_kept += 1
                continue
            label_path = lbl_dir / f"{stem}.txt"
            image_path = None
            for ext in IMAGE_EXTS:
                candidate = img_dir / f"{stem}{ext}"
                if candidate.exists():
                    image_path = candidate
                    break

            print(f"  {'[DRY] ' if dry_run else ''}Remove empty: {split}/labels/{stem}.txt"
                  f"{f' + {split}/images/{image_path.name}' if image_path else ''}")
            if not dry_run:
                label_path.unlink()
                if image_path:
                    image_path.unlink()
            stats.empty_labels_removed += 1

    # 3. Delete .cache files
    for cache in dataset_dir.rglob("*.cache"):
        print(f"  {'[DRY] ' if dry_run else ''}Remove cache: {cache.relative_to(dataset_dir)}")
        if not dry_run:
            cache.unlink()
        stats.cache_files_removed += 1

    print(f"\n--- Cleanup {'preview' if dry_run else 'complete'} ---")
    print(f"  Orphan labels removed: {stats.orphan_labels_removed}")
    print(f"  Empty labels removed:  {stats.empty_labels_removed}")
    print(f"  Empty labels kept:     {stats.empty_labels_kept} (background negatives)")
    print(f"  Cache files removed:   {stats.cache_files_removed}")

    return stats


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

def build_cuphead_remap(source_class_names):
    """
    Heuristic mapper: source class names → our 4-class gameplay schema.

    Args:
        source_class_names: dict {class_id: class_name} from source data.yaml

    Returns:
        (remap_table, unmapped) where remap_table is {src_id: target_id}
    """
    remap = {}
    unmapped = []

    for src_id, src_name in source_class_names.items():
        src_lower = src_name.lower().replace("-", "_").replace(" ", "_")
        matched = False
        for target_id, keywords in _REMAP_RULES:
            for kw in keywords:
                if kw in src_lower:
                    remap[int(src_id)] = target_id
                    matched = True
                    break
            if matched:
                break
        if not matched:
            unmapped.append((src_id, src_name))

    return remap, unmapped


def _discover_source_layout(source_dir):
    """Find images and labels dirs in a Roboflow-style dataset."""
    source_dir = Path(source_dir)
    splits = {}

    # Roboflow exports: train/, valid/, test/ each with images/ and labels/
    for split_name in ["train", "valid", "test"]:
        img_dir = source_dir / split_name / "images"
        lbl_dir = source_dir / split_name / "labels"
        if img_dir.exists() and lbl_dir.exists():
            splits[split_name] = (img_dir, lbl_dir)

    # Flat layout: images/ and labels/ at root
    if not splits:
        img_dir = source_dir / "images"
        lbl_dir = source_dir / "labels"
        if img_dir.exists() and lbl_dir.exists():
            splits["all"] = (img_dir, lbl_dir)

    return splits


def _load_source_classes(source_dir):
    """Load class names from source data.yaml."""
    source_dir = Path(source_dir)
    for name in ["data.yaml", "dataset.yaml"]:
        yaml_path = source_dir / name
        if yaml_path.exists():
            with open(yaml_path) as f:
                data = yaml.safe_load(f)
            names = data.get("names", {})
            # Handle both dict and list formats
            if isinstance(names, list):
                return {i: n for i, n in enumerate(names)}
            return {int(k): v for k, v in names.items()}
    return None


def merge_external_dataset(
    source_dir,
    dataset_dir=None,
    class_remap=None,
    auto_remap=False,
    val_split=0.2,
    dry_run=False,
    prefix="ext_",
):
    """
    Merge an external YOLO dataset into ours with class remapping.

    Args:
        source_dir: Path to downloaded external dataset root
        dataset_dir: Target dataset dir (defaults to yolo_dataset/)
        class_remap: Dict {src_class_id: target_class_id}. None = auto or identity.
        auto_remap: Use heuristic Cuphead class name matching
        val_split: Fraction of images for validation
        dry_run: Preview without copying
        prefix: Filename prefix to avoid collisions

    Returns:
        MergeStats
    """
    source_dir = Path(source_dir)
    dataset_dir = Path(dataset_dir) if dataset_dir else PROJECT_ROOT / "yolo_dataset"
    stats = MergeStats()

    # Load source class names
    source_classes = _load_source_classes(source_dir)
    if source_classes is None:
        print("ERROR: No data.yaml or dataset.yaml found in source directory")
        return stats

    print(f"Source classes: {source_classes}")

    # Build remap table
    if class_remap:
        remap = {int(k): int(v) for k, v in class_remap.items()}
        unmapped = [(k, source_classes.get(k, "?")) for k in source_classes if k not in remap]
    elif auto_remap:
        remap, unmapped = build_cuphead_remap(source_classes)
    else:
        # Identity mapping for classes 0-3, drop the rest
        remap = {i: i for i in range(4) if i in source_classes}
        unmapped = [(k, v) for k, v in source_classes.items() if k not in remap]

    stats.remap_table = remap
    stats.unmapped_classes = unmapped

    print(f"\nClass remapping:")
    for src_id, target_id in sorted(remap.items()):
        src_name = source_classes.get(src_id, "?")
        target_name = GAMEPLAY_CLASSES.get(target_id, "?")
        print(f"  {src_id} ({src_name}) → {target_id} ({target_name})")
    if unmapped:
        print(f"\nUnmapped (will be dropped):")
        for src_id, src_name in unmapped:
            print(f"  {src_id} ({src_name})")

    # Discover source layout
    splits = _discover_source_layout(source_dir)
    if not splits:
        print("ERROR: No valid image/label directories found in source")
        return stats

    # Build hash set of existing images for dedup
    existing_hashes = set()
    for split in ["train", "val"]:
        img_dir = dataset_dir / split / "images"
        if img_dir.exists():
            for f in img_dir.iterdir():
                if f.suffix.lower() in IMAGE_EXTS:
                    existing_hashes.add(_file_hash(f))

    print(f"\nExisting images hashed: {len(existing_hashes)}")

    # Ensure target dirs exist
    for split in ["train", "val"]:
        (dataset_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (dataset_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    # Process source images
    random.seed(42)
    all_pairs = []
    for split_name, (img_dir, lbl_dir) in splits.items():
        for img_path in sorted(img_dir.iterdir()):
            if img_path.suffix.lower() not in IMAGE_EXTS:
                continue
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            if lbl_path.exists():
                all_pairs.append((img_path, lbl_path))

    stats.source_images = len(all_pairs)
    print(f"Source image-label pairs: {len(all_pairs)}")

    for img_path, lbl_path in all_pairs:
        # Dedup check
        img_hash = _file_hash(img_path)
        if img_hash in existing_hashes:
            stats.images_skipped_duplicate += 1
            continue

        # Remap labels
        text = lbl_path.read_text().strip()
        remapped_lines = []
        if text:
            for line in text.split("\n"):
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                src_class = int(parts[0])
                if src_class in remap:
                    target_class = remap[src_class]
                    remapped_lines.append(f"{target_class} {' '.join(parts[1:])}")
                    stats.class_counts_added[target_class] += 1

        if not remapped_lines:
            stats.images_skipped_no_gameplay += 1
            continue

        # Decide split
        target_split = "val" if random.random() < val_split else "train"
        target_img_dir = dataset_dir / target_split / "images"
        target_lbl_dir = dataset_dir / target_split / "labels"

        # Copy with prefix
        new_name = f"{prefix}{img_path.stem}"
        target_img = target_img_dir / f"{new_name}{img_path.suffix}"
        target_lbl = target_lbl_dir / f"{new_name}.txt"

        if not dry_run:
            shutil.copy2(img_path, target_img)
            target_lbl.write_text("\n".join(remapped_lines) + "\n")
            existing_hashes.add(img_hash)

        stats.labels_remapped += len(remapped_lines)
        if target_split == "train":
            stats.images_added_train += 1
        else:
            stats.images_added_val += 1

    # Delete .cache files
    if not dry_run:
        for cache in dataset_dir.rglob("*.cache"):
            cache.unlink()

    print(f"\n--- Merge {'preview' if dry_run else 'complete'} ---")
    print(f"  Source images:          {stats.source_images}")
    print(f"  Added to train:         {stats.images_added_train}")
    print(f"  Added to val:           {stats.images_added_val}")
    print(f"  Skipped (duplicate):    {stats.images_skipped_duplicate}")
    print(f"  Skipped (no gameplay):  {stats.images_skipped_no_gameplay}")
    print(f"  Labels remapped:        {stats.labels_remapped}")
    print(f"\n  Classes added:")
    for cid in sorted(stats.class_counts_added.keys()):
        name = GAMEPLAY_CLASSES.get(cid, f"class_{cid}")
        print(f"    {cid} ({name}): {stats.class_counts_added[cid]}")

    return stats


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

def _parse_yolo_labels(label_path):
    """Parse YOLO label file into list of (class_id, cx, cy, w, h)."""
    labels = []
    text = label_path.read_text().strip()
    if not text:
        return labels
    for line in text.split("\n"):
        parts = line.strip().split()
        if len(parts) >= 5:
            labels.append((int(parts[0]), *[float(x) for x in parts[1:5]]))
    return labels


def _yolo_to_albumentations(labels):
    """Convert YOLO (cx, cy, w, h) to albumentations (x_min, y_min, x_max, y_max) normalized."""
    bboxes = []
    class_ids = []
    for cid, cx, cy, w, h in labels:
        x_min = max(0.0, cx - w / 2)
        y_min = max(0.0, cy - h / 2)
        x_max = min(1.0, cx + w / 2)
        y_max = min(1.0, cy + h / 2)
        if x_max > x_min and y_max > y_min:
            bboxes.append([x_min, y_min, x_max, y_max])
            class_ids.append(cid)
    return bboxes, class_ids


def _albumentations_to_yolo(bboxes, class_ids):
    """Convert albumentations bboxes back to YOLO label lines."""
    lines = []
    for (x_min, y_min, x_max, y_max), cid in zip(bboxes, class_ids):
        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2
        w = x_max - x_min
        h = y_max - y_min
        if w > 0.001 and h > 0.001:  # skip degenerate boxes
            lines.append(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines


def augment_dataset(dataset_dir=None, base_count=2, dry_run=False, train_only=True):
    """
    Generate augmented copies of training images with proper transforms.

    Augmentations: horizontal flip, brightness/contrast, hue/saturation,
    gaussian noise, slight blur, random scale crop. All bbox-safe.

    Images with rare classes (platform) get extra augments for balance.

    Args:
        dataset_dir: Dataset root (default: yolo_dataset/)
        base_count: Base augmentations per image (rare classes get more)
        dry_run: Preview without writing
        train_only: Only augment train split

    Returns:
        dict with augmentation statistics
    """
    import albumentations as A
    import cv2

    dataset_dir = Path(dataset_dir) if dataset_dir else PROJECT_ROOT / "yolo_dataset"

    # Define transform pipeline — no kaleidoscope, just clean augments
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=30, val_shift_limit=20, p=0.6),
        A.OneOf([
            A.GaussNoise(std_range=(0.01, 0.03), p=1.0),
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
        ], p=0.3),
        A.RandomResizedCrop(
            size=(640, 640),
            scale=(0.7, 1.0),
            ratio=(0.9, 1.1),
            p=0.4,
        ),
    ], bbox_params=A.BboxParams(
        format="albumentations",
        min_area=100,           # drop tiny boxes that become noise after crop
        min_visibility=0.3,     # keep boxes that are at least 30% visible
        label_fields=["class_ids"],
    ))

    splits = ["train"] if train_only else ["train", "val"]
    stats = {"images_processed": 0, "augments_created": 0, "skipped_empty": 0,
             "class_boost": Counter()}

    for split in splits:
        img_dir = dataset_dir / split / "images"
        lbl_dir = dataset_dir / split / "labels"
        if not img_dir.exists():
            continue

        # Collect populated label files (skip empty/augmented)
        pairs = []
        for lbl_path in sorted(lbl_dir.iterdir()):
            if lbl_path.suffix != ".txt":
                continue
            if "_aug" in lbl_path.stem:
                continue  # don't augment existing augments
            labels = _parse_yolo_labels(lbl_path)
            if not labels:
                stats["skipped_empty"] += 1
                continue
            # Find matching image
            img_path = None
            for ext in IMAGE_EXTS:
                candidate = img_dir / f"{lbl_path.stem}{ext}"
                if candidate.exists():
                    img_path = candidate
                    break
            if img_path:
                class_set = {cid for cid, *_ in labels}
                pairs.append((img_path, lbl_path, labels, class_set))

        print(f"\n  {split}: {len(pairs)} images to augment")

        for img_path, lbl_path, labels, class_set in pairs:
            # More augments for rare classes
            if 3 in class_set:      # platform — very rare
                n_aug = base_count + 3
            elif 2 in class_set:    # projectile — weak class
                n_aug = base_count + 1
            else:
                n_aug = base_count

            if dry_run:
                stats["images_processed"] += 1
                stats["augments_created"] += n_aug
                for cid in class_set:
                    stats["class_boost"][cid] += n_aug
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                continue

            bboxes, class_ids = _yolo_to_albumentations(labels)

            for i in range(n_aug):
                try:
                    result = transform(image=img, bboxes=bboxes, class_ids=class_ids)
                except Exception:
                    continue

                aug_bboxes = result["bboxes"]
                aug_class_ids = result["class_ids"]

                # Skip if all boxes were lost
                if not aug_bboxes:
                    continue

                yolo_lines = _albumentations_to_yolo(aug_bboxes, aug_class_ids)
                if not yolo_lines:
                    continue

                aug_name = f"{img_path.stem}_aug{i}"
                aug_img_path = img_dir / f"{aug_name}{img_path.suffix}"
                aug_lbl_path = lbl_dir / f"{aug_name}.txt"

                cv2.imwrite(str(aug_img_path), result["image"])
                aug_lbl_path.write_text("\n".join(yolo_lines) + "\n")
                stats["augments_created"] += 1

            stats["images_processed"] += 1
            for cid in class_set:
                stats["class_boost"][cid] += n_aug

    # Clear cache
    if not dry_run:
        for cache in dataset_dir.rglob("*.cache"):
            cache.unlink()

    print(f"\n--- Augmentation {'preview' if dry_run else 'complete'} ---")
    print(f"  Images processed:  {stats['images_processed']}")
    print(f"  Augments created:  {stats['augments_created']}")
    print(f"  Skipped (empty):   {stats['skipped_empty']}")
    print(f"\n  Extra instances per class:")
    for cid in sorted(stats["class_boost"].keys()):
        name = GAMEPLAY_CLASSES.get(cid, f"class_{cid}")
        print(f"    {cid} ({name}): ~{stats['class_boost'][cid]} augmented images")

    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Paper Engine dataset tools")
    parser.add_argument("--summary", action="store_true", help="Print dataset statistics")
    parser.add_argument("--cleanup", action="store_true", help="Remove orphan/empty labels")
    parser.add_argument("--merge", type=str, metavar="DIR", help="Merge external dataset from DIR")
    parser.add_argument("--augment", action="store_true", help="Generate augmented training images")
    parser.add_argument("--aug-count", type=int, default=2,
                        help="Base augmentations per image (rare classes get more)")
    parser.add_argument("--auto-remap", action="store_true",
                        help="Auto-remap class names for Cuphead datasets")
    parser.add_argument("--dry-run", action="store_true", help="Preview without making changes")
    parser.add_argument("--dataset-dir", type=str, default=None,
                        help="Target dataset directory (default: yolo_dataset/)")

    args = parser.parse_args()

    if not any([args.summary, args.cleanup, args.merge, args.augment]):
        parser.print_help()
        return

    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else None

    if args.cleanup:
        print("=" * 50)
        print("  DATASET CLEANUP")
        print("=" * 50)
        cleanup_dataset(dataset_dir=dataset_dir, dry_run=args.dry_run)

    if args.merge:
        print("\n" + "=" * 50)
        print("  DATASET MERGE")
        print("=" * 50)
        merge_external_dataset(
            source_dir=args.merge,
            dataset_dir=dataset_dir,
            auto_remap=args.auto_remap,
            dry_run=args.dry_run,
        )

    if args.augment:
        print("\n" + "=" * 50)
        print("  DATASET AUGMENTATION")
        print("=" * 50)
        augment_dataset(dataset_dir=dataset_dir, base_count=args.aug_count,
                        dry_run=args.dry_run)

    if args.summary or args.cleanup or args.merge or args.augment:
        print("\n" + "=" * 50)
        print("  DATASET SUMMARY")
        print("=" * 50)
        dataset_summary(dataset_dir=dataset_dir)


if __name__ == "__main__":
    main()
