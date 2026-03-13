"""
Video Dataset Builder — download gameplay videos, extract frames, dedup, label, build YOLO dataset.

End-to-end pipeline:
  1. download  — YouTube search + download at target resolution (yt-dlp)
  2. extract   — ffmpeg frame extraction at configurable FPS
  3. dedup     — perceptual hash deduplication (imagehash)
   4. label     — YOLO-World auto-labeling (pipeline/auto_labeler.py)
  5. build     — train/val split + dataset.yaml for YOLO training

Usage:
    # Full pipeline:
    python -m pipeline.video_dataset --game cuphead --queries "cuphead bosses no commentary"

    # Individual steps:
    python -m pipeline.video_dataset download --queries "cuphead bosses" --max-videos 5
    python -m pipeline.video_dataset extract --videos-dir videos/ --fps 1
    python -m pipeline.video_dataset dedup --frames-dir frames/
    python -m pipeline.video_dataset label --frames-dir frames/
    python -m pipeline.video_dataset build --frames-dir frames/ --labels-dir labels/

    # From URLs file:
    python -m pipeline.video_dataset download --urls-file urls.txt --max-videos 10
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_BASE_DIR = _PROJECT_ROOT / "screenshots"
DEFAULT_RESOLUTION = 480  # Max height in pixels
DEFAULT_FPS = 1  # Frames per second to extract
DEFAULT_JPEG_QUALITY = 75  # JPEG quality (1-100)
DEFAULT_DEDUP_THRESHOLD = 8  # Hamming distance threshold for perceptual hash
DEFAULT_VAL_SPLIT = 0.15  # Fraction of data for validation
DEFAULT_SKIP_START_FRAMES = 3  # Skip first N frames per video (watermarks, intros)


# ---------------------------------------------------------------------------
# Step 1: Download
# ---------------------------------------------------------------------------


def download_videos(
    queries: list[str] | None = None,
    urls: list[str] | None = None,
    output_dir: Path = DEFAULT_BASE_DIR / "videos",
    max_videos: int = 10,
    max_height: int = DEFAULT_RESOLUTION,
) -> list[Path]:
    """Download gameplay videos from YouTube.

    Args:
        queries: Search queries to find videos.
        urls: Direct video URLs.
        output_dir: Where to save downloaded videos.
        max_videos: Maximum number of videos to download total.
        max_height: Maximum video height (e.g. 480 for 480p).

    Returns:
        List of downloaded video file paths.
    """
    import yt_dlp

    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded: list[Path] = []

    # Collect URLs from search queries
    all_urls: list[str] = list(urls or [])

    if queries:
        logger.info("Searching YouTube for %d queries...", len(queries))
        for query in queries:
            if len(all_urls) >= max_videos:
                break
            search_opts = {
                "quiet": True,
                "no_warnings": True,
                "extract_flat": True,
                "default_search": "ytsearch10",  # Up to 10 results per query
            }
            try:
                with yt_dlp.YoutubeDL(search_opts) as ydl:
                    result = ydl.extract_info(f"ytsearch10:{query}", download=False)
                    if result and "entries" in result:
                        for entry in result["entries"]:
                            if len(all_urls) >= max_videos:
                                break
                            url = entry.get("url") or entry.get("webpage_url")
                            if url and url not in all_urls:
                                title = entry.get("title", "?")
                                duration = entry.get("duration", 0)
                                # Skip very short (<60s) or very long (>30min) videos
                                if duration and (duration < 60 or duration > 1800):
                                    logger.debug("Skipping %s (%ds)", title, duration)
                                    continue
                                all_urls.append(url)
                                logger.info("  Found: %s (%ds)", title, duration)
            except Exception as e:
                logger.warning("Search failed for '%s': %s", query, e)

    if not all_urls:
        logger.warning("No videos found to download")
        return downloaded

    logger.info("Downloading %d videos at max %dp...", len(all_urls), max_height)

    dl_opts = {
        "format": f"best[height<={max_height}][ext=mp4]/best[height<={max_height}]/best",
        "outtmpl": str(output_dir / "%(id)s.%(ext)s"),
        "quiet": False,
        "no_warnings": True,
        "ignoreerrors": True,
        "noplaylist": True,
    }

    with yt_dlp.YoutubeDL(dl_opts) as ydl:
        for url in all_urls:
            try:
                info = ydl.extract_info(url, download=True)
                if info:
                    filename = ydl.prepare_filename(info)
                    fpath = Path(filename)
                    if fpath.exists():
                        size_mb = fpath.stat().st_size / (1024 * 1024)
                        downloaded.append(fpath)
                        logger.info(
                            "  Downloaded: %s (%.1f MB)",
                            fpath.name,
                            size_mb,
                        )
            except Exception as e:
                logger.warning("Failed to download %s: %s", url, e)

    logger.info("Downloaded %d videos", len(downloaded))
    return downloaded


# ---------------------------------------------------------------------------
# Step 2: Extract frames
# ---------------------------------------------------------------------------


def extract_frames(
    videos_dir: Path,
    output_dir: Path = DEFAULT_BASE_DIR / "frames_raw",
    fps: float = DEFAULT_FPS,
    quality: int = DEFAULT_JPEG_QUALITY,
    skip_start_frames: int = DEFAULT_SKIP_START_FRAMES,
) -> int:
    """Extract frames from videos using ffmpeg.

    Args:
        videos_dir: Directory containing video files.
        output_dir: Where to save extracted JPEG frames.
        fps: Frames per second to extract.
        quality: JPEG quality (2=best, 31=worst; we use -q:v scale).
        skip_start_frames: Delete first N frames per video (watermarks, intros).

    Returns:
        Total number of frames extracted.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    videos = sorted(
        p
        for p in videos_dir.iterdir()
        if p.suffix.lower() in (".mp4", ".mkv", ".webm", ".avi", ".mov")
    )

    if not videos:
        logger.warning("No video files found in %s", videos_dir)
        return 0

    total_frames = 0

    for video_path in videos:
        # Unique prefix per video (short hash of filename)
        vid_hash = hashlib.md5(video_path.name.encode()).hexdigest()[:8]
        prefix = output_dir / f"{vid_hash}_"

        logger.info("Extracting frames from %s at %.1f FPS...", video_path.name, fps)

        # Map quality 1-100 to ffmpeg -q:v 2-31 (inverted scale)
        qv = max(2, min(31, int(31 - (quality / 100) * 29)))

        cmd = [
            "ffmpeg",
            "-i",
            str(video_path),
            "-vf",
            f"fps={fps}",
            "-q:v",
            str(qv),
            "-y",
            f"{prefix}%06d.jpg",
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,  # 10 min per video
            )
            if result.returncode != 0:
                logger.warning(
                    "ffmpeg error for %s: %s",
                    video_path.name,
                    result.stderr[-200:] if result.stderr else "unknown",
                )
                continue
        except subprocess.TimeoutExpired:
            logger.warning("ffmpeg timed out for %s", video_path.name)
            continue

        # Delete first N frames per video (watermarks, intro cards)
        if skip_start_frames > 0:
            early = sorted(
                p for p in output_dir.iterdir() if p.name.startswith(vid_hash)
            )[:skip_start_frames]
            for p in early:
                p.unlink()
            if early:
                logger.info(
                    "  Skipped first %d frames from %s",
                    len(early),
                    video_path.name,
                )

        # Count extracted frames for this video
        new_frames = len(
            [p for p in output_dir.iterdir() if p.name.startswith(vid_hash)]
        )
        total_frames += new_frames
        logger.info("  Extracted %d frames from %s", new_frames, video_path.name)

    logger.info("Total frames extracted: %d", total_frames)
    return total_frames


# ---------------------------------------------------------------------------
# Step 3: Deduplication
# ---------------------------------------------------------------------------


def deduplicate_frames(
    frames_dir: Path,
    output_dir: Optional[Path] = None,
    threshold: int = DEFAULT_DEDUP_THRESHOLD,
) -> dict:
    """Remove near-duplicate frames using perceptual hashing.

    If output_dir is None, duplicates are deleted in-place.
    If output_dir is set, unique frames are COPIED there.

    Args:
        frames_dir: Directory of extracted frames.
        output_dir: Optional output directory for unique frames.
        threshold: Hamming distance threshold (lower = stricter).

    Returns:
        Stats dict.
    """
    import imagehash

    images = sorted(
        p for p in frames_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )

    if not images:
        logger.warning("No images found in %s", frames_dir)
        return {"total": 0, "unique": 0, "duplicates": 0}

    logger.info("Deduplicating %d frames (threshold=%d)...", len(images), threshold)

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    from PIL import Image

    hashes: list[tuple[imagehash.ImageHash, Path]] = []
    stats = {"total": len(images), "unique": 0, "duplicates": 0}

    for img_path in images:
        try:
            img = Image.open(img_path)
            h = imagehash.phash(img)
        except Exception as e:
            logger.debug("Hash failed for %s: %s", img_path.name, e)
            stats["duplicates"] += 1
            if not output_dir:
                img_path.unlink(missing_ok=True)
            continue

        # Check against existing hashes
        is_dup = False
        for existing_hash, _ in hashes:
            if h - existing_hash < threshold:
                is_dup = True
                break

        if is_dup:
            stats["duplicates"] += 1
            if not output_dir:
                img_path.unlink(missing_ok=True)
        else:
            stats["unique"] += 1
            hashes.append((h, img_path))
            if output_dir:
                shutil.copy2(img_path, output_dir / img_path.name)

        # Progress every 500 images
        done = stats["unique"] + stats["duplicates"]
        if done % 500 == 0:
            print(
                f"\r  [{done}/{stats['total']}] "
                f"{stats['unique']} unique, {stats['duplicates']} dups",
                end="",
                flush=True,
            )

    print()
    logger.info(
        "Dedup: %d total -> %d unique (%d duplicates removed, %.0f%% reduction)",
        stats["total"],
        stats["unique"],
        stats["duplicates"],
        stats["duplicates"] / max(stats["total"], 1) * 100,
    )
    return stats


# ---------------------------------------------------------------------------
# Step 5a: Validate YOLO labels
# ---------------------------------------------------------------------------


def validate_labels(
    labels_dir: Path,
    classes: list[str],
    max_per_image: dict[str, int] | None = None,
) -> dict:
    """Validate YOLO label files against class count and per-image rules.

    Args:
        labels_dir: Directory containing .txt YOLO label files.
        classes: Class name list (index = class ID).
        max_per_image: Per-class max instance rules {class_name: max_count}.

    Returns:
        Stats dict with pass/fail counts and details.
    """
    from collections import Counter

    labels_dir = Path(labels_dir)
    nc = len(classes)

    stats = {
        "total": 0,
        "valid": 0,
        "invalid_class_id": 0,
        "invalid_max_per_image": 0,
        "invalid_degenerate_box": 0,
        "empty": 0,
        "violations": [],
    }

    for label_path in sorted(labels_dir.glob("*.txt")):
        stats["total"] += 1
        lines = label_path.read_text().strip().split("\n")
        lines = [l for l in lines if l.strip()]

        if not lines:
            stats["empty"] += 1
            stats["valid"] += 1
            continue

        is_valid = True
        class_counts = Counter()

        for line in lines:
            parts = line.split()
            if len(parts) < 5:
                continue
            cid = int(parts[0])
            w, h = float(parts[3]), float(parts[4])

            # Check class ID range
            if cid < 0 or cid >= nc:
                stats["invalid_class_id"] += 1
                stats["violations"].append(
                    f"{label_path.name}: class {cid} out of range 0-{nc - 1}"
                )
                is_valid = False
                continue

            # Check degenerate box
            if w < 0.005 or h < 0.005:
                stats["invalid_degenerate_box"] += 1
                is_valid = False
                continue

            class_counts[classes[cid]] += 1

        # Check per-image max
        if max_per_image:
            for cname, max_count in max_per_image.items():
                if class_counts.get(cname, 0) > max_count:
                    stats["invalid_max_per_image"] += 1
                    stats["violations"].append(
                        f"{label_path.name}: {cname}={class_counts[cname]} "
                        f"(max {max_count})"
                    )
                    is_valid = False

        if is_valid:
            stats["valid"] += 1

    logger.info(
        "Validation: %d/%d valid, %d bad class IDs, %d max-per-image violations, "
        "%d degenerate boxes, %d empty",
        stats["valid"],
        stats["total"],
        stats["invalid_class_id"],
        stats["invalid_max_per_image"],
        stats["invalid_degenerate_box"],
        stats["empty"],
    )
    if stats["violations"]:
        for v in stats["violations"][:20]:
            logger.warning("  %s", v)
        if len(stats["violations"]) > 20:
            logger.warning("  ... and %d more", len(stats["violations"]) - 20)

    return stats


# ---------------------------------------------------------------------------
# Step 5c: Build YOLO dataset
# ---------------------------------------------------------------------------


def build_yolo_dataset(
    frames_dirs: list[Path] | Path,
    labels_dirs: list[Path] | Path,
    output_dir: Path = _PROJECT_ROOT / "yolo_dataset",
    classes: list[str] | None = None,
    max_per_image: dict[str, int] | None = None,
    val_split: float = DEFAULT_VAL_SPLIT,
    wipe: bool = False,
) -> Path:
    """Organize labeled frames into YOLO train/val structure.

    Accepts multiple source directories (e.g., remapped legacy data +
    auto-labeled YouTube data). Validates labels before including them.

    Creates:
        output_dir/
            train/images/
            train/labels/
            val/images/
            val/labels/
            dataset.yaml

    Returns:
        Path to dataset.yaml.
    """
    import random

    # Normalize to lists
    if isinstance(frames_dirs, Path):
        frames_dirs = [frames_dirs]
    if isinstance(labels_dirs, Path):
        labels_dirs = [labels_dirs]

    if classes is None:
        from conf.game_configs.cuphead import DETECTION_CLASSES

        classes = DETECTION_CLASSES

    nc = len(classes)

    # Wipe old data if requested
    if wipe:
        for split in ("train", "val"):
            split_dir = output_dir / split
            if split_dir.exists():
                shutil.rmtree(split_dir)
                logger.info("Wiped %s", split_dir)
        # Remove stale cache files
        for cache in output_dir.rglob("*.cache"):
            cache.unlink()
            logger.info("Removed cache: %s", cache)

    # Collect all image+label pairs from all source directories
    pairs = []
    for frames_dir, labels_dir in zip(frames_dirs, labels_dirs):
        frames_dir = Path(frames_dir)
        labels_dir = Path(labels_dir)
        if not frames_dir.exists():
            logger.warning("Frames dir not found: %s", frames_dir)
            continue
        if not labels_dir.exists():
            logger.warning("Labels dir not found: %s", labels_dir)
            continue

        for img_path in sorted(frames_dir.iterdir()):
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            label_path = labels_dir / (img_path.stem + ".txt")
            if not label_path.exists() or label_path.stat().st_size == 0:
                continue

            # Validate this label file inline
            lines = label_path.read_text().strip().split("\n")
            lines = [l for l in lines if l.strip()]
            valid = True
            from collections import Counter

            class_counts = Counter()

            for line in lines:
                parts = line.split()
                if len(parts) < 5:
                    continue
                cid = int(parts[0])
                w, h = float(parts[3]), float(parts[4])
                if cid < 0 or cid >= nc:
                    valid = False
                    break
                if w < 0.005 or h < 0.005:
                    valid = False
                    break
                class_counts[classes[cid]] += 1

            if valid and max_per_image:
                for cname, max_count in max_per_image.items():
                    if class_counts.get(cname, 0) > max_count:
                        valid = False
                        break

            if valid:
                pairs.append((img_path, label_path))

    if not pairs:
        logger.warning("No valid image+label pairs found")
        return output_dir / "dataset.yaml"

    logger.info("Building YOLO dataset from %d valid labeled images...", len(pairs))

    # Shuffle and split
    random.seed(42)
    random.shuffle(pairs)
    val_count = max(1, int(len(pairs) * val_split))
    val_pairs = pairs[:val_count]
    train_pairs = pairs[val_count:]

    # Create directory structure
    for split in ("train", "val"):
        (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    # Copy files
    for split_name, split_pairs in [("train", train_pairs), ("val", val_pairs)]:
        for img_path, label_path in split_pairs:
            shutil.copy2(img_path, output_dir / split_name / "images" / img_path.name)
            shutil.copy2(
                label_path, output_dir / split_name / "labels" / label_path.name
            )

    # Write dataset.yaml
    yaml_path = output_dir / "dataset.yaml"
    import yaml

    dataset_config = {
        "path": str(output_dir.resolve()),
        "train": "train/images",
        "val": "val/images",
        "nc": nc,
        "names": classes,
    }
    with open(yaml_path, "w") as f:
        yaml.dump(dataset_config, f, default_flow_style=False)

    # Print per-class summary
    from collections import Counter

    total_class_counts = Counter()
    for _, label_path in pairs:
        for line in label_path.read_text().strip().split("\n"):
            parts = line.split()
            if len(parts) >= 5:
                total_class_counts[classes[int(parts[0])]] += 1

    logger.info(
        "Dataset built: %d train, %d val, %d classes -> %s",
        len(train_pairs),
        len(val_pairs),
        nc,
        yaml_path,
    )
    logger.info("Per-class counts: %s", dict(total_class_counts.most_common()))
    return yaml_path


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


def run_pipeline(
    game: str = "cuphead",
    queries: list[str] | None = None,
    urls: list[str] | None = None,
    urls_file: Path | None = None,
    base_dir: Path = DEFAULT_BASE_DIR,
    max_videos: int = 10,
    max_height: int = DEFAULT_RESOLUTION,
    fps: float = DEFAULT_FPS,
    dedup_threshold: int = DEFAULT_DEDUP_THRESHOLD,
    model_name: str = "yolov8s-worldv2.pt",
    device: str | None = None,
    skip_download: bool = False,
    skip_extract: bool = False,
    skip_dedup: bool = False,
    skip_label: bool = False,
    skip_build: bool = False,
):
    """Run the full video dataset pipeline."""
    # Load game config
    if game == "cuphead":
        from conf.game_configs.cuphead import (
            DETECTION_CLASSES,
            DETECTION_PROMPTS,
            DATASET_SEARCH_QUERIES,
        )
    else:
        logger.error("Unknown game: %s", game)
        return

    if queries is None and urls is None and urls_file is None:
        queries = DATASET_SEARCH_QUERIES

    # Load URLs from file
    if urls_file:
        with open(urls_file) as f:
            file_urls = [
                line.strip() for line in f if line.strip() and not line.startswith("#")
            ]
            urls = (urls or []) + file_urls

    videos_dir = base_dir / "videos"
    frames_raw_dir = base_dir / "frames_raw"
    frames_dir = base_dir / "frames"
    labels_dir = base_dir / "labels"
    dataset_dir = _PROJECT_ROOT / "yolo_dataset"

    # Step 1: Download
    if not skip_download:
        logger.info("=" * 60)
        logger.info("STEP 1: Download videos")
        logger.info("=" * 60)
        download_videos(
            queries=queries,
            urls=urls,
            output_dir=videos_dir,
            max_videos=max_videos,
            max_height=max_height,
        )

    # Step 2: Extract frames
    if not skip_extract:
        logger.info("=" * 60)
        logger.info("STEP 2: Extract frames at %.1f FPS", fps)
        logger.info("=" * 60)
        extract_frames(
            videos_dir=videos_dir,
            output_dir=frames_raw_dir,
            fps=fps,
        )

    # Step 3: Dedup
    if not skip_dedup:
        logger.info("=" * 60)
        logger.info("STEP 3: Deduplicate frames")
        logger.info("=" * 60)
        deduplicate_frames(
            frames_dir=frames_raw_dir,
            output_dir=frames_dir,
            threshold=dedup_threshold,
        )

    # Step 4: Auto-label
    if not skip_label:
        logger.info("=" * 60)
        logger.info("STEP 4: Auto-label with YOLO-World")
        logger.info("=" * 60)
        from pipeline.auto_labeler import AutoLabeler

        labeler = AutoLabeler(
            classes=DETECTION_CLASSES,
            prompts=DETECTION_PROMPTS,
            model_name=model_name,
            device=device,
        )
        labeler.load_model()
        labeler.label_directory(frames_dir, labels_dir)

    # Step 5: Build dataset
    if not skip_build:
        logger.info("=" * 60)
        logger.info("STEP 5: Build YOLO dataset")
        logger.info("=" * 60)
        yaml_path = build_yolo_dataset(
            frames_dirs=[frames_dir],
            labels_dirs=[labels_dir],
            output_dir=dataset_dir,
            classes=DETECTION_CLASSES,
        )
        print(f"\nDataset ready: {yaml_path}")
        print(f"Train with:  yolo detect train data={yaml_path} model=yolo11n.pt")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Build YOLO training dataset from YouTube gameplay videos"
    )
    sub = parser.add_subparsers(dest="command")

    # -- Full pipeline --
    p_all = sub.add_parser("all", help="Run full pipeline")
    p_all.add_argument("--game", default="cuphead")
    p_all.add_argument("--queries", nargs="+", help="YouTube search queries")
    p_all.add_argument("--urls", nargs="+", help="Direct video URLs")
    p_all.add_argument("--urls-file", type=Path, help="File with URLs (one per line)")
    p_all.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR)
    p_all.add_argument("--max-videos", type=int, default=10)
    p_all.add_argument("--max-height", type=int, default=DEFAULT_RESOLUTION)
    p_all.add_argument("--fps", type=float, default=DEFAULT_FPS)
    p_all.add_argument("--dedup-threshold", type=int, default=DEFAULT_DEDUP_THRESHOLD)
    p_all.add_argument("--model", default="yolov8s-worldv2.pt")
    p_all.add_argument("--device", choices=["cpu", "cuda"])
    p_all.add_argument("--skip-download", action="store_true")
    p_all.add_argument("--skip-extract", action="store_true")
    p_all.add_argument("--skip-dedup", action="store_true")
    p_all.add_argument("--skip-label", action="store_true")
    p_all.add_argument("--skip-build", action="store_true")

    # -- Individual steps --
    p_dl = sub.add_parser("download", help="Download videos only")
    p_dl.add_argument("--queries", nargs="+")
    p_dl.add_argument("--urls", nargs="+")
    p_dl.add_argument("--urls-file", type=Path)
    p_dl.add_argument("--output-dir", type=Path, default=DEFAULT_BASE_DIR / "videos")
    p_dl.add_argument("--max-videos", type=int, default=10)
    p_dl.add_argument("--max-height", type=int, default=DEFAULT_RESOLUTION)

    p_ex = sub.add_parser("extract", help="Extract frames from videos")
    p_ex.add_argument("--videos-dir", type=Path, default=DEFAULT_BASE_DIR / "videos")
    p_ex.add_argument(
        "--output-dir", type=Path, default=DEFAULT_BASE_DIR / "frames_raw"
    )
    p_ex.add_argument("--fps", type=float, default=DEFAULT_FPS)
    p_ex.add_argument(
        "--skip-start-frames",
        type=int,
        default=DEFAULT_SKIP_START_FRAMES,
        help="Delete first N frames per video (watermarks/intros)",
    )

    p_dd = sub.add_parser("dedup", help="Deduplicate frames")
    p_dd.add_argument(
        "--frames-dir", type=Path, default=DEFAULT_BASE_DIR / "frames_raw"
    )
    p_dd.add_argument("--output-dir", type=Path, default=DEFAULT_BASE_DIR / "frames")
    p_dd.add_argument("--threshold", type=int, default=DEFAULT_DEDUP_THRESHOLD)

    p_lb = sub.add_parser("label", help="Auto-label frames")
    p_lb.add_argument("--frames-dir", type=Path, default=DEFAULT_BASE_DIR / "frames")
    p_lb.add_argument("--labels-dir", type=Path, default=DEFAULT_BASE_DIR / "labels")
    p_lb.add_argument("--model", default="yolov8s-worldv2.pt")
    p_lb.add_argument("--device", choices=["cpu", "cuda"])
    p_lb.add_argument("--game", default="cuphead")

    p_bd = sub.add_parser("build", help="Build YOLO dataset from labeled frames")
    p_bd.add_argument("--frames-dir", type=Path, default=DEFAULT_BASE_DIR / "frames")
    p_bd.add_argument("--labels-dir", type=Path, default=DEFAULT_BASE_DIR / "labels")
    p_bd.add_argument("--output-dir", type=Path, default=_PROJECT_ROOT / "yolo_dataset")
    p_bd.add_argument("--game", default="cuphead")
    p_bd.add_argument(
        "--wipe", action="store_true", help="Wipe output dir before building"
    )

    p_vl = sub.add_parser("validate", help="Validate YOLO label files")
    p_vl.add_argument("--labels-dir", type=Path, required=True)
    p_vl.add_argument("--game", default="cuphead")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.command is None:
        parser.print_help()
        return

    if args.command == "all":
        run_pipeline(
            game=args.game,
            queries=args.queries,
            urls=args.urls,
            urls_file=args.urls_file,
            base_dir=args.base_dir,
            max_videos=args.max_videos,
            max_height=args.max_height,
            fps=args.fps,
            dedup_threshold=args.dedup_threshold,
            model_name=args.model,
            device=args.device,
            skip_download=args.skip_download,
            skip_extract=args.skip_extract,
            skip_dedup=args.skip_dedup,
            skip_label=args.skip_label,
            skip_build=args.skip_build,
        )
    elif args.command == "download":
        urls = args.urls or []
        if args.urls_file:
            with open(args.urls_file) as f:
                urls += [l.strip() for l in f if l.strip() and not l.startswith("#")]
        download_videos(
            queries=args.queries,
            urls=urls or None,
            output_dir=args.output_dir,
            max_videos=args.max_videos,
            max_height=args.max_height,
        )
    elif args.command == "extract":
        extract_frames(
            videos_dir=args.videos_dir,
            output_dir=args.output_dir,
            fps=args.fps,
            skip_start_frames=args.skip_start_frames,
        )
    elif args.command == "dedup":
        deduplicate_frames(
            frames_dir=args.frames_dir,
            output_dir=args.output_dir,
            threshold=args.threshold,
        )
    elif args.command == "label":
        game = args.game
        if game == "cuphead":
            from conf.game_configs.cuphead import DETECTION_CLASSES, DETECTION_PROMPTS
        else:
            logger.error("Unknown game: %s", game)
            return

        from pipeline.auto_labeler import AutoLabeler

        labeler = AutoLabeler(
            classes=DETECTION_CLASSES,
            prompts=DETECTION_PROMPTS,
            model_name=args.model,
            device=args.device,
        )
        labeler.load_model()
        labeler.label_directory(args.frames_dir, args.labels_dir)
    elif args.command == "build":
        game = args.game
        if game == "cuphead":
            from conf.game_configs.cuphead import DETECTION_CLASSES, MAX_PER_IMAGE
        else:
            logger.error("Unknown game: %s", game)
            return

        build_yolo_dataset(
            frames_dirs=[args.frames_dir],
            labels_dirs=[args.labels_dir],
            output_dir=args.output_dir,
            classes=DETECTION_CLASSES,
            max_per_image=MAX_PER_IMAGE,
            wipe=args.wipe,
        )
    elif args.command == "validate":
        game = args.game
        if game == "cuphead":
            from conf.game_configs.cuphead import DETECTION_CLASSES, MAX_PER_IMAGE
        else:
            logger.error("Unknown game: %s", game)
            return

        validate_labels(
            labels_dir=args.labels_dir,
            classes=DETECTION_CLASSES,
            max_per_image=MAX_PER_IMAGE,
        )


if __name__ == "__main__":
    main()
