"""
Auto-labeler — generates YOLO annotations using YOLO-World open-vocabulary detection.

Uses Ultralytics YOLO-World for zero-shot detection with text prompts.
Much faster and more reliable than VLM-based approaches (Florence-2, etc.)
while providing good quality bounding boxes for scaffolding datasets.

Game-agnostic: class names and text prompts are passed in from the game
config (e.g., games/cuphead.py).

Usage:
    # From game config:
    from conf.game_configs.cuphead import DETECTION_CLASSES, DETECTION_PROMPTS
    from pipeline.auto_labeler import AutoLabeler

    labeler = AutoLabeler(DETECTION_CLASSES, DETECTION_PROMPTS)
    labeler.load_model()
    labeler.label_directory("path/to/frames", "path/to/labels")

    # Standalone:
    python -m pipeline.auto_labeler --frames path/to/frames --labels path/to/labels
"""

from __future__ import annotations

import argparse
import logging
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
# AutoLabeler
# ---------------------------------------------------------------------------


class AutoLabeler:
    """YOLO auto-labeler using YOLO-World open-vocabulary detection.

    YOLO-World accepts text prompts describing what to detect, and outputs
    bounding boxes. The prompts are mapped to YOLO class IDs for the output
    label files.

    Attributes:
        classes: List of class names (index = YOLO class ID).
        prompts: List of text prompts for YOLO-World (one per class).
        model_name: YOLO-World model variant.
        conf: Confidence threshold for detections.
    """

    SUPPORTED_MODELS = {
        "yolov8s-worldv2": "yolov8s-worldv2.pt",
        "yolov8m-worldv2": "yolov8m-worldv2.pt",
        "yolov8l-worldv2": "yolov8l-worldv2.pt",
        "yolov8x-worldv2": "yolov8x-worldv2.pt",
    }

    def __init__(
        self,
        classes: list[str],
        prompts: list[str],
        model_name: str = "yolov8s-worldv2.pt",
        conf: float = 0.15,
        device: Optional[str] = None,
    ):
        self.classes = classes
        self.prompts = prompts
        self.model_name = model_name
        self.conf = conf
        self.device = device  # None = auto-detect
        self.model = None

        logger.info(
            "AutoLabeler: %d classes, model=%s, conf=%.2f",
            len(classes),
            model_name,
            conf,
        )

    def load_model(self):
        """Download and load the YOLO-World model."""
        from ultralytics import YOLO

        logger.info("Loading model %s ...", self.model_name)
        t0 = time.monotonic()

        self.model = YOLO(self.model_name)
        # Set the text prompts for open-vocabulary detection
        self.model.set_classes(self.prompts)

        dt = time.monotonic() - t0
        logger.info("Model loaded in %.1fs, prompts: %s", dt, self.prompts)

    def label_image(
        self, image_path: str | Path
    ) -> list[tuple[int, float, float, float, float]]:
        """Run open-vocabulary detection on a single image.

        Returns:
            List of (class_id, cx, cy, w, h) in YOLO format (normalized 0-1).
            Empty list if no detections.
        """
        if self.model is None:
            raise RuntimeError("Call load_model() first")

        results = self.model.predict(
            str(image_path),
            conf=self.conf,
            verbose=False,
            device=self.device or "",
        )

        detections = []
        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            for box in boxes:
                # YOLO-World class index maps to our prompt index
                cls_id = int(box.cls.item())
                if cls_id >= len(self.classes):
                    continue
                # xywhn = normalized center x, y, width, height
                xywhn = box.xywhn[0]
                cx = float(xywhn[0])
                cy = float(xywhn[1])
                w = float(xywhn[2])
                h = float(xywhn[3])
                if w < 0.005 or h < 0.005:
                    continue  # Skip degenerate boxes
                detections.append((cls_id, cx, cy, w, h))

        return detections

    @staticmethod
    def save_yolo_label(
        detections: list[tuple[int, float, float, float, float]],
        output_path: str | Path,
    ):
        """Write detections to a YOLO-format .txt label file."""
        with open(output_path, "w") as f:
            for class_id, cx, cy, w, h in detections:
                f.write(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

    def label_directory(
        self,
        frames_dir: str | Path,
        labels_dir: str | Path,
        skip_existing: bool = True,
        extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png"),
    ) -> dict:
        """Label all images in a directory, saving YOLO .txt files.

        Args:
            frames_dir: Directory containing input images.
            labels_dir: Directory to write .txt label files.
            skip_existing: Skip images that already have a label file.
            extensions: Image file extensions to process.

        Returns:
            Stats dict with counts.
        """
        frames_dir = Path(frames_dir)
        labels_dir = Path(labels_dir)
        labels_dir.mkdir(parents=True, exist_ok=True)

        images = sorted(
            p for p in frames_dir.iterdir() if p.suffix.lower() in extensions
        )

        stats = {
            "total": len(images),
            "labeled": 0,
            "skipped": 0,
            "empty": 0,
            "errors": 0,
            "total_detections": 0,
        }

        if not images:
            logger.warning("No images found in %s", frames_dir)
            return stats

        logger.info(
            "Labeling %d images from %s -> %s",
            len(images),
            frames_dir,
            labels_dir,
        )

        t0 = time.monotonic()
        for i, img_path in enumerate(images):
            label_path = labels_dir / (img_path.stem + ".txt")

            if skip_existing and label_path.exists():
                stats["skipped"] += 1
                continue

            try:
                detections = self.label_image(img_path)
                self.save_yolo_label(detections, label_path)
                stats["labeled"] += 1
                stats["total_detections"] += len(detections)
                if not detections:
                    stats["empty"] += 1
            except Exception as e:
                logger.error("Error labeling %s: %s", img_path.name, e)
                stats["errors"] += 1

            # Progress
            done = stats["labeled"] + stats["skipped"] + stats["errors"]
            if done % 10 == 0 or done == stats["total"]:
                elapsed = time.monotonic() - t0
                rate = stats["labeled"] / max(elapsed, 0.01)
                remaining = (stats["total"] - done) / max(rate, 0.001)
                print(
                    f"\r  [{done}/{stats['total']}] "
                    f"{rate:.1f} img/s, "
                    f"~{remaining / 60:.0f}m remaining, "
                    f"{stats['total_detections']} detections",
                    end="",
                    flush=True,
                )

        elapsed = time.monotonic() - t0
        print()  # newline after progress
        logger.info(
            "Done: %d labeled, %d skipped, %d empty, %d errors in %.1fs "
            "(%.1f img/s, %d total detections)",
            stats["labeled"],
            stats["skipped"],
            stats["empty"],
            stats["errors"],
            elapsed,
            stats["labeled"] / max(elapsed, 0.01),
            stats["total_detections"],
        )
        return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Auto-label game screenshots with YOLO-World"
    )
    parser.add_argument(
        "--frames",
        required=True,
        help="Directory of input images",
    )
    parser.add_argument(
        "--labels",
        help="Output directory for YOLO label files (default: <frames>/../labels)",
    )
    parser.add_argument(
        "--model",
        default="yolov8s-worldv2.pt",
        help="YOLO-World model variant (s/m/l/x)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.15,
        help="Confidence threshold (default: 0.15)",
    )
    parser.add_argument(
        "--device",
        help="Force device (cpu, 0, cuda:0, etc.)",
    )
    parser.add_argument(
        "--game",
        default="cuphead",
        help="Game config to use for class ontology",
    )
    parser.add_argument("--no-skip", action="store_true", help="Re-label all images")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    # Load game config
    if args.game == "cuphead":
        from conf.game_configs.cuphead import DETECTION_CLASSES, DETECTION_PROMPTS
    else:
        logger.error("Unknown game: %s", args.game)
        sys.exit(1)

    frames_dir = Path(args.frames)
    labels_dir = Path(args.labels) if args.labels else frames_dir.parent / "labels"

    labeler = AutoLabeler(
        classes=DETECTION_CLASSES,
        prompts=DETECTION_PROMPTS,
        model_name=args.model,
        conf=args.conf,
        device=args.device,
    )
    labeler.load_model()
    stats = labeler.label_directory(
        frames_dir, labels_dir, skip_existing=not args.no_skip
    )
    print(f"\nStats: {stats}")


if __name__ == "__main__":
    main()
