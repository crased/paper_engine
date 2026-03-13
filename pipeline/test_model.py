"""
Test YOLO11 Model - Run inference and return structured results
"""

from ultralytics import YOLO
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _resolve_model_path():
    """
    Find the most recently trained model by scanning runs/detect/ for
    model directories matching the configured model_output_name pattern.

    Falls back to the configured best_model_path if no directories found.

    Returns:
        str: path to best.pt
    """
    try:
        from conf.config_parser import training_conf

        configured = training_conf.get("BEST_MODEL_PATH", "")
        model_name = training_conf.get("MODEL_OUTPUT_NAME", "paper_engine_model")
    except Exception:
        configured = "runs/detect/paper_engine_model/weights/best.pt"
        model_name = "paper_engine_model"

    # Scan for the newest model directory matching the name pattern
    runs_dir = Path("runs/detect")
    if runs_dir.is_dir():
        candidates = sorted(
            runs_dir.glob(f"{model_name}*/weights/best.pt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            chosen = str(candidates[0])
            if configured and chosen != str(configured):
                print(f"Note: using latest model {chosen} (config has {configured})")
            return chosen

    # Nothing found via scan — return configured path as fallback
    return (
        str(configured)
        if configured
        else "runs/detect/paper_engine_model/weights/best.pt"
    )


def test_model(
    model_path=None,
    source="screenshots/captures",
    conf_threshold=0.25,
    save_results=True,
):
    """
    Run YOLO11 model inference on test images.

    Args:
        model_path: Path to trained model weights
        source: Path to test images (directory or single image)
        conf_threshold: Confidence threshold for detections (0.0-1.0)
        save_results: Save annotated images

    Returns:
        list[dict]: Each dict has:
            - 'image_path': str, absolute path to the source image
            - 'image_name': str, filename only
            - 'detections': list[dict] each with:
                - 'class_id': int
                - 'class_name': str
                - 'confidence': float
                - 'bbox': tuple (x1, y1, x2, y2) in pixels
                - 'bbox_norm': tuple (x_center, y_center, w, h) normalised 0-1
            - 'image_width': int
            - 'image_height': int
        Returns None on error.
    """
    if model_path is None:
        model_path = _resolve_model_path()

    # Check if model exists
    if not Path(model_path).exists():
        print(f"ERROR: Model not found at {model_path}")
        print("Please train a model first using: python pipeline/training_model.py")
        return None

    # Load model
    print(f"\n{'=' * 60}")
    print(f"Loading YOLO11 model from: {model_path}")
    print(f"{'=' * 60}\n")

    model = YOLO(model_path)

    # Check source
    source_path = Path(source)
    if not source_path.exists():
        print(f"ERROR: Source not found: {source}")
        return None

    # Get test images
    if source_path.is_dir():
        test_images = list(source_path.glob("*.png")) + list(source_path.glob("*.jpg"))
        print(f"Found {len(test_images)} images in {source}")
    else:
        test_images = [source_path]
        print(f"Testing single image: {source}")

    if not test_images:
        print(f"No images found in {source}")
        return None

    # Run inference
    print(f"\nRunning inference with confidence threshold: {conf_threshold}")
    print(f"{'=' * 60}\n")

    results = model.predict(
        source=str(source),
        conf=conf_threshold,
        save=save_results,
        project="runs/detect",
        name="test_results",
        exist_ok=True,
    )

    # Build structured output
    structured = []
    total_detections = 0
    class_counts = {}

    for i, result in enumerate(results):
        image_name = Path(result.path).name
        boxes = result.boxes
        img_h, img_w = result.orig_shape  # (height, width)

        entry = {
            "image_path": str(Path(result.path).resolve()),
            "image_name": image_name,
            "image_width": img_w,
            "image_height": img_h,
            "detections": [],
        }

        if len(boxes) > 0:
            print(f"Image {i + 1}/{len(results)}: {image_name}")
            print(f"  Detections: {len(boxes)}")

            for box in boxes:
                class_id = int(box.cls[0])
                class_name = result.names[class_id]
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                # Normalised centre-format for YOLO label files
                xc = ((x1 + x2) / 2) / img_w
                yc = ((y1 + y2) / 2) / img_h
                w = (x2 - x1) / img_w
                h = (y2 - y1) / img_h

                entry["detections"].append(
                    {
                        "class_id": class_id,
                        "class_name": class_name,
                        "confidence": confidence,
                        "bbox": (x1, y1, x2, y2),
                        "bbox_norm": (xc, yc, w, h),
                    }
                )

                class_counts[class_name] = class_counts.get(class_name, 0) + 1
                print(f"    - {class_name}: {confidence:.2%}")

            total_detections += len(boxes)
            print()
        else:
            print(f"Image {i + 1}/{len(results)}: {image_name} - No detections")

        structured.append(entry)

    # Summary
    print(f"{'=' * 60}")
    print("Summary")
    print(f"{'=' * 60}")
    print(f"Total images processed: {len(results)}")
    print(f"Total detections: {total_detections}")
    if results:
        print(f"Average detections per image: {total_detections / len(results):.1f}")
    print(f"\nDetections by class:")
    for class_name, count in sorted(
        class_counts.items(), key=lambda x: x[1], reverse=True
    ):
        print(f"  {class_name}: {count}")

    if save_results:
        output_dir = Path("runs/detect/test_results")
        print(f"\nAnnotated images saved to: {output_dir}")

    print(f"\n{'=' * 60}\n")

    return structured


def get_class_names(model_path=None):
    """Return the class name mapping {id: name} from a trained model."""
    if model_path is None:
        model_path = _resolve_model_path()
    if not Path(model_path).exists():
        return {}
    model = YOLO(model_path)
    return dict(model.names) if hasattr(model, "names") else {}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test YOLO11 model on images")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to model weights (auto-detects latest if not specified)",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="screenshots/captures",
        help="Path to test images (directory or single image)",
    )
    parser.add_argument(
        "--conf", type=float, default=0.25, help="Confidence threshold (0.0-1.0)"
    )
    parser.add_argument(
        "--no-save", action="store_true", help="Do not save annotated images"
    )

    args = parser.parse_args()

    test_model(
        model_path=args.model,
        source=args.source,
        conf_threshold=args.conf,
        save_results=not args.no_save,
    )
