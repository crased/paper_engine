"""
Test YOLO11 Model - Run inference and visualize detections
"""

from ultralytics import YOLO
from pathlib import Path
import sys

def test_model(model_path='runs/detect/paper_engine_model/weights/best.pt',
               source='screenshots',
               conf_threshold=0.25,
               save_results=True):
    """
    Run YOLO11 model inference on test images

    Args:
        model_path: Path to trained model weights
        source: Path to test images (directory or single image)
        conf_threshold: Confidence threshold for detections (0.0-1.0)
        save_results: Save annotated images
    """

    # Check if model exists
    if not Path(model_path).exists():
        print(f"ERROR: Model not found at {model_path}")
        print("Please train a model first using: python training_model.py")
        sys.exit(1)

    # Load model
    print(f"\n{'='*60}")
    print(f"Loading YOLO11 model from: {model_path}")
    print(f"{'='*60}\n")

    model = YOLO(model_path)

    # Check source
    source_path = Path(source)
    if not source_path.exists():
        print(f"ERROR: Source not found: {source}")
        sys.exit(1)

    # Get test images
    if source_path.is_dir():
        test_images = list(source_path.glob('*.png')) + list(source_path.glob('*.jpg'))
        print(f"Found {len(test_images)} images in {source}")
    else:
        test_images = [source_path]
        print(f"Testing single image: {source}")

    if not test_images:
        print(f"No images found in {source}")
        sys.exit(1)

    # Run inference
    print(f"\nRunning inference with confidence threshold: {conf_threshold}")
    print(f"{'='*60}\n")

    results = model.predict(
        source=str(source),
        conf=conf_threshold,
        save=save_results,
        project='runs/detect',
        name='test_results',
        exist_ok=True
    )

    # Print results
    print(f"\n{'='*60}")
    print("Detection Results")
    print(f"{'='*60}\n")

    total_detections = 0
    class_counts = {}

    for i, result in enumerate(results):
        image_name = Path(result.path).name
        boxes = result.boxes

        if len(boxes) > 0:
            print(f"Image {i+1}/{len(results)}: {image_name}")
            print(f"  Detections: {len(boxes)}")

            for box in boxes:
                class_id = int(box.cls[0])
                class_name = result.names[class_id]
                confidence = float(box.conf[0])

                # Count detections per class
                class_counts[class_name] = class_counts.get(class_name, 0) + 1

                print(f"    - {class_name}: {confidence:.2%}")

            total_detections += len(boxes)
            print()
        else:
            print(f"Image {i+1}/{len(results)}: {image_name} - No detections")

    # Summary
    print(f"{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"Total images processed: {len(results)}")
    print(f"Total detections: {total_detections}")
    print(f"Average detections per image: {total_detections/len(results):.1f}")
    print(f"\nDetections by class:")
    for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {class_name}: {count}")

    if save_results:
        output_dir = Path('runs/detect/test_results')
        print(f"\n✓ Annotated images saved to: {output_dir}")
        print(f"  View them to see bounding boxes and confidence scores!")

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Test YOLO11 model on images')
    parser.add_argument('--model', type=str,
                       default='runs/detect/paper_engine_model/weights/best.pt',
                       help='Path to model weights')
    parser.add_argument('--source', type=str,
                       default='screenshots',
                       help='Path to test images (directory or single image)')
    parser.add_argument('--conf', type=float,
                       default=0.25,
                       help='Confidence threshold (0.0-1.0)')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save annotated images')

    args = parser.parse_args()

    test_model(
        model_path=args.model,
        source=args.source,
        conf_threshold=args.conf,
        save_results=not args.no_save
    )
