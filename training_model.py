"""
YOLO Model Training Script for Paper Engine

Converts Label Studio annotations to YOLO format, trains a YOLOv8 model,
and exports it for inference.
"""

from ultralytics import YOLO
import torch
import json
import shutil
from pathlib import Path
import random
import yaml


def convert_label_studio_to_yolo(json_file, output_labels_dir, class_mapping):
    """
    Convert a single Label Studio JSON file to YOLO format.

    Args:
        json_file: Path to Label Studio JSON annotation file
        output_labels_dir: Directory to save YOLO format labels
        class_mapping: Dict mapping class names to class IDs

    Returns:
        str: Image filename referenced in the annotation
    """
    with open(json_file, 'r') as f:
        data = json.load(f)

    # Extract image path
    image_path = data['task']['data']['image']
    # Parse the Label Studio local file path format
    # Format: "/data/local-files/?d=home/biel/workspace/paper_engine/screenshots/screenshot_20251223_103851.png"
    image_filename = Path(image_path.split('?d=')[-1]).name

    # Get image dimensions
    annotations = data['result']
    if not annotations:
        return None

    img_width = annotations[0]['original_width']
    img_height = annotations[0]['original_height']

    # Convert annotations to YOLO format
    yolo_lines = []
    for annotation in annotations:
        if annotation['type'] != 'rectanglelabels':
            continue

        value = annotation['value']
        label = value['rectanglelabels'][0]

        if label not in class_mapping:
            print(f"Warning: Unknown label '{label}' in {json_file}")
            continue

        class_id = class_mapping[label]

        # Label Studio uses percentage coordinates (0-100)
        # Convert to YOLO format (normalized 0-1)
        x_center = (value['x'] + value['width'] / 2) / 100.0
        y_center = (value['y'] + value['height'] / 2) / 100.0
        width = value['width'] / 100.0
        height = value['height'] / 100.0

        # YOLO format: class_id x_center y_center width height
        yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    # Save YOLO format label file
    if yolo_lines:
        label_filename = Path(image_filename).stem + '.txt'
        label_path = output_labels_dir / label_filename
        with open(label_path, 'w') as f:
            f.write('\n'.join(yolo_lines))

        return image_filename

    return None


def prepare_dataset(dataset_dir='dataset', screenshots_dir='screenshots', output_dir='yolo_dataset', train_split=0.8):
    """
    Prepare YOLO dataset from Label Studio annotations.

    Args:
        dataset_dir: Directory containing Label Studio JSON files
        screenshots_dir: Directory containing screenshot images
        output_dir: Output directory for YOLO format dataset
        train_split: Fraction of data to use for training (rest for validation)

    Returns:
        dict: Class mapping {class_name: class_id}
    """
    dataset_path = Path(dataset_dir)
    screenshots_path = Path(screenshots_dir)
    output_path = Path(output_dir)

    # Create output directory structure
    for split in ['train', 'val']:
        (output_path / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_path / split / 'labels').mkdir(parents=True, exist_ok=True)

    # Collect all unique class names
    all_classes = set()
    json_files = list(dataset_path.glob('[0-9]*'))

    print(f"\nScanning {len(json_files)} annotation files...")
    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)
        for annotation in data.get('result', []):
            if annotation['type'] == 'rectanglelabels':
                for label in annotation['value']['rectanglelabels']:
                    all_classes.add(label)

    # Create class mapping
    class_mapping = {cls: idx for idx, cls in enumerate(sorted(all_classes))}
    print(f"Found {len(class_mapping)} classes: {list(class_mapping.keys())}")

    # Process all annotations
    processed_images = []
    for json_file in json_files:
        temp_labels_dir = output_path / 'temp_labels'
        temp_labels_dir.mkdir(exist_ok=True)

        image_filename = convert_label_studio_to_yolo(json_file, temp_labels_dir, class_mapping)
        if image_filename:
            processed_images.append(image_filename)

    # Shuffle and split dataset
    random.shuffle(processed_images)
    split_idx = int(len(processed_images) * train_split)
    train_images = processed_images[:split_idx]
    val_images = processed_images[split_idx:]

    print(f"\nDataset split: {len(train_images)} training, {len(val_images)} validation")

    # Copy images and labels to appropriate directories
    for split, images in [('train', train_images), ('val', val_images)]:
        for image_filename in images:
            # Copy image
            src_image = screenshots_path / image_filename
            dst_image = output_path / split / 'images' / image_filename
            if src_image.exists():
                shutil.copy2(src_image, dst_image)
            else:
                print(f"Warning: Image not found: {src_image}")

            # Copy label
            label_filename = Path(image_filename).stem + '.txt'
            src_label = output_path / 'temp_labels' / label_filename
            dst_label = output_path / split / 'labels' / label_filename
            if src_label.exists():
                shutil.copy2(src_label, dst_label)

    # Clean up temporary directory
    shutil.rmtree(output_path / 'temp_labels', ignore_errors=True)

    print(f"Dataset prepared in: {output_path}")
    return class_mapping


def create_dataset_yaml(output_dir, class_mapping):
    """
    Create YOLO dataset configuration YAML file.

    Args:
        output_dir: Root directory of the YOLO dataset
        class_mapping: Dict mapping class names to class IDs
    """
    output_path = Path(output_dir)

    # Get absolute paths
    train_path = (output_path / 'train' / 'images').absolute()
    val_path = (output_path / 'val' / 'images').absolute()

    # Create YAML content
    yaml_content = {
        'path': str(output_path.absolute()),
        'train': str(train_path),
        'val': str(val_path),
        'names': {v: k for k, v in class_mapping.items()}
    }

    yaml_file = output_path / 'dataset.yaml'
    with open(yaml_file, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)

    print(f"Dataset YAML created: {yaml_file}")
    return yaml_file


def train_model(dataset_yaml, epochs=50, img_size=640, batch_size=16, model_name='yolov8n.pt'):
    """
    Train YOLO model on the prepared dataset.

    Args:
        dataset_yaml: Path to dataset YAML configuration
        epochs: Number of training epochs
        img_size: Input image size
        batch_size: Batch size for training
        model_name: Pretrained model to start from

    Returns:
        YOLO: Trained model object
    """
    print(f"\nInitializing {model_name} model...")
    model = YOLO(model_name)

    print(f"\nStarting training for {epochs} epochs...")
    print(f"Dataset: {dataset_yaml}")
    print(f"Image size: {img_size}")
    print(f"Batch size: {batch_size}")

    # Train the model
    results = model.train(
        data=str(dataset_yaml),
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        name='paper_engine_model',
        patience=10,  # Early stopping patience
        save=True,
        plots=True,
        verbose=True
    )

    print("\nTraining completed!")
    print(f"Best model saved to: runs/detect/paper_engine_model/weights/best.pt")

    return model


def export_model(model_path, formats=['torchscript', 'onnx']):
    """
    Export trained model to specified formats.

    Args:
        model_path: Path to trained model weights
        formats: List of export formats

    Returns:
        list: Paths to exported models
    """
    print(f"\nLoading model from: {model_path}")
    model = YOLO(model_path)

    exported_models = []

    if 'torchscript' in formats:
        print("\nExporting to TorchScript...")
        torchscript_path = model.export(format='torchscript')
        exported_models.append(torchscript_path)
        print(f"TorchScript model saved: {torchscript_path}")

    if 'onnx' in formats:
        print("\nExporting to ONNX...")
        onnx_path = model.export(format='onnx')
        exported_models.append(onnx_path)
        print(f"ONNX model saved: {onnx_path}")

    if 'tensorrt' in formats:
        print("\nExporting to TensorRT...")
        try:
            trt_path = model.export(format='engine')
            exported_models.append(trt_path)
            print(f"TensorRT model saved: {trt_path}")
        except Exception as e:
            print(f"TensorRT export failed: {e}")
            print("TensorRT requires NVIDIA GPU and proper drivers.")

    if 'pytorch' in formats or 'pt' in formats:
        # PyTorch format is already saved, just copy it
        print(f"\nPyTorch model already saved: {model_path}")
        exported_models.append(model_path)

    return exported_models


def main():
    """Main training pipeline."""
    print("=" * 60)
    print("Paper Engine - YOLO Model Training")
    print("=" * 60)

    # Configuration
    DATASET_DIR = 'dataset'
    SCREENSHOTS_DIR = 'screenshots'
    OUTPUT_DIR = 'yolo_dataset'
    TRAIN_SPLIT = 0.8
    EPOCHS = 50
    IMG_SIZE = 640
    BATCH_SIZE = 16
    MODEL_NAME = 'yolov8n.pt'
    EXPORT_FORMATS = ['torchscript', 'onnx']

    # Step 1: Prepare dataset
    print("\n[Step 1/4] Preparing dataset...")
    class_mapping = prepare_dataset(
        dataset_dir=DATASET_DIR,
        screenshots_dir=SCREENSHOTS_DIR,
        output_dir=OUTPUT_DIR,
        train_split=TRAIN_SPLIT
    )

    # Step 2: Create dataset YAML
    print("\n[Step 2/4] Creating dataset configuration...")
    dataset_yaml = create_dataset_yaml(OUTPUT_DIR, class_mapping)

    # Step 3: Train model
    print("\n[Step 3/4] Training model...")
    model = train_model(
        dataset_yaml=dataset_yaml,
        epochs=EPOCHS,
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        model_name=MODEL_NAME
    )

    # Step 4: Export model
    print("\n[Step 4/4] Exporting model...")
    best_model_path = 'runs/detect/paper_engine_model/weights/best.pt'
    exported_models = export_model(best_model_path, formats=EXPORT_FORMATS)

    print("\n" + "=" * 60)
    print("Training Pipeline Complete!")
    print("=" * 60)
    print(f"\nTrained classes: {list(class_mapping.keys())}")
    print(f"Training results: runs/detect/paper_engine_model/")
    print(f"Best model: {best_model_path}")
    print(f"\nExported models:")
    for model_path in exported_models:
        print(f"  - {model_path}")
    print("\nYou can now use these models for inference in your game bot!")


if __name__ == "__main__":
    main()
