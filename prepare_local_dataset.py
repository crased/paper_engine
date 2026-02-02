#!/usr/bin/env python3
"""
Convert local Label Studio annotations to YOLO format
"""

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
    image_filename = Path(image_path.split('?d=')[-1]).name

    # Get annotations
    annotations = data['result']
    if not annotations:
        return None

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


def prepare_dataset(dataset_dir='dataset', screenshots_dir='screenshots', output_dir='dataset', train_split=0.8):
    """
    Prepare YOLO dataset from Label Studio annotations.

    Args:
        dataset_dir: Directory containing Label Studio JSON files
        screenshots_dir: Directory containing screenshot images
        output_dir: Output directory for YOLO format dataset (will be dataset/)
        train_split: Fraction of data to use for training

    Returns:
        dict: Class mapping {class_name: class_id}
    """
    dataset_path = Path(dataset_dir)
    screenshots_path = Path(screenshots_dir)
    output_path = Path(output_dir)

    # Create output directory structure
    for split in ['train', 'valid']:
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
    temp_labels_dir = output_path / 'temp_labels'
    temp_labels_dir.mkdir(exist_ok=True)

    for json_file in json_files:
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
    for split, images in [('train', train_images), ('valid', val_images)]:
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
            src_label = temp_labels_dir / label_filename
            dst_label = output_path / split / 'labels' / label_filename
            if src_label.exists():
                shutil.copy2(src_label, dst_label)

    # Clean up temporary directory
    shutil.rmtree(temp_labels_dir, ignore_errors=True)

    print(f"Dataset prepared in: {output_path}")
    return class_mapping


def create_dataset_yaml(output_dir, class_mapping):
    """
    Create YOLO dataset configuration YAML file.

    Args:
        output_dir: Root directory of the YOLO dataset
        class_mapping: Dict mapping class names to class IDs
    """
    output_path = Path(output_dir).absolute()

    # Create YAML content
    yaml_content = {
        'path': str(output_path),
        'train': 'train/images',
        'val': 'valid/images',
        'names': {v: k for k, v in class_mapping.items()}
    }

    yaml_file = output_path / 'data.yaml'
    with open(yaml_file, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)

    print(f"Dataset YAML created: {yaml_file}")
    return yaml_file


if __name__ == "__main__":
    print("=" * 60)
    print("Converting Local Dataset to YOLO Format")
    print("=" * 60)

    # Prepare dataset
    class_mapping = prepare_dataset(
        dataset_dir='dataset',
        screenshots_dir='screenshots',
        output_dir='dataset',
        train_split=0.8
    )

    # Create YAML config
    create_dataset_yaml('dataset', class_mapping)

    print("\n" + "=" * 60)
    print("Dataset Ready for Training!")
    print("=" * 60)
    print(f"Classes: {list(class_mapping.keys())}")
    print("\nRun: python training_model.py")
