#!/usr/bin/env python3
"""
Convert local Label Studio annotations to YOLO format
"""

import json
import shutil
from pathlib import Path
import random
import yaml
from conf.config_parser import prepare_dataset_conf as config

def convert_label_studio_to_yolo(json_file, output_labels_dir, class_mapping, label_mapping=None):
    """
    Convert a single Label Studio JSON file to YOLO format.

    Args:
        json_file: Path to Label Studio JSON annotation file
        output_labels_dir: Directory to save YOLO format labels
        class_mapping: Dict mapping class names to class IDs
        label_mapping: Optional dict for renaming labels {old_name: new_name}

    Returns:
        str: Image filename referenced in the annotation
    """
    if label_mapping is None:
        label_mapping = {}
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

        # Apply label renaming if configured
        label = label_mapping.get(label, label)

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


def prepare_dataset(dataset_dir=None, screenshots_dir=None, output_dir=None, train_split=None):
    """
    Prepare YOLO dataset from Label Studio annotations.

    Args:
        dataset_dir: Directory containing Label Studio JSON files (default: from config)
        screenshots_dir: Directory containing screenshot images (default: from config)
        output_dir: Output directory for YOLO format dataset (default: from config)
        train_split: Fraction of data to use for training (default: from config)

    Returns:
        dict: Class mapping {class_name: class_id}
    """
    # Use config defaults if not specified
    if dataset_dir is None:
        dataset_dir = config.DATASET_DIR
    if screenshots_dir is None:
        screenshots_dir = config.SCREENSHOTS_DIR
    if output_dir is None:
        output_dir = config.OUTPUT_DIR
    if train_split is None:
        train_split = config.TRAIN_SPLIT
    dataset_path = Path(dataset_dir)
    screenshots_path = Path(screenshots_dir)
    output_path = Path(output_dir)

    # Create output directory structure
    for split in ['train', 'valid']:
        (output_path / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_path / split / 'labels').mkdir(parents=True, exist_ok=True)

    # Collect all unique class names from annotations
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

    print(f"Discovered {len(all_classes)} unique labels: {sorted(all_classes)}")

    # Create class mapping using config (supports custom grouping and renaming)
    class_mapping = config.get_class_mapping(all_classes)

    if not class_mapping:
        print("WARNING: No class mapping generated!")
        return {}

    print(f"\nClass mapping ({len(class_mapping)} classes):")
    for class_name, class_id in sorted(class_mapping.items(), key=lambda x: x[1]):
        print(f"  {class_id}: {class_name}")

    # Process all annotations
    processed_images = []
    temp_labels_dir = output_path / 'temp_labels'
    temp_labels_dir.mkdir(exist_ok=True)

    # Get label mapping for renaming
    label_mapping = config.LABEL_MAPPING

    for json_file in json_files:
        image_filename = convert_label_studio_to_yolo(
            json_file, temp_labels_dir, class_mapping, label_mapping
        )
        if image_filename:
            processed_images.append(image_filename)

    # Shuffle and split dataset
    random_seed = config.RANDOM_SEED
    if random_seed and random_seed > 0:
        random.seed(random_seed)
        print(f"\nUsing random seed: {random_seed} (reproducible split)")
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

    # Show configuration
    print(f"\nConfiguration:")
    print(f"  Dataset directory: {config.DATASET_DIR}")
    print(f"  Screenshots directory: {config.SCREENSHOTS_DIR}")
    print(f"  Output directory: {config.OUTPUT_DIR}")
    print(f"  Train split: {config.TRAIN_SPLIT * 100}%")
    print(f"  Custom grouping: {'Enabled' if config.ENABLE_CUSTOM_GROUPING else 'Disabled (auto-discovery)'}")

    # Prepare dataset (uses config defaults)
    class_mapping = prepare_dataset()

    if not class_mapping:
        print("\nERROR: No classes mapped. Check your configuration.")
        import sys
        sys.exit(1)

    # Create YAML config
    create_dataset_yaml(config.OUTPUT_DIR, class_mapping)

    print("\n" + "=" * 60)
    print("Dataset Ready for Training!")
    print("=" * 60)
    print(f"Classes: {list(class_mapping.keys())}")
    print("\nRun: python training_model.py")
