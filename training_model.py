"""
YOLO Model Training Script for Paper Engine

Trains a YOLO11 model on pre-prepared YOLO format datasets and exports it for inference.

YOLO11 Benefits:
- 5x faster training convergence
- 36% faster CPU inference
- +2.2% mAP improvement over YOLOv8
- 22% fewer parameters

Compatible with YOLO format datasets from:
- Manually created YOLO datasets (see prepare_local_dataset.py)
- Any source with train/valid/test splits in YOLO format
"""

from ultralytics import YOLO
import torch
import yaml
from pathlib import Path
from conf.config_parser import training_conf as config


def train_model(dataset_yaml, epochs=50, img_size=640, batch_size=16, model_name='yolo11n.pt'):
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
        name=config.MODEL_OUTPUT_NAME,
        patience=config.PATIENCE,
        save=config.SAVE_CHECKPOINTS,
        plots=config.GENERATE_PLOTS,
        verbose=config.VERBOSE
    )

    print("\nTraining completed!")
    print(f"Best model saved to: runs/detect/{config.MODEL_OUTPUT_NAME}/weights/best.pt")

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
    """Main training pipeline for pre-prepared YOLO datasets."""
    print("=" * 60)
    print("Paper Engine - YOLO Model Training")
    print("=" * 60)

    # Check if dataset directory exists
    dataset_path = Path(config.DATASET_DIR)
    if not dataset_path.exists():
        print(f"\nERROR: Dataset directory not found: {dataset_path}")
        print("\nPlease prepare your dataset using prepare_local_dataset.py")
        print("Or place a YOLO format dataset in the dataset/ directory")
        return

    # Look for data.yaml or dataset.yaml
    yaml_file = None
    for yaml_name in ['data.yaml', 'dataset.yaml']:
        potential_yaml = dataset_path / yaml_name
        if potential_yaml.exists():
            yaml_file = potential_yaml
            break

    if not yaml_file:
        print(f"\nERROR: No dataset configuration YAML found in {dataset_path}")
        print("Expected: data.yaml or dataset.yaml")
        return

    print(f"\nFound dataset configuration: {yaml_file}")

    # Step 1: Train model
    print("\n[Step 1/2] Training model...")
    model = train_model(
        dataset_yaml=yaml_file,
        epochs=config.EPOCHS,
        img_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        model_name=config.MODEL_NAME
    )

    # Step 2: Export model
    print("\n[Step 2/2] Exporting model...")
    best_model_path = config.BEST_MODEL_PATH
    exported_models = export_model(best_model_path, formats=config.EXPORT_FORMATS)

    print("\n" + "=" * 60)
    print("Training Pipeline Complete!")
    print("=" * 60)
    print(f"Training results: runs/detect/{config.MODEL_OUTPUT_NAME}/")
    print(f"Best model: {best_model_path}")
    print(f"\nExported models:")
    for model_path in exported_models:
        print(f"  - {model_path}")
    print("\nYou can now use these models for inference in your game bot!")


if __name__ == "__main__":
    main()
