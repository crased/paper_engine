"""
Configuration file for training_model.py
Paper Engine - YOLO model training settings
"""

# Dataset directories
DATASET_DIR = 'dataset'  # Label Studio annotation JSON files
SCREENSHOTS_DIR = 'screenshots'  # Source images for training
OUTPUT_DIR = 'yolo_dataset'  # YOLO format dataset output

# Dataset split
TRAIN_SPLIT = 0.8  # 80% training, 20% validation

# YOLO model settings
MODEL_NAME = 'yolov8n.pt'  # YOLOv8 nano (fastest, smallest)
# Other options: 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt'

# Training hyperparameters
EPOCHS = 50  # Number of training epochs
IMG_SIZE = 640  # Input image size (pixels)
BATCH_SIZE = 16  # Batch size for training
PATIENCE = 10  # Early stopping patience (epochs without improvement)

# Training options
SAVE_CHECKPOINTS = True  # Save model checkpoints during training
GENERATE_PLOTS = True  # Generate training plots
VERBOSE = True  # Print detailed training progress

# Model export settings
EXPORT_FORMATS = ['torchscript', 'onnx']
# Available formats: 'torchscript', 'onnx', 'tensorrt', 'pytorch', 'pt'
# TensorRT requires NVIDIA GPU and proper drivers

# Output settings
MODEL_OUTPUT_NAME = 'paper_engine_model'  # Name for training run
BEST_MODEL_PATH = 'runs/detect/paper_engine_model/weights/best.pt'

# Advanced training settings (optional)
OPTIMIZER = 'auto'  # Options: 'SGD', 'Adam', 'AdamW', 'auto'
LEARNING_RATE = 0.01  # Initial learning rate
MOMENTUM = 0.937  # SGD momentum/Adam beta1
WEIGHT_DECAY = 0.0005  # Optimizer weight decay

# Data augmentation (YOLO defaults)
HSV_H = 0.015  # HSV-Hue augmentation
HSV_S = 0.7  # HSV-Saturation augmentation
HSV_V = 0.4  # HSV-Value augmentation
DEGREES = 0.0  # Rotation augmentation degrees
TRANSLATE = 0.1  # Translation augmentation
SCALE = 0.5  # Scaling augmentation
SHEAR = 0.0  # Shear augmentation degrees
PERSPECTIVE = 0.0  # Perspective augmentation
FLIPUD = 0.0  # Vertical flip probability
FLIPLR = 0.5  # Horizontal flip probability (50%)
MOSAIC = 1.0  # Mosaic augmentation probability
MIXUP = 0.0  # MixUp augmentation probability
