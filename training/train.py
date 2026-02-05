"""
YOLOv8 Fine-tuning Script for Bus Person Detection
===================================================

This script fine-tunes YOLOv8 on your custom bus dataset using
transfer learning from COCO pretrained weights.

Features:
- Transfer learning from pretrained weights
- TensorBoard monitoring
- Early stopping
- Best model checkpoint saving
- Configurable hyperparameters

Usage:
    python training/train.py --data datasets/dataset.yaml
    python training/train.py --data datasets/dataset.yaml --epochs 100 --batch 16
    tensorboard --logdir runs/train  # Monitor training progress
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

try:
    from ultralytics import YOLO
except ImportError:
    print("ERROR: ultralytics not installed. Run: pip install ultralytics")
    sys.exit(1)


def check_gpu():
    """Check GPU availability and print info."""
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"✅ GPU Available: {device_name} ({memory_gb:.1f} GB)")
            return True
        else:
            print("⚠️  No GPU detected, training will use CPU (much slower)")
            return False
    except:
        print("⚠️  Could not detect GPU")
        return False


def train(
    data_yaml: str,
    model: str = "yolov8m.pt",
    epochs: int = 100,
    batch_size: int = 16,
    image_size: int = 640,
    patience: int = 20,
    project: str = "runs/train",
    name: str = None,
    device: str = None,
    workers: int = 4,
    resume: bool = False
):
    """
    Fine-tune YOLOv8 on custom dataset.
    
    Args:
        data_yaml: Path to dataset YAML configuration
        model: Pretrained model to start from
        epochs: Number of training epochs
        batch_size: Batch size for training
        image_size: Input image size
        patience: Early stopping patience
        project: Output project directory
        name: Run name (auto-generated if not provided)
        device: Device to train on ('0' for GPU, 'cpu' for CPU)
        workers: Number of data loading workers
        resume: Resume from last checkpoint
    """
    # Validate data yaml exists
    data_path = Path(data_yaml)
    if not data_path.exists():
        print(f"❌ Dataset config not found: {data_yaml}")
        print("  Run prepare_dataset.py first to create the dataset")
        sys.exit(1)
    
    # Check GPU
    has_gpu = check_gpu()
    if device is None:
        device = 0 if has_gpu else 'cpu'
    
    # Generate run name if not provided
    if name is None:
        name = f"bus_finetune_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print("\n" + "=" * 50)
    print("YOLOV8 FINE-TUNING")
    print("=" * 50)
    print(f"  Dataset: {data_yaml}")
    print(f"  Base model: {model}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Image size: {image_size}")
    print(f"  Patience: {patience}")
    print(f"  Device: {device}")
    print(f"  Output: {project}/{name}")
    print("=" * 50 + "\n")
    
    # Load model
    print(f"Loading base model: {model}")
    yolo_model = YOLO(model)
    
    # Train
    print("\nStarting training...")
    print("Monitor with TensorBoard: tensorboard --logdir runs/train\n")
    
    results = yolo_model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=image_size,
        patience=patience,
        device=device,
        workers=workers,
        project=project,
        name=name,
        save=True,
        save_period=10,  # Save checkpoint every 10 epochs
        exist_ok=True,
        pretrained=True,
        resume=resume,
        
        # Augmentation (suitable for person detection)
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
        
        # Performance
        amp=True,  # Automatic mixed precision
        
        # Logging
        verbose=True
    )
    
    # Print results
    print("\n" + "=" * 50)
    print("TRAINING COMPLETE")
    print("=" * 50)
    
    best_model_path = Path(project) / name / "weights" / "best.pt"
    last_model_path = Path(project) / name / "weights" / "last.pt"
    
    print(f"\n📁 Results saved to: {project}/{name}")
    print(f"📊 Best model: {best_model_path}")
    print(f"📊 Last model: {last_model_path}")
    
    if best_model_path.exists():
        # Validate best model
        print("\n📈 Validating best model on validation set...")
        best_model = YOLO(str(best_model_path))
        val_results = best_model.val(data=data_yaml)
        
        print(f"\n📊 Validation Results:")
        print(f"   mAP@50: {val_results.box.map50:.4f}")
        print(f"   mAP@50-95: {val_results.box.map:.4f}")
        print(f"   Precision: {val_results.box.mp:.4f}")
        print(f"   Recall: {val_results.box.mr:.4f}")
    
    print("\n🎉 Next steps:")
    print(f"   1. Evaluate: python training/evaluate.py --model {best_model_path}")
    print(f"   2. Use for detection: python -m bus_tracker.detector --model {best_model_path}")
    
    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Fine-tune YOLOv8 for bus person detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Basic training:
    python train.py --data datasets/dataset.yaml
    
  Custom hyperparameters:
    python train.py --data datasets/dataset.yaml --epochs 50 --batch 8 --imgsz 800
    
  Use smaller model for faster training:
    python train.py --data datasets/dataset.yaml --model yolov8s.pt
    
  Resume interrupted training:
    python train.py --data datasets/dataset.yaml --resume
    
  Monitor training:
    tensorboard --logdir runs/train
        """
    )
    
    parser.add_argument(
        "--data", "-d",
        default="datasets/dataset.yaml",
        help="Path to dataset YAML configuration"
    )
    parser.add_argument(
        "--model", "-m",
        default="yolov8m.pt",
        help="Base model: yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt"
    )
    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=100,
        help="Number of training epochs (default: 100)"
    )
    parser.add_argument(
        "--batch", "-b",
        type=int,
        default=16,
        help="Batch size (default: 16, reduce if OOM)"
    )
    parser.add_argument(
        "--imgsz", "-s",
        type=int,
        default=640,
        help="Input image size (default: 640)"
    )
    parser.add_argument(
        "--patience", "-p",
        type=int,
        default=20,
        help="Early stopping patience (default: 20)"
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device: 0 for GPU, cpu for CPU"
    )
    parser.add_argument(
        "--name", "-n",
        default=None,
        help="Run name (auto-generated if not provided)"
    )
    parser.add_argument(
        "--resume", "-r",
        action="store_true",
        help="Resume training from last checkpoint"
    )
    
    args = parser.parse_args()
    
    train(
        data_yaml=args.data,
        model=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        image_size=args.imgsz,
        patience=args.patience,
        device=args.device,
        name=args.name,
        resume=args.resume
    )


if __name__ == "__main__":
    main()
