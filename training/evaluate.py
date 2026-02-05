"""
Model Evaluation Script for Bus Person Detection
=================================================

Evaluates and compares model performance:
- Test pretrained vs fine-tuned models
- Generate metrics: mAP, precision, recall
- Create comparison visualizations

Usage:
    python training/evaluate.py --model runs/train/bus_finetune/weights/best.pt
    python training/evaluate.py --compare yolov8m.pt runs/train/bus_finetune/weights/best.pt
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any
import json

try:
    from ultralytics import YOLO
except ImportError:
    print("ERROR: ultralytics not installed. Run: pip install ultralytics")
    sys.exit(1)


def evaluate_model(
    model_path: str,
    data_yaml: str,
    split: str = "test",
    device: str = None
) -> Dict[str, float]:
    """
    Evaluate a model on the specified dataset split.
    
    Args:
        model_path: Path to model weights
        data_yaml: Path to dataset YAML
        split: Dataset split to evaluate on ('val' or 'test')
        device: Device to use
        
    Returns:
        Dictionary of metrics
    """
    print(f"\n📊 Evaluating: {model_path}")
    print(f"   Dataset: {data_yaml}")
    print(f"   Split: {split}")
    
    # Load model
    model = YOLO(model_path)
    
    # Detect device
    if device is None:
        try:
            import torch
            device = 0 if torch.cuda.is_available() else 'cpu'
        except:
            device = 'cpu'
    
    # Run validation
    results = model.val(
        data=data_yaml,
        split=split,
        device=device,
        verbose=False
    )
    
    metrics = {
        'model': str(model_path),
        'mAP50': float(results.box.map50),
        'mAP50-95': float(results.box.map),
        'precision': float(results.box.mp),
        'recall': float(results.box.mr),
        'f1': 2 * (results.box.mp * results.box.mr) / (results.box.mp + results.box.mr + 1e-6)
    }
    
    return metrics


def print_metrics(metrics: Dict[str, float], name: str = None):
    """Print formatted metrics."""
    title = name or metrics.get('model', 'Model')
    
    print(f"\n{'=' * 50}")
    print(f" {title}")
    print(f"{'=' * 50}")
    print(f"  mAP@50:     {metrics['mAP50']:.4f}")
    print(f"  mAP@50-95:  {metrics['mAP50-95']:.4f}")
    print(f"  Precision:  {metrics['precision']:.4f}")
    print(f"  Recall:     {metrics['recall']:.4f}")
    print(f"  F1 Score:   {metrics['f1']:.4f}")
    print(f"{'=' * 50}")


def compare_models(
    models: List[str],
    data_yaml: str,
    split: str = "test",
    output_path: str = None
) -> List[Dict[str, float]]:
    """
    Compare multiple models on the same dataset.
    
    Args:
        models: List of model paths to compare
        data_yaml: Path to dataset YAML
        split: Dataset split to evaluate on
        output_path: Optional path to save comparison results
        
    Returns:
        List of metrics dictionaries
    """
    all_metrics = []
    
    for model_path in models:
        if not Path(model_path).exists():
            print(f"⚠️  Model not found: {model_path}")
            continue
        
        metrics = evaluate_model(model_path, data_yaml, split)
        all_metrics.append(metrics)
    
    if len(all_metrics) < 2:
        print("Need at least 2 models to compare")
        return all_metrics
    
    # Print comparison table
    print("\n" + "=" * 80)
    print(" MODEL COMPARISON")
    print("=" * 80)
    print(f"{'Model':<40} {'mAP50':>8} {'mAP':>8} {'Prec':>8} {'Recall':>8} {'F1':>8}")
    print("-" * 80)
    
    for m in all_metrics:
        model_name = Path(m['model']).stem[:38]
        print(f"{model_name:<40} {m['mAP50']:>8.4f} {m['mAP50-95']:>8.4f} "
              f"{m['precision']:>8.4f} {m['recall']:>8.4f} {m['f1']:>8.4f}")
    
    print("=" * 80)
    
    # Calculate improvement
    if len(all_metrics) >= 2:
        base = all_metrics[0]
        best = max(all_metrics, key=lambda x: x['mAP50'])
        
        if best != base:
            improvement = (best['mAP50'] - base['mAP50']) / base['mAP50'] * 100
            print(f"\n📈 Best model improvement over baseline: {improvement:+.1f}% mAP@50")
    
    # Save results
    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(all_metrics, f, indent=2)
        print(f"\n📁 Results saved to: {output_path}")
    
    return all_metrics


def test_on_images(
    model_path: str,
    images_dir: str,
    output_dir: str = None,
    conf: float = 0.5
):
    """
    Run inference on test images and save visualizations.
    
    Args:
        model_path: Path to model weights
        images_dir: Directory containing test images
        output_dir: Directory to save results
        conf: Confidence threshold
    """
    print(f"\n🔍 Testing model on images from: {images_dir}")
    
    model = YOLO(model_path)
    
    # Get image files
    images_path = Path(images_dir)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    images = [
        f for f in images_path.iterdir()
        if f.suffix.lower() in image_extensions
    ]
    
    if not images:
        print(f"No images found in: {images_dir}")
        return
    
    print(f"   Found {len(images)} images")
    
    # Run inference
    results = model.predict(
        source=images_dir,
        conf=conf,
        save=output_dir is not None,
        project=output_dir,
        name="test_predictions"
    )
    
    # Print summary
    total_detections = sum(len(r.boxes) for r in results)
    print(f"\n📊 Detection Summary:")
    print(f"   Images processed: {len(results)}")
    print(f"   Total detections: {total_detections}")
    print(f"   Avg per image: {total_detections / len(results):.1f}")
    
    if output_dir:
        print(f"\n📁 Predictions saved to: {output_dir}/test_predictions")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate YOLOv8 models for person detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Evaluate single model:
    python evaluate.py --model runs/train/bus_finetune/weights/best.pt
    
  Compare pretrained vs fine-tuned:
    python evaluate.py --compare yolov8m.pt runs/train/bus_finetune/weights/best.pt
    
  Test on images:
    python evaluate.py --model best.pt --test-images datasets/test/images
        """
    )
    
    parser.add_argument(
        "--model", "-m",
        help="Path to model to evaluate"
    )
    parser.add_argument(
        "--compare", "-c",
        nargs='+',
        help="Multiple models to compare"
    )
    parser.add_argument(
        "--data", "-d",
        default="datasets/dataset.yaml",
        help="Path to dataset YAML"
    )
    parser.add_argument(
        "--split", "-s",
        default="test",
        choices=['val', 'test'],
        help="Dataset split to evaluate on"
    )
    parser.add_argument(
        "--test-images", "-t",
        help="Directory of images to run inference on"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output path for results"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="Confidence threshold for inference"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.model and not args.compare:
        parser.error("Either --model or --compare is required")
    
    if not Path(args.data).exists():
        print(f"⚠️  Dataset config not found: {args.data}")
        print("   Continuing with model testing only...")
    
    # Compare models
    if args.compare:
        compare_models(
            models=args.compare,
            data_yaml=args.data,
            split=args.split,
            output_path=args.output
        )
    
    # Evaluate single model
    elif args.model:
        if not Path(args.model).exists():
            print(f"❌ Model not found: {args.model}")
            sys.exit(1)
        
        metrics = evaluate_model(args.model, args.data, args.split)
        print_metrics(metrics)
        
        # Test on images if provided
        if args.test_images:
            test_on_images(
                args.model,
                args.test_images,
                args.output,
                args.conf
            )


if __name__ == "__main__":
    main()
