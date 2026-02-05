"""
Dataset Preparation Script for YOLOv8 Fine-tuning
=================================================

This script prepares your bus video footage for training by:
1. Extracting frames at specified FPS
2. Splitting into train/val/test sets
3. Generating YOLO dataset configuration

Usage:
    python training/prepare_dataset.py --input input/ --output datasets/
    python training/prepare_dataset.py --input video.mp4 --fps 2 --split 0.7 0.2 0.1
"""

import os
import sys
import cv2
import shutil
import random
import argparse
from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm


def extract_frames(
    video_path: str,
    output_dir: str,
    target_fps: float = 1.0,
    max_frames: int = None
) -> List[str]:
    """
    Extract frames from video at specified FPS.
    
    Args:
        video_path: Path to input video
        output_dir: Directory to save frames
        target_fps: Frames per second to extract
        max_frames: Maximum frames to extract (optional)
        
    Returns:
        List of extracted frame paths
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate frame interval
    frame_interval = max(1, int(video_fps / target_fps))
    
    print(f"Video: {video_path}")
    print(f"  Original FPS: {video_fps:.1f}")
    print(f"  Total frames: {total_frames}")
    print(f"  Extracting every {frame_interval} frames ({target_fps} FPS)")
    
    extracted_paths = []
    frame_num = 0
    extracted_count = 0
    video_name = Path(video_path).stem
    
    # Progress bar
    pbar = tqdm(total=total_frames // frame_interval, desc="Extracting frames")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Check if we should extract this frame
        if frame_num % frame_interval == 0:
            # Generate unique filename
            frame_filename = f"{video_name}_frame_{frame_num:08d}.jpg"
            frame_path = output_path / frame_filename
            
            # Save frame
            cv2.imwrite(str(frame_path), frame)
            extracted_paths.append(str(frame_path))
            
            extracted_count += 1
            pbar.update(1)
            
            if max_frames and extracted_count >= max_frames:
                break
        
        frame_num += 1
    
    pbar.close()
    cap.release()
    
    print(f"  Extracted {extracted_count} frames to {output_dir}")
    return extracted_paths


def split_dataset(
    frames_dir: str,
    output_base: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[int, int, int]:
    """
    Split frames into train/val/test sets.
    
    Args:
        frames_dir: Directory containing extracted frames
        output_base: Base output directory for splits
        train_ratio: Proportion for training (0.0-1.0)
        val_ratio: Proportion for validation (0.0-1.0)
        test_ratio: Proportion for testing (0.0-1.0)
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_count, val_count, test_count)
    """
    # Validate ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 0.01:
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")
    
    # Get all image files
    frames_path = Path(frames_dir)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    images = [
        f for f in frames_path.iterdir()
        if f.suffix.lower() in image_extensions
    ]
    
    if not images:
        raise ValueError(f"No images found in {frames_dir}")
    
    print(f"\nSplitting {len(images)} images...")
    print(f"  Train: {train_ratio*100:.0f}%")
    print(f"  Val: {val_ratio*100:.0f}%")
    print(f"  Test: {test_ratio*100:.0f}%")
    
    # Shuffle with seed
    random.seed(seed)
    random.shuffle(images)
    
    # Calculate split indices
    n = len(images)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    train_images = images[:train_end]
    val_images = images[train_end:val_end]
    test_images = images[val_end:]
    
    # Create output directories and copy files
    output_path = Path(output_base)
    splits = [
        ('train', train_images),
        ('val', val_images),
        ('test', test_images)
    ]
    
    for split_name, split_images in splits:
        images_dir = output_path / split_name / 'images'
        labels_dir = output_path / split_name / 'labels'
        
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        
        for img_path in tqdm(split_images, desc=f"Copying {split_name}"):
            # Copy image
            dest_path = images_dir / img_path.name
            shutil.copy2(img_path, dest_path)
            
            # Create empty label file (will be filled during annotation)
            label_path = labels_dir / img_path.with_suffix('.txt').name
            label_path.touch()
    
    print(f"\nDataset split complete:")
    print(f"  Train: {len(train_images)} images")
    print(f"  Val: {len(val_images)} images")
    print(f"  Test: {len(test_images)} images")
    
    return len(train_images), len(val_images), len(test_images)


def generate_dataset_yaml(
    output_dir: str,
    class_names: List[str] = None
) -> str:
    """
    Generate YOLO dataset configuration YAML file.
    
    Args:
        output_dir: Directory containing train/val/test splits
        class_names: List of class names (default: ['person'])
        
    Returns:
        Path to generated YAML file
    """
    if class_names is None:
        class_names = ['person']
    
    output_path = Path(output_dir)
    yaml_path = output_path / 'dataset.yaml'
    
    # Build YAML content
    yaml_content = f"""# YOLOv8 Dataset Configuration
# Generated by Bus Tracker Dataset Preparation Script

# Dataset root directory (relative to this file)
path: {output_path.absolute()}

# Train/val/test directories (relative to path)
train: train/images
val: val/images
test: test/images

# Number of classes
nc: {len(class_names)}

# Class names
names:
"""
    
    for i, name in enumerate(class_names):
        yaml_content += f"  {i}: {name}\n"
    
    # Write file
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"\nGenerated dataset config: {yaml_path}")
    return str(yaml_path)


def get_video_files(input_path: str) -> List[str]:
    """Get all video files from path (file or directory)."""
    path = Path(input_path)
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
    
    if path.is_file():
        if path.suffix.lower() in video_extensions:
            return [str(path)]
        return []
    
    videos = []
    for ext in video_extensions:
        videos.extend(path.glob(f'*{ext}'))
        videos.extend(path.glob(f'*{ext.upper()}'))
    
    return [str(v) for v in sorted(videos)]


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Prepare bus video footage for YOLOv8 fine-tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Extract at 1 FPS from all videos in input/:
    python prepare_dataset.py --input input/ --output datasets/
    
  Extract at 2 FPS with custom split:
    python prepare_dataset.py --input video.mp4 --fps 2 --split 0.8 0.1 0.1
    
  Extract maximum 1000 frames:
    python prepare_dataset.py --input input/ --max-frames 1000
        """
    )
    
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input video file or directory"
    )
    parser.add_argument(
        "--output", "-o",
        default="datasets",
        help="Output directory for dataset (default: datasets/)"
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=1.0,
        help="Frames per second to extract (default: 1.0)"
    )
    parser.add_argument(
        "--split",
        nargs=3,
        type=float,
        default=[0.7, 0.2, 0.1],
        metavar=("TRAIN", "VAL", "TEST"),
        help="Train/val/test split ratios (default: 0.7 0.2 0.1)"
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum frames to extract per video"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    args = parser.parse_args()
    
    # Get videos
    videos = get_video_files(args.input)
    if not videos:
        print(f"No video files found in: {args.input}")
        sys.exit(1)
    
    print(f"Found {len(videos)} video(s) to process\n")
    
    # Create output directories
    output_path = Path(args.output)
    raw_frames_dir = output_path / "raw_frames"
    
    # Extract frames from all videos
    all_frames = []
    for video in videos:
        frames = extract_frames(
            video,
            str(raw_frames_dir),
            target_fps=args.fps,
            max_frames=args.max_frames
        )
        all_frames.extend(frames)
    
    print(f"\nTotal frames extracted: {len(all_frames)}")
    
    # Split dataset
    train_ratio, val_ratio, test_ratio = args.split
    split_dataset(
        str(raw_frames_dir),
        str(output_path),
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=args.seed
    )
    
    # Generate YAML config
    generate_dataset_yaml(str(output_path))
    
    print("\n" + "=" * 50)
    print("DATASET PREPARATION COMPLETE")
    print("=" * 50)
    print(f"\nNext steps:")
    print(f"1. Annotate images in {output_path}/train/images/")
    print(f"   Use Roboflow or CVAT for annotation")
    print(f"2. Export annotations to YOLO format")
    print(f"3. Place label files in corresponding labels/ directories")
    print(f"4. Run training: python training/train.py --data {output_path}/dataset.yaml")


if __name__ == "__main__":
    main()
