"""
Annotation Conversion Script for YOLOv8 Training
=================================================

Converts annotations from various formats to YOLO format:
- CVAT XML format
- Roboflow export (already YOLO format, just copies)
- COCO JSON format

Also includes validation to check annotation quality.

Usage:
    python training/convert_annotations.py --input annotations.xml --format cvat --output datasets/
    python training/convert_annotations.py --input coco.json --format coco --output datasets/
"""

import os
import sys
import json
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


def convert_cvat_xml(
    xml_path: str,
    output_dir: str,
    class_mapping: Dict[str, int] = None
) -> int:
    """
    Convert CVAT XML annotations to YOLO format.
    
    Args:
        xml_path: Path to CVAT XML file
        output_dir: Directory for output label files
        class_mapping: Optional class name to class ID mapping
        
    Returns:
        Number of annotations converted
    """
    if class_mapping is None:
        class_mapping = {'person': 0}
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Parse XML
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    total_annotations = 0
    files_processed = 0
    
    # CVAT format has <image> elements with <box> children
    for image in root.findall('.//image'):
        image_name = image.get('name')
        width = float(image.get('width', 1920))
        height = float(image.get('height', 1080))
        
        # Get all boxes for this image
        annotations = []
        for box in image.findall('box'):
            label = box.get('label', 'person')
            
            if label not in class_mapping:
                continue
            
            class_id = class_mapping[label]
            
            # Get box coordinates
            xtl = float(box.get('xtl'))
            ytl = float(box.get('ytl'))
            xbr = float(box.get('xbr'))
            ybr = float(box.get('ybr'))
            
            # Convert to YOLO format (center x, center y, width, height) normalized
            x_center = ((xtl + xbr) / 2) / width
            y_center = ((ytl + ybr) / 2) / height
            box_width = (xbr - xtl) / width
            box_height = (ybr - ytl) / height
            
            annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}")
            total_annotations += 1
        
        # Write label file
        if annotations:
            label_filename = Path(image_name).stem + '.txt'
            label_path = output_path / label_filename
            
            with open(label_path, 'w') as f:
                f.write('\n'.join(annotations))
            
            files_processed += 1
    
    print(f"Converted {total_annotations} annotations from {files_processed} images")
    return total_annotations


def convert_coco_json(
    json_path: str,
    output_dir: str
) -> int:
    """
    Convert COCO JSON annotations to YOLO format.
    
    Args:
        json_path: Path to COCO JSON file
        output_dir: Directory for output label files
        
    Returns:
        Number of annotations converted
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load COCO JSON
    with open(json_path, 'r') as f:
        coco_data = json.load(f)
    
    # Build category mapping
    category_map = {}
    for cat in coco_data.get('categories', []):
        if cat['name'].lower() == 'person':
            category_map[cat['id']] = 0  # person = class 0
    
    # Build image info mapping
    image_info = {}
    for img in coco_data.get('images', []):
        image_info[img['id']] = {
            'filename': img['file_name'],
            'width': img['width'],
            'height': img['height']
        }
    
    # Group annotations by image
    annotations_by_image = defaultdict(list)
    for ann in coco_data.get('annotations', []):
        if ann['category_id'] in category_map:
            annotations_by_image[ann['image_id']].append(ann)
    
    total_annotations = 0
    files_processed = 0
    
    # Convert each image's annotations
    for image_id, annotations in annotations_by_image.items():
        if image_id not in image_info:
            continue
        
        img = image_info[image_id]
        width = img['width']
        height = img['height']
        
        yolo_annotations = []
        for ann in annotations:
            class_id = category_map[ann['category_id']]
            
            # COCO bbox is [x, y, width, height] in pixels
            bbox = ann['bbox']
            x, y, w, h = bbox
            
            # Convert to YOLO format
            x_center = (x + w / 2) / width
            y_center = (y + h / 2) / height
            box_width = w / width
            box_height = h / height
            
            yolo_annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}")
            total_annotations += 1
        
        # Write label file
        if yolo_annotations:
            label_filename = Path(img['filename']).stem + '.txt'
            label_path = output_path / label_filename
            
            with open(label_path, 'w') as f:
                f.write('\n'.join(yolo_annotations))
            
            files_processed += 1
    
    print(f"Converted {total_annotations} annotations from {files_processed} images")
    return total_annotations


def copy_roboflow_labels(
    source_dir: str,
    output_dir: str
) -> int:
    """
    Copy Roboflow YOLO format labels to output directory.
    
    Roboflow exports labels in YOLO format, so we just copy them.
    
    Args:
        source_dir: Directory containing Roboflow label files
        output_dir: Destination directory
        
    Returns:
        Number of files copied
    """
    import shutil
    
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    count = 0
    for label_file in source_path.glob('*.txt'):
        dest = output_path / label_file.name
        shutil.copy2(label_file, dest)
        count += 1
    
    print(f"Copied {count} label files")
    return count


def validate_annotations(
    labels_dir: str,
    images_dir: str = None
) -> Dict[str, any]:
    """
    Validate annotation quality and report issues.
    
    Checks:
    - Valid class IDs
    - Valid coordinate ranges (0-1)
    - Matching image files
    - Empty label files
    
    Args:
        labels_dir: Directory containing label files
        images_dir: Optional directory containing images
        
    Returns:
        Validation report dictionary
    """
    labels_path = Path(labels_dir)
    images_path = Path(images_dir) if images_dir else None
    
    report = {
        'total_files': 0,
        'total_annotations': 0,
        'empty_files': [],
        'invalid_coordinates': [],
        'invalid_class_ids': [],
        'missing_images': [],
        'valid': True
    }
    
    for label_file in labels_path.glob('*.txt'):
        report['total_files'] += 1
        
        # Check for matching image
        if images_path:
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
            image_found = False
            for ext in image_extensions:
                if (images_path / (label_file.stem + ext)).exists():
                    image_found = True
                    break
            
            if not image_found:
                report['missing_images'].append(str(label_file))
        
        # Parse label file
        with open(label_file, 'r') as f:
            lines = f.readlines()
        
        if not lines or all(line.strip() == '' for line in lines):
            report['empty_files'].append(str(label_file))
            continue
        
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            if len(parts) != 5:
                report['invalid_coordinates'].append(f"{label_file}:{line_num}")
                report['valid'] = False
                continue
            
            try:
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                
                report['total_annotations'] += 1
                
                # Check class ID
                if class_id < 0:
                    report['invalid_class_ids'].append(f"{label_file}:{line_num}")
                    report['valid'] = False
                
                # Check coordinates
                coords = [x_center, y_center, width, height]
                if any(c < 0 or c > 1 for c in coords):
                    report['invalid_coordinates'].append(f"{label_file}:{line_num}")
                    report['valid'] = False
                    
            except ValueError:
                report['invalid_coordinates'].append(f"{label_file}:{line_num}")
                report['valid'] = False
    
    return report


def print_validation_report(report: Dict[str, any]):
    """Print formatted validation report."""
    print("\n" + "=" * 50)
    print("ANNOTATION VALIDATION REPORT")
    print("=" * 50)
    
    print(f"\nTotal files: {report['total_files']}")
    print(f"Total annotations: {report['total_annotations']}")
    
    if report['empty_files']:
        print(f"\n⚠️  Empty files ({len(report['empty_files'])}):")
        for f in report['empty_files'][:5]:
            print(f"   - {f}")
        if len(report['empty_files']) > 5:
            print(f"   ... and {len(report['empty_files']) - 5} more")
    
    if report['invalid_coordinates']:
        print(f"\n❌ Invalid coordinates ({len(report['invalid_coordinates'])}):")
        for f in report['invalid_coordinates'][:5]:
            print(f"   - {f}")
        if len(report['invalid_coordinates']) > 5:
            print(f"   ... and {len(report['invalid_coordinates']) - 5} more")
    
    if report['invalid_class_ids']:
        print(f"\n❌ Invalid class IDs ({len(report['invalid_class_ids'])}):")
        for f in report['invalid_class_ids'][:5]:
            print(f"   - {f}")
    
    if report['missing_images']:
        print(f"\n⚠️  Missing images ({len(report['missing_images'])}):")
        for f in report['missing_images'][:5]:
            print(f"   - {f}")
        if len(report['missing_images']) > 5:
            print(f"   ... and {len(report['missing_images']) - 5} more")
    
    if report['valid'] and not report['empty_files'] and not report['missing_images']:
        print("\n✅ All annotations are valid!")
    elif not report['valid']:
        print("\n❌ Some annotations have errors. Please fix before training.")
    
    print("=" * 50)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Convert annotations to YOLO format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Formats:
  cvat     - CVAT XML export
  coco     - COCO JSON format
  roboflow - Roboflow YOLO export (just copies)
  
Examples:
  From CVAT:
    python convert_annotations.py --input annotations.xml --format cvat --output datasets/train/labels
    
  From COCO:
    python convert_annotations.py --input instances.json --format coco --output datasets/train/labels
    
  Validate existing:
    python convert_annotations.py --validate datasets/train/labels --images datasets/train/images
        """
    )
    
    parser.add_argument(
        "--input", "-i",
        help="Input annotation file or directory"
    )
    parser.add_argument(
        "--format", "-f",
        choices=['cvat', 'coco', 'roboflow'],
        help="Input annotation format"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output directory for YOLO labels"
    )
    parser.add_argument(
        "--validate", "-v",
        help="Validate labels in this directory"
    )
    parser.add_argument(
        "--images",
        help="Images directory (for validation)"
    )
    
    args = parser.parse_args()
    
    # Validation mode
    if args.validate:
        report = validate_annotations(args.validate, args.images)
        print_validation_report(report)
        sys.exit(0 if report['valid'] else 1)
    
    # Conversion mode
    if not args.input or not args.format or not args.output:
        parser.error("--input, --format, and --output are required for conversion")
    
    if args.format == 'cvat':
        convert_cvat_xml(args.input, args.output)
    elif args.format == 'coco':
        convert_coco_json(args.input, args.output)
    elif args.format == 'roboflow':
        copy_roboflow_labels(args.input, args.output)
    
    # Validate output
    print("\nValidating converted annotations...")
    report = validate_annotations(args.output)
    print_validation_report(report)


if __name__ == "__main__":
    main()
