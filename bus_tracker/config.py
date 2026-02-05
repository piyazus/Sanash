"""
Configuration Settings for Bus Person Detection System
======================================================

This file contains all configurable parameters for the detection system.
Adjust these values to optimize for your specific use case.

Performance Optimization Notes:
- Lower CONFIDENCE_THRESHOLD = more detections but more false positives
- Higher FRAME_SKIP = faster processing but may miss fast-moving people
- Use YOLOv8s for speed, YOLOv8m for accuracy
"""

import os
from pathlib import Path

# =============================================================================
# PATH CONFIGURATION
# =============================================================================

# Base directory (auto-detected from this file's location)
BASE_DIR = Path(__file__).parent.parent

# Input/Output directories
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_VIDEOS_DIR = OUTPUT_DIR / "videos"
OUTPUT_REPORTS_DIR = OUTPUT_DIR / "reports"
OUTPUT_DATA_DIR = OUTPUT_DIR / "data"

# Create directories if they don't exist
for directory in [INPUT_DIR, OUTPUT_VIDEOS_DIR, OUTPUT_REPORTS_DIR, OUTPUT_DATA_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

# Model selection: 'yolov8n.pt' (fastest), 'yolov8s.pt' (fast), 
#                  'yolov8m.pt' (balanced), 'yolov8l.pt' (accurate)
MODEL_PATH = "yolov8m.pt"

# Detection thresholds
# TIP: If getting too many false positives, increase CONFIDENCE_THRESHOLD
# TIP: If missing detections in crowds, decrease CONFIDENCE_THRESHOLD
CONFIDENCE_THRESHOLD = 0.5  # Range: 0.0 - 1.0
IOU_THRESHOLD = 0.45        # Non-max suppression IoU threshold

# Only detect 'person' class (class 0 in COCO dataset)
TARGET_CLASSES = [0]  # 0 = person

# =============================================================================
# PERFORMANCE OPTIMIZATION
# =============================================================================

# Frame skipping: Process every Nth frame
# FRAME_SKIP=1: Process all frames (slowest, most accurate)
# FRAME_SKIP=2: Process every 2nd frame (2x faster)
# FRAME_SKIP=3: Process every 3rd frame (3x faster)
FRAME_SKIP = 2

# Batch processing: Process multiple frames at once
# Higher values = faster but more GPU memory
# Set to 1 if running out of GPU memory
BATCH_SIZE = 4

# Resolution scaling: Resize input for faster processing
# 1.0 = original resolution, 0.5 = half resolution
INPUT_SCALE = 1.0

# GPU settings
USE_GPU = True              # Set False to force CPU
GPU_MEMORY_FRACTION = 0.8   # Max GPU memory to use (0.0-1.0)

# Multi-threading
NUM_WORKERS = 4             # Worker threads for data loading

# Memory management for long videos
CLEAR_MEMORY_INTERVAL = 1000  # Clear GPU cache every N frames

# =============================================================================
# TRACKING CONFIGURATION (ByteTrack)
# =============================================================================

# Tracker settings - adjust for crowded scenes
TRACKER_TYPE = "bytetrack"  # Options: 'bytetrack', 'botsort'

# ByteTrack specific parameters
TRACK_HIGH_THRESH = 0.5     # Detection threshold for first association
TRACK_LOW_THRESH = 0.1      # Detection threshold for second association
TRACK_BUFFER = 30           # Frames to keep lost tracks (higher = more ID retention)
MATCH_THRESH = 0.8          # Matching threshold for tracking

# =============================================================================
# ZONE DETECTION (Entry/Exit Counting)
# =============================================================================

# Define zones as [x1, y1, x2, y2] in relative coordinates (0.0 - 1.0)
# Set to None to disable zone counting
ENTRY_ZONE = None  # Example: [0.0, 0.3, 0.3, 0.7]  # Left side
EXIT_ZONE = None   # Example: [0.7, 0.3, 1.0, 0.7]  # Right side

# =============================================================================
# DASHBOARD & VISUALIZATION
# =============================================================================

SHOW_LIVE_DASHBOARD = True   # Show real-time visualization window
DASHBOARD_FPS = 30           # Dashboard refresh rate

# Heatmap settings
ENABLE_HEATMAP = True
HEATMAP_RESOLUTION = (64, 48)  # Grid resolution for heatmap
HEATMAP_DECAY = 0.995          # Decay factor (lower = faster fade)

# Display settings
SHOW_BOUNDING_BOXES = True
SHOW_TRACK_IDS = True
SHOW_CONFIDENCE = False
SHOW_COUNT = True
SHOW_FPS = True

# Bounding box colors (BGR format)
BOX_COLOR = (0, 255, 0)        # Green
BOX_THICKNESS = 2
FONT_SCALE = 0.6

# =============================================================================
# OUTPUT CONFIGURATION
# =============================================================================

# Video output
SAVE_OUTPUT_VIDEO = True
OUTPUT_VIDEO_CODEC = "mp4v"  # Options: 'mp4v', 'avc1', 'XVID'
OUTPUT_FPS = None            # None = same as input

# Reports
GENERATE_HTML_REPORT = True
GENERATE_CSV_REPORT = True
GENERATE_JSON_EXPORT = True

# Timestamps
ADD_TIMESTAMPS = True
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"

# =============================================================================
# LOGGING & DEBUGGING
# =============================================================================

LOG_LEVEL = "INFO"  # Options: 'DEBUG', 'INFO', 'WARNING', 'ERROR'
SAVE_DEBUG_FRAMES = False     # Save frames with issues for debugging
DEBUG_FRAMES_DIR = OUTPUT_DIR / "debug"

# Profiling (for performance optimization)
ENABLE_PROFILING = False
PROFILE_OUTPUT = OUTPUT_DIR / "profile_results.txt"

# =============================================================================
# CROWDED SCENE OPTIMIZATION
# =============================================================================
# These settings improve detection in buses with heavy occlusion, low lighting,
# and motion blur. Enable CROWDED_SCENE_MODE for best results on crowded footage.

CROWDED_SCENE_MODE = True  # Enable all crowded scene optimizations

# -----------------------------------------------------------------------------
# Density-Based Confidence Adjustment
# -----------------------------------------------------------------------------
# Automatically lower confidence threshold when crowd density increases
# to catch partially occluded people
ENABLE_DENSITY_CONFIDENCE = True
DENSITY_CONFIDENCE_MIN = 0.25      # Min confidence when very crowded
DENSITY_CONFIDENCE_MAX = 0.6       # Max confidence when sparse
DENSITY_THRESHOLD_HIGH = 25        # Above this count = high density
DENSITY_THRESHOLD_LOW = 5          # Below this count = low density
DENSITY_SMOOTHING_WINDOW = 10      # Frames to average for density estimation

# -----------------------------------------------------------------------------
# Temporal Smoothing (Motion Blur Handling)
# -----------------------------------------------------------------------------
# Smooth detections across frames to reduce false positives from motion blur
ENABLE_TEMPORAL_SMOOTHING = True
TEMPORAL_WINDOW = 3                # Frames to look back/forward
TEMPORAL_IOU_THRESHOLD = 0.5       # IoU to consider same detection across frames
MIN_FRAME_APPEARANCES = 2          # Min appearances in window to keep detection
BLUR_CONFIDENCE_BOOST = 0.1        # Confidence boost for stable detections

# -----------------------------------------------------------------------------
# Multi-Scale Detection
# -----------------------------------------------------------------------------
# Process at multiple scales to catch people at different distances
ENABLE_MULTI_SCALE = False         # Disabled by default (slow, but accurate)
DETECTION_SCALES = [1.0, 0.75, 1.25]  # Scale factors for multi-scale
SCALE_WEIGHT_DECAY = 0.8           # Weight reduction for non-primary scales

# -----------------------------------------------------------------------------
# Overlapping Detection Merger (Soft-NMS)
# -----------------------------------------------------------------------------
# Merge overlapping detections instead of hard suppression
ENABLE_SOFT_NMS = True
SOFT_NMS_SIGMA = 0.5               # Gaussian sigma for soft-NMS
SOFT_NMS_THRESHOLD = 0.001         # Remove boxes below this after soft-NMS
MERGE_OVERLAP_THRESHOLD = 0.6      # IoU threshold for merging boxes

# -----------------------------------------------------------------------------
# Kalman Filtering for Smoother Tracking
# -----------------------------------------------------------------------------
# Use Kalman filter to predict locations and smooth track trajectories
ENABLE_KALMAN_SMOOTHING = True
KALMAN_PROCESS_NOISE = 0.03        # Process noise covariance
KALMAN_MEASUREMENT_NOISE = 0.1     # Measurement noise covariance
KALMAN_PREDICTION_FRAMES = 3       # Frames to predict when detection missing
USE_PREDICTED_FOR_DISPLAY = True   # Show Kalman-predicted positions

# -----------------------------------------------------------------------------
# Low Light Enhancement
# -----------------------------------------------------------------------------
# Enhance dark areas for better detection
ENABLE_LOW_LIGHT_ENHANCE = False   # CPU intensive, enable if needed
LOW_LIGHT_THRESHOLD = 50           # Mean brightness to trigger enhancement
CLAHE_CLIP_LIMIT = 2.0             # CLAHE contrast limiting
CLAHE_GRID_SIZE = (8, 8)           # CLAHE tile grid size

# =============================================================================
# TELEMETRY & METRICS
# =============================================================================

ENABLE_TELEMETRY = True            # Collect improvement metrics
TELEMETRY_OUTPUT = OUTPUT_DIR / "telemetry"
COMPARE_BASELINE = True            # Run baseline comparison when profiling
SAVE_DETECTION_LOGS = True         # Log per-frame detection details

