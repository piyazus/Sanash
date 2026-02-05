"""
Utility Functions for Bus Person Detection System
=================================================

Helper functions for video processing, annotation, profiling,
and report generation.
"""

import os
import cv2
import time
import logging
import psutil
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False


# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """
    Set up logging with console and file output.
    
    Args:
        log_level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger("BusTracker")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Console handler with formatting
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_format = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    return logger


# =============================================================================
# VIDEO VALIDATION
# =============================================================================

def validate_video(video_path: str) -> Dict[str, Any]:
    """
    Validate video file and extract metadata.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        Dictionary with video metadata or error info
        
    Raises:
        FileNotFoundError: If video file doesn't exist
        ValueError: If video file is invalid
    """
    path = Path(video_path)
    
    # Check file exists
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Check file extension
    valid_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
    if path.suffix.lower() not in valid_extensions:
        raise ValueError(f"Invalid video format: {path.suffix}. Supported: {valid_extensions}")
    
    # Open video and get properties
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")
    
    metadata = {
        'path': str(path.absolute()),
        'filename': path.name,
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'duration_seconds': None,
        'duration_formatted': None,
        'file_size_mb': path.stat().st_size / (1024 * 1024)
    }
    
    # Calculate duration
    if metadata['fps'] > 0:
        duration_sec = metadata['total_frames'] / metadata['fps']
        metadata['duration_seconds'] = duration_sec
        metadata['duration_formatted'] = str(timedelta(seconds=int(duration_sec)))
    
    cap.release()
    return metadata


def get_video_files(input_dir: str) -> List[str]:
    """
    Get all valid video files from a directory.
    
    Args:
        input_dir: Directory to search for videos
        
    Returns:
        List of video file paths
    """
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
    input_path = Path(input_dir)
    
    if not input_path.exists():
        return []
    
    videos = []
    for ext in video_extensions:
        videos.extend(input_path.glob(f'*{ext}'))
        videos.extend(input_path.glob(f'*{ext.upper()}'))
    
    return [str(v) for v in sorted(videos)]


# =============================================================================
# FRAME ANNOTATION
# =============================================================================

def annotate_frame(
    frame: np.ndarray,
    detections: List[Dict],
    current_count: int,
    total_unique: int,
    fps: float = 0.0,
    timestamp: str = "",
    show_boxes: bool = True,
    show_ids: bool = True,
    show_confidence: bool = False,
    box_color: Tuple[int, int, int] = (0, 255, 0),
    box_thickness: int = 2,
    font_scale: float = 0.6
) -> np.ndarray:
    """
    Annotate a frame with bounding boxes, track IDs, and statistics.
    
    Args:
        frame: Input frame (BGR format)
        detections: List of detection dictionaries with keys:
                   'bbox' (x1, y1, x2, y2), 'track_id', 'confidence'
        current_count: Current number of people in frame
        total_unique: Total unique people tracked so far
        fps: Current processing FPS
        timestamp: Video timestamp string
        show_boxes: Whether to draw bounding boxes
        show_ids: Whether to show track IDs
        show_confidence: Whether to show confidence scores
        box_color: BGR color tuple for boxes
        box_thickness: Line thickness for boxes
        font_scale: Font size scale
        
    Returns:
        Annotated frame
    """
    annotated = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Draw bounding boxes and labels
    if show_boxes:
        for det in detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            track_id = det.get('track_id', -1)
            confidence = det.get('confidence', 0.0)
            
            # Draw box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, box_thickness)
            
            # Build label
            label_parts = []
            if show_ids and track_id >= 0:
                label_parts.append(f"ID:{track_id}")
            if show_confidence:
                label_parts.append(f"{confidence:.2f}")
            
            if label_parts:
                label = " ".join(label_parts)
                
                # Draw label background
                (label_w, label_h), baseline = cv2.getTextSize(
                    label, font, font_scale, 1
                )
                cv2.rectangle(
                    annotated,
                    (x1, y1 - label_h - 10),
                    (x1 + label_w + 5, y1),
                    box_color,
                    -1
                )
                
                # Draw label text
                cv2.putText(
                    annotated, label,
                    (x1 + 2, y1 - 5),
                    font, font_scale, (0, 0, 0), 2
                )
    
    # Draw info overlay (top-left)
    info_lines = [
        f"Current: {current_count} people",
        f"Total Unique: {total_unique}",
    ]
    if fps > 0:
        info_lines.append(f"FPS: {fps:.1f}")
    if timestamp:
        info_lines.append(timestamp)
    
    # Draw semi-transparent background for info
    overlay = annotated.copy()
    info_height = len(info_lines) * 30 + 20
    cv2.rectangle(overlay, (10, 10), (250, info_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, annotated, 0.4, 0, annotated)
    
    # Draw info text
    for i, line in enumerate(info_lines):
        cv2.putText(
            annotated, line,
            (20, 35 + i * 28),
            font, 0.7, (255, 255, 255), 2
        )
    
    return annotated


# =============================================================================
# PERFORMANCE PROFILING
# =============================================================================

class PerformanceProfiler:
    """
    Profile system performance and identify bottlenecks.
    
    Usage:
        profiler = PerformanceProfiler()
        
        with profiler.measure("detection"):
            # detection code here
            
        profiler.print_summary()
    """
    
    def __init__(self):
        self.measurements: Dict[str, List[float]] = {}
        self.start_times: Dict[str, float] = {}
        self.gpu_memory: List[float] = []
        self.cpu_usage: List[float] = []
        
    def start(self, name: str):
        """Start timing a section."""
        self.start_times[name] = time.perf_counter()
        
    def stop(self, name: str):
        """Stop timing and record measurement."""
        if name in self.start_times:
            elapsed = time.perf_counter() - self.start_times[name]
            if name not in self.measurements:
                self.measurements[name] = []
            self.measurements[name].append(elapsed * 1000)  # Convert to ms
            
    class measure:
        """Context manager for timing code sections."""
        def __init__(self, profiler: 'PerformanceProfiler', name: str):
            self.profiler = profiler
            self.name = name
            
        def __enter__(self):
            self.profiler.start(self.name)
            return self
            
        def __exit__(self, *args):
            self.profiler.stop(self.name)
    
    def record_system_stats(self):
        """Record current CPU and GPU usage."""
        self.cpu_usage.append(psutil.cpu_percent())
        
        if GPU_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    self.gpu_memory.append(gpus[0].memoryUsed)
            except:
                pass
    
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics for all measurements."""
        summary = {}
        for name, times in self.measurements.items():
            if times:
                summary[name] = {
                    'mean_ms': np.mean(times),
                    'std_ms': np.std(times),
                    'min_ms': np.min(times),
                    'max_ms': np.max(times),
                    'total_ms': np.sum(times),
                    'count': len(times)
                }
        return summary
    
    def print_summary(self):
        """Print formatted performance summary."""
        print("\n" + "=" * 60)
        print("PERFORMANCE PROFILING RESULTS")
        print("=" * 60)
        
        summary = self.get_summary()
        
        # Sort by total time
        sorted_items = sorted(
            summary.items(),
            key=lambda x: x[1]['total_ms'],
            reverse=True
        )
        
        for name, stats in sorted_items:
            print(f"\n{name}:")
            print(f"  Mean: {stats['mean_ms']:.2f} ms")
            print(f"  Total: {stats['total_ms']:.0f} ms ({stats['count']} calls)")
            print(f"  Range: {stats['min_ms']:.2f} - {stats['max_ms']:.2f} ms")
        
        if self.cpu_usage:
            print(f"\nCPU Usage: {np.mean(self.cpu_usage):.1f}% (avg)")
        
        if self.gpu_memory:
            print(f"GPU Memory: {np.mean(self.gpu_memory):.0f} MB (avg)")
        
        print("=" * 60)
    
    def identify_bottlenecks(self) -> List[str]:
        """Identify performance bottlenecks."""
        bottlenecks = []
        summary = self.get_summary()
        
        total_time = sum(s['total_ms'] for s in summary.values())
        
        for name, stats in summary.items():
            proportion = stats['total_ms'] / total_time if total_time > 0 else 0
            
            if proportion > 0.3:  # More than 30% of total time
                bottlenecks.append(
                    f"HIGH: '{name}' takes {proportion*100:.1f}% of processing time"
                )
            elif stats['std_ms'] > stats['mean_ms']:  # High variance
                bottlenecks.append(
                    f"VARIABLE: '{name}' has high variance (std > mean)"
                )
        
        return bottlenecks


# =============================================================================
# MEMORY MANAGEMENT
# =============================================================================

def clear_gpu_memory():
    """Clear GPU memory cache (for PyTorch/CUDA)."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except ImportError:
        pass


def get_memory_usage() -> Dict[str, float]:
    """
    Get current memory usage statistics.
    
    Returns:
        Dictionary with memory usage in MB
    """
    stats = {
        'ram_used_mb': psutil.Process().memory_info().rss / (1024 * 1024),
        'ram_percent': psutil.virtual_memory().percent,
    }
    
    if GPU_AVAILABLE:
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                stats['gpu_used_mb'] = gpu.memoryUsed
                stats['gpu_total_mb'] = gpu.memoryTotal
                stats['gpu_percent'] = gpu.memoryUtil * 100
        except:
            pass
    
    return stats


def check_gpu_available() -> Tuple[bool, str]:
    """
    Check if GPU is available and return device info.
    
    Returns:
        Tuple of (is_available, device_info_string)
    """
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            return True, f"GPU: {device_name} ({memory_gb:.1f} GB)"
        else:
            return False, "CUDA not available, using CPU"
    except ImportError:
        return False, "PyTorch not installed, GPU check unavailable"


# =============================================================================
# TIME UTILITIES
# =============================================================================

def format_timestamp(frame_number: int, fps: float, fmt: str = "%H:%M:%S") -> str:
    """
    Convert frame number to timestamp string.
    
    Args:
        frame_number: Current frame number
        fps: Video frames per second
        fmt: Time format string
        
    Returns:
        Formatted timestamp string
    """
    if fps <= 0:
        return "00:00:00"
    
    seconds = frame_number / fps
    td = timedelta(seconds=seconds)
    
    # Format as HH:MM:SS
    hours, remainder = divmod(int(td.total_seconds()), 3600)
    minutes, secs = divmod(remainder, 60)
    
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted string like "2h 30m 45s"
    """
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    
    parts = []
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    parts.append(f"{secs}s")
    
    return " ".join(parts)


# =============================================================================
# PROGRESS TRACKING
# =============================================================================

class ProgressTracker:
    """
    Track processing progress with ETA calculation.
    
    Usage:
        tracker = ProgressTracker(total_frames=10000)
        
        for frame_num in range(10000):
            # process frame...
            tracker.update(frame_num)
            print(tracker.get_progress_string())
    """
    
    def __init__(self, total_frames: int, fps: float = 30.0):
        self.total_frames = total_frames
        self.video_fps = fps
        self.start_time = time.time()
        self.current_frame = 0
        self.processing_fps_history: List[float] = []
        
    def update(self, frame_number: int):
        """Update progress with current frame number."""
        self.current_frame = frame_number
        
        elapsed = time.time() - self.start_time
        if elapsed > 0:
            current_fps = frame_number / elapsed
            self.processing_fps_history.append(current_fps)
            # Keep only last 100 measurements for smoothing
            if len(self.processing_fps_history) > 100:
                self.processing_fps_history.pop(0)
    
    @property
    def progress_percent(self) -> float:
        """Get current progress as percentage."""
        if self.total_frames <= 0:
            return 0.0
        return (self.current_frame / self.total_frames) * 100
    
    @property
    def processing_fps(self) -> float:
        """Get current processing FPS (smoothed)."""
        if not self.processing_fps_history:
            return 0.0
        return np.mean(self.processing_fps_history[-20:])  # Average of last 20
    
    @property
    def eta_seconds(self) -> float:
        """Get estimated time remaining in seconds."""
        if self.processing_fps <= 0:
            return float('inf')
        remaining_frames = self.total_frames - self.current_frame
        return remaining_frames / self.processing_fps
    
    def get_progress_string(self) -> str:
        """Get formatted progress string."""
        eta_str = format_duration(self.eta_seconds) if self.eta_seconds < float('inf') else "calculating..."
        return (
            f"Progress: {self.progress_percent:.1f}% | "
            f"Frame: {self.current_frame}/{self.total_frames} | "
            f"FPS: {self.processing_fps:.1f} | "
            f"ETA: {eta_str}"
        )
