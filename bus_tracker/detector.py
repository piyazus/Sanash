"""
Main Person Detection & Tracking Script
========================================

This script processes video files to detect and track people using YOLOv8
and ByteTrack. It includes extensive optimizations for handling long videos.

Performance Optimizations Included:
1. Batch processing for faster GPU inference
2. Frame skipping to reduce processing load
3. Dynamic confidence threshold adjustment
4. Memory management for long videos
5. Multi-threaded video reading

Usage:
    python -m bus_tracker.detector --input video.mp4
    python -m bus_tracker.detector --input input/ --dashboard
    python -m bus_tracker.detector --input video.mp4 --frame-skip 3 --batch 8

Author: Bus Tracker System
"""

import os
import sys
import cv2
import time
import argparse
import threading
import queue
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import numpy as np

# Suppress unnecessary warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

try:
    from ultralytics import YOLO
except ImportError:
    print("ERROR: ultralytics not installed. Run: pip install ultralytics")
    sys.exit(1)

from . import config
from .utils import (
    setup_logging,
    validate_video,
    get_video_files,
    annotate_frame,
    PerformanceProfiler,
    ProgressTracker,
    clear_gpu_memory,
    get_memory_usage,
    check_gpu_available,
    format_timestamp,
    format_duration
)

# Crowded scene enhancement (optional but recommended for buses)
try:
    from .crowded_enhancer import CrowdedSceneEnhancer, TelemetryCollector
    CROWDED_ENHANCER_AVAILABLE = True
except ImportError:
    CROWDED_ENHANCER_AVAILABLE = False

# Zone management for entry/exit counting
try:
    from .zones import ZoneManager
    ZONES_AVAILABLE = True
except ImportError:
    ZONES_AVAILABLE = False


# =============================================================================
# VIDEO READER WITH MULTI-THREADING
# =============================================================================

class ThreadedVideoReader:
    """
    Multi-threaded video reader for faster frame loading.
    
    Reads frames in a background thread while the main thread processes,
    reducing I/O wait time.
    """
    
    def __init__(self, video_path: str, queue_size: int = 128, frame_skip: int = 1):
        """
        Initialize threaded video reader.
        
        Args:
            video_path: Path to video file
            queue_size: Maximum frames to buffer
            frame_skip: Read every Nth frame
        """
        self.video_path = video_path
        self.queue_size = queue_size
        self.frame_skip = frame_skip
        
        # Open video
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Video properties
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Threading setup
        self.frame_queue = queue.Queue(maxsize=queue_size)
        self.stopped = False
        self.thread = None
        
    def start(self):
        """Start the background reading thread."""
        self.thread = threading.Thread(target=self._read_frames, daemon=True)
        self.thread.start()
        return self
    
    def _read_frames(self):
        """Background thread function to read frames."""
        frame_num = 0
        
        while not self.stopped:
            if self.frame_queue.full():
                time.sleep(0.001)  # Small wait if queue is full
                continue
            
            ret, frame = self.cap.read()
            
            if not ret:
                self.stopped = True
                break
            
            # Apply frame skipping
            if frame_num % self.frame_skip == 0:
                self.frame_queue.put((frame_num, frame))
            
            frame_num += 1
        
        # Signal end of video
        self.frame_queue.put((None, None))
    
    def read(self) -> Tuple[Optional[int], Optional[np.ndarray]]:
        """
        Read the next frame from the queue.
        
        Returns:
            Tuple of (frame_number, frame) or (None, None) if done
        """
        return self.frame_queue.get()
    
    def stop(self):
        """Stop the reader thread."""
        self.stopped = True
        if self.thread:
            self.thread.join(timeout=1)
        self.cap.release()
    
    def __enter__(self):
        return self.start()
    
    def __exit__(self, *args):
        self.stop()


# =============================================================================
# BATCH PROCESSOR
# =============================================================================

class BatchProcessor:
    """
    Collect frames into batches for more efficient GPU inference.
    
    Processing frames in batches is significantly faster than one-by-one
    due to better GPU utilization.
    """
    
    def __init__(self, batch_size: int = 4):
        self.batch_size = batch_size
        self.frames: List[np.ndarray] = []
        self.frame_numbers: List[int] = []
    
    def add(self, frame_num: int, frame: np.ndarray) -> bool:
        """
        Add a frame to the batch.
        
        Returns:
            True if batch is full and ready for processing
        """
        self.frames.append(frame)
        self.frame_numbers.append(frame_num)
        return len(self.frames) >= self.batch_size
    
    def get_batch(self) -> Tuple[List[int], List[np.ndarray]]:
        """Get the current batch and clear it."""
        frame_nums = self.frame_numbers.copy()
        frames = self.frames.copy()
        self.frames.clear()
        self.frame_numbers.clear()
        return frame_nums, frames
    
    def has_remaining(self) -> bool:
        """Check if there are remaining frames in the batch."""
        return len(self.frames) > 0


# =============================================================================
# DYNAMIC CONFIDENCE ADJUSTER
# =============================================================================

class DynamicConfidenceAdjuster:
    """
    Automatically adjust confidence threshold based on detection patterns.
    
    - Too many detections (likely false positives) -> increase threshold
    - Too few detections -> decrease threshold (within limits)
    """
    
    def __init__(
        self,
        initial_conf: float = 0.5,
        min_conf: float = 0.3,
        max_conf: float = 0.8,
        target_detections: int = 20,  # Target detections per frame
        adjustment_rate: float = 0.01
    ):
        self.confidence = initial_conf
        self.min_conf = min_conf
        self.max_conf = max_conf
        self.target = target_detections
        self.rate = adjustment_rate
        self.history: List[int] = []
    
    def update(self, num_detections: int) -> float:
        """
        Update confidence based on recent detection count.
        
        Args:
            num_detections: Number of detections in current frame
            
        Returns:
            Updated confidence threshold
        """
        self.history.append(num_detections)
        
        # Keep last 30 frames for averaging
        if len(self.history) > 30:
            self.history.pop(0)
        
        avg_detections = np.mean(self.history)
        
        # Adjust confidence
        if avg_detections > self.target * 1.5:  # Too many detections
            self.confidence = min(self.max_conf, self.confidence + self.rate)
        elif avg_detections < self.target * 0.5:  # Too few detections
            self.confidence = max(self.min_conf, self.confidence - self.rate)
        
        return self.confidence
    
    def get_confidence(self) -> float:
        """Get current confidence threshold."""
        return self.confidence


# =============================================================================
# MAIN DETECTOR CLASS
# =============================================================================

class PersonDetector:
    """
    Main class for person detection and tracking in videos.
    
    Features:
    - YOLOv8 detection with ByteTrack tracking
    - Batch processing for performance
    - Frame skipping option
    - Dynamic confidence adjustment
    - Memory management for long videos
    - Progress tracking and profiling
    """
    
    def __init__(
        self,
        model_path: str = None,
        confidence: float = None,
        frame_skip: int = None,
        batch_size: int = None,
        use_gpu: bool = None,
        enable_profiling: bool = False,
        dynamic_confidence: bool = False
    ):
        """
        Initialize the detector.
        
        Args:
            model_path: Path to YOLO model weights
            confidence: Detection confidence threshold
            frame_skip: Process every Nth frame
            batch_size: Number of frames per batch
            use_gpu: Whether to use GPU acceleration
            enable_profiling: Enable performance profiling
            dynamic_confidence: Enable dynamic confidence adjustment
        """
        # Use config defaults if not specified
        self.model_path = model_path or config.MODEL_PATH
        self.confidence = confidence or config.CONFIDENCE_THRESHOLD
        self.frame_skip = frame_skip or config.FRAME_SKIP
        self.batch_size = batch_size or config.BATCH_SIZE
        self.use_gpu = use_gpu if use_gpu is not None else config.USE_GPU
        
        # Setup logging
        self.logger = setup_logging(config.LOG_LEVEL)
        
        # Check GPU availability
        gpu_available, gpu_info = check_gpu_available()
        self.logger.info(gpu_info)
        
        if self.use_gpu and not gpu_available:
            self.logger.warning("GPU requested but not available, using CPU")
            self.use_gpu = False
        
        # Load model
        self.logger.info(f"Loading model: {self.model_path}")
        self.model = YOLO(self.model_path)
        
        # Set device
        self.device = 0 if self.use_gpu else 'cpu'
        
        # Profiling
        self.profiler = PerformanceProfiler() if enable_profiling else None
        self.enable_profiling = enable_profiling
        
        # Dynamic confidence
        self.dynamic_confidence = dynamic_confidence
        self.conf_adjuster = DynamicConfidenceAdjuster(
            initial_conf=self.confidence
        ) if dynamic_confidence else None
        
        # Tracking data
        self.unique_track_ids: set = set()
        self.track_history: Dict[int, List[Dict]] = defaultdict(list)
        
        # Crowded scene enhancer for occlusion/blur handling
        self.crowded_enhancer = None
        self.telemetry = None
        if config.CROWDED_SCENE_MODE and CROWDED_ENHANCER_AVAILABLE:
            self.logger.info("Crowded scene enhancement: ENABLED")
            self.crowded_enhancer = CrowdedSceneEnhancer()
            if config.ENABLE_TELEMETRY:
                self.telemetry = TelemetryCollector()
        elif config.CROWDED_SCENE_MODE:
            self.logger.warning("Crowded enhancer not available")
        
        # Zone manager for entry/exit counting
        self.zone_manager = None
    
    def load_zones(self, zones_path: str):
        """
        Load zones from a JSON file for entry/exit counting.
        
        Args:
            zones_path: Path to zones JSON file
        """
        if not ZONES_AVAILABLE:
            self.logger.warning("Zone manager not available")
            return
        
        from pathlib import Path
        if not Path(zones_path).exists():
            self.logger.warning(f"Zones file not found: {zones_path}")
            return
        
        self.zone_manager = ZoneManager()
        self.zone_manager.load_zones(zones_path)
        self.logger.info(f"Loaded {len(self.zone_manager.zones)} zone(s) from {zones_path}")
        
    def process_video(
        self,
        video_path: str,
        output_path: str = None,
        show_dashboard: bool = False
    ) -> Dict[str, Any]:
        """
        Process a video file for person detection and tracking.
        
        Args:
            video_path: Path to input video
            output_path: Path for output video (optional)
            show_dashboard: Whether to show live visualization
            
        Returns:
            Dictionary with processing results and statistics
        """
        # Validate video
        self.logger.info(f"Processing video: {video_path}")
        video_info = validate_video(video_path)
        self.logger.info(
            f"Video: {video_info['width']}x{video_info['height']} @ "
            f"{video_info['fps']:.1f} FPS, Duration: {video_info['duration_formatted']}"
        )
        
        # Reset tracking state
        self.unique_track_ids.clear()
        self.track_history.clear()
        
        # Setup output video writer
        writer = None
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            fourcc = cv2.VideoWriter_fourcc(*config.OUTPUT_VIDEO_CODEC)
            output_fps = config.OUTPUT_FPS or video_info['fps']
            writer = cv2.VideoWriter(
                str(output_path),
                fourcc,
                output_fps / self.frame_skip,  # Adjust for frame skipping
                (video_info['width'], video_info['height'])
            )
            self.logger.info(f"Output video: {output_path}")
        
        # Initialize progress tracking
        effective_frames = video_info['total_frames'] // self.frame_skip
        progress = ProgressTracker(effective_frames, video_info['fps'])
        
        # Initialize batch processor
        batch_processor = BatchProcessor(self.batch_size)
        
        # Statistics
        stats = {
            'start_time': datetime.now(),
            'total_frames_processed': 0,
            'total_detections': 0,
            'frame_times': [],
            'detection_counts': []
        }
        
        # Process video with threaded reader
        with ThreadedVideoReader(
            video_path, 
            frame_skip=self.frame_skip
        ) as reader:
            
            processed_count = 0
            last_results = []  # Store results for batch processing
            
            while True:
                frame_num, frame = reader.read()
                
                # End of video
                if frame is None:
                    # Process any remaining frames in batch
                    if batch_processor.has_remaining():
                        frame_nums, frames = batch_processor.get_batch()
                        batch_results = self._process_batch(frames)
                        self._handle_batch_results(
                            frame_nums, frames, batch_results, 
                            writer, show_dashboard, video_info['fps']
                        )
                    break
                
                # Add frame to batch
                if batch_processor.add(frame_num, frame):
                    # Batch is full, process it
                    frame_nums, frames = batch_processor.get_batch()
                    
                    # Profile batch processing
                    if self.enable_profiling:
                        self.profiler.start("batch_inference")
                    
                    batch_results = self._process_batch(frames)
                    
                    if self.enable_profiling:
                        self.profiler.stop("batch_inference")
                        self.profiler.start("post_processing")
                    
                    # Handle results
                    self._handle_batch_results(
                        frame_nums, frames, batch_results,
                        writer, show_dashboard, video_info['fps']
                    )
                    
                    if self.enable_profiling:
                        self.profiler.stop("post_processing")
                        self.profiler.record_system_stats()
                    
                    processed_count += len(frames)
                    stats['total_frames_processed'] = processed_count
                
                # Update progress
                progress.update(processed_count)
                
                # Log progress every 100 frames
                if processed_count % 100 == 0 and processed_count > 0:
                    self.logger.info(progress.get_progress_string())
                
                # Memory management for long videos
                if processed_count % config.CLEAR_MEMORY_INTERVAL == 0:
                    clear_gpu_memory()
                
                # Handle dashboard key events
                if show_dashboard:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        self.logger.info("Processing stopped by user")
                        break
                    elif key == ord('p'):  # Pause
                        cv2.waitKey(0)
        
        # Cleanup
        if writer:
            writer.release()
        if show_dashboard:
            cv2.destroyAllWindows()
        
        # Calculate final statistics
        stats['end_time'] = datetime.now()
        stats['processing_duration'] = (stats['end_time'] - stats['start_time']).total_seconds()
        stats['unique_people_count'] = len(self.unique_track_ids)
        stats['average_fps'] = stats['total_frames_processed'] / stats['processing_duration'] if stats['processing_duration'] > 0 else 0
        
        # Print profiling results
        if self.enable_profiling:
            self.profiler.print_summary()
            bottlenecks = self.profiler.identify_bottlenecks()
            if bottlenecks:
                self.logger.warning("Detected bottlenecks:")
                for b in bottlenecks:
                    self.logger.warning(f"  - {b}")
        
        # Generate enhancement telemetry report
        if self.telemetry:
            enhancement_report = self.telemetry.generate_report()
            report_path = self.telemetry.save_report()
            self.logger.info(f"Enhancement telemetry saved: {report_path}")
            stats['enhancement_metrics'] = enhancement_report
            
            # Print enhancement summary
            self.logger.info("-" * 50)
            self.logger.info("CROWDED SCENE ENHANCEMENT SUMMARY")
            self.logger.info(f"  Baseline detections: {enhancement_report.get('total_baseline_detections', 0)}")
            self.logger.info(f"  Enhanced detections: {enhancement_report.get('total_enhanced_detections', 0)}")
            self.logger.info(f"  Detections recovered: {enhancement_report.get('detections_recovered', 0)}")
            self.logger.info(f"  False positives filtered: {enhancement_report.get('false_positives_filtered', 0)}")
            self.logger.info(f"  Net improvement: {enhancement_report.get('improvement_percent', 0):.1f}%")
            self.logger.info(f"  Avg enhancement time: {enhancement_report.get('avg_processing_time_ms', 0):.2f}ms")
            self.logger.info("-" * 50)
        
        # Final summary
        self.logger.info("=" * 50)
        self.logger.info("PROCESSING COMPLETE")
        self.logger.info(f"Frames processed: {stats['total_frames_processed']}")
        self.logger.info(f"Unique people detected: {stats['unique_people_count']}")
        self.logger.info(f"Processing time: {format_duration(stats['processing_duration'])}")
        self.logger.info(f"Average FPS: {stats['average_fps']:.1f}")
        self.logger.info("=" * 50)
        
        return stats
    
    def _process_batch(self, frames: List[np.ndarray]) -> List[Any]:
        """
        Process a batch of frames through the detector.
        
        Args:
            frames: List of frames to process
            
        Returns:
            List of detection results
        """
        # Get current confidence - prioritize crowded enhancer's density-based
        # confidence over basic dynamic confidence for better crowd handling
        if self.crowded_enhancer is not None:
            conf = self.crowded_enhancer.get_recommended_confidence()
        elif self.dynamic_confidence:
            conf = self.conf_adjuster.get_confidence()
        else:
            conf = self.confidence
        
        # Run inference with tracking
        results = self.model.track(
            frames,
            persist=True,  # Maintain track IDs across frames
            conf=conf,
            iou=config.IOU_THRESHOLD,
            classes=config.TARGET_CLASSES,
            device=self.device,
            tracker=f"{config.TRACKER_TYPE}.yaml",
            verbose=False
        )
        
        return results
    
    def _handle_batch_results(
        self,
        frame_nums: List[int],
        frames: List[np.ndarray],
        results: List[Any],
        writer: Optional[cv2.VideoWriter],
        show_dashboard: bool,
        video_fps: float
    ):
        """
        Handle batch detection results - annotate, track, and output.
        
        Args:
            frame_nums: Frame numbers in batch
            frames: Original frames
            results: Detection results from model
            writer: Video writer (optional)
            show_dashboard: Whether to show live view
            video_fps: Original video FPS
        """
        for i, (frame_num, frame, result) in enumerate(zip(frame_nums, frames, results)):
            detections = []
            baseline_count = 0  # For telemetry
            
            # Extract detections from result
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                
                # Get track IDs if available
                track_ids = None
                if result.boxes.id is not None:
                    track_ids = result.boxes.id.cpu().numpy().astype(int)
                
                for j, box in enumerate(boxes):
                    det = {
                        'bbox': box.tolist(),
                        'confidence': float(confidences[j]),
                        'track_id': int(track_ids[j]) if track_ids is not None else -1,
                        'frame_num': frame_num,
                        'timestamp': format_timestamp(frame_num, video_fps)
                    }
                    detections.append(det)
            
            baseline_count = len(detections)
            
            # =================================================================
            # CROWDED SCENE ENHANCEMENT
            # Apply density-based confidence, temporal smoothing, soft-NMS,
            # and Kalman filtering to improve detection in crowded conditions
            # =================================================================
            if self.crowded_enhancer is not None:
                detections, metrics = self.crowded_enhancer.process(
                    detections, frame, frame_num
                )
                
                # Record telemetry
                if self.telemetry:
                    self.telemetry.record_frame(
                        frame_num, baseline_count, len(detections), metrics
                    )
            
            # Track unique IDs (after enhancement to include recovered tracks)
            for det in detections:
                if det.get('track_id', -1) >= 0:
                    self.unique_track_ids.add(det['track_id'])
                    self.track_history[det['track_id']].append(det)
            
            # =================================================================
            # ZONE ENTRY/EXIT COUNTING
            # Update zone manager with current detections to detect crossings
            # =================================================================
            if self.zone_manager is not None:
                crossings = self.zone_manager.update(
                    detections, frame_num, video_fps
                )
                # Log significant crossings
                for crossing in crossings:
                    direction_symbol = '→' if crossing.direction == 'entry' else '←'
                    self.logger.debug(
                        f"{direction_symbol} Track {crossing.track_id} {crossing.direction} "
                        f"at {crossing.zone_name}"
                    )
            
            # Update dynamic confidence
            if self.dynamic_confidence:
                self.conf_adjuster.update(len(detections))
            
            # Draw zones on frame before other annotations
            if self.zone_manager is not None:
                frame = self.zone_manager.draw_zones(
                    frame, show_counts=True, show_arrows=True
                )
            
            # Annotate frame
            annotated = annotate_frame(
                frame,
                detections,
                current_count=len(detections),
                total_unique=len(self.unique_track_ids),
                fps=0,  # Will be calculated elsewhere
                timestamp=format_timestamp(frame_num, video_fps),
                show_boxes=config.SHOW_BOUNDING_BOXES,
                show_ids=config.SHOW_TRACK_IDS,
                show_confidence=config.SHOW_CONFIDENCE,
                box_color=config.BOX_COLOR,
                box_thickness=config.BOX_THICKNESS,
                font_scale=config.FONT_SCALE
            )
            
            # Write output video
            if writer:
                writer.write(annotated)
            
            # Show dashboard
            if show_dashboard:
                # Resize for display if too large
                display_frame = annotated
                max_height = 720
                if annotated.shape[0] > max_height:
                    scale = max_height / annotated.shape[0]
                    display_frame = cv2.resize(
                        annotated, 
                        None, 
                        fx=scale, 
                        fy=scale
                    )
                
                cv2.imshow("Bus Person Tracker", display_frame)
    
    def get_tracking_data(self) -> Dict[int, List[Dict]]:
        """Get all tracking data for analysis."""
        return dict(self.track_history)
    
    def get_unique_count(self) -> int:
        """Get count of unique people tracked."""
        return len(self.unique_track_ids)


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Person Detection & Tracking System for Bus Videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Process single video:
    python -m bus_tracker.detector --input video.mp4
    
  Process folder of videos:
    python -m bus_tracker.detector --input input/
    
  With live dashboard:
    python -m bus_tracker.detector --input video.mp4 --dashboard
    
  Optimize for speed:
    python -m bus_tracker.detector --input video.mp4 --frame-skip 3 --batch 8 --model yolov8s.pt
    
  Enable profiling to find bottlenecks:
    python -m bus_tracker.detector --input video.mp4 --profile
        """
    )
    
    # Input/Output
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input video file or directory"
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output directory (default: output/videos/)"
    )
    
    # Model settings
    parser.add_argument(
        "--model", "-m",
        default=None,
        help=f"Model path (default: {config.MODEL_PATH})"
    )
    parser.add_argument(
        "--confidence", "-c",
        type=float,
        default=None,
        help=f"Confidence threshold (default: {config.CONFIDENCE_THRESHOLD})"
    )
    
    # Performance options
    parser.add_argument(
        "--frame-skip", "-fs",
        type=int,
        default=None,
        help=f"Process every Nth frame (default: {config.FRAME_SKIP})"
    )
    parser.add_argument(
        "--batch", "-b",
        type=int,
        default=None,
        help=f"Batch size for inference (default: {config.BATCH_SIZE})"
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU mode (no GPU)"
    )
    
    # Features
    parser.add_argument(
        "--dashboard", "-d",
        action="store_true",
        help="Show live visualization dashboard"
    )
    parser.add_argument(
        "--profile", "-p",
        action="store_true",
        help="Enable performance profiling"
    )
    parser.add_argument(
        "--dynamic-conf",
        action="store_true",
        help="Enable dynamic confidence adjustment"
    )
    parser.add_argument(
        "--zones", "-z",
        default=None,
        help="Path to zones JSON file for entry/exit counting"
    )
    parser.add_argument(
        "--draw-zones",
        action="store_true",
        help="Draw zones interactively on first frame"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Initialize detector
    detector = PersonDetector(
        model_path=args.model,
        confidence=args.confidence,
        frame_skip=args.frame_skip,
        batch_size=args.batch,
        use_gpu=not args.cpu,
        enable_profiling=args.profile,
        dynamic_confidence=args.dynamic_conf
    )
    
    # Load zones if specified
    if args.zones:
        detector.load_zones(args.zones)
    
    # Interactive zone drawing
    if args.draw_zones:
        print("\nStarting interactive zone drawer...")
        print("Draw zones on the first frame, then press S to save and continue.")
        
        # Get first frame from first video
        input_path = Path(args.input)
        if input_path.is_file():
            video_for_zones = str(input_path)
        else:
            videos_temp = get_video_files(str(input_path))
            video_for_zones = videos_temp[0] if videos_temp else None
        
        if video_for_zones:
            import cv2
            cap = cv2.VideoCapture(video_for_zones)
            ret, first_frame = cap.read()
            cap.release()
            
            if ret and ZONES_AVAILABLE:
                from .zones import ZoneDrawer, ZoneManager
                drawer = ZoneDrawer(first_frame)
                zones = drawer.run()
                
                if zones:
                    zones_output = str(config.OUTPUT_DIR / "zones.json")
                    manager = ZoneManager((first_frame.shape[1], first_frame.shape[0]))
                    for z in zones:
                        manager.add_zone(z)
                    manager.save_zones(zones_output)
                    print(f"Zones saved to: {zones_output}")
                    detector.load_zones(zones_output)
    
    # Get input video(s)
    input_path = Path(args.input)
    
    if input_path.is_file():
        videos = [str(input_path)]
    elif input_path.is_dir():
        videos = get_video_files(str(input_path))
        if not videos:
            print(f"No video files found in: {input_path}")
            sys.exit(1)
        print(f"Found {len(videos)} video(s) to process")
    else:
        print(f"Input not found: {input_path}")
        sys.exit(1)
    
    # Process each video
    all_stats = []
    
    for video_path in videos:
        # Determine output path
        output_path = None
        if args.output or config.SAVE_OUTPUT_VIDEO:
            output_dir = Path(args.output) if args.output else config.OUTPUT_VIDEOS_DIR
            output_name = Path(video_path).stem + "_tracked.mp4"
            output_path = str(output_dir / output_name)
        
        # Process video
        stats = detector.process_video(
            video_path,
            output_path=output_path,
            show_dashboard=args.dashboard
        )
        
        all_stats.append({
            'video': video_path,
            **stats
        })
        
        # Print zone stats if available
        if detector.zone_manager:
            zone_report = detector.zone_manager.get_full_report()
            totals = zone_report['totals']
            print("\n" + "-" * 50)
            print("ZONE ENTRY/EXIT SUMMARY")
            print(f"  Total entries: {totals['total_entries']}")
            print(f"  Total exits: {totals['total_exits']}")
            print(f"  Net occupancy change: {totals['net_occupancy_change']:+d}")
            
            dwell = zone_report['dwell_times']
            if dwell['count'] > 0:
                print(f"  Avg dwell time: {dwell['average']:.1f}s")
            print("-" * 50)
    
    # Summary for multiple videos
    if len(videos) > 1:
        print("\n" + "=" * 60)
        print("BATCH PROCESSING SUMMARY")
        print("=" * 60)
        total_people = sum(s['unique_people_count'] for s in all_stats)
        total_time = sum(s['processing_duration'] for s in all_stats)
        print(f"Videos processed: {len(videos)}")
        print(f"Total unique people: {total_people}")
        print(f"Total processing time: {format_duration(total_time)}")


if __name__ == "__main__":
    main()
