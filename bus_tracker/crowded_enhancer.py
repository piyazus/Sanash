"""
Crowded Scene Enhancement Module
=================================

Advanced algorithms for improving person detection in extremely crowded
bus conditions with:
1. Density-based confidence adjustment
2. Temporal smoothing to handle motion blur
3. Multi-scale detection for varying distances
4. Soft-NMS to merge overlapping detections
5. Kalman filtering for smoother tracking

These optimizations specifically address:
- Partial occlusion in crowds
- Low lighting in bus areas
- Motion blur from bus movement

Usage:
    from bus_tracker.crowded_enhancer import CrowdedSceneEnhancer
    
    enhancer = CrowdedSceneEnhancer(frame_size=(1920, 1080))
    enhanced_detections = enhancer.process(detections, frame)
"""

import cv2
import numpy as np
from collections import deque
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import time

from . import config


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class KalmanTrack:
    """Kalman filter state for a single track."""
    track_id: int
    kf: cv2.KalmanFilter = None
    last_detection: Dict = None
    frames_missing: int = 0
    prediction_count: int = 0
    
    def __post_init__(self):
        """Initialize Kalman filter with 4 state variables (x, y, vx, vy)."""
        self.kf = cv2.KalmanFilter(4, 2)  # 4 state vars, 2 measurements
        
        # State transition matrix (constant velocity model)
        self.kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        # Measurement matrix
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)
        
        # Process noise
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * config.KALMAN_PROCESS_NOISE
        
        # Measurement noise
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * config.KALMAN_MEASUREMENT_NOISE
        
        # Initial state covariance
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)


@dataclass
class EnhancementMetrics:
    """Telemetry metrics for measuring improvement."""
    baseline_detections: int = 0
    enhanced_detections: int = 0
    detections_added: int = 0
    detections_removed: int = 0
    false_positives_filtered: int = 0
    occluded_recovered: int = 0
    blur_smoothed: int = 0
    kalman_predictions: int = 0
    processing_time_ms: float = 0.0
    density_level: str = "normal"
    current_confidence: float = 0.5


# =============================================================================
# CROWDED SCENE ENHANCER
# =============================================================================

class CrowdedSceneEnhancer:
    """
    Enhances detection results for crowded bus conditions.
    
    This class post-processes YOLO detections to improve accuracy
    in challenging conditions like occlusion, low light, and motion blur.
    """
    
    def __init__(
        self,
        frame_size: Tuple[int, int] = (1920, 1080),
        enable_all: bool = None
    ):
        """
        Initialize enhancer.
        
        Args:
            frame_size: (width, height) of video frames
            enable_all: Override to enable/disable all enhancements
        """
        self.frame_width, self.frame_height = frame_size
        
        # Feature flags (from config or override)
        if enable_all is not None:
            self.enable_density_conf = enable_all
            self.enable_temporal = enable_all
            self.enable_multi_scale = enable_all
            self.enable_soft_nms = enable_all
            self.enable_kalman = enable_all
        else:
            self.enable_density_conf = config.ENABLE_DENSITY_CONFIDENCE
            self.enable_temporal = config.ENABLE_TEMPORAL_SMOOTHING
            self.enable_multi_scale = config.ENABLE_MULTI_SCALE
            self.enable_soft_nms = config.ENABLE_SOFT_NMS
            self.enable_kalman = config.ENABLE_KALMAN_SMOOTHING
        
        # Density-based confidence state
        # -------------------------------------------------------------------------
        # In crowded scenes, people are partially occluded, resulting in lower
        # confidence scores. We dynamically lower the threshold when density is high.
        # -------------------------------------------------------------------------
        self.density_history: deque = deque(maxlen=config.DENSITY_SMOOTHING_WINDOW)
        self.current_dynamic_conf = config.CONFIDENCE_THRESHOLD
        
        # Temporal smoothing state
        # -------------------------------------------------------------------------
        # Motion blur causes flickering detections. We track detections across
        # multiple frames and only keep those that appear consistently.
        # -------------------------------------------------------------------------
        self.temporal_buffer: deque = deque(maxlen=config.TEMPORAL_WINDOW * 2 + 1)
        
        # Kalman filter state per track
        # -------------------------------------------------------------------------
        # Kalman filters predict position when detection is missing and smooth
        # noisy bounding box coordinates for more stable tracking.
        # -------------------------------------------------------------------------
        self.kalman_tracks: Dict[int, KalmanTrack] = {}
        
        # Telemetry
        self.metrics = EnhancementMetrics()
        self.frame_number = 0
    
    # =========================================================================
    # MAIN PROCESSING PIPELINE
    # =========================================================================
    
    def process(
        self,
        detections: List[Dict],
        frame: np.ndarray = None,
        frame_num: int = None
    ) -> Tuple[List[Dict], EnhancementMetrics]:
        """
        Process detections through all enhancement stages.
        
        Args:
            detections: Raw detections from YOLO
                       Each dict has: bbox, confidence, track_id, frame_num
            frame: Original frame (for low-light enhancement)
            frame_num: Current frame number
            
        Returns:
            Tuple of (enhanced_detections, metrics)
        """
        start_time = time.perf_counter()
        self.frame_number = frame_num or self.frame_number + 1
        
        # Reset metrics
        self.metrics = EnhancementMetrics()
        self.metrics.baseline_detections = len(detections)
        
        # Stage 1: Low-light enhancement (preprocessing)
        if frame is not None and config.ENABLE_LOW_LIGHT_ENHANCE:
            frame = self._enhance_low_light(frame)
        
        # Stage 2: Density-based confidence adjustment
        # This doesn't filter detections, but adjusts the confidence threshold
        # for the next frame's inference
        if self.enable_density_conf:
            self._update_density_confidence(len(detections))
        
        # Stage 3: Soft-NMS to merge overlapping detections
        # Prevents removing occluded people that overlap significantly
        if self.enable_soft_nms and len(detections) > 0:
            detections = self._apply_soft_nms(detections)
        
        # Stage 4: Temporal smoothing
        # Filter out flickering detections from motion blur
        if self.enable_temporal:
            detections = self._apply_temporal_smoothing(detections)
        
        # Stage 5: Kalman filtering for track smoothing
        # Predict missing detections and smooth trajectories
        if self.enable_kalman:
            detections = self._apply_kalman_smoothing(detections)
        
        # Record metrics
        self.metrics.enhanced_detections = len(detections)
        self.metrics.processing_time_ms = (time.perf_counter() - start_time) * 1000
        self.metrics.current_confidence = self.current_dynamic_conf
        
        return detections, self.metrics
    
    # =========================================================================
    # DENSITY-BASED CONFIDENCE ADJUSTMENT
    # =========================================================================
    
    def _update_density_confidence(self, detection_count: int):
        """
        Adjust confidence threshold based on crowd density.
        
        In crowded scenes, lower confidence catches more partially
        occluded people. In sparse scenes, higher confidence reduces
        false positives.
        
        Args:
            detection_count: Number of detections in current frame
        """
        self.density_history.append(detection_count)
        
        if len(self.density_history) < 3:
            return
        
        avg_density = np.mean(self.density_history)
        
        # Determine density level and adjust confidence
        if avg_density >= config.DENSITY_THRESHOLD_HIGH:
            # High density = lower confidence to catch occluded people
            self.current_dynamic_conf = config.DENSITY_CONFIDENCE_MIN
            self.metrics.density_level = "high"
        elif avg_density <= config.DENSITY_THRESHOLD_LOW:
            # Low density = higher confidence to reduce false positives
            self.current_dynamic_conf = config.DENSITY_CONFIDENCE_MAX
            self.metrics.density_level = "low"
        else:
            # Interpolate between min and max based on density
            density_range = config.DENSITY_THRESHOLD_HIGH - config.DENSITY_THRESHOLD_LOW
            density_ratio = (avg_density - config.DENSITY_THRESHOLD_LOW) / density_range
            conf_range = config.DENSITY_CONFIDENCE_MAX - config.DENSITY_CONFIDENCE_MIN
            self.current_dynamic_conf = config.DENSITY_CONFIDENCE_MAX - (density_ratio * conf_range)
            self.metrics.density_level = "normal"
    
    def get_recommended_confidence(self) -> float:
        """Get the dynamically adjusted confidence threshold."""
        return self.current_dynamic_conf
    
    # =========================================================================
    # SOFT-NMS (Overlapping Detection Merger)
    # =========================================================================
    
    def _apply_soft_nms(self, detections: List[Dict]) -> List[Dict]:
        """
        Apply Soft-NMS to merge overlapping detections.
        
        Unlike hard NMS which removes overlapping boxes entirely,
        soft-NMS reduces their confidence based on overlap. This
        preserves detections of people standing close together.
        
        Args:
            detections: List of detections
            
        Returns:
            Detections with adjusted confidences
        """
        if len(detections) == 0:
            return detections
        
        # Extract boxes and scores
        boxes = np.array([d['bbox'] for d in detections])
        scores = np.array([d['confidence'] for d in detections])
        
        # Apply soft-NMS
        new_scores = self._soft_nms_cpu(
            boxes, 
            scores, 
            sigma=config.SOFT_NMS_SIGMA,
            threshold=config.SOFT_NMS_THRESHOLD
        )
        
        # Update detections and filter low scores
        enhanced_detections = []
        for i, det in enumerate(detections):
            if new_scores[i] >= config.SOFT_NMS_THRESHOLD:
                det = det.copy()
                det['confidence'] = float(new_scores[i])
                enhanced_detections.append(det)
            else:
                self.metrics.false_positives_filtered += 1
        
        return enhanced_detections
    
    def _soft_nms_cpu(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        sigma: float = 0.5,
        threshold: float = 0.001
    ) -> np.ndarray:
        """
        CPU implementation of Soft-NMS using Gaussian penalty.
        
        For each box, reduce scores of overlapping boxes based on IoU.
        Higher IoU = more score reduction.
        
        Args:
            boxes: Nx4 array of [x1, y1, x2, y2]
            scores: N array of confidence scores
            sigma: Gaussian sigma (higher = less suppression)
            threshold: Remove boxes below this score
            
        Returns:
            Updated scores array
        """
        N = len(boxes)
        if N == 0:
            return scores
        
        # Sort by score (descending)
        indices = np.argsort(scores)[::-1]
        new_scores = scores.copy()
        
        for i in range(N):
            max_idx = indices[i]
            max_box = boxes[max_idx]
            
            for j in range(i + 1, N):
                idx = indices[j]
                iou = self._calculate_iou(max_box, boxes[idx])
                
                # Gaussian penalty based on IoU
                penalty = np.exp(-(iou ** 2) / sigma)
                new_scores[idx] *= penalty
        
        return new_scores
    
    def _calculate_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """Calculate Intersection over Union between two boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    # =========================================================================
    # TEMPORAL SMOOTHING (Motion Blur Handling)
    # =========================================================================
    
    def _apply_temporal_smoothing(self, detections: List[Dict]) -> List[Dict]:
        """
        Apply temporal smoothing to filter flickering detections.
        
        Motion blur causes detections to appear/disappear randomly.
        We keep track of detections across frames and only retain
        those that appear consistently.
        
        Args:
            detections: Current frame detections
            
        Returns:
            Temporally smoothed detections
        """
        # Add current frame to buffer
        self.temporal_buffer.append({
            'frame_num': self.frame_number,
            'detections': detections.copy()
        })
        
        if len(self.temporal_buffer) < config.MIN_FRAME_APPEARANCES:
            return detections
        
        # For each current detection, check if it appears in enough frames
        smoothed_detections = []
        
        for det in detections:
            appearances = 1  # Current frame counts
            confidence_boost = 0
            
            # Look back through buffer
            for past_frame in list(self.temporal_buffer)[:-1]:
                for past_det in past_frame['detections']:
                    iou = self._calculate_iou(
                        np.array(det['bbox']),
                        np.array(past_det['bbox'])
                    )
                    
                    if iou >= config.TEMPORAL_IOU_THRESHOLD:
                        appearances += 1
                        confidence_boost += config.BLUR_CONFIDENCE_BOOST
                        break
            
            # Keep detection if it appears enough times
            if appearances >= config.MIN_FRAME_APPEARANCES:
                det = det.copy()
                # Boost confidence for stable detections
                det['confidence'] = min(1.0, det['confidence'] + confidence_boost)
                det['temporal_stability'] = appearances
                smoothed_detections.append(det)
                self.metrics.blur_smoothed += 1
            else:
                self.metrics.false_positives_filtered += 1
        
        return smoothed_detections
    
    # =========================================================================
    # KALMAN FILTERING FOR SMOOTH TRACKING
    # =========================================================================
    
    def _apply_kalman_smoothing(self, detections: List[Dict]) -> List[Dict]:
        """
        Apply Kalman filtering to smooth track trajectories.
        
        Kalman filters predict where a person should be based on
        their velocity, even if detection is temporarily lost.
        This helps maintain consistent tracking through occlusions.
        
        Args:
            detections: Current detections with track_ids
            
        Returns:
            Detections with smoothed positions
        """
        current_track_ids = set()
        smoothed_detections = []
        
        for det in detections:
            track_id = det.get('track_id', -1)
            if track_id < 0:
                smoothed_detections.append(det)
                continue
            
            current_track_ids.add(track_id)
            bbox = det['bbox']
            
            # Calculate center point
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            
            # Get or create Kalman track
            if track_id not in self.kalman_tracks:
                self.kalman_tracks[track_id] = KalmanTrack(track_id=track_id)
                # Initialize state with first measurement
                self.kalman_tracks[track_id].kf.statePost = np.array(
                    [[cx], [cy], [0], [0]], dtype=np.float32
                )
            
            ktrack = self.kalman_tracks[track_id]
            ktrack.frames_missing = 0
            ktrack.last_detection = det
            ktrack.prediction_count = 0
            
            # Predict then correct with measurement
            ktrack.kf.predict()
            measurement = np.array([[cx], [cy]], dtype=np.float32)
            corrected = ktrack.kf.correct(measurement)
            
            # Use corrected position for smoother display
            det = det.copy()
            if config.USE_PREDICTED_FOR_DISPLAY:
                # Calculate smoothed bbox from Kalman state
                smooth_cx = corrected[0, 0]
                smooth_cy = corrected[1, 0]
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                
                det['bbox'] = [
                    smooth_cx - width / 2,
                    smooth_cy - height / 2,
                    smooth_cx + width / 2,
                    smooth_cy + height / 2
                ]
                det['kalman_smoothed'] = True
            
            smoothed_detections.append(det)
        
        # Predict positions for missing tracks (handle temporary occlusions)
        for track_id, ktrack in list(self.kalman_tracks.items()):
            if track_id not in current_track_ids:
                ktrack.frames_missing += 1
                
                if ktrack.frames_missing <= config.KALMAN_PREDICTION_FRAMES:
                    # Predict position using Kalman
                    predicted = ktrack.kf.predict()
                    
                    if ktrack.last_detection:
                        # Create predicted detection
                        pred_det = ktrack.last_detection.copy()
                        old_bbox = pred_det['bbox']
                        width = old_bbox[2] - old_bbox[0]
                        height = old_bbox[3] - old_bbox[1]
                        
                        pred_cx = predicted[0, 0]
                        pred_cy = predicted[1, 0]
                        
                        pred_det['bbox'] = [
                            pred_cx - width / 2,
                            pred_cy - height / 2,
                            pred_cx + width / 2,
                            pred_cy + height / 2
                        ]
                        pred_det['is_prediction'] = True
                        pred_det['confidence'] = max(0.3, pred_det['confidence'] - 0.1)
                        
                        smoothed_detections.append(pred_det)
                        ktrack.prediction_count += 1
                        self.metrics.kalman_predictions += 1
                        self.metrics.occluded_recovered += 1
                else:
                    # Track lost for too long, remove
                    del self.kalman_tracks[track_id]
        
        return smoothed_detections
    
    # =========================================================================
    # LOW-LIGHT ENHANCEMENT
    # =========================================================================
    
    def _enhance_low_light(self, frame: np.ndarray) -> np.ndarray:
        """
        Enhance frame in low-light conditions using CLAHE.
        
        Args:
            frame: Input BGR frame
            
        Returns:
            Enhanced frame
        """
        # Check if enhancement is needed
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        
        if mean_brightness >= config.LOW_LIGHT_THRESHOLD:
            return frame
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]
        
        clahe = cv2.createCLAHE(
            clipLimit=config.CLAHE_CLIP_LIMIT,
            tileGridSize=config.CLAHE_GRID_SIZE
        )
        l_enhanced = clahe.apply(l_channel)
        
        lab[:, :, 0] = l_enhanced
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    # =========================================================================
    # MULTI-SCALE DETECTION HELPER
    # =========================================================================
    
    def merge_multi_scale_detections(
        self,
        detections_list: List[List[Dict]],
        scales: List[float]
    ) -> List[Dict]:
        """
        Merge detections from multiple scales.
        
        Args:
            detections_list: List of detections per scale
            scales: Scale factors used
            
        Returns:
            Merged detections
        """
        if len(detections_list) == 0:
            return []
        
        # Collect all detections with scale-adjusted confidence
        all_detections = []
        
        for i, (dets, scale) in enumerate(zip(detections_list, scales)):
            weight = 1.0 if scale == 1.0 else config.SCALE_WEIGHT_DECAY
            
            for det in dets:
                det = det.copy()
                
                # Rescale bbox back to original resolution
                if scale != 1.0:
                    bbox = det['bbox']
                    det['bbox'] = [
                        bbox[0] / scale,
                        bbox[1] / scale,
                        bbox[2] / scale,
                        bbox[3] / scale
                    ]
                
                # Adjust confidence by scale weight
                det['confidence'] *= weight
                det['source_scale'] = scale
                all_detections.append(det)
        
        # Apply soft-NMS to merge
        if self.enable_soft_nms:
            all_detections = self._apply_soft_nms(all_detections)
        
        return all_detections
    
    # =========================================================================
    # TELEMETRY
    # =========================================================================
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current enhancement metrics as dictionary."""
        return {
            'baseline_detections': self.metrics.baseline_detections,
            'enhanced_detections': self.metrics.enhanced_detections,
            'detections_added': self.metrics.occluded_recovered,
            'detections_removed': self.metrics.false_positives_filtered,
            'kalman_predictions': self.metrics.kalman_predictions,
            'blur_smoothed': self.metrics.blur_smoothed,
            'processing_time_ms': self.metrics.processing_time_ms,
            'density_level': self.metrics.density_level,
            'current_confidence': self.metrics.current_confidence,
            'improvement_ratio': (
                self.metrics.enhanced_detections / max(1, self.metrics.baseline_detections)
            )
        }
    
    def reset(self):
        """Reset all state for processing a new video."""
        self.density_history.clear()
        self.temporal_buffer.clear()
        self.kalman_tracks.clear()
        self.current_dynamic_conf = config.CONFIDENCE_THRESHOLD
        self.frame_number = 0


# =============================================================================
# TELEMETRY COLLECTOR
# =============================================================================

class TelemetryCollector:
    """
    Collects and reports telemetry for measuring improvements.
    
    Tracks before/after metrics to quantify enhancement effectiveness.
    """
    
    def __init__(self, output_dir: str = None):
        self.output_dir = output_dir or str(config.TELEMETRY_OUTPUT)
        self.frame_metrics: List[Dict] = []
        self.baseline_metrics: List[Dict] = []
        
    def record_frame(
        self,
        frame_num: int,
        baseline_count: int,
        enhanced_count: int,
        metrics: EnhancementMetrics
    ):
        """Record metrics for a single frame."""
        self.frame_metrics.append({
            'frame': frame_num,
            'baseline': baseline_count,
            'enhanced': enhanced_count,
            'added': metrics.occluded_recovered,
            'removed': metrics.false_positives_filtered,
            'density': metrics.density_level,
            'confidence': metrics.current_confidence,
            'time_ms': metrics.processing_time_ms
        })
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate summary report of improvements."""
        if not self.frame_metrics:
            return {}
        
        total_baseline = sum(m['baseline'] for m in self.frame_metrics)
        total_enhanced = sum(m['enhanced'] for m in self.frame_metrics)
        total_added = sum(m['added'] for m in self.frame_metrics)
        total_removed = sum(m['removed'] for m in self.frame_metrics)
        avg_time = np.mean([m['time_ms'] for m in self.frame_metrics])
        
        return {
            'total_frames': len(self.frame_metrics),
            'total_baseline_detections': total_baseline,
            'total_enhanced_detections': total_enhanced,
            'detections_recovered': total_added,
            'false_positives_filtered': total_removed,
            'net_change': total_enhanced - total_baseline,
            'improvement_percent': (total_enhanced - total_baseline) / max(1, total_baseline) * 100,
            'avg_processing_time_ms': avg_time
        }
    
    def save_report(self, filename: str = "enhancement_report.json"):
        """Save detailed report to file."""
        import json
        from pathlib import Path
        
        output_path = Path(self.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        report = {
            'summary': self.generate_report(),
            'frame_details': self.frame_metrics
        }
        
        with open(output_path / filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        return str(output_path / filename)
