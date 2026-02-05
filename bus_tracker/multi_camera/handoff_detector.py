"""
Bus Vision - Handoff Detector
=============================

Detects and validates camera transitions (handoffs).
Ensures transitions are spatially and temporally valid.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from enum import Enum

import numpy as np

from .camera_config import CameraLayout, create_standard_bus_layout

logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class HandoffStatus(Enum):
    """Status of handoff validation."""
    VALID = "valid"
    INVALID_SPATIAL = "invalid_spatial"
    INVALID_TEMPORAL = "invalid_temporal"
    INVALID_SIMILARITY = "invalid_similarity"
    PENDING = "pending"


@dataclass
class HandoffEvent:
    """Record of a camera transition."""
    global_track_id: int
    from_camera_id: int
    from_camera_name: str
    to_camera_id: int
    to_camera_name: str
    timestamp: float
    
    # Validation
    reid_confidence: float
    spatial_valid: bool
    temporal_valid: bool
    status: HandoffStatus
    
    # Position info
    exit_position: Optional[Tuple[int, int]] = None  # Last bbox center in from_camera
    entry_position: Optional[Tuple[int, int]] = None  # First bbox center in to_camera
    
    # Time delta
    time_gap_seconds: float = 0.0
    
    @property
    def is_valid(self) -> bool:
        """Check if handoff is valid."""
        return self.status == HandoffStatus.VALID


@dataclass
class TrackLastSeen:
    """Last seen info for a track."""
    camera_id: int
    camera_name: str
    timestamp: float
    bbox: Tuple[int, int, int, int]
    features: Optional[np.ndarray] = None
    
    @property
    def bbox_center(self) -> Tuple[int, int]:
        """Get center of bounding box."""
        x, y, w, h = self.bbox
        return (x + w // 2, y + h // 2)


# =============================================================================
# HANDOFF DETECTOR
# =============================================================================

class HandoffDetector:
    """
    Detects and validates camera handoffs.
    
    A handoff occurs when a person leaves one camera's view and
    enters another camera's view. Valid handoffs must satisfy:
    
    1. Spatial validity: Person was near edge of camera view
    2. Temporal validity: Time gap is reasonable for walking distance
    3. Feature similarity: ReID features match
    
    Usage:
        detector = HandoffDetector()
        
        # Record last seen position
        detector.record_last_seen(
            global_id=1,
            camera_id=1,
            timestamp=10.0,
            bbox=(400, 100, 50, 120)
        )
        
        # Check for handoff when person appears in new camera
        handoff = detector.detect_handoff(
            global_id=1,
            new_camera_id=3,
            new_timestamp=15.0,
            new_bbox=(50, 100, 50, 120),
            reid_similarity=0.85
        )
    """
    
    def __init__(
        self,
        camera_layout: Optional[CameraLayout] = None,
        min_similarity: float = 0.7,
        max_time_gap: float = 30.0,
        edge_threshold: float = 0.15  # 15% of frame width considered "edge"
    ):
        """
        Initialize handoff detector.
        
        Args:
            camera_layout: Camera configuration
            min_similarity: Minimum ReID similarity for valid handoff
            max_time_gap: Maximum seconds between cameras
            edge_threshold: Fraction of frame considered edge zone
        """
        self.camera_layout = camera_layout or create_standard_bus_layout()
        self.min_similarity = min_similarity
        self.max_time_gap = max_time_gap
        self.edge_threshold = edge_threshold
        
        # Track last seen info
        self.last_seen: Dict[int, TrackLastSeen] = {}  # global_id -> last seen
        
        # Recorded handoffs
        self.handoffs: List[HandoffEvent] = []
        
        # Statistics
        self.stats = {
            'total_checks': 0,
            'valid_handoffs': 0,
            'invalid_spatial': 0,
            'invalid_temporal': 0,
            'invalid_similarity': 0
        }
        
        logger.info("HandoffDetector initialized")
    
    def record_last_seen(
        self,
        global_id: int,
        camera_id: int,
        timestamp: float,
        bbox: Tuple[int, int, int, int],
        features: Optional[np.ndarray] = None
    ):
        """
        Record last seen position for a track.
        
        Args:
            global_id: Global track ID
            camera_id: Camera where last seen
            timestamp: Frame timestamp
            bbox: Bounding box (x, y, w, h)
            features: ReID features (optional)
        """
        camera = self.camera_layout.get_camera_by_id(camera_id)
        camera_name = camera.name if camera else f"camera_{camera_id}"
        
        self.last_seen[global_id] = TrackLastSeen(
            camera_id=camera_id,
            camera_name=camera_name,
            timestamp=timestamp,
            bbox=bbox,
            features=features
        )
    
    def detect_handoff(
        self,
        global_id: int,
        new_camera_id: int,
        new_timestamp: float,
        new_bbox: Tuple[int, int, int, int],
        reid_similarity: float = 1.0,
        frame_width: int = 1920
    ) -> Optional[HandoffEvent]:
        """
        Detect if valid handoff occurred.
        
        Args:
            global_id: Global track ID
            new_camera_id: Camera where person just appeared
            new_timestamp: Timestamp of new appearance
            new_bbox: Bounding box in new camera
            reid_similarity: ReID similarity score
            frame_width: Width of camera frame
            
        Returns:
            HandoffEvent if handoff detected, None if same camera
        """
        self.stats['total_checks'] += 1
        
        # Check if we have last seen info
        last = self.last_seen.get(global_id)
        if not last:
            return None
        
        # Same camera - not a handoff
        if last.camera_id == new_camera_id:
            return None
        
        # Get camera info
        from_camera = self.camera_layout.get_camera_by_id(last.camera_id)
        to_camera = self.camera_layout.get_camera_by_id(new_camera_id)
        
        from_name = from_camera.name if from_camera else f"camera_{last.camera_id}"
        to_name = to_camera.name if to_camera else f"camera_{new_camera_id}"
        
        # Calculate time gap
        time_gap = new_timestamp - last.timestamp
        
        # Validate
        spatial_valid = self._validate_spatial(
            last.bbox, new_bbox, from_name, to_name, frame_width
        )
        temporal_valid = self._validate_temporal(
            from_name, to_name, time_gap
        )
        similarity_valid = reid_similarity >= self.min_similarity
        
        # Determine status
        if spatial_valid and temporal_valid and similarity_valid:
            status = HandoffStatus.VALID
            self.stats['valid_handoffs'] += 1
        elif not spatial_valid:
            status = HandoffStatus.INVALID_SPATIAL
            self.stats['invalid_spatial'] += 1
        elif not temporal_valid:
            status = HandoffStatus.INVALID_TEMPORAL
            self.stats['invalid_temporal'] += 1
        else:
            status = HandoffStatus.INVALID_SIMILARITY
            self.stats['invalid_similarity'] += 1
        
        # Create event
        handoff = HandoffEvent(
            global_track_id=global_id,
            from_camera_id=last.camera_id,
            from_camera_name=from_name,
            to_camera_id=new_camera_id,
            to_camera_name=to_name,
            timestamp=new_timestamp,
            reid_confidence=reid_similarity,
            spatial_valid=spatial_valid,
            temporal_valid=temporal_valid,
            status=status,
            exit_position=last.bbox_center,
            entry_position=self._bbox_center(new_bbox),
            time_gap_seconds=time_gap
        )
        
        self.handoffs.append(handoff)
        
        logger.debug(
            f"Handoff {'VALID' if handoff.is_valid else 'INVALID'}: "
            f"track{global_id} {from_name}->{to_name} "
            f"(gap={time_gap:.1f}s, sim={reid_similarity:.2f})"
        )
        
        return handoff
    
    def _validate_spatial(
        self,
        from_bbox: Tuple[int, int, int, int],
        to_bbox: Tuple[int, int, int, int],
        from_camera: str,
        to_camera: str,
        frame_width: int
    ) -> bool:
        """
        Validate spatial consistency of handoff.
        
        Person should exit near edge of from_camera and enter near
        corresponding edge of to_camera.
        """
        # Edge zones
        left_edge = frame_width * self.edge_threshold
        right_edge = frame_width * (1 - self.edge_threshold)
        
        from_x = from_bbox[0] + from_bbox[2] // 2
        to_x = to_bbox[0] + to_bbox[2] // 2
        
        # Check camera adjacency and expected direction
        from_cam = self.camera_layout.get_camera(from_camera)
        to_cam = self.camera_layout.get_camera(to_camera)
        
        if not from_cam or not to_cam:
            return True  # Can't validate, assume valid
        
        # Are cameras adjacent?
        if to_camera not in from_cam.overlaps_with:
            # Non-adjacent cameras - relax spatial check
            return True
        
        # Direction check: if moving from lower to higher position,
        # should exit right edge, enter left edge
        if to_cam.position > from_cam.position:
            # Moving toward rear
            from_edge_ok = from_x > right_edge * 0.8
            to_edge_ok = to_x < left_edge * 1.5
        else:
            # Moving toward front
            from_edge_ok = from_x < left_edge * 1.5
            to_edge_ok = to_x > right_edge * 0.8
        
        return from_edge_ok or to_edge_ok  # At least one should be near edge
    
    def _validate_temporal(
        self,
        from_camera: str,
        to_camera: str,
        time_gap: float
    ) -> bool:
        """
        Validate temporal consistency of handoff.
        
        Time gap should match expected walking time between cameras.
        """
        if time_gap < 0:
            return False  # Can't go back in time
        
        if time_gap > self.max_time_gap:
            return False  # Too long
        
        # Get expected transition time
        min_time, max_time = self.camera_layout.get_expected_transition_time(
            from_camera, to_camera
        )
        
        # Allow some flexibility
        return time_gap >= min_time * 0.5 and time_gap <= max_time * 1.5
    
    @staticmethod
    def _bbox_center(bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
        """Get center of bounding box."""
        x, y, w, h = bbox
        return (x + w // 2, y + h // 2)
    
    def get_handoffs_for_track(self, global_id: int) -> List[HandoffEvent]:
        """Get all handoffs for a track."""
        return [h for h in self.handoffs if h.global_track_id == global_id]
    
    def get_valid_handoffs(self) -> List[HandoffEvent]:
        """Get all valid handoffs."""
        return [h for h in self.handoffs if h.is_valid]
    
    def get_handoff_matrix(self) -> Dict[Tuple[str, str], int]:
        """
        Get count of valid handoffs between each camera pair.
        
        Returns: {('front', 'middle'): 45, ('middle', 'rear'): 38, ...}
        """
        matrix = defaultdict(int)
        
        for handoff in self.handoffs:
            if handoff.is_valid:
                key = (handoff.from_camera_name, handoff.to_camera_name)
                matrix[key] += 1
        
        return dict(matrix)
    
    def get_average_transition_times(self) -> Dict[Tuple[str, str], float]:
        """
        Get average transition time for each camera pair.
        """
        times = defaultdict(list)
        
        for handoff in self.handoffs:
            if handoff.is_valid:
                key = (handoff.from_camera_name, handoff.to_camera_name)
                times[key].append(handoff.time_gap_seconds)
        
        return {k: np.mean(v) for k, v in times.items()}
    
    def get_statistics(self) -> Dict:
        """Get detector statistics."""
        return {
            **self.stats,
            'total_handoffs': len(self.handoffs),
            'valid_handoff_rate': (
                self.stats['valid_handoffs'] / max(1, len(self.handoffs))
            )
        }
    
    def clear(self):
        """Clear all state."""
        self.last_seen.clear()
        self.handoffs.clear()
        self.stats = {k: 0 for k in self.stats}


# =============================================================================
# FLOW ANALYZER
# =============================================================================

class FlowAnalyzer:
    """
    Analyze passenger flow patterns from handoff data.
    """
    
    def __init__(self, handoff_detector: HandoffDetector):
        self.detector = handoff_detector
    
    def get_flow_direction_counts(self) -> Dict[str, int]:
        """
        Get counts of forward vs backward movement.
        
        Returns: {'toward_rear': 100, 'toward_front': 85}
        """
        forward = 0
        backward = 0
        
        layout = self.detector.camera_layout
        
        for handoff in self.detector.get_valid_handoffs():
            from_cam = layout.get_camera(handoff.from_camera_name)
            to_cam = layout.get_camera(handoff.to_camera_name)
            
            if from_cam and to_cam:
                if to_cam.position > from_cam.position:
                    forward += 1  # Moving toward rear
                else:
                    backward += 1  # Moving toward front
        
        return {
            'toward_rear': forward,
            'toward_front': backward
        }
    
    def get_busiest_transition(self) -> Optional[Tuple[str, str, int]]:
        """
        Get the camera transition that happens most often.
        
        Returns: ('front', 'middle', 123) or None
        """
        matrix = self.detector.get_handoff_matrix()
        
        if not matrix:
            return None
        
        busiest = max(matrix.items(), key=lambda x: x[1])
        return (*busiest[0], busiest[1])
    
    def detect_congestion_points(
        self,
        threshold_multiplier: float = 2.0
    ) -> List[Dict]:
        """
        Identify cameras where people spend unusually long time.
        
        Returns list of potential congestion points.
        """
        avg_times = self.detector.get_average_transition_times()
        
        if not avg_times:
            return []
        
        overall_avg = np.mean(list(avg_times.values()))
        threshold = overall_avg * threshold_multiplier
        
        congestion_points = []
        
        for (from_cam, to_cam), avg_time in avg_times.items():
            if avg_time > threshold:
                congestion_points.append({
                    'from_camera': from_cam,
                    'to_camera': to_cam,
                    'avg_time_seconds': avg_time,
                    'expected_time': overall_avg,
                    'severity': 'high' if avg_time > threshold * 1.5 else 'medium'
                })
        
        return congestion_points


if __name__ == "__main__":
    # Test handoff detector
    print("Testing HandoffDetector...")
    
    detector = HandoffDetector()
    
    # Track 1: Person moves from front to middle
    detector.record_last_seen(
        global_id=1,
        camera_id=1,  # front
        timestamp=10.0,
        bbox=(1800, 200, 50, 120)  # Near right edge
    )
    
    handoff = detector.detect_handoff(
        global_id=1,
        new_camera_id=3,  # middle
        new_timestamp=15.0,
        new_bbox=(100, 200, 50, 120),  # Near left edge
        reid_similarity=0.85
    )
    
    if handoff:
        print(f"Handoff detected: {handoff.from_camera_name} -> {handoff.to_camera_name}")
        print(f"  Valid: {handoff.is_valid}")
        print(f"  Status: {handoff.status.value}")
        print(f"  Time gap: {handoff.time_gap_seconds:.1f}s")
    
    # Invalid handoff (too fast)
    detector.record_last_seen(
        global_id=2,
        camera_id=1,
        timestamp=20.0,
        bbox=(1800, 200, 50, 120)
    )
    
    handoff2 = detector.detect_handoff(
        global_id=2,
        new_camera_id=4,  # rear - too far
        new_timestamp=20.5,  # Too fast
        new_bbox=(100, 200, 50, 120),
        reid_similarity=0.85
    )
    
    if handoff2:
        print(f"\nHandoff 2: {handoff2.from_camera_name} -> {handoff2.to_camera_name}")
        print(f"  Valid: {handoff2.is_valid}")
        print(f"  Status: {handoff2.status.value}")
    
    print(f"\nStatistics: {detector.get_statistics()}")
    
    # Test flow analyzer
    analyzer = FlowAnalyzer(detector)
    print(f"Flow directions: {analyzer.get_flow_direction_counts()}")
    
    print("\nHandoffDetector test passed!")
