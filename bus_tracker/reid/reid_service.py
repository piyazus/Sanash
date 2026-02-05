"""
Bus Vision - ReID Service
=========================

High-level service integrating ReID model with tracking.
Provides unified API for cross-camera person matching.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import cv2
import numpy as np

from .reid_model import ReIDModel, FeatureCache
from .feature_matcher import FeatureMatcher, TemporalFeatureMatcher

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class PersonAppearance:
    """Record of a person appearing in a camera."""
    camera_id: int
    local_track_id: int
    timestamp: float
    features: np.ndarray
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    confidence: float = 0.0
    
    def __hash__(self):
        return hash((self.camera_id, self.local_track_id, self.timestamp))


@dataclass
class GlobalTrack:
    """A person tracked across multiple cameras."""
    global_id: int
    appearances: List[PersonAppearance] = field(default_factory=list)
    aggregated_features: Optional[np.ndarray] = None
    first_seen: float = 0.0
    last_seen: float = 0.0
    cameras_visited: List[int] = field(default_factory=list)
    
    def add_appearance(self, appearance: PersonAppearance, alpha: float = 0.9):
        """Add new appearance and update aggregated features."""
        self.appearances.append(appearance)
        self.last_seen = appearance.timestamp
        
        if appearance.camera_id not in self.cameras_visited:
            self.cameras_visited.append(appearance.camera_id)
        
        if self.first_seen == 0:
            self.first_seen = appearance.timestamp
        
        # Update features with EMA
        if self.aggregated_features is None:
            self.aggregated_features = appearance.features
        else:
            self.aggregated_features = (
                alpha * self.aggregated_features + 
                (1 - alpha) * appearance.features
            )
            # Re-normalize
            norm = np.linalg.norm(self.aggregated_features)
            if norm > 0:
                self.aggregated_features = self.aggregated_features / norm
    
    @property
    def total_duration(self) -> float:
        """Total time tracked in seconds."""
        return self.last_seen - self.first_seen
    
    @property
    def journey_path(self) -> List[Dict]:
        """Get journey through cameras."""
        path = []
        current_camera = None
        entry_time = 0.0
        
        for app in sorted(self.appearances, key=lambda x: x.timestamp):
            if app.camera_id != current_camera:
                if current_camera is not None:
                    path.append({
                        'camera_id': current_camera,
                        'entry_time': entry_time,
                        'exit_time': app.timestamp
                    })
                current_camera = app.camera_id
                entry_time = app.timestamp
        
        if current_camera is not None:
            path.append({
                'camera_id': current_camera,
                'entry_time': entry_time,
                'exit_time': self.last_seen
            })
        
        return path


# =============================================================================
# REID SERVICE
# =============================================================================

class ReIDService:
    """
    High-level ReID service for cross-camera tracking.
    
    Workflow:
    1. Extract features from person crops
    2. Match against known tracks
    3. Assign global IDs
    4. Track camera transitions
    
    Usage:
        service = ReIDService()
        
        # Process detections from each camera
        for camera_id, detections in camera_detections.items():
            for det in detections:
                crop = extract_crop(frame, det.bbox)
                global_id = service.process_detection(
                    camera_id=camera_id,
                    local_track_id=det.track_id,
                    person_crop=crop,
                    timestamp=current_time,
                    bbox=det.bbox
                )
                det.global_track_id = global_id
    """
    
    def __init__(
        self,
        similarity_threshold: float = 0.7,
        time_window: float = 5.0,
        feature_update_alpha: float = 0.9,
        min_crop_size: Tuple[int, int] = (32, 64)
    ):
        """
        Initialize ReID service.
        
        Args:
            similarity_threshold: Minimum cosine similarity for matching
            time_window: Maximum seconds between camera appearances
            feature_update_alpha: EMA weight for feature updates
            min_crop_size: Minimum (width, height) to extract features
        """
        self.similarity_threshold = similarity_threshold
        self.time_window = time_window
        self.feature_update_alpha = feature_update_alpha
        self.min_crop_size = min_crop_size
        
        # Initialize components
        self.reid_model = ReIDModel()
        self.feature_cache = FeatureCache(max_size=50000)
        
        # Matcher for cross-camera matching
        self.cross_camera_matcher = TemporalFeatureMatcher(
            threshold=similarity_threshold,
            time_window=time_window
        )
        
        # Storage
        self.global_tracks: Dict[int, GlobalTrack] = {}
        self.local_to_global: Dict[Tuple[int, int], int] = {}  # (camera_id, local_id) -> global_id
        self.next_global_id = 1
        
        # Statistics
        self.stats = {
            'total_detections': 0,
            'features_extracted': 0,
            'matches_found': 0,
            'new_tracks_created': 0,
            'handoffs_detected': 0
        }
        
        logger.info("ReID Service initialized")
    
    def extract_features(self, person_crop: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract ReID features from person crop.
        
        Args:
            person_crop: BGR image of person
            
        Returns:
            Feature vector (512,) or None if crop too small
        """
        h, w = person_crop.shape[:2]
        
        if w < self.min_crop_size[0] or h < self.min_crop_size[1]:
            return None
        
        features = self.reid_model.extract_features(person_crop)
        self.stats['features_extracted'] += 1
        
        return features
    
    def process_detection(
        self,
        camera_id: int,
        local_track_id: int,
        person_crop: np.ndarray,
        timestamp: float,
        bbox: Tuple[int, int, int, int],
        confidence: float = 0.0
    ) -> Optional[int]:
        """
        Process a single detection and assign global track ID.
        
        Args:
            camera_id: Camera this detection is from
            local_track_id: Local track ID from ByteTrack
            person_crop: Cropped person image
            timestamp: Frame timestamp
            bbox: Bounding box (x, y, w, h)
            confidence: Detection confidence
            
        Returns:
            Global track ID or None if features couldn't be extracted
        """
        self.stats['total_detections'] += 1
        
        # Check if we already have a mapping for this local track
        key = (camera_id, local_track_id)
        if key in self.local_to_global:
            global_id = self.local_to_global[key]
            
            # Extract features to update the track
            features = self.extract_features(person_crop)
            if features is not None:
                appearance = PersonAppearance(
                    camera_id=camera_id,
                    local_track_id=local_track_id,
                    timestamp=timestamp,
                    features=features,
                    bbox=bbox,
                    confidence=confidence
                )
                self.global_tracks[global_id].add_appearance(
                    appearance, self.feature_update_alpha
                )
                
                # Update matcher
                self.cross_camera_matcher.update_feature(
                    global_id, features, self.feature_update_alpha
                )
            
            return global_id
        
        # Extract features
        features = self.extract_features(person_crop)
        if features is None:
            return None
        
        # Try to match to existing track from other cameras
        match_id, match_score = self.cross_camera_matcher.find_match(
            query=features,
            query_time=timestamp,
            exclude_ids=self._get_active_tracks_in_camera(camera_id)
        )
        
        if match_id is not None:
            # Found match - use existing global ID
            global_id = match_id
            self.stats['matches_found'] += 1
            self.stats['handoffs_detected'] += 1
            
            logger.debug(
                f"Camera handoff detected: cam{camera_id} track{local_track_id} "
                f"-> global{global_id} (score: {match_score:.3f})"
            )
        else:
            # No match - create new global track
            global_id = self._create_new_global_track()
            self.stats['new_tracks_created'] += 1
        
        # Create appearance
        appearance = PersonAppearance(
            camera_id=camera_id,
            local_track_id=local_track_id,
            timestamp=timestamp,
            features=features,
            bbox=bbox,
            confidence=confidence
        )
        
        # Add to global track
        if global_id not in self.global_tracks:
            self.global_tracks[global_id] = GlobalTrack(global_id=global_id)
        
        self.global_tracks[global_id].add_appearance(
            appearance, self.feature_update_alpha
        )
        
        # Update mappings
        self.local_to_global[key] = global_id
        
        # Add to matcher for future cross-camera matching
        self.cross_camera_matcher.add_feature(
            track_id=global_id,
            features=features,
            timestamp=timestamp,
            metadata={'camera_id': camera_id, 'local_track_id': local_track_id}
        )
        
        return global_id
    
    def _create_new_global_track(self) -> int:
        """Create new global track ID."""
        global_id = self.next_global_id
        self.next_global_id += 1
        return global_id
    
    def _get_active_tracks_in_camera(self, camera_id: int) -> List[int]:
        """Get global IDs of tracks currently active in a camera."""
        active = []
        for (cam, local), global_id in self.local_to_global.items():
            if cam == camera_id:
                active.append(global_id)
        return active
    
    def get_global_track(self, global_id: int) -> Optional[GlobalTrack]:
        """Get global track by ID."""
        return self.global_tracks.get(global_id)
    
    def get_all_journeys(self) -> List[Dict]:
        """Get all person journeys across cameras."""
        journeys = []
        
        for global_id, track in self.global_tracks.items():
            if len(track.cameras_visited) > 1:
                journeys.append({
                    'global_id': global_id,
                    'cameras_visited': track.cameras_visited,
                    'duration_seconds': track.total_duration,
                    'first_seen': track.first_seen,
                    'last_seen': track.last_seen,
                    'path': track.journey_path
                })
        
        return journeys
    
    def cleanup_old_tracks(self, current_time: float, max_age: float = 300.0):
        """
        Remove tracks not seen for max_age seconds.
        
        Args:
            current_time: Current timestamp
            max_age: Maximum seconds since last seen
        """
        cutoff = current_time - max_age
        
        to_remove = [
            gid for gid, track in self.global_tracks.items()
            if track.last_seen < cutoff
        ]
        
        for gid in to_remove:
            del self.global_tracks[gid]
            self.cross_camera_matcher.remove_track(gid)
        
        # Clean up local mappings
        self.local_to_global = {
            k: v for k, v in self.local_to_global.items()
            if v not in to_remove
        }
        
        # Clean up matcher
        self.cross_camera_matcher.cleanup_old(current_time, max_age)
        
        if to_remove:
            logger.debug(f"Cleaned up {len(to_remove)} old tracks")
    
    def get_statistics(self) -> Dict:
        """Get service statistics."""
        return {
            **self.stats,
            'active_global_tracks': len(self.global_tracks),
            'local_mappings': len(self.local_to_global),
            'matcher_gallery_size': len(self.cross_camera_matcher)
        }
    
    def reset(self):
        """Reset service state."""
        self.global_tracks.clear()
        self.local_to_global.clear()
        self.cross_camera_matcher.clear()
        self.feature_cache.clear()
        self.next_global_id = 1
        
        self.stats = {k: 0 for k in self.stats}
        
        logger.info("ReID Service reset")


# =============================================================================
# UTILITY FUNCTIONS  
# =============================================================================

def extract_person_crop(
    frame: np.ndarray,
    bbox: Tuple[int, int, int, int],
    padding: float = 0.1
) -> np.ndarray:
    """
    Extract person crop from frame with padding.
    
    Args:
        frame: Full frame image
        bbox: Bounding box (x, y, w, h)
        padding: Padding ratio to add around bbox
        
    Returns:
        Cropped person image
    """
    x, y, w, h = bbox
    H, W = frame.shape[:2]
    
    # Add padding
    pad_w = int(w * padding)
    pad_h = int(h * padding)
    
    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h)
    x2 = min(W, x + w + pad_w)
    y2 = min(H, y + h + pad_h)
    
    return frame[y1:y2, x1:x2].copy()


if __name__ == "__main__":
    # Test ReID service
    print("Testing ReID Service...")
    
    service = ReIDService(similarity_threshold=0.6)
    
    # Simulate detections from 2 cameras
    test_crop = np.random.randint(0, 255, (256, 128, 3), dtype=np.uint8)
    
    # Person 1 in camera 1
    gid1 = service.process_detection(
        camera_id=1,
        local_track_id=101,
        person_crop=test_crop,
        timestamp=0.0,
        bbox=(100, 100, 50, 120)
    )
    print(f"Camera 1, Track 101 -> Global ID: {gid1}")
    
    # Same person appears in camera 2 (similar features)
    # In real scenario, features would be similar
    gid2 = service.process_detection(
        camera_id=2,
        local_track_id=201,
        person_crop=test_crop,  # Same crop = same features
        timestamp=2.0,
        bbox=(200, 100, 50, 120)
    )
    print(f"Camera 2, Track 201 -> Global ID: {gid2}")
    
    # Different person in camera 1
    different_crop = np.random.randint(0, 255, (256, 128, 3), dtype=np.uint8)
    gid3 = service.process_detection(
        camera_id=1,
        local_track_id=102,
        person_crop=different_crop,
        timestamp=1.0,
        bbox=(300, 100, 50, 120)
    )
    print(f"Camera 1, Track 102 -> Global ID: {gid3}")
    
    print(f"\nStatistics: {service.get_statistics()}")
    print(f"Journeys: {service.get_all_journeys()}")
    
    print("\nReID Service test passed!")
