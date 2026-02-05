"""
Bus Vision - Global Tracker
===========================

Manages consistent person IDs across all cameras.
Implements cross-camera matching and track merging.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict

import numpy as np

from .camera_config import CameraLayout, create_standard_bus_layout
from ..reid import ReIDService

logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class CameraAppearance:
    """Record of a person appearing in a specific camera."""
    camera_id: int
    camera_name: str
    local_track_id: int
    first_seen: float  # timestamp
    last_seen: float
    entry_bbox: Tuple[int, int, int, int]  # First detection bbox
    exit_bbox: Tuple[int, int, int, int]   # Last detection bbox
    frame_count: int = 0
    avg_confidence: float = 0.0


@dataclass
class GlobalTrackInfo:
    """Complete information about a globally tracked person."""
    global_id: int
    appearances: Dict[int, CameraAppearance] = field(default_factory=dict)  # camera_id -> appearance
    current_camera: Optional[int] = None
    features: Optional[np.ndarray] = None
    
    # Journey tracking
    journey: List[Dict] = field(default_factory=list)
    total_distance: float = 0.0  # meters traveled
    
    # Timestamps
    first_seen_time: float = 0.0
    last_seen_time: float = 0.0
    
    # Status
    is_active: bool = True
    exited: bool = False
    exit_camera: Optional[int] = None
    
    @property
    def cameras_visited(self) -> List[int]:
        """List of camera IDs visited."""
        return list(self.appearances.keys())
    
    @property
    def total_dwell_time(self) -> float:
        """Total time on bus in seconds."""
        return self.last_seen_time - self.first_seen_time
    
    def add_appearance(
        self,
        camera_id: int,
        camera_name: str,
        local_track_id: int,
        timestamp: float,
        bbox: Tuple[int, int, int, int],
        confidence: float
    ):
        """Record appearance in camera."""
        if camera_id not in self.appearances:
            # New camera
            self.appearances[camera_id] = CameraAppearance(
                camera_id=camera_id,
                camera_name=camera_name,
                local_track_id=local_track_id,
                first_seen=timestamp,
                last_seen=timestamp,
                entry_bbox=bbox,
                exit_bbox=bbox,
                frame_count=1,
                avg_confidence=confidence
            )
            
            # Record in journey
            if self.current_camera is not None and self.current_camera != camera_id:
                self.journey.append({
                    'from_camera': self.current_camera,
                    'to_camera': camera_id,
                    'timestamp': timestamp
                })
            
            self.current_camera = camera_id
        else:
            # Update existing
            app = self.appearances[camera_id]
            app.last_seen = timestamp
            app.exit_bbox = bbox
            app.frame_count += 1
            # Running average of confidence
            app.avg_confidence = (
                app.avg_confidence * (app.frame_count - 1) + confidence
            ) / app.frame_count
        
        self.last_seen_time = timestamp
        if self.first_seen_time == 0:
            self.first_seen_time = timestamp


# =============================================================================
# GLOBAL TRACK MANAGER
# =============================================================================

class GlobalTrackManager:
    """
    Manages globally consistent person IDs across all cameras.
    
    Responsibilities:
    1. Assign global IDs to local tracks
    2. Detect camera handoffs using ReID
    3. Merge tracks when same person detected
    4. Track complete passenger journeys
    
    Usage:
        manager = GlobalTrackManager()
        
        # For each detection
        global_id = manager.assign_global_id(
            camera_id=1,
            local_track_id=45,
            features=feature_vector,
            timestamp=10.5,
            bbox=(100, 100, 50, 120)
        )
        
        # Get journey
        journey = manager.get_journey(global_id)
    """
    
    def __init__(
        self,
        camera_layout: Optional[CameraLayout] = None,
        similarity_threshold: float = 0.7,
        time_window: float = 10.0,
        inactive_timeout: float = 300.0
    ):
        """
        Initialize global track manager.
        
        Args:
            camera_layout: Camera configuration (uses default if not provided)
            similarity_threshold: Minimum ReID similarity for matching
            time_window: Max seconds between cameras for same person
            inactive_timeout: Seconds before track considered lost
        """
        self.camera_layout = camera_layout or create_standard_bus_layout()
        self.similarity_threshold = similarity_threshold
        self.time_window = time_window
        self.inactive_timeout = inactive_timeout
        
        # Track storage
        self.global_tracks: Dict[int, GlobalTrackInfo] = {}
        self.local_to_global: Dict[Tuple[int, int], int] = {}  # (cam_id, local_id) -> global_id
        self.next_global_id = 1
        
        # Feature database for cross-camera matching
        self.camera_features: Dict[int, Dict[int, Tuple[np.ndarray, float]]] = defaultdict(dict)
        # camera_features[camera_id][local_track_id] = (features, timestamp)
        
        # Statistics
        self.stats = {
            'total_assignments': 0,
            'new_tracks': 0,
            'matched_tracks': 0,
            'handoffs': 0,
            'lost_tracks': 0
        }
        
        logger.info("GlobalTrackManager initialized")
    
    def assign_global_id(
        self,
        camera_id: int,
        local_track_id: int,
        features: np.ndarray,
        timestamp: float,
        bbox: Tuple[int, int, int, int],
        confidence: float = 0.0
    ) -> int:
        """
        Assign global track ID to a local detection.
        
        Args:
            camera_id: Camera ID
            local_track_id: Local track ID from detector
            features: ReID feature vector
            timestamp: Frame timestamp
            bbox: Bounding box (x, y, w, h)
            confidence: Detection confidence
            
        Returns:
            Global track ID
        """
        self.stats['total_assignments'] += 1
        
        key = (camera_id, local_track_id)
        camera = self.camera_layout.get_camera_by_id(camera_id)
        camera_name = camera.name if camera else f"camera_{camera_id}"
        
        # Check existing mapping
        if key in self.local_to_global:
            global_id = self.local_to_global[key]
            self._update_track(global_id, camera_id, camera_name, local_track_id, 
                             features, timestamp, bbox, confidence)
            return global_id
        
        # Try to match with tracks from other cameras
        global_id = self._find_cross_camera_match(
            camera_id, features, timestamp
        )
        
        if global_id is not None:
            # Found match - detected camera handoff
            self.stats['matched_tracks'] += 1
            self.stats['handoffs'] += 1
            
            logger.debug(
                f"Handoff detected: cam{camera_id} track{local_track_id} "
                f"matched to global{global_id}"
            )
        else:
            # No match - create new global track
            global_id = self._create_new_track(timestamp)
            self.stats['new_tracks'] += 1
        
        # Update mappings and track info
        self.local_to_global[key] = global_id
        self._update_track(global_id, camera_id, camera_name, local_track_id,
                          features, timestamp, bbox, confidence)
        
        # Store features for future matching
        self._store_features(camera_id, local_track_id, features, timestamp)
        
        return global_id
    
    def _create_new_track(self, timestamp: float) -> int:
        """Create new global track."""
        global_id = self.next_global_id
        self.next_global_id += 1
        
        self.global_tracks[global_id] = GlobalTrackInfo(
            global_id=global_id,
            first_seen_time=timestamp
        )
        
        return global_id
    
    def _update_track(
        self,
        global_id: int,
        camera_id: int,
        camera_name: str,
        local_track_id: int,
        features: np.ndarray,
        timestamp: float,
        bbox: Tuple[int, int, int, int],
        confidence: float
    ):
        """Update global track with new detection."""
        track = self.global_tracks.get(global_id)
        if not track:
            return
        
        track.add_appearance(
            camera_id=camera_id,
            camera_name=camera_name,
            local_track_id=local_track_id,
            timestamp=timestamp,
            bbox=bbox,
            confidence=confidence
        )
        
        # Update features with EMA
        if track.features is None:
            track.features = features
        else:
            alpha = 0.9
            track.features = alpha * track.features + (1 - alpha) * features
            # Re-normalize
            norm = np.linalg.norm(track.features)
            if norm > 0:
                track.features = track.features / norm
    
    def _store_features(
        self,
        camera_id: int,
        local_track_id: int,
        features: np.ndarray,
        timestamp: float
    ):
        """Store features for cross-camera matching."""
        self.camera_features[camera_id][local_track_id] = (features, timestamp)
    
    def _find_cross_camera_match(
        self,
        camera_id: int,
        features: np.ndarray,
        timestamp: float
    ) -> Optional[int]:
        """
        Find matching track from adjacent cameras.
        
        Returns global_id if match found, None otherwise.
        """
        camera = self.camera_layout.get_camera_by_id(camera_id)
        if not camera:
            return None
        
        best_match = None
        best_score = self.similarity_threshold
        
        # Check adjacent cameras
        for adjacent_name in camera.overlaps_with:
            adjacent_cam = self.camera_layout.get_camera(adjacent_name)
            if not adjacent_cam:
                continue
            
            adj_cam_id = adjacent_cam.camera_id
            
            # Check recent tracks in adjacent camera
            for local_id, (stored_features, stored_time) in self.camera_features.get(adj_cam_id, {}).items():
                # Check time window
                time_diff = abs(timestamp - stored_time)
                if time_diff > self.time_window:
                    continue
                
                # Check if transition is valid
                if not self.camera_layout.is_transition_valid(
                    adjacent_cam.name, camera.name, time_diff
                ):
                    continue
                
                # Compute similarity
                similarity = self._compute_similarity(features, stored_features)
                
                if similarity > best_score:
                    key = (adj_cam_id, local_id)
                    if key in self.local_to_global:
                        best_match = self.local_to_global[key]
                        best_score = similarity
        
        return best_match
    
    def _compute_similarity(self, feat1: np.ndarray, feat2: np.ndarray) -> float:
        """Compute cosine similarity."""
        norm1 = np.linalg.norm(feat1)
        norm2 = np.linalg.norm(feat2)
        
        if norm1 < 1e-6 or norm2 < 1e-6:
            return 0.0
        
        return float(np.dot(feat1 / norm1, feat2 / norm2))
    
    def get_track(self, global_id: int) -> Optional[GlobalTrackInfo]:
        """Get global track info."""
        return self.global_tracks.get(global_id)
    
    def get_journey(self, global_id: int) -> Optional[Dict]:
        """
        Get complete journey of a person.
        
        Returns:
            {
                'global_id': 1,
                'cameras_visited': ['front', 'middle', 'rear'],
                'total_time_seconds': 180.5,
                'journey': [
                    {'camera': 'front', 'entry': 10.5, 'exit': 45.2},
                    {'camera': 'middle', 'entry': 45.8, 'exit': 120.0},
                    {'camera': 'rear', 'entry': 120.5, 'exit': 191.0}
                ]
            }
        """
        track = self.global_tracks.get(global_id)
        if not track:
            return None
        
        journey = []
        for cam_id, appearance in sorted(
            track.appearances.items(),
            key=lambda x: x[1].first_seen
        ):
            journey.append({
                'camera_id': cam_id,
                'camera_name': appearance.camera_name,
                'entry_time': appearance.first_seen,
                'exit_time': appearance.last_seen,
                'duration': appearance.last_seen - appearance.first_seen,
                'frames': appearance.frame_count
            })
        
        return {
            'global_id': global_id,
            'cameras_visited': [j['camera_name'] for j in journey],
            'total_time_seconds': track.total_dwell_time,
            'first_seen': track.first_seen_time,
            'last_seen': track.last_seen_time,
            'is_active': track.is_active,
            'journey': journey
        }
    
    def get_all_active_tracks(self) -> List[GlobalTrackInfo]:
        """Get all currently active tracks."""
        return [t for t in self.global_tracks.values() if t.is_active]
    
    def get_occupancy_by_camera(self) -> Dict[int, int]:
        """Get current person count per camera."""
        counts = defaultdict(int)
        
        for track in self.global_tracks.values():
            if track.is_active and track.current_camera is not None:
                counts[track.current_camera] += 1
        
        return dict(counts)
    
    def cleanup_inactive_tracks(self, current_time: float):
        """Mark tracks as inactive if not seen recently."""
        cutoff = current_time - self.inactive_timeout
        
        for track in self.global_tracks.values():
            if track.is_active and track.last_seen_time < cutoff:
                track.is_active = False
                self.stats['lost_tracks'] += 1
        
        # Clean up old features
        for cam_id in list(self.camera_features.keys()):
            for local_id in list(self.camera_features[cam_id].keys()):
                _, timestamp = self.camera_features[cam_id][local_id]
                if timestamp < cutoff:
                    del self.camera_features[cam_id][local_id]
    
    def mark_exit(self, global_id: int, exit_camera_id: int):
        """Mark track as exited through specific camera."""
        track = self.global_tracks.get(global_id)
        if track:
            track.exited = True
            track.exit_camera = exit_camera_id
            track.is_active = False
    
    def get_handoff_events(self) -> List[Dict]:
        """Get all camera handoff events."""
        events = []
        
        for track in self.global_tracks.values():
            for handoff in track.journey:
                events.append({
                    'global_id': track.global_id,
                    'from_camera': handoff['from_camera'],
                    'to_camera': handoff['to_camera'],
                    'timestamp': handoff['timestamp']
                })
        
        return sorted(events, key=lambda x: x['timestamp'])
    
    def get_flow_matrix(self) -> Dict[Tuple[int, int], int]:
        """
        Get flow counts between cameras.
        
        Returns dict: (from_camera, to_camera) -> count
        """
        flow = defaultdict(int)
        
        for track in self.global_tracks.values():
            for handoff in track.journey:
                key = (handoff['from_camera'], handoff['to_camera'])
                flow[key] += 1
        
        return dict(flow)
    
    def get_statistics(self) -> Dict:
        """Get manager statistics."""
        active_count = len([t for t in self.global_tracks.values() if t.is_active])
        
        return {
            **self.stats,
            'total_tracks': len(self.global_tracks),
            'active_tracks': active_count,
            'total_handoffs': len(self.get_handoff_events())
        }
    
    def reset(self):
        """Reset all tracking state."""
        self.global_tracks.clear()
        self.local_to_global.clear()
        self.camera_features.clear()
        self.next_global_id = 1
        self.stats = {k: 0 for k in self.stats}
        
        logger.info("GlobalTrackManager reset")


if __name__ == "__main__":
    # Test global tracker
    print("Testing GlobalTrackManager...")
    
    manager = GlobalTrackManager()
    
    # Simulate person moving through cameras
    feat1 = np.random.randn(512).astype(np.float32)
    feat1 = feat1 / np.linalg.norm(feat1)
    
    # Appear in front camera
    gid1 = manager.assign_global_id(
        camera_id=1,
        local_track_id=101,
        features=feat1,
        timestamp=0.0,
        bbox=(100, 100, 50, 120)
    )
    print(f"Front camera detection -> Global ID: {gid1}")
    
    # Same person appears in middle camera (similar features)
    feat2 = feat1 + np.random.randn(512).astype(np.float32) * 0.1
    feat2 = feat2 / np.linalg.norm(feat2)
    
    gid2 = manager.assign_global_id(
        camera_id=3,  # Middle camera
        local_track_id=201,
        features=feat2,
        timestamp=5.0,
        bbox=(200, 100, 50, 120)
    )
    print(f"Middle camera detection -> Global ID: {gid2}")
    print(f"Same person? {gid1 == gid2}")
    
    # Get journey
    journey = manager.get_journey(gid1)
    print(f"\nJourney: {journey}")
    
    # Get statistics
    print(f"\nStatistics: {manager.get_statistics()}")
    
    print("\nGlobalTrackManager test passed!")
