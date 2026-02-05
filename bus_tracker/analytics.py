"""
Analytics Module for Bus Person Detection System
=================================================

This module calculates advanced analytics from the tracking data:
- Peak occupancy times
- Average dwell time per person
- Entry/exit zone counting
- Heatmap generation

Usage:
    from bus_tracker.analytics import AnalyticsEngine
    
    analytics = AnalyticsEngine(entry_zone, exit_zone)
    analytics.process_tracking_data(track_history)
    report = analytics.generate_report()
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
from dataclasses import dataclass, field

from . import config


@dataclass
class PersonStats:
    """Statistics for a single tracked person."""
    track_id: int
    first_seen_frame: int
    last_seen_frame: int
    first_seen_time: str
    last_seen_time: str
    total_frames: int
    dwell_time_seconds: float
    avg_position: Tuple[float, float]
    entered_via_zone: Optional[str] = None
    exited_via_zone: Optional[str] = None
    positions: List[Tuple[float, float]] = field(default_factory=list)


@dataclass
class TimeSlotStats:
    """Statistics for a time slot."""
    start_time: str
    end_time: str
    start_frame: int
    end_frame: int
    avg_count: float
    max_count: int
    min_count: int
    entries: int
    exits: int


class AnalyticsEngine:
    """
    Engine for calculating analytics from tracking data.
    
    Features:
    - Peak occupancy detection
    - Dwell time calculation
    - Entry/exit zone counting
    - Heatmap generation
    - Time-based statistics
    """
    
    def __init__(
        self,
        video_fps: float = 30.0,
        entry_zone: Optional[List[float]] = None,
        exit_zone: Optional[List[float]] = None,
        frame_width: int = 1920,
        frame_height: int = 1080
    ):
        """
        Initialize analytics engine.
        
        Args:
            video_fps: Video frames per second
            entry_zone: Entry zone [x1, y1, x2, y2] in relative coords (0-1)
            exit_zone: Exit zone [x1, y1, x2, y2] in relative coords (0-1)
            frame_width: Video frame width in pixels
            frame_height: Video frame height in pixels
        """
        self.video_fps = video_fps
        self.entry_zone = entry_zone or config.ENTRY_ZONE
        self.exit_zone = exit_zone or config.EXIT_ZONE
        self.frame_width = frame_width
        self.frame_height = frame_height
        
        # Results storage
        self.person_stats: Dict[int, PersonStats] = {}
        self.frame_counts: Dict[int, int] = {}
        self.heatmap: Optional[np.ndarray] = None
        self.entry_count = 0
        self.exit_count = 0
        
        # Time slot analysis
        self.time_slot_duration = 60  # seconds
        self.time_slots: List[TimeSlotStats] = []
    
    def process_tracking_data(self, track_history: Dict[int, List[Dict]]):
        """
        Process tracking data and calculate all analytics.
        
        Args:
            track_history: Dictionary mapping track_id to list of detections
                          Each detection has: bbox, frame_num, timestamp, confidence
        """
        # Initialize heatmap
        heatmap_h, heatmap_w = config.HEATMAP_RESOLUTION
        self.heatmap = np.zeros((heatmap_h, heatmap_w), dtype=np.float32)
        
        # Process each person's track
        for track_id, detections in track_history.items():
            if not detections:
                continue
            
            # Sort by frame number
            detections = sorted(detections, key=lambda d: d['frame_num'])
            
            # Calculate person statistics
            stats = self._calculate_person_stats(track_id, detections)
            self.person_stats[track_id] = stats
            
            # Update heatmap
            for det in detections:
                self._update_heatmap(det['bbox'])
            
            # Track frame counts
            for det in detections:
                frame_num = det['frame_num']
                self.frame_counts[frame_num] = self.frame_counts.get(frame_num, 0) + 1
            
            # Check zone crossings
            if self.entry_zone or self.exit_zone:
                self._check_zone_crossings(stats, detections)
        
        # Normalize heatmap
        if self.heatmap.max() > 0:
            self.heatmap = self.heatmap / self.heatmap.max()
        
        # Calculate time slots
        self._calculate_time_slots()
    
    def _calculate_person_stats(
        self, 
        track_id: int, 
        detections: List[Dict]
    ) -> PersonStats:
        """Calculate statistics for a single person."""
        first_det = detections[0]
        last_det = detections[-1]
        
        # Calculate duration
        frame_diff = last_det['frame_num'] - first_det['frame_num']
        dwell_time = frame_diff / self.video_fps if self.video_fps > 0 else 0
        
        # Calculate average position
        positions = []
        for det in detections:
            bbox = det['bbox']
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            positions.append((cx, cy))
        
        avg_x = np.mean([p[0] for p in positions])
        avg_y = np.mean([p[1] for p in positions])
        
        return PersonStats(
            track_id=track_id,
            first_seen_frame=first_det['frame_num'],
            last_seen_frame=last_det['frame_num'],
            first_seen_time=first_det.get('timestamp', ''),
            last_seen_time=last_det.get('timestamp', ''),
            total_frames=len(detections),
            dwell_time_seconds=dwell_time,
            avg_position=(avg_x, avg_y),
            positions=positions
        )
    
    def _update_heatmap(self, bbox: List[float]):
        """Update heatmap with detection bounding box."""
        heatmap_h, heatmap_w = self.heatmap.shape
        
        # Calculate center point
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        
        # Convert to heatmap coordinates
        hx = int((cx / self.frame_width) * heatmap_w)
        hy = int((cy / self.frame_height) * heatmap_h)
        
        # Clamp to valid range
        hx = max(0, min(heatmap_w - 1, hx))
        hy = max(0, min(heatmap_h - 1, hy))
        
        # Add to heatmap with Gaussian spread
        self._add_gaussian(self.heatmap, hx, hy, sigma=2)
    
    def _add_gaussian(
        self, 
        heatmap: np.ndarray, 
        cx: int, 
        cy: int, 
        sigma: float = 2
    ):
        """Add a Gaussian blob to heatmap at specified location."""
        h, w = heatmap.shape
        
        # Create a small kernel
        size = int(sigma * 3)
        x = np.arange(max(0, cx - size), min(w, cx + size + 1))
        y = np.arange(max(0, cy - size), min(h, cy + size + 1))
        
        if len(x) == 0 or len(y) == 0:
            return
        
        xx, yy = np.meshgrid(x, y)
        gaussian = np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * sigma**2))
        
        # Add to heatmap
        heatmap[
            max(0, cy - size):min(h, cy + size + 1),
            max(0, cx - size):min(w, cx + size + 1)
        ] += gaussian
    
    def _check_zone_crossings(
        self, 
        stats: PersonStats, 
        detections: List[Dict]
    ):
        """Check if person crossed entry/exit zones."""
        if len(detections) < 2:
            return
        
        first_pos = stats.positions[0]
        last_pos = stats.positions[-1]
        
        # Check entry zone
        if self.entry_zone:
            if self._point_in_zone(first_pos, self.entry_zone):
                stats.entered_via_zone = "entry"
                self.entry_count += 1
        
        # Check exit zone
        if self.exit_zone:
            if self._point_in_zone(last_pos, self.exit_zone):
                stats.exited_via_zone = "exit"
                self.exit_count += 1
    
    def _point_in_zone(
        self, 
        point: Tuple[float, float], 
        zone: List[float]
    ) -> bool:
        """Check if a point is inside a zone."""
        # Convert zone from relative to absolute coords
        x1 = zone[0] * self.frame_width
        y1 = zone[1] * self.frame_height
        x2 = zone[2] * self.frame_width
        y2 = zone[3] * self.frame_height
        
        px, py = point
        return x1 <= px <= x2 and y1 <= py <= y2
    
    def _calculate_time_slots(self):
        """Calculate statistics for time slots."""
        if not self.frame_counts:
            return
        
        # Get frame range
        min_frame = min(self.frame_counts.keys())
        max_frame = max(self.frame_counts.keys())
        
        # Calculate slot size in frames
        slot_frames = int(self.time_slot_duration * self.video_fps)
        
        # Process each time slot
        current_frame = min_frame
        while current_frame < max_frame:
            slot_end = min(current_frame + slot_frames, max_frame)
            
            # Get counts for this slot
            slot_counts = [
                self.frame_counts.get(f, 0)
                for f in range(current_frame, slot_end)
                if f in self.frame_counts
            ]
            
            if slot_counts:
                # Calculate start/end times
                start_seconds = current_frame / self.video_fps
                end_seconds = slot_end / self.video_fps
                
                slot_stats = TimeSlotStats(
                    start_time=str(timedelta(seconds=int(start_seconds))),
                    end_time=str(timedelta(seconds=int(end_seconds))),
                    start_frame=current_frame,
                    end_frame=slot_end,
                    avg_count=float(np.mean(slot_counts)),
                    max_count=int(np.max(slot_counts)),
                    min_count=int(np.min(slot_counts)),
                    entries=0,  # Would need frame-level entry tracking
                    exits=0
                )
                self.time_slots.append(slot_stats)
            
            current_frame = slot_end
    
    # =========================================================================
    # PUBLIC ANALYSIS METHODS
    # =========================================================================
    
    def get_peak_occupancy(self) -> Dict[str, Any]:
        """
        Get peak occupancy information.
        
        Returns:
            Dictionary with peak count, time, and frame number
        """
        if not self.frame_counts:
            return {'peak_count': 0, 'peak_frame': 0, 'peak_time': '00:00:00'}
        
        peak_frame = max(self.frame_counts.keys(), key=lambda k: self.frame_counts[k])
        peak_count = self.frame_counts[peak_frame]
        peak_time = str(timedelta(seconds=int(peak_frame / self.video_fps)))
        
        return {
            'peak_count': peak_count,
            'peak_frame': peak_frame,
            'peak_time': peak_time
        }
    
    def get_average_dwell_time(self) -> float:
        """Get average dwell time in seconds across all tracked people."""
        if not self.person_stats:
            return 0.0
        
        dwell_times = [s.dwell_time_seconds for s in self.person_stats.values()]
        return float(np.mean(dwell_times))
    
    def get_dwell_time_distribution(self) -> Dict[str, int]:
        """
        Get distribution of dwell times in buckets.
        
        Returns:
            Dictionary mapping time ranges to person counts
        """
        buckets = {
            '0-10s': 0,
            '10-30s': 0,
            '30-60s': 0,
            '1-5min': 0,
            '5-15min': 0,
            '15min+': 0
        }
        
        for stats in self.person_stats.values():
            dt = stats.dwell_time_seconds
            
            if dt < 10:
                buckets['0-10s'] += 1
            elif dt < 30:
                buckets['10-30s'] += 1
            elif dt < 60:
                buckets['30-60s'] += 1
            elif dt < 300:
                buckets['1-5min'] += 1
            elif dt < 900:
                buckets['5-15min'] += 1
            else:
                buckets['15min+'] += 1
        
        return buckets
    
    def get_entry_exit_counts(self) -> Dict[str, int]:
        """Get entry and exit counts from zones."""
        return {
            'entries': self.entry_count,
            'exits': self.exit_count,
            'net_change': self.entry_count - self.exit_count
        }
    
    def get_occupancy_over_time(self) -> List[Dict[str, Any]]:
        """
        Get occupancy counts over time for graphing.
        
        Returns:
            List of dictionaries with timestamp and count
        """
        data = []
        for slot in self.time_slots:
            data.append({
                'time': slot.start_time,
                'avg_count': slot.avg_count,
                'max_count': slot.max_count,
                'min_count': slot.min_count
            })
        return data
    
    def get_heatmap(self) -> np.ndarray:
        """Get the congregation heatmap as numpy array."""
        return self.heatmap if self.heatmap is not None else np.zeros((48, 64))
    
    def get_heatmap_image(self, target_size: Tuple[int, int] = None) -> np.ndarray:
        """
        Get heatmap as a color image.
        
        Args:
            target_size: Optional (width, height) to resize to
            
        Returns:
            BGR color image of heatmap
        """
        import cv2
        
        heatmap = self.get_heatmap()
        
        # Normalize to 0-255
        heatmap_uint8 = (heatmap * 255).astype(np.uint8)
        
        # Apply colormap
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        
        # Resize if needed
        if target_size:
            heatmap_color = cv2.resize(heatmap_color, target_size)
        
        return heatmap_color
    
    def generate_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive analytics report.
        
        Returns:
            Dictionary containing all analytics data
        """
        peak = self.get_peak_occupancy()
        entry_exit = self.get_entry_exit_counts()
        
        return {
            'summary': {
                'total_unique_people': len(self.person_stats),
                'peak_occupancy': peak['peak_count'],
                'peak_occupancy_time': peak['peak_time'],
                'average_dwell_time_seconds': self.get_average_dwell_time(),
                'entries': entry_exit['entries'],
                'exits': entry_exit['exits'],
            },
            'dwell_time_distribution': self.get_dwell_time_distribution(),
            'occupancy_over_time': self.get_occupancy_over_time(),
            'person_details': {
                track_id: {
                    'first_seen': stats.first_seen_time,
                    'last_seen': stats.last_seen_time,
                    'dwell_time_seconds': stats.dwell_time_seconds,
                    'entered_via': stats.entered_via_zone,
                    'exited_via': stats.exited_via_zone
                }
                for track_id, stats in self.person_stats.items()
            }
        }
