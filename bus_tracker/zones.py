"""
Zone Management for Entry/Exit Counting
=========================================

Manages entry/exit zones for tracking people entering and leaving
the bus through specific doors (front, back, emergency).

Features:
1. Interactive zone drawing on first frame
2. Polygon-based zones with entry/exit direction
3. Crossing detection with edge case handling
4. Zone statistics and dwell time tracking

Usage:
    # Define zones interactively
    python -m bus_tracker.zones --video input.mp4 --output zones.json
    
    # Use in detection
    python -m bus_tracker.detector --input video.mp4 --zones zones.json
"""

import cv2
import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set, Any
from collections import defaultdict
from datetime import datetime
import time


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class Zone:
    """
    Represents a counting zone (entry/exit area).
    
    A zone is a polygon with:
    - Defined boundary points
    - An entry direction (which side is "inside" the bus)
    - Associated statistics
    """
    name: str
    zone_type: str  # 'entry', 'exit', or 'bidirectional'
    points: List[Tuple[int, int]]  # Polygon vertices
    color: Tuple[int, int, int] = (0, 255, 0)  # BGR
    
    # Direction vector pointing "into" the bus
    # Crossings in this direction = entry, opposite = exit
    entry_direction: Tuple[float, float] = (0, -1)  # Default: upward
    
    # Statistics
    entry_count: int = 0
    exit_count: int = 0
    
    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            'name': self.name,
            'zone_type': self.zone_type,
            'points': self.points,
            'color': list(self.color),
            'entry_direction': list(self.entry_direction)
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Zone':
        """Create Zone from dict."""
        return cls(
            name=data['name'],
            zone_type=data['zone_type'],
            points=[tuple(p) for p in data['points']],
            color=tuple(data.get('color', [0, 255, 0])),
            entry_direction=tuple(data.get('entry_direction', [0, -1]))
        )


@dataclass
class PersonCrossing:
    """Record of a person crossing a zone."""
    track_id: int
    zone_name: str
    direction: str  # 'entry' or 'exit'
    timestamp: float
    frame_num: int
    position: Tuple[float, float]


@dataclass
class PersonJourney:
    """
    Complete journey of a person through the bus.
    
    Tracks entry, exit, and dwell time.
    """
    track_id: int
    entry_zone: Optional[str] = None
    exit_zone: Optional[str] = None
    entry_time: Optional[float] = None
    exit_time: Optional[float] = None
    entry_frame: Optional[int] = None
    exit_frame: Optional[int] = None
    
    @property
    def dwell_time(self) -> Optional[float]:
        """Calculate time between entry and exit."""
        if self.entry_time and self.exit_time:
            return self.exit_time - self.entry_time
        return None
    
    @property
    def is_complete(self) -> bool:
        """Check if person has both entered and exited."""
        return self.entry_zone is not None and self.exit_zone is not None


# =============================================================================
# ZONE MANAGER
# =============================================================================

class ZoneManager:
    """
    Manages zones and tracks crossings for entry/exit counting.
    
    Key features:
    - Multiple named zones (front door, back door, etc.)
    - Direction-aware crossing detection
    - Edge case handling (lingering, partial crossings)
    - Statistics accumulation
    """
    
    def __init__(self, frame_size: Tuple[int, int] = (1920, 1080)):
        """
        Initialize ZoneManager.
        
        Args:
            frame_size: (width, height) for coordinate normalization
        """
        self.frame_width, self.frame_height = frame_size
        
        # All defined zones
        self.zones: Dict[str, Zone] = {}
        
        # Tracking state per person
        # Maps track_id -> last known center position
        self.track_positions: Dict[int, Tuple[float, float]] = {}
        
        # Maps track_id -> set of zones they are currently inside
        self.track_zones_inside: Dict[int, Set[str]] = defaultdict(set)
        
        # Maps track_id -> last frame seen in each zone (for lingering detection)
        self.track_zone_frames: Dict[int, Dict[str, int]] = defaultdict(dict)
        
        # All crossings recorded
        self.crossings: List[PersonCrossing] = []
        
        # Journey tracking per person
        self.journeys: Dict[int, PersonJourney] = {}
        
        # Time slot statistics
        self.hourly_entries: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self.hourly_exits: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        
        # Minimum frames between same-direction crossings (debounce)
        self.crossing_cooldown = 30  # ~1 second at 30fps
        self.last_crossing: Dict[Tuple[int, str], int] = {}  # (track_id, zone) -> frame
        
    # =========================================================================
    # ZONE MANAGEMENT
    # =========================================================================
    
    def add_zone(self, zone: Zone):
        """Add a zone for tracking."""
        self.zones[zone.name] = zone
    
    def remove_zone(self, name: str):
        """Remove a zone by name."""
        if name in self.zones:
            del self.zones[name]
    
    def clear_zones(self):
        """Remove all zones."""
        self.zones.clear()
    
    def get_zone(self, name: str) -> Optional[Zone]:
        """Get zone by name."""
        return self.zones.get(name)
    
    def save_zones(self, filepath: str):
        """Save zones to JSON file."""
        data = {
            'zones': [z.to_dict() for z in self.zones.values()],
            'frame_size': [self.frame_width, self.frame_height],
            'created': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_zones(self, filepath: str):
        """Load zones from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.zones.clear()
        for zone_data in data.get('zones', []):
            zone = Zone.from_dict(zone_data)
            self.zones[zone.name] = zone
        
        # Update frame size if specified
        if 'frame_size' in data:
            self.frame_width, self.frame_height = data['frame_size']
    
    # =========================================================================
    # CROSSING DETECTION
    # =========================================================================
    
    def update(
        self,
        detections: List[Dict],
        frame_num: int,
        video_fps: float = 30.0
    ) -> List[PersonCrossing]:
        """
        Update zone tracking with new detections.
        
        Detects crossings by checking if person center moved from
        inside to outside a zone (or vice versa).
        
        Args:
            detections: List of detection dicts with bbox and track_id
            frame_num: Current frame number
            video_fps: Video FPS for timestamp calculation
            
        Returns:
            List of new crossings detected this frame
        """
        new_crossings = []
        timestamp = frame_num / video_fps
        
        for det in detections:
            track_id = det.get('track_id', -1)
            if track_id < 0:
                continue
            
            # Calculate center point of detection
            bbox = det['bbox']
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            current_pos = (cx, cy)
            
            # Get previous position
            prev_pos = self.track_positions.get(track_id)
            
            # Check each zone for crossings
            for zone_name, zone in self.zones.items():
                currently_inside = self._point_in_polygon(current_pos, zone.points)
                was_inside = zone_name in self.track_zones_inside[track_id]
                
                # Detect zone boundary crossing
                if currently_inside != was_inside:
                    # Check cooldown to prevent rapid re-crossings
                    cooldown_key = (track_id, zone_name)
                    last_cross = self.last_crossing.get(cooldown_key, -999)
                    
                    if frame_num - last_cross >= self.crossing_cooldown:
                        # Determine direction
                        direction = self._determine_direction(
                            prev_pos, current_pos, zone
                        ) if prev_pos else ('entry' if currently_inside else 'exit')
                        
                        # Record crossing
                        crossing = PersonCrossing(
                            track_id=track_id,
                            zone_name=zone_name,
                            direction=direction,
                            timestamp=timestamp,
                            frame_num=frame_num,
                            position=current_pos
                        )
                        new_crossings.append(crossing)
                        self.crossings.append(crossing)
                        
                        # Update zone counts
                        if direction == 'entry':
                            zone.entry_count += 1
                            hour = int(timestamp // 3600)
                            self.hourly_entries[zone_name][hour] += 1
                        else:
                            zone.exit_count += 1
                            hour = int(timestamp // 3600)
                            self.hourly_exits[zone_name][hour] += 1
                        
                        # Update journey
                        self._update_journey(track_id, zone_name, direction, timestamp, frame_num)
                        
                        # Set cooldown
                        self.last_crossing[cooldown_key] = frame_num
                
                # Update zone membership
                if currently_inside:
                    self.track_zones_inside[track_id].add(zone_name)
                    self.track_zone_frames[track_id][zone_name] = frame_num
                else:
                    self.track_zones_inside[track_id].discard(zone_name)
            
            # Store current position for next frame
            self.track_positions[track_id] = current_pos
        
        return new_crossings
    
    def _point_in_polygon(
        self,
        point: Tuple[float, float],
        polygon: List[Tuple[int, int]]
    ) -> bool:
        """
        Check if point is inside polygon using ray casting algorithm.
        
        Args:
            point: (x, y) point to check
            polygon: List of (x, y) vertices
            
        Returns:
            True if point is inside polygon
        """
        x, y = point
        n = len(polygon)
        inside = False
        
        j = n - 1
        for i in range(n):
            xi, yi = polygon[i]
            xj, yj = polygon[j]
            
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            
            j = i
        
        return inside
    
    def _determine_direction(
        self,
        prev_pos: Tuple[float, float],
        current_pos: Tuple[float, float],
        zone: Zone
    ) -> str:
        """
        Determine if crossing is entry or exit based on movement direction.
        
        Compares movement vector with zone's entry direction.
        
        Args:
            prev_pos: Previous center position
            current_pos: Current center position
            zone: Zone being crossed
            
        Returns:
            'entry' or 'exit'
        """
        if prev_pos is None:
            return 'entry'  # Default for new tracks
        
        # Movement vector
        dx = current_pos[0] - prev_pos[0]
        dy = current_pos[1] - prev_pos[1]
        
        # Normalize
        mag = np.sqrt(dx**2 + dy**2)
        if mag < 1e-6:
            return 'entry'
        
        dx /= mag
        dy /= mag
        
        # Dot product with entry direction
        entry_dx, entry_dy = zone.entry_direction
        dot = dx * entry_dx + dy * entry_dy
        
        return 'entry' if dot > 0 else 'exit'
    
    def _update_journey(
        self,
        track_id: int,
        zone_name: str,
        direction: str,
        timestamp: float,
        frame_num: int
    ):
        """Update person's journey record."""
        if track_id not in self.journeys:
            self.journeys[track_id] = PersonJourney(track_id=track_id)
        
        journey = self.journeys[track_id]
        
        if direction == 'entry' and journey.entry_zone is None:
            journey.entry_zone = zone_name
            journey.entry_time = timestamp
            journey.entry_frame = frame_num
        elif direction == 'exit':
            journey.exit_zone = zone_name
            journey.exit_time = timestamp
            journey.exit_frame = frame_num
    
    # =========================================================================
    # STATISTICS
    # =========================================================================
    
    def get_zone_stats(self, zone_name: str = None) -> Dict[str, Any]:
        """
        Get statistics for a zone or all zones.
        
        Returns entries, exits, net change, and hourly breakdown.
        """
        if zone_name:
            zones_to_check = [self.zones[zone_name]] if zone_name in self.zones else []
        else:
            zones_to_check = list(self.zones.values())
        
        stats = {}
        for zone in zones_to_check:
            stats[zone.name] = {
                'entries': zone.entry_count,
                'exits': zone.exit_count,
                'net_change': zone.entry_count - zone.exit_count,
                'hourly_entries': dict(self.hourly_entries[zone.name]),
                'hourly_exits': dict(self.hourly_exits[zone.name])
            }
        
        return stats
    
    def get_total_counts(self) -> Dict[str, int]:
        """Get total entries and exits across all zones."""
        total_entries = sum(z.entry_count for z in self.zones.values())
        total_exits = sum(z.exit_count for z in self.zones.values())
        
        return {
            'total_entries': total_entries,
            'total_exits': total_exits,
            'net_occupancy_change': total_entries - total_exits,
            'current_occupancy': max(0, total_entries - total_exits)
        }
    
    def get_peak_times(self) -> Dict[str, Dict[str, int]]:
        """Find peak entry and exit hours for each zone."""
        peaks = {}
        
        for zone_name in self.zones:
            entry_hours = self.hourly_entries[zone_name]
            exit_hours = self.hourly_exits[zone_name]
            
            peaks[zone_name] = {
                'peak_entry_hour': max(entry_hours, key=entry_hours.get, default=None) if entry_hours else None,
                'peak_entry_count': max(entry_hours.values(), default=0),
                'peak_exit_hour': max(exit_hours, key=exit_hours.get, default=None) if exit_hours else None,
                'peak_exit_count': max(exit_hours.values(), default=0)
            }
        
        return peaks
    
    def get_dwell_times(self) -> Dict[str, Any]:
        """Calculate dwell time statistics for complete journeys."""
        complete_journeys = [j for j in self.journeys.values() if j.is_complete]
        
        if not complete_journeys:
            return {'count': 0, 'average': 0, 'min': 0, 'max': 0}
        
        dwell_times = [j.dwell_time for j in complete_journeys if j.dwell_time]
        
        return {
            'count': len(dwell_times),
            'average': np.mean(dwell_times) if dwell_times else 0,
            'min': min(dwell_times) if dwell_times else 0,
            'max': max(dwell_times) if dwell_times else 0,
            'median': np.median(dwell_times) if dwell_times else 0
        }
    
    def get_full_report(self) -> Dict[str, Any]:
        """Generate complete zone analytics report."""
        return {
            'zones': self.get_zone_stats(),
            'totals': self.get_total_counts(),
            'peak_times': self.get_peak_times(),
            'dwell_times': self.get_dwell_times(),
            'total_crossings': len(self.crossings),
            'unique_people': len(self.journeys)
        }
    
    # =========================================================================
    # VISUALIZATION
    # =========================================================================
    
    def draw_zones(
        self,
        frame: np.ndarray,
        show_counts: bool = True,
        show_arrows: bool = True,
        alpha: float = 0.3
    ) -> np.ndarray:
        """
        Draw zones on frame with counts and direction arrows.
        
        Args:
            frame: Frame to draw on
            show_counts: Show entry/exit counts
            show_arrows: Show direction arrows
            alpha: Zone fill transparency
            
        Returns:
            Annotated frame
        """
        overlay = frame.copy()
        
        for zone in self.zones.values():
            pts = np.array(zone.points, dtype=np.int32)
            
            # Fill zone with transparent color
            cv2.fillPoly(overlay, [pts], zone.color)
            
            # Draw zone border
            cv2.polylines(frame, [pts], True, zone.color, 2)
            
            # Zone name and counts
            if len(zone.points) > 0:
                # Find centroid for text placement
                cx = int(np.mean([p[0] for p in zone.points]))
                cy = int(np.mean([p[1] for p in zone.points]))
                
                # Zone name
                cv2.putText(
                    frame, zone.name,
                    (cx - 40, cy - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2
                )
                
                if show_counts:
                    # Entry count (green)
                    cv2.putText(
                        frame, f"IN: {zone.entry_count}",
                        (cx - 40, cy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2
                    )
                    
                    # Exit count (red)
                    cv2.putText(
                        frame, f"OUT: {zone.exit_count}",
                        (cx - 40, cy + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 255), 2
                    )
                
                if show_arrows:
                    # Draw entry direction arrow
                    arrow_len = 40
                    dx, dy = zone.entry_direction
                    end_x = int(cx + dx * arrow_len)
                    end_y = int(cy + dy * arrow_len)
                    cv2.arrowedLine(
                        frame, (cx, cy), (end_x, end_y),
                        (0, 255, 0), 2, tipLength=0.3
                    )
        
        # Blend overlay
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        return frame
    
    def reset_stats(self):
        """Reset all statistics while keeping zones."""
        for zone in self.zones.values():
            zone.entry_count = 0
            zone.exit_count = 0
        
        self.crossings.clear()
        self.journeys.clear()
        self.hourly_entries.clear()
        self.hourly_exits.clear()
        self.track_positions.clear()
        self.track_zones_inside.clear()
        self.last_crossing.clear()


# =============================================================================
# INTERACTIVE ZONE DRAWER
# =============================================================================

class ZoneDrawer:
    """
    Interactive UI for drawing zones on a frame.
    
    Features:
    - Click to add polygon points
    - Right-click to finish current zone
    - 'n' for new zone
    - 's' to save
    - 'u' to undo last point
    - 'c' to clear current zone
    - 'q' to quit
    """
    
    def __init__(self, frame: np.ndarray, window_name: str = "Zone Drawer"):
        self.original_frame = frame.copy()
        self.display_frame = frame.copy()
        self.window_name = window_name
        
        self.zones: List[Zone] = []
        self.current_points: List[Tuple[int, int]] = []
        self.zone_counter = 1
        
        # Zone colors
        self.colors = [
            (0, 255, 0),    # Green
            (0, 255, 255),  # Yellow
            (0, 165, 255),  # Orange
            (255, 0, 0),    # Blue
            (255, 0, 255),  # Magenta
        ]
        
        self.running = True
        
    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for zone drawing."""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Add point
            self.current_points.append((x, y))
            self._update_display()
            
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Finish current zone
            if len(self.current_points) >= 3:
                self._finish_zone()
    
    def _update_display(self):
        """Update display with current drawing state."""
        self.display_frame = self.original_frame.copy()
        
        # Draw completed zones
        for zone in self.zones:
            pts = np.array(zone.points, dtype=np.int32)
            cv2.fillPoly(self.display_frame, [pts], (*zone.color[:3], 100))
            cv2.polylines(self.display_frame, [pts], True, zone.color, 2)
            
            # Zone name
            cx = int(np.mean([p[0] for p in zone.points]))
            cy = int(np.mean([p[1] for p in zone.points]))
            cv2.putText(
                self.display_frame, zone.name,
                (cx - 30, cy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2
            )
        
        # Draw current zone being drawn
        if self.current_points:
            color = self.colors[len(self.zones) % len(self.colors)]
            
            # Draw points
            for pt in self.current_points:
                cv2.circle(self.display_frame, pt, 5, color, -1)
            
            # Draw lines connecting points
            if len(self.current_points) > 1:
                pts = np.array(self.current_points, dtype=np.int32)
                cv2.polylines(self.display_frame, [pts], False, color, 2)
        
        # Instructions
        instructions = [
            "Left-click: Add point",
            "Right-click: Finish zone",
            "N: New zone | U: Undo | C: Clear",
            "D: Set direction | S: Save | Q: Quit"
        ]
        for i, text in enumerate(instructions):
            cv2.putText(
                self.display_frame, text,
                (10, 25 + i * 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1
            )
    
    def _finish_zone(self):
        """Complete the current zone."""
        if len(self.current_points) < 3:
            return
        
        color = self.colors[len(self.zones) % len(self.colors)]
        
        zone = Zone(
            name=f"Zone {self.zone_counter}",
            zone_type='bidirectional',
            points=self.current_points.copy(),
            color=color
        )
        
        self.zones.append(zone)
        self.zone_counter += 1
        self.current_points.clear()
        self._update_display()
    
    def _set_entry_direction(self, zone_idx: int = -1):
        """
        Set entry direction for the last zone interactively.
        
        Shows arrows, user clicks to select direction.
        """
        if not self.zones:
            return
        
        zone = self.zones[zone_idx]
        
        # Calculate centroid
        cx = int(np.mean([p[0] for p in zone.points]))
        cy = int(np.mean([p[1] for p in zone.points]))
        
        # Show direction options
        temp_frame = self.display_frame.copy()
        directions = {
            'Up': (0, -1),
            'Down': (0, 1),
            'Left': (-1, 0),
            'Right': (1, 0)
        }
        
        option_positions = {}
        for i, (name, (dx, dy)) in enumerate(directions.items()):
            end_x = int(cx + dx * 60)
            end_y = int(cy + dy * 60)
            cv2.arrowedLine(temp_frame, (cx, cy), (end_x, end_y), (0, 255, 0), 3, tipLength=0.3)
            
            text_x = int(cx + dx * 80)
            text_y = int(cy + dy * 80)
            cv2.putText(temp_frame, f"{i+1}:{name}", (text_x - 30, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            option_positions[str(i+1)] = (dx, dy)
        
        cv2.putText(temp_frame, "Press 1-4 to select entry direction",
                   (10, temp_frame.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow(self.window_name, temp_frame)
        
        while True:
            key = cv2.waitKey(0) & 0xFF
            key_char = chr(key)
            
            if key_char in option_positions:
                zone.entry_direction = option_positions[key_char]
                break
            elif key == 27:  # ESC
                break
        
        self._update_display()
    
    def run(self) -> List[Zone]:
        """
        Run the interactive zone drawer.
        
        Returns:
            List of drawn zones
        """
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)
        self._update_display()
        
        while self.running:
            cv2.imshow(self.window_name, self.display_frame)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                # Finish current zone if in progress
                if len(self.current_points) >= 3:
                    self._finish_zone()
                self.running = False
                
            elif key == ord('n'):
                # Finish current and start new
                if len(self.current_points) >= 3:
                    self._finish_zone()
                self.current_points.clear()
                self._update_display()
                
            elif key == ord('u'):
                # Undo last point
                if self.current_points:
                    self.current_points.pop()
                    self._update_display()
                    
            elif key == ord('c'):
                # Clear current zone
                self.current_points.clear()
                self._update_display()
                
            elif key == ord('d'):
                # Set direction for last zone
                if len(self.current_points) >= 3:
                    self._finish_zone()
                self._set_entry_direction()
                
            elif key == ord('s'):
                # Save (handled by caller)
                if len(self.current_points) >= 3:
                    self._finish_zone()
                self.running = False
        
        cv2.destroyWindow(self.window_name)
        return self.zones


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def main():
    """CLI for interactive zone definition."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Draw entry/exit zones on video frame",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Draw zones on first frame:
    python -m bus_tracker.zones --video input.mp4 --output zones.json
    
  Draw zones on specific frame:
    python -m bus_tracker.zones --video input.mp4 --frame 100 --output zones.json
    
  Load and display existing zones:
    python -m bus_tracker.zones --video input.mp4 --load zones.json
        """
    )
    
    parser.add_argument(
        "--video", "-v",
        required=True,
        help="Input video file"
    )
    parser.add_argument(
        "--output", "-o",
        default="zones.json",
        help="Output JSON file for zones"
    )
    parser.add_argument(
        "--frame", "-f",
        type=int,
        default=0,
        help="Frame number to use for drawing"
    )
    parser.add_argument(
        "--load", "-l",
        help="Load existing zones from JSON"
    )
    
    args = parser.parse_args()
    
    # Open video and get frame
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Cannot open video {args.video}")
        return
    
    # Seek to specified frame
    if args.frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("Error: Cannot read frame")
        return
    
    # Load existing zones if specified
    if args.load:
        manager = ZoneManager((frame.shape[1], frame.shape[0]))
        manager.load_zones(args.load)
        
        # Display zones
        display = manager.draw_zones(frame)
        cv2.imshow("Zones", display)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return
    
    # Draw new zones
    print("\n" + "=" * 50)
    print("ZONE DRAWER")
    print("=" * 50)
    print("Instructions:")
    print("  Left-click: Add point to zone")
    print("  Right-click: Complete current zone")
    print("  N: Start new zone")
    print("  D: Set entry direction for last zone")
    print("  U: Undo last point")
    print("  C: Clear current zone")
    print("  S/Q: Save and quit")
    print("=" * 50 + "\n")
    
    drawer = ZoneDrawer(frame)
    zones = drawer.run()
    
    if zones:
        # Create manager and save
        manager = ZoneManager((frame.shape[1], frame.shape[0]))
        for zone in zones:
            manager.add_zone(zone)
        
        manager.save_zones(args.output)
        print(f"\nSaved {len(zones)} zone(s) to: {args.output}")
        
        # Print zone summary
        for zone in zones:
            print(f"  - {zone.name}: {len(zone.points)} points, direction={zone.entry_direction}")
    else:
        print("\nNo zones created")


if __name__ == "__main__":
    main()
