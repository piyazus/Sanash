"""
Real-time Dashboard for Bus Person Detection System
====================================================

This module provides real-time visualization with:
- Live person count display
- Rolling graph of people count over time
- Heatmap overlay showing congregation areas
- Keyboard controls for interaction

Usage:
    from bus_tracker.dashboard import Dashboard
    
    dashboard = Dashboard(frame_size=(1920, 1080))
    dashboard.update(frame, detections, current_count, total_unique)
    dashboard.show()
"""

import cv2
import numpy as np
from collections import deque
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime

from . import config


class Dashboard:
    """
    Real-time visualization dashboard for person detection.
    
    Features:
    - Live count overlay
    - 60-second rolling graph of occupancy
    - Heatmap overlay toggle
    - FPS display
    - Keyboard controls
    """
    
    def __init__(
        self,
        frame_size: Tuple[int, int] = (1920, 1080),
        graph_width: int = 300,
        graph_height: int = 100,
        history_seconds: int = 60
    ):
        """
        Initialize dashboard.
        
        Args:
            frame_size: (width, height) of input frames
            graph_width: Width of the occupancy graph
            graph_height: Height of the occupancy graph
            history_seconds: Seconds of history to show in graph
        """
        self.frame_width, self.frame_height = frame_size
        self.graph_width = graph_width
        self.graph_height = graph_height
        self.history_seconds = history_seconds
        
        # State
        self.show_heatmap = config.ENABLE_HEATMAP
        self.show_graph = True
        self.paused = False
        
        # Count history for graph (deque with max length)
        self.count_history: deque = deque(maxlen=history_seconds * 30)  # Assume 30 FPS
        
        # Heatmap accumulator
        heatmap_h, heatmap_w = config.HEATMAP_RESOLUTION
        self.heatmap = np.zeros((heatmap_h, heatmap_w), dtype=np.float32)
        
        # FPS tracking
        self.fps_history: deque = deque(maxlen=30)
        self.last_frame_time = datetime.now()
        
        # Colors
        self.graph_color = (0, 255, 0)  # Green
        self.text_color = (255, 255, 255)  # White
        self.overlay_alpha = 0.3
    
    def update_heatmap(self, detections: List[Dict]):
        """
        Update accumulated heatmap with new detections.
        
        Args:
            detections: List of detection dictionaries with 'bbox' key
        """
        heatmap_h, heatmap_w = self.heatmap.shape
        
        for det in detections:
            bbox = det['bbox']
            
            # Calculate center
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            
            # Convert to heatmap coordinates
            hx = int((cx / self.frame_width) * heatmap_w)
            hy = int((cy / self.frame_height) * heatmap_h)
            
            # Clamp
            hx = max(0, min(heatmap_w - 1, hx))
            hy = max(0, min(heatmap_h - 1, hy))
            
            # Add point with small gaussian
            self._add_point(hx, hy)
        
        # Apply decay to fade old detections
        self.heatmap *= config.HEATMAP_DECAY
    
    def _add_point(self, cx: int, cy: int, sigma: float = 2):
        """Add a point to heatmap with Gaussian spread."""
        h, w = self.heatmap.shape
        size = int(sigma * 3)
        
        y_start, y_end = max(0, cy - size), min(h, cy + size + 1)
        x_start, x_end = max(0, cx - size), min(w, cx + size + 1)
        
        for y in range(y_start, y_end):
            for x in range(x_start, x_end):
                dist_sq = (x - cx)**2 + (y - cy)**2
                self.heatmap[y, x] += np.exp(-dist_sq / (2 * sigma**2))
    
    def update(
        self,
        frame: np.ndarray,
        detections: List[Dict],
        current_count: int,
        total_unique: int,
        timestamp: str = ""
    ) -> np.ndarray:
        """
        Update dashboard and return annotated frame.
        
        Args:
            frame: Input frame (BGR)
            detections: Current detections
            current_count: People currently in frame
            total_unique: Total unique people tracked
            timestamp: Current video timestamp
            
        Returns:
            Annotated frame with dashboard overlays
        """
        # Calculate FPS
        now = datetime.now()
        delta = (now - self.last_frame_time).total_seconds()
        if delta > 0:
            self.fps_history.append(1.0 / delta)
        self.last_frame_time = now
        
        current_fps = np.mean(self.fps_history) if self.fps_history else 0
        
        # Update history
        self.count_history.append(current_count)
        
        # Update heatmap
        if self.show_heatmap:
            self.update_heatmap(detections)
        
        # Create output frame
        output = frame.copy()
        
        # Draw heatmap overlay
        if self.show_heatmap and self.heatmap.max() > 0:
            output = self._draw_heatmap_overlay(output)
        
        # Draw count info (top-left)
        output = self._draw_info_panel(
            output, current_count, total_unique, 
            current_fps, timestamp
        )
        
        # Draw graph (bottom-right)
        if self.show_graph:
            output = self._draw_graph(output)
        
        # Draw controls hint (bottom-left)
        output = self._draw_controls_hint(output)
        
        return output
    
    def _draw_heatmap_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Draw heatmap as semi-transparent overlay."""
        # Normalize and colorize heatmap
        heatmap_norm = self.heatmap / (self.heatmap.max() + 1e-6)
        heatmap_uint8 = (heatmap_norm * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        
        # Resize to frame size
        heatmap_resized = cv2.resize(
            heatmap_color, 
            (self.frame_width, self.frame_height)
        )
        
        # Blend with frame
        output = cv2.addWeighted(
            frame, 1 - self.overlay_alpha,
            heatmap_resized, self.overlay_alpha,
            0
        )
        
        return output
    
    def _draw_info_panel(
        self,
        frame: np.ndarray,
        current_count: int,
        total_unique: int,
        fps: float,
        timestamp: str
    ) -> np.ndarray:
        """Draw information panel in top-left corner."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Info lines
        lines = [
            f"Current: {current_count}",
            f"Total Unique: {total_unique}",
            f"FPS: {fps:.1f}",
        ]
        if timestamp:
            lines.append(timestamp)
        
        # Calculate panel size
        padding = 15
        line_height = 32
        panel_height = len(lines) * line_height + padding * 2
        panel_width = 220
        
        # Draw semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (10, 10),
            (10 + panel_width, 10 + panel_height),
            (0, 0, 0),
            -1
        )
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        
        # Draw text
        for i, line in enumerate(lines):
            y = 35 + i * line_height
            cv2.putText(
                frame, line,
                (20, y),
                font, 0.7,
                self.text_color, 2
            )
        
        return frame
    
    def _draw_graph(self, frame: np.ndarray) -> np.ndarray:
        """Draw rolling occupancy graph in bottom-right corner."""
        if len(self.count_history) < 2:
            return frame
        
        # Graph position
        margin = 20
        graph_x = self.frame_width - self.graph_width - margin
        graph_y = self.frame_height - self.graph_height - margin
        
        # Draw background
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (graph_x - 10, graph_y - 30),
            (graph_x + self.graph_width + 10, graph_y + self.graph_height + 10),
            (0, 0, 0),
            -1
        )
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        
        # Title
        cv2.putText(
            frame, "People Over Time",
            (graph_x, graph_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            self.text_color, 1
        )
        
        # Draw axes
        cv2.line(
            frame,
            (graph_x, graph_y + self.graph_height),
            (graph_x + self.graph_width, graph_y + self.graph_height),
            (100, 100, 100), 1
        )
        cv2.line(
            frame,
            (graph_x, graph_y),
            (graph_x, graph_y + self.graph_height),
            (100, 100, 100), 1
        )
        
        # Normalize and draw graph line
        counts = list(self.count_history)
        max_count = max(counts) if counts else 1
        
        points = []
        for i, count in enumerate(counts):
            x = graph_x + int((i / len(counts)) * self.graph_width)
            y = graph_y + self.graph_height - int((count / max_count) * self.graph_height * 0.9)
            points.append((x, y))
        
        if len(points) > 1:
            for i in range(len(points) - 1):
                cv2.line(frame, points[i], points[i + 1], self.graph_color, 2)
        
        # Draw current value
        if counts:
            cv2.putText(
                frame, str(counts[-1]),
                (graph_x + self.graph_width + 5, points[-1][1] + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                self.graph_color, 1
            )
        
        return frame
    
    def _draw_controls_hint(self, frame: np.ndarray) -> np.ndarray:
        """Draw controls hint in bottom-left corner."""
        hint = "[Q] Quit  [P] Pause  [H] Heatmap  [G] Graph"
        
        cv2.putText(
            frame, hint,
            (10, self.frame_height - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (150, 150, 150), 1
        )
        
        return frame
    
    def handle_key(self, key: int) -> str:
        """
        Handle keyboard input.
        
        Args:
            key: Key code from cv2.waitKey()
            
        Returns:
            Action string: 'quit', 'pause', 'continue', or 'none'
        """
        key = key & 0xFF
        
        if key == ord('q'):
            return 'quit'
        elif key == ord('p'):
            self.paused = not self.paused
            return 'pause' if self.paused else 'continue'
        elif key == ord('h'):
            self.show_heatmap = not self.show_heatmap
        elif key == ord('g'):
            self.show_graph = not self.show_graph
        
        return 'none'
    
    def show(self, frame: np.ndarray, window_name: str = "Bus Person Tracker"):
        """
        Display frame in window.
        
        Args:
            frame: Frame to display
            window_name: Window title
        """
        # Resize for display if needed
        display_frame = frame
        max_height = 720
        
        if frame.shape[0] > max_height:
            scale = max_height / frame.shape[0]
            display_frame = cv2.resize(frame, None, fx=scale, fy=scale)
        
        cv2.imshow(window_name, display_frame)
    
    def close(self):
        """Close all windows."""
        cv2.destroyAllWindows()
    
    def get_current_heatmap(self) -> np.ndarray:
        """Get current heatmap as normalized array."""
        if self.heatmap.max() > 0:
            return self.heatmap / self.heatmap.max()
        return self.heatmap
