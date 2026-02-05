"""
Multi-Camera Module
===================

Cross-camera person tracking for bus monitoring.
"""

from .camera_config import CameraLayout, CameraConfig, load_camera_config
from .global_tracker import GlobalTrackManager
from .handoff_detector import HandoffDetector, HandoffEvent

__all__ = [
    'CameraLayout',
    'CameraConfig',
    'load_camera_config',
    'GlobalTrackManager',
    'HandoffDetector',
    'HandoffEvent',
]
