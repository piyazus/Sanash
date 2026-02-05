"""
Bus Vision - Camera Configuration
=================================

Defines camera positions, overlap zones, and relationships.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

logger = logging.getLogger(__name__)


# =============================================================================
# CAMERA CONFIGURATION
# =============================================================================

@dataclass
class CameraConfig:
    """Configuration for a single camera."""
    
    camera_id: int
    name: str  # 'front', 'middle', 'rear', 'door'
    position: float  # Position along bus (meters from front)
    
    # Overlap zones with adjacent cameras
    overlaps_with: List[str] = field(default_factory=list)
    overlap_zone_start: float = 0.0  # Start of overlap (meters)
    overlap_zone_end: float = 0.0    # End of overlap (meters)
    
    # Camera properties
    resolution: Tuple[int, int] = (1920, 1080)
    fps: int = 30
    field_of_view: float = 90.0  # degrees
    
    # Mounting
    height: float = 2.2  # meters from floor
    angle: float = 15.0  # degrees downward from horizontal
    
    # Calibration (optional)
    intrinsic_matrix: Optional[List[List[float]]] = None
    distortion_coeffs: Optional[List[float]] = None


@dataclass
class CameraLayout:
    """
    Complete layout of cameras in a bus.
    
    Standard bus layout:
    - Front camera: Covers driver area and front door
    - Middle camera: Covers middle section
    - Rear camera: Covers rear section and emergency exit
    - Door camera: Focused on main passenger door
    """
    
    bus_length: float = 12.0  # meters
    bus_width: float = 2.5    # meters
    cameras: Dict[str, CameraConfig] = field(default_factory=dict)
    
    # Expected transition times between cameras (seconds)
    transition_times: Dict[Tuple[str, str], Tuple[float, float]] = field(default_factory=dict)
    
    def get_camera(self, name: str) -> Optional[CameraConfig]:
        """Get camera by name."""
        return self.cameras.get(name)
    
    def get_camera_by_id(self, camera_id: int) -> Optional[CameraConfig]:
        """Get camera by ID."""
        for cam in self.cameras.values():
            if cam.camera_id == camera_id:
                return cam
        return None
    
    def get_adjacent_cameras(self, camera_name: str) -> List[str]:
        """Get names of cameras that overlap with given camera."""
        camera = self.cameras.get(camera_name)
        if camera:
            return camera.overlaps_with
        return []
    
    def is_transition_valid(
        self,
        from_camera: str,
        to_camera: str,
        time_diff: float
    ) -> bool:
        """
        Check if transition between cameras is physically possible.
        
        Args:
            from_camera: Source camera name
            to_camera: Destination camera name
            time_diff: Time between last seen in from_camera and first seen in to_camera
            
        Returns:
            True if transition is plausible
        """
        key = (from_camera, to_camera)
        
        # Check if cameras are adjacent
        from_cam = self.cameras.get(from_camera)
        if from_cam and to_camera not in from_cam.overlaps_with:
            # Non-adjacent cameras - need more time
            min_time, max_time = self.transition_times.get(key, (5.0, 60.0))
        else:
            # Adjacent cameras
            min_time, max_time = self.transition_times.get(key, (0.5, 15.0))
        
        return min_time <= time_diff <= max_time
    
    def get_expected_transition_time(
        self,
        from_camera: str,
        to_camera: str
    ) -> Tuple[float, float]:
        """Get expected (min, max) transition time in seconds."""
        key = (from_camera, to_camera)
        
        if key in self.transition_times:
            return self.transition_times[key]
        
        # Estimate based on distance
        from_cam = self.cameras.get(from_camera)
        to_cam = self.cameras.get(to_camera)
        
        if from_cam and to_cam:
            distance = abs(to_cam.position - from_cam.position)
            # Walking speed: 0.5 - 1.5 m/s in bus
            min_time = distance / 1.5
            max_time = distance / 0.3
            return (max(0.5, min_time), min(60.0, max_time))
        
        return (0.5, 15.0)  # Default
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'bus_length': self.bus_length,
            'bus_width': self.bus_width,
            'cameras': {
                name: {
                    'camera_id': cam.camera_id,
                    'name': cam.name,
                    'position': cam.position,
                    'overlaps_with': cam.overlaps_with,
                    'overlap_zone': (cam.overlap_zone_start, cam.overlap_zone_end),
                    'resolution': cam.resolution,
                    'fps': cam.fps,
                    'fov': cam.field_of_view,
                    'height': cam.height,
                    'angle': cam.angle
                }
                for name, cam in self.cameras.items()
            }
        }


# =============================================================================
# DEFAULT LAYOUTS
# =============================================================================

def create_standard_bus_layout() -> CameraLayout:
    """
    Create standard 4-camera bus layout.
    
    Layout (12m bus):
    
    FRONT                                                    REAR
    [CAM1]----[DOOR]-------[CAM2]--------------[CAM3]
      0m       2m           4m                   8m          12m
    
    Overlap zones:
    - Front <-> Door: 1.5m - 2.5m
    - Front <-> Middle: 3.5m - 4.5m  
    - Middle <-> Rear: 7.5m - 8.5m
    """
    layout = CameraLayout(bus_length=12.0, bus_width=2.5)
    
    # Front camera
    layout.cameras['front'] = CameraConfig(
        camera_id=1,
        name='front',
        position=0.0,
        overlaps_with=['door', 'middle'],
        overlap_zone_start=0.0,
        overlap_zone_end=4.5
    )
    
    # Door camera (focused on main entrance)
    layout.cameras['door'] = CameraConfig(
        camera_id=2,
        name='door',
        position=2.0,
        overlaps_with=['front'],
        overlap_zone_start=1.5,
        overlap_zone_end=2.5
    )
    
    # Middle camera
    layout.cameras['middle'] = CameraConfig(
        camera_id=3,
        name='middle',
        position=4.0,
        overlaps_with=['front', 'rear'],
        overlap_zone_start=3.5,
        overlap_zone_end=8.5
    )
    
    # Rear camera
    layout.cameras['rear'] = CameraConfig(
        camera_id=4,
        name='rear',
        position=8.0,
        overlaps_with=['middle'],
        overlap_zone_start=7.5,
        overlap_zone_end=12.0
    )
    
    # Expected transition times (seconds)
    layout.transition_times = {
        ('front', 'door'): (0.5, 5.0),
        ('door', 'front'): (0.5, 5.0),
        ('front', 'middle'): (2.0, 15.0),
        ('middle', 'front'): (2.0, 15.0),
        ('middle', 'rear'): (2.0, 15.0),
        ('rear', 'middle'): (2.0, 15.0),
        ('front', 'rear'): (5.0, 45.0),
        ('rear', 'front'): (5.0, 45.0),
    }
    
    return layout


def create_articulated_bus_layout() -> CameraLayout:
    """
    Create layout for articulated (bendy) bus with 6 cameras.
    """
    layout = CameraLayout(bus_length=18.0, bus_width=2.5)
    
    # Front section
    layout.cameras['front'] = CameraConfig(
        camera_id=1, name='front', position=0.0,
        overlaps_with=['front_middle']
    )
    
    layout.cameras['front_door'] = CameraConfig(
        camera_id=2, name='front_door', position=2.0,
        overlaps_with=['front']
    )
    
    layout.cameras['front_middle'] = CameraConfig(
        camera_id=3, name='front_middle', position=5.0,
        overlaps_with=['front', 'articulation']
    )
    
    # Articulation point
    layout.cameras['articulation'] = CameraConfig(
        camera_id=4, name='articulation', position=9.0,
        overlaps_with=['front_middle', 'rear_middle']
    )
    
    # Rear section
    layout.cameras['rear_middle'] = CameraConfig(
        camera_id=5, name='rear_middle', position=13.0,
        overlaps_with=['articulation', 'rear']
    )
    
    layout.cameras['rear'] = CameraConfig(
        camera_id=6, name='rear', position=17.0,
        overlaps_with=['rear_middle']
    )
    
    return layout


# =============================================================================
# CONFIGURATION LOADING
# =============================================================================

def load_camera_config(config_path: str) -> CameraLayout:
    """
    Load camera configuration from YAML file.
    
    Example YAML:
    ```yaml
    bus_length: 12.0
    bus_width: 2.5
    cameras:
      front:
        camera_id: 1
        position: 0.0
        overlaps_with: ["door", "middle"]
        resolution: [1920, 1080]
        fps: 30
      middle:
        camera_id: 2
        position: 4.0
        overlaps_with: ["front", "rear"]
    transition_times:
      front_to_middle: [2.0, 15.0]
    ```
    """
    path = Path(config_path)
    
    if not path.exists():
        logger.warning(f"Config not found: {config_path}. Using default layout.")
        return create_standard_bus_layout()
    
    with open(path) as f:
        data = yaml.safe_load(f)
    
    layout = CameraLayout(
        bus_length=data.get('bus_length', 12.0),
        bus_width=data.get('bus_width', 2.5)
    )
    
    # Parse cameras
    for name, cam_data in data.get('cameras', {}).items():
        resolution = cam_data.get('resolution', [1920, 1080])
        
        layout.cameras[name] = CameraConfig(
            camera_id=cam_data['camera_id'],
            name=name,
            position=cam_data.get('position', 0.0),
            overlaps_with=cam_data.get('overlaps_with', []),
            overlap_zone_start=cam_data.get('overlap_zone', [0, 0])[0],
            overlap_zone_end=cam_data.get('overlap_zone', [0, 0])[1],
            resolution=tuple(resolution),
            fps=cam_data.get('fps', 30),
            field_of_view=cam_data.get('fov', 90.0),
            height=cam_data.get('height', 2.2),
            angle=cam_data.get('angle', 15.0)
        )
    
    # Parse transition times
    for key, times in data.get('transition_times', {}).items():
        parts = key.split('_to_')
        if len(parts) == 2:
            layout.transition_times[(parts[0], parts[1])] = tuple(times)
    
    logger.info(f"Loaded camera config: {len(layout.cameras)} cameras")
    
    return layout


def save_camera_config(layout: CameraLayout, config_path: str):
    """Save camera configuration to YAML file."""
    data = {
        'bus_length': layout.bus_length,
        'bus_width': layout.bus_width,
        'cameras': {},
        'transition_times': {}
    }
    
    for name, cam in layout.cameras.items():
        data['cameras'][name] = {
            'camera_id': cam.camera_id,
            'position': cam.position,
            'overlaps_with': cam.overlaps_with,
            'overlap_zone': [cam.overlap_zone_start, cam.overlap_zone_end],
            'resolution': list(cam.resolution),
            'fps': cam.fps,
            'fov': cam.field_of_view,
            'height': cam.height,
            'angle': cam.angle
        }
    
    for (from_cam, to_cam), times in layout.transition_times.items():
        key = f"{from_cam}_to_{to_cam}"
        data['transition_times'][key] = list(times)
    
    with open(config_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)
    
    logger.info(f"Saved camera config to {config_path}")


if __name__ == "__main__":
    # Test configuration
    print("Testing Camera Configuration...")
    
    layout = create_standard_bus_layout()
    
    print(f"Bus: {layout.bus_length}m x {layout.bus_width}m")
    print(f"Cameras: {list(layout.cameras.keys())}")
    
    for name, cam in layout.cameras.items():
        print(f"  {name}: position={cam.position}m, overlaps={cam.overlaps_with}")
    
    # Test transitions
    print("\nTransition validation:")
    print(f"  front->middle in 5s: {layout.is_transition_valid('front', 'middle', 5.0)}")
    print(f"  front->rear in 2s: {layout.is_transition_valid('front', 'rear', 2.0)}")
    print(f"  front->rear in 20s: {layout.is_transition_valid('front', 'rear', 20.0)}")
    
    print("\nCamera Configuration test passed!")
