"""
Multi-Camera API Endpoints
==========================

API endpoints for cross-camera tracking and analysis.
"""

from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from api.core.database import get_db
from api.core.security import get_current_user
from api.models.multi_camera import GlobalTrack, TrackMapping, CameraHandoff, MultiCameraSummary
from api.models.detection import Detection
from api.models.bus import Camera


router = APIRouter(prefix="/multi-camera", tags=["Multi-Camera Tracking"])


# =============================================================================
# SCHEMAS
# =============================================================================

class CameraView(BaseModel):
    """Camera info for multi-camera view."""
    id: int
    position: str
    video_url: Optional[str] = None
    total_detections: int
    peak_occupancy: int
    status: str = "active"


class JourneySegment(BaseModel):
    """One segment of a person's journey."""
    camera_id: int
    camera_name: str
    entry_time: datetime
    exit_time: datetime
    duration_seconds: float
    frame_count: int


class GlobalTrackResponse(BaseModel):
    """Complete journey of a tracked person."""
    global_id: int
    cameras_visited: List[str]
    total_time_seconds: float
    first_seen: datetime
    last_seen: datetime
    is_active: bool
    journey: List[JourneySegment]


class HandoffResponse(BaseModel):
    """Camera handoff event."""
    from_camera: str
    to_camera: str
    timestamp: datetime
    confidence: float
    is_valid: bool
    time_gap_seconds: float


class MultiCameraResponse(BaseModel):
    """Complete multi-camera data for a job."""
    job_id: int
    cameras: List[CameraView]
    global_tracks_count: int
    handoffs_count: int
    multi_camera_tracks: int  # Tracks seen in 2+ cameras


class FlowMatrixResponse(BaseModel):
    """Flow between cameras."""
    matrix: dict  # {from_camera: {to_camera: count}}
    total_transitions: int
    busiest_route: Optional[dict] = None


# =============================================================================
# ENDPOINTS
# =============================================================================

@router.get("/jobs/{job_id}", response_model=MultiCameraResponse)
async def get_multi_camera_data(
    job_id: int,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """
    Get synchronized data from all cameras for a job.
    
    Returns camera details, track counts, and handoff statistics.
    """
    # Get cameras for this job
    cameras_query = select(Camera).where(Camera.status == "active")
    cameras_result = await db.execute(cameras_query)
    cameras = cameras_result.scalars().all()
    
    camera_views = []
    for cam in cameras:
        # Count detections per camera
        det_count = await db.execute(
            select(func.count(Detection.id))
            .where(Detection.job_id == job_id)
            .where(Detection.camera_id == cam.id)
        )
        total_dets = det_count.scalar() or 0
        
        # Get peak occupancy (max people in single frame)
        peak_query = select(
            func.max(func.count(Detection.id))
        ).where(
            Detection.job_id == job_id,
            Detection.camera_id == cam.id
        ).group_by(Detection.frame_number)
        
        camera_views.append(CameraView(
            id=cam.id,
            position=cam.position or f"camera_{cam.id}",
            total_detections=total_dets,
            peak_occupancy=0,  # Simplified
            status=cam.status
        ))
    
    # Count global tracks
    tracks_count = await db.execute(
        select(func.count(GlobalTrack.id))
        .where(GlobalTrack.job_id == job_id)
    )
    
    # Count handoffs
    handoffs_count = await db.execute(
        select(func.count(CameraHandoff.id))
        .where(CameraHandoff.job_id == job_id)
    )
    
    # Count multi-camera tracks
    multi_cam_count = await db.execute(
        select(func.count(GlobalTrack.id))
        .where(GlobalTrack.job_id == job_id)
        .where(GlobalTrack.total_cameras_visited > 1)
    )
    
    return MultiCameraResponse(
        job_id=job_id,
        cameras=camera_views,
        global_tracks_count=tracks_count.scalar() or 0,
        handoffs_count=handoffs_count.scalar() or 0,
        multi_camera_tracks=multi_cam_count.scalar() or 0
    )


@router.get("/jobs/{job_id}/tracks", response_model=List[GlobalTrackResponse])
async def get_global_tracks(
    job_id: int,
    limit: int = Query(100, le=500),
    offset: int = Query(0),
    multi_camera_only: bool = Query(False),
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """
    Get all global tracks for a job.
    
    Optionally filter to only show tracks that appeared in multiple cameras.
    """
    query = select(GlobalTrack).where(GlobalTrack.job_id == job_id)
    
    if multi_camera_only:
        query = query.where(GlobalTrack.total_cameras_visited > 1)
    
    query = query.order_by(GlobalTrack.first_seen_at).offset(offset).limit(limit)
    
    result = await db.execute(query)
    tracks = result.scalars().all()
    
    responses = []
    for track in tracks:
        journey = []
        
        # Get mappings for this track
        mappings_result = await db.execute(
            select(TrackMapping)
            .where(TrackMapping.global_track_id == track.id)
            .order_by(TrackMapping.first_seen)
        )
        mappings = mappings_result.scalars().all()
        
        for mapping in mappings:
            camera_result = await db.execute(
                select(Camera).where(Camera.id == mapping.camera_id)
            )
            camera = camera_result.scalar_one_or_none()
            
            journey.append(JourneySegment(
                camera_id=mapping.camera_id,
                camera_name=camera.position if camera else f"camera_{mapping.camera_id}",
                entry_time=mapping.first_seen,
                exit_time=mapping.last_seen,
                duration_seconds=(mapping.last_seen - mapping.first_seen).total_seconds(),
                frame_count=mapping.last_frame - mapping.first_frame + 1
            ))
        
        total_time = (track.last_seen_at - track.first_seen_at).total_seconds()
        
        responses.append(GlobalTrackResponse(
            global_id=track.id,
            cameras_visited=[j.camera_name for j in journey],
            total_time_seconds=total_time,
            first_seen=track.first_seen_at,
            last_seen=track.last_seen_at,
            is_active=track.is_active,
            journey=journey
        ))
    
    return responses


@router.get("/jobs/{job_id}/journey/{global_track_id}", response_model=GlobalTrackResponse)
async def get_person_journey(
    job_id: int,
    global_track_id: int,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """
    Get complete journey of one person across all cameras.
    
    Returns detailed timeline with video timestamps for each camera appearance.
    """
    track_result = await db.execute(
        select(GlobalTrack)
        .where(GlobalTrack.id == global_track_id)
        .where(GlobalTrack.job_id == job_id)
    )
    track = track_result.scalar_one_or_none()
    
    if not track:
        raise HTTPException(status_code=404, detail="Track not found")
    
    # Get all mappings
    mappings_result = await db.execute(
        select(TrackMapping)
        .where(TrackMapping.global_track_id == global_track_id)
        .order_by(TrackMapping.first_seen)
    )
    mappings = mappings_result.scalars().all()
    
    journey = []
    for mapping in mappings:
        camera_result = await db.execute(
            select(Camera).where(Camera.id == mapping.camera_id)
        )
        camera = camera_result.scalar_one_or_none()
        
        journey.append(JourneySegment(
            camera_id=mapping.camera_id,
            camera_name=camera.position if camera else f"camera_{mapping.camera_id}",
            entry_time=mapping.first_seen,
            exit_time=mapping.last_seen,
            duration_seconds=(mapping.last_seen - mapping.first_seen).total_seconds(),
            frame_count=mapping.last_frame - mapping.first_frame + 1
        ))
    
    return GlobalTrackResponse(
        global_id=track.id,
        cameras_visited=[j.camera_name for j in journey],
        total_time_seconds=(track.last_seen_at - track.first_seen_at).total_seconds(),
        first_seen=track.first_seen_at,
        last_seen=track.last_seen_at,
        is_active=track.is_active,
        journey=journey
    )


@router.get("/jobs/{job_id}/handoffs", response_model=List[HandoffResponse])
async def get_handoffs(
    job_id: int,
    valid_only: bool = Query(False),
    limit: int = Query(100, le=500),
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """
    Get camera handoff events for a job.
    
    Handoffs are detected transitions between cameras.
    """
    query = select(CameraHandoff).where(CameraHandoff.job_id == job_id)
    
    if valid_only:
        query = query.where(
            CameraHandoff.spatial_valid == True,
            CameraHandoff.temporal_valid == True
        )
    
    query = query.order_by(CameraHandoff.timestamp).limit(limit)
    
    result = await db.execute(query)
    handoffs = result.scalars().all()
    
    responses = []
    for handoff in handoffs:
        # Get camera names
        from_cam = await db.execute(select(Camera).where(Camera.id == handoff.from_camera_id))
        to_cam = await db.execute(select(Camera).where(Camera.id == handoff.to_camera_id))
        
        from_camera = from_cam.scalar_one_or_none()
        to_camera = to_cam.scalar_one_or_none()
        
        responses.append(HandoffResponse(
            from_camera=from_camera.position if from_camera else f"camera_{handoff.from_camera_id}",
            to_camera=to_camera.position if to_camera else f"camera_{handoff.to_camera_id}",
            timestamp=handoff.timestamp,
            confidence=handoff.reid_confidence or 0.0,
            is_valid=handoff.is_valid,
            time_gap_seconds=handoff.time_gap_seconds or 0.0
        ))
    
    return responses


@router.get("/jobs/{job_id}/flow-matrix", response_model=FlowMatrixResponse)
async def get_flow_matrix(
    job_id: int,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """
    Get movement flow between cameras as matrix.
    
    Useful for Sankey diagram visualization.
    """
    # Get summary if exists
    summary_result = await db.execute(
        select(MultiCameraSummary).where(MultiCameraSummary.job_id == job_id)
    )
    summary = summary_result.scalar_one_or_none()
    
    if summary and summary.flow_matrix:
        matrix = summary.flow_matrix
    else:
        # Calculate from handoffs
        handoffs_result = await db.execute(
            select(CameraHandoff)
            .where(CameraHandoff.job_id == job_id)
            .where(CameraHandoff.spatial_valid == True)
            .where(CameraHandoff.temporal_valid == True)
        )
        handoffs = handoffs_result.scalars().all()
        
        matrix = {}
        for handoff in handoffs:
            from_key = str(handoff.from_camera_id)
            to_key = str(handoff.to_camera_id)
            
            if from_key not in matrix:
                matrix[from_key] = {}
            
            matrix[from_key][to_key] = matrix[from_key].get(to_key, 0) + 1
    
    # Calculate totals
    total = sum(
        sum(destinations.values())
        for destinations in matrix.values()
    )
    
    # Find busiest route
    busiest = None
    max_count = 0
    for from_cam, destinations in matrix.items():
        for to_cam, count in destinations.items():
            if count > max_count:
                max_count = count
                busiest = {"from": from_cam, "to": to_cam, "count": count}
    
    return FlowMatrixResponse(
        matrix=matrix,
        total_transitions=total,
        busiest_route=busiest
    )


@router.get("/analytics/coverage/{job_id}")
async def get_camera_coverage(
    job_id: int,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """
    Calculate camera coverage statistics.
    
    Shows what percentage of people were tracked across multiple cameras.
    """
    # Total tracks
    total_result = await db.execute(
        select(func.count(GlobalTrack.id))
        .where(GlobalTrack.job_id == job_id)
    )
    total_tracks = total_result.scalar() or 0
    
    # Multi-camera tracks
    multi_result = await db.execute(
        select(func.count(GlobalTrack.id))
        .where(GlobalTrack.job_id == job_id)
        .where(GlobalTrack.total_cameras_visited > 1)
    )
    multi_tracks = multi_result.scalar() or 0
    
    # Average cameras per track
    avg_result = await db.execute(
        select(func.avg(GlobalTrack.total_cameras_visited))
        .where(GlobalTrack.job_id == job_id)
    )
    avg_cameras = avg_result.scalar() or 1.0
    
    return {
        "job_id": job_id,
        "total_tracks": total_tracks,
        "single_camera_tracks": total_tracks - multi_tracks,
        "multi_camera_tracks": multi_tracks,
        "multi_camera_rate": multi_tracks / max(1, total_tracks),
        "avg_cameras_per_person": round(avg_cameras, 2),
        "coverage_score": min(1.0, avg_cameras / 4)  # 4 cameras = 100%
    }
