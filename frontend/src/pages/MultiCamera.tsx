import { useEffect, useState, useCallback } from 'react'
import { useParams } from 'react-router-dom'
import { api } from '../services/api'
import {
    VideoCameraIcon,
    UserGroupIcon,
    ArrowsRightLeftIcon,
    MapIcon
} from '@heroicons/react/24/outline'

interface CameraView {
    id: number
    position: string
    videoUrl?: string
    totalDetections: number
    peakOccupancy: number
    status: string
}

interface JourneySegment {
    cameraId: number
    cameraName: string
    entryTime: string
    exitTime: string
    durationSeconds: number
    frameCount: number
}

interface GlobalTrack {
    globalId: number
    camerasVisited: string[]
    totalTimeSeconds: number
    firstSeen: string
    lastSeen: string
    isActive: boolean
    journey: JourneySegment[]
}

interface MultiCameraData {
    jobId: number
    cameras: CameraView[]
    globalTracksCount: number
    handoffsCount: number
    multiCameraTracks: number
}

export default function MultiCameraView() {
    const { jobId } = useParams<{ jobId: string }>()
    const [data, setData] = useState<MultiCameraData | null>(null)
    const [tracks, setTracks] = useState<GlobalTrack[]>([])
    const [selectedTrack, setSelectedTrack] = useState<number | null>(null)
    const [loading, setLoading] = useState(true)
    const [activeCamera, setActiveCamera] = useState<number | null>(null)

    useEffect(() => {
        if (jobId) {
            loadData()
        }
    }, [jobId])

    const loadData = async () => {
        try {
            setLoading(true)
            const [multiCamRes, tracksRes] = await Promise.all([
                api.get(`/api/v1/multi-camera/jobs/${jobId}`),
                api.get(`/api/v1/multi-camera/jobs/${jobId}/tracks?multi_camera_only=true&limit=50`)
            ])
            setData(multiCamRes.data)
            setTracks(tracksRes.data)
        } catch (error) {
            console.error('Failed to load multi-camera data:', error)
        } finally {
            setLoading(false)
        }
    }

    const getCameraStatusColor = (status: string) => {
        switch (status) {
            case 'active': return 'bg-green-500'
            case 'offline': return 'bg-red-500'
            default: return 'bg-yellow-500'
        }
    }

    const formatDuration = (seconds: number) => {
        if (seconds < 60) return `${seconds.toFixed(0)}s`
        if (seconds < 3600) return `${(seconds / 60).toFixed(1)}m`
        return `${(seconds / 3600).toFixed(1)}h`
    }

    if (loading) {
        return (
            <div className="flex items-center justify-center h-96">
                <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600"></div>
            </div>
        )
    }

    return (
        <div className="space-y-6">
            {/* Header */}
            <div className="flex items-center justify-between">
                <div>
                    <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
                        Multi-Camera Tracking
                    </h1>
                    <p className="text-gray-500 dark:text-gray-400">
                        Job #{jobId} - Cross-camera person tracking
                    </p>
                </div>
            </div>

            {/* Stats Cards */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <div className="card p-6">
                    <div className="flex items-center justify-between">
                        <div>
                            <p className="text-sm text-gray-500 dark:text-gray-400">Total Tracks</p>
                            <p className="text-3xl font-bold text-gray-900 dark:text-white">
                                {data?.globalTracksCount ?? 0}
                            </p>
                        </div>
                        <div className="bg-blue-500 p-3 rounded-lg">
                            <UserGroupIcon className="w-6 h-6 text-white" />
                        </div>
                    </div>
                </div>

                <div className="card p-6">
                    <div className="flex items-center justify-between">
                        <div>
                            <p className="text-sm text-gray-500 dark:text-gray-400">Multi-Camera</p>
                            <p className="text-3xl font-bold text-gray-900 dark:text-white">
                                {data?.multiCameraTracks ?? 0}
                            </p>
                        </div>
                        <div className="bg-purple-500 p-3 rounded-lg">
                            <VideoCameraIcon className="w-6 h-6 text-white" />
                        </div>
                    </div>
                </div>

                <div className="card p-6">
                    <div className="flex items-center justify-between">
                        <div>
                            <p className="text-sm text-gray-500 dark:text-gray-400">Handoffs</p>
                            <p className="text-3xl font-bold text-gray-900 dark:text-white">
                                {data?.handoffsCount ?? 0}
                            </p>
                        </div>
                        <div className="bg-green-500 p-3 rounded-lg">
                            <ArrowsRightLeftIcon className="w-6 h-6 text-white" />
                        </div>
                    </div>
                </div>

                <div className="card p-6">
                    <div className="flex items-center justify-between">
                        <div>
                            <p className="text-sm text-gray-500 dark:text-gray-400">Cameras</p>
                            <p className="text-3xl font-bold text-gray-900 dark:text-white">
                                {data?.cameras?.length ?? 0}
                            </p>
                        </div>
                        <div className="bg-orange-500 p-3 rounded-lg">
                            <MapIcon className="w-6 h-6 text-white" />
                        </div>
                    </div>
                </div>
            </div>

            {/* Camera Grid */}
            <div className="card p-6">
                <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                    Camera Views
                </h2>
                <div className="grid grid-cols-2 gap-4">
                    {data?.cameras?.map((camera) => (
                        <div
                            key={camera.id}
                            className={`relative border-2 rounded-lg overflow-hidden cursor-pointer transition-all ${activeCamera === camera.id
                                ? 'border-primary-500 ring-2 ring-primary-200'
                                : 'border-gray-200 dark:border-gray-700 hover:border-primary-300'
                                }`}
                            onClick={() => setActiveCamera(camera.id)}
                        >
                            {/* Placeholder for video */}
                            <div className="aspect-video bg-gray-900 flex items-center justify-center">
                                <VideoCameraIcon className="w-16 h-16 text-gray-600" />
                            </div>

                            {/* Camera info overlay */}
                            <div className="absolute top-4 left-4 bg-black/70 text-white px-3 py-2 rounded">
                                <div className="flex items-center gap-2">
                                    <span className={`w-2 h-2 rounded-full ${getCameraStatusColor(camera.status)}`}></span>
                                    <p className="font-bold uppercase">{camera.position}</p>
                                </div>
                                <p className="text-sm opacity-80">
                                    {camera.totalDetections.toLocaleString()} detections
                                </p>
                            </div>

                            {/* Peak badge */}
                            <div className="absolute top-4 right-4 bg-primary-600 text-white px-2 py-1 rounded text-sm">
                                Peak: {camera.peakOccupancy}
                            </div>

                            {/* Selection highlight overlay */}
                            {selectedTrack && (
                                <div className="absolute inset-0 bg-primary-500/10 pointer-events-none">
                                    {/* Highlight would show bounding boxes here */}
                                </div>
                            )}
                        </div>
                    ))}
                </div>
            </div>

            {/* Journey Timeline */}
            <div className="card p-6">
                <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                    Passenger Journeys (Multi-Camera)
                </h2>

                {tracks.length === 0 ? (
                    <p className="text-gray-500 dark:text-gray-400 text-center py-8">
                        No multi-camera journeys detected
                    </p>
                ) : (
                    <div className="space-y-3 max-h-96 overflow-y-auto">
                        {tracks.map((track) => (
                            <div
                                key={track.globalId}
                                className={`p-4 rounded-lg border cursor-pointer transition-all ${selectedTrack === track.globalId
                                    ? 'border-primary-500 bg-primary-50 dark:bg-primary-900/20'
                                    : 'border-gray-200 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-800'
                                    }`}
                                onClick={() => setSelectedTrack(
                                    selectedTrack === track.globalId ? null : track.globalId
                                )}
                            >
                                <div className="flex items-center justify-between">
                                    <div className="flex items-center gap-4">
                                        <div className="w-10 h-10 rounded-full bg-primary-100 dark:bg-primary-900 flex items-center justify-center">
                                            <span className="text-primary-600 dark:text-primary-400 font-bold">
                                                #{track.globalId}
                                            </span>
                                        </div>

                                        {/* Camera path visualization */}
                                        <div className="flex items-center gap-1">
                                            {track.camerasVisited.map((cam, idx) => (
                                                <div key={idx} className="flex items-center">
                                                    <span className="px-2 py-1 bg-gray-100 dark:bg-gray-700 rounded text-sm font-medium">
                                                        {cam}
                                                    </span>
                                                    {idx < track.camerasVisited.length - 1 && (
                                                        <ArrowsRightLeftIcon className="w-4 h-4 mx-1 text-gray-400" />
                                                    )}
                                                </div>
                                            ))}
                                        </div>
                                    </div>

                                    <div className="flex items-center gap-4 text-sm text-gray-500">
                                        <span>{formatDuration(track.totalTimeSeconds)}</span>
                                        <span className={`px-2 py-1 rounded ${track.isActive
                                            ? 'bg-green-100 text-green-700'
                                            : 'bg-gray-100 text-gray-600'
                                            }`}>
                                            {track.isActive ? 'Active' : 'Completed'}
                                        </span>
                                    </div>
                                </div>

                                {/* Expanded journey details */}
                                {selectedTrack === track.globalId && (
                                    <div className="mt-4 pt-4 border-t border-gray-200 dark:border-gray-700">
                                        <div className="flex gap-2 overflow-x-auto pb-2">
                                            {track.journey.map((segment, idx) => (
                                                <div
                                                    key={idx}
                                                    className="flex-shrink-0 p-3 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-600 rounded-lg min-w-[180px]"
                                                >
                                                    <p className="font-semibold text-gray-900 dark:text-white">
                                                        {segment.cameraName}
                                                    </p>
                                                    <p className="text-xs text-gray-500 mt-1">
                                                        Entry: {new Date(segment.entryTime).toLocaleTimeString()}
                                                    </p>
                                                    <p className="text-xs text-gray-500">
                                                        Exit: {new Date(segment.exitTime).toLocaleTimeString()}
                                                    </p>
                                                    <p className="text-xs text-primary-600 mt-1">
                                                        Duration: {formatDuration(segment.durationSeconds)}
                                                    </p>
                                                </div>
                                            ))}
                                        </div>
                                    </div>
                                )}
                            </div>
                        ))}
                    </div>
                )}
            </div>
        </div>
    )
}
