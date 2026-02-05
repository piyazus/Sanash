import { useEffect, useState } from 'react'
import { useParams } from 'react-router-dom'
import { useDispatch, useSelector } from 'react-redux'
import { fetchJob, updateJobProgress } from '../store/jobsSlice'
import { RootState, AppDispatch } from '../store/store'

export default function JobDetails() {
    const { id } = useParams<{ id: string }>()
    const dispatch = useDispatch<AppDispatch>()
    const { currentJob } = useSelector((state: RootState) => state.jobs)
    const [wsConnected, setWsConnected] = useState(false)

    useEffect(() => {
        if (id) {
            dispatch(fetchJob(parseInt(id)))
        }
    }, [id, dispatch])

    // WebSocket for live updates
    useEffect(() => {
        if (!id || currentJob?.status === 'completed' || currentJob?.status === 'failed') return

        const ws = new WebSocket(`ws://${window.location.host}/api/v1/jobs/${id}/live`)

        ws.onopen = () => {
            setWsConnected(true)
        }

        ws.onmessage = (event) => {
            const data = JSON.parse(event.data)
            if (data.type === 'progress') {
                dispatch(updateJobProgress({
                    id: parseInt(id),
                    progress: data.progress,
                    currentFrame: data.frame,
                }))
            }
        }

        ws.onclose = () => {
            setWsConnected(false)
        }

        return () => {
            ws.close()
        }
    }, [id, currentJob?.status, dispatch])

    if (!currentJob) {
        return <div className="text-center py-12 text-gray-500">Loading...</div>
    }

    return (
        <div className="space-y-6">
            <div className="flex items-center justify-between">
                <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Job #{currentJob.id}</h1>
                {wsConnected && (
                    <span className="flex items-center gap-2 text-sm text-green-600">
                        <span className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
                        Live
                    </span>
                )}
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* Status Card */}
                <div className="card p-6">
                    <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">Status</h2>
                    <div className="space-y-4">
                        <div>
                            <p className="text-sm text-gray-500">Current Status</p>
                            <p className="text-lg font-medium capitalize">{currentJob.status}</p>
                        </div>
                        <div>
                            <p className="text-sm text-gray-500">Progress</p>
                            <div className="mt-2">
                                <div className="h-3 bg-gray-200 rounded-full overflow-hidden">
                                    <div
                                        className="h-full bg-primary-600 transition-all duration-300"
                                        style={{ width: `${currentJob.progress}%` }}
                                    />
                                </div>
                                <p className="text-sm text-gray-500 mt-1">{currentJob.progress.toFixed(1)}%</p>
                            </div>
                        </div>
                        <div>
                            <p className="text-sm text-gray-500">Frames</p>
                            <p className="text-lg font-medium">
                                {currentJob.currentFrame.toLocaleString()} / {currentJob.totalFrames?.toLocaleString() || '?'}
                            </p>
                        </div>
                    </div>
                </div>

                {/* Configuration Card */}
                <div className="card p-6">
                    <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">Configuration</h2>
                    <div className="space-y-3">
                        <div className="flex justify-between">
                            <span className="text-gray-500">Model</span>
                            <span className="font-medium">{(currentJob as any).config?.model || 'yolov8m.pt'}</span>
                        </div>
                        <div className="flex justify-between">
                            <span className="text-gray-500">Confidence</span>
                            <span className="font-medium">{(currentJob as any).config?.confidence || 0.5}</span>
                        </div>
                        <div className="flex justify-between">
                            <span className="text-gray-500">Frame Skip</span>
                            <span className="font-medium">{(currentJob as any).config?.frame_skip || 2}</span>
                        </div>
                    </div>
                </div>

                {/* Timestamps Card */}
                <div className="card p-6">
                    <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">Timestamps</h2>
                    <div className="space-y-3">
                        <div>
                            <p className="text-sm text-gray-500">Created</p>
                            <p className="font-medium">{new Date(currentJob.createdAt).toLocaleString()}</p>
                        </div>
                    </div>
                </div>
            </div>

            {/* Results section (shows when completed) */}
            {currentJob.status === 'completed' && (
                <div className="card p-6">
                    <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">Results</h2>
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                        <a href={`/api/v1/jobs/${currentJob.id}/video`} className="btn btn-secondary justify-center">
                            Download Video
                        </a>
                        <a href={`/api/v1/jobs/${currentJob.id}/report`} className="btn btn-secondary justify-center">
                            View Report
                        </a>
                        <a href={`/api/v1/jobs/${currentJob.id}/detections.csv`} className="btn btn-secondary justify-center">
                            Export CSV
                        </a>
                    </div>
                </div>
            )}
        </div>
    )
}
