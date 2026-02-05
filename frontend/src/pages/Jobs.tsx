import { useEffect, useState, useCallback } from 'react'
import { Link } from 'react-router-dom'
import { useDispatch, useSelector } from 'react-redux'
import { fetchJobs } from '../store/jobsSlice'
import { RootState, AppDispatch } from '../store/store'
import { api } from '../services/api'
import { PlusIcon, ArrowPathIcon } from '@heroicons/react/24/outline'

export default function Jobs() {
    const dispatch = useDispatch<AppDispatch>()
    const { items: jobs, loading } = useSelector((state: RootState) => state.jobs)
    const [uploading, setUploading] = useState(false)
    const [uploadProgress, setUploadProgress] = useState(0)

    useEffect(() => {
        dispatch(fetchJobs())
    }, [dispatch])

    const handleFileUpload = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0]
        if (!file) return

        setUploading(true)
        setUploadProgress(0)

        const formData = new FormData()
        formData.append('file', file)

        try {
            // Upload video
            const uploadResponse = await api.post('/api/v1/videos/upload', formData, {
                headers: { 'Content-Type': 'multipart/form-data' },
                onUploadProgress: (progressEvent) => {
                    const percent = Math.round((progressEvent.loaded * 100) / (progressEvent.total || 1))
                    setUploadProgress(percent)
                },
            })

            // Create job
            await api.post('/api/v1/jobs', {
                video_id: uploadResponse.data.id,
                config: {
                    model: 'yolov8m.pt',
                    confidence: 0.5,
                    frame_skip: 2,
                },
            })

            // Refresh jobs list
            dispatch(fetchJobs())
        } catch (error) {
            console.error('Upload failed:', error)
        } finally {
            setUploading(false)
            setUploadProgress(0)
        }
    }, [dispatch])

    const getStatusColor = (status: string) => {
        switch (status) {
            case 'completed': return 'bg-green-100 text-green-800'
            case 'processing': return 'bg-blue-100 text-blue-800'
            case 'failed': return 'bg-red-100 text-red-800'
            case 'queued': return 'bg-yellow-100 text-yellow-800'
            default: return 'bg-gray-100 text-gray-800'
        }
    }

    return (
        <div className="space-y-6">
            <div className="flex items-center justify-between">
                <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Detection Jobs</h1>
                <div className="flex items-center gap-3">
                    <button onClick={() => dispatch(fetchJobs())} className="btn btn-secondary">
                        <ArrowPathIcon className="w-5 h-5 mr-2" />
                        Refresh
                    </button>
                    <label className="btn btn-primary cursor-pointer">
                        <PlusIcon className="w-5 h-5 mr-2" />
                        Upload Video
                        <input type="file" accept="video/*" className="hidden" onChange={handleFileUpload} />
                    </label>
                </div>
            </div>

            {/* Upload Progress */}
            {uploading && (
                <div className="card p-4">
                    <div className="flex items-center gap-4">
                        <div className="flex-1">
                            <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
                                <div
                                    className="h-full bg-primary-600 transition-all duration-300"
                                    style={{ width: `${uploadProgress}%` }}
                                />
                            </div>
                        </div>
                        <span className="text-sm text-gray-500">{uploadProgress}%</span>
                    </div>
                </div>
            )}

            {/* Jobs Table */}
            <div className="card overflow-hidden">
                <table className="w-full">
                    <thead className="bg-gray-50 dark:bg-gray-700">
                        <tr>
                            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase">ID</th>
                            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase">Status</th>
                            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase">Progress</th>
                            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase">Created</th>
                            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase">Actions</th>
                        </tr>
                    </thead>
                    <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
                        {loading ? (
                            <tr>
                                <td colSpan={5} className="px-6 py-8 text-center text-gray-500">Loading...</td>
                            </tr>
                        ) : jobs.length === 0 ? (
                            <tr>
                                <td colSpan={5} className="px-6 py-8 text-center text-gray-500">No jobs yet. Upload a video to get started.</td>
                            </tr>
                        ) : (
                            jobs.map((job: any) => (
                                <tr key={job.id} className="hover:bg-gray-50 dark:hover:bg-gray-700">
                                    <td className="px-6 py-4 text-sm text-gray-900 dark:text-gray-100">#{job.id}</td>
                                    <td className="px-6 py-4">
                                        <span className={`px-2 py-1 text-xs rounded-full ${getStatusColor(job.status)}`}>
                                            {job.status}
                                        </span>
                                    </td>
                                    <td className="px-6 py-4">
                                        <div className="flex items-center gap-2">
                                            <div className="w-24 h-2 bg-gray-200 rounded-full overflow-hidden">
                                                <div className="h-full bg-primary-600" style={{ width: `${job.progress}%` }} />
                                            </div>
                                            <span className="text-sm text-gray-500">{job.progress.toFixed(0)}%</span>
                                        </div>
                                    </td>
                                    <td className="px-6 py-4 text-sm text-gray-500">
                                        {new Date(job.createdAt).toLocaleString()}
                                    </td>
                                    <td className="px-6 py-4">
                                        <Link to={`/jobs/${job.id}`} className="text-primary-600 hover:text-primary-700 text-sm font-medium">
                                            View Details
                                        </Link>
                                    </td>
                                </tr>
                            ))
                        )}
                    </tbody>
                </table>
            </div>
        </div>
    )
}
