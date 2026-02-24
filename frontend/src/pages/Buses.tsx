import { useState, useEffect } from 'react'
import { api } from '../services/api'

interface BusStatus {
    bus_id: string
    plate_number: string
    current_count: number
    capacity: number
    occupancy_ratio: number
    status: string
    latitude: number | null
    longitude: number | null
    last_seen: string | null
    route_number: string | null
}

const STATUS_CLASSES: Record<string, string> = {
    green: 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400',
    yellow: 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-400',
    red: 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400',
}

const BAR_CLASSES: Record<string, string> = {
    green: 'bg-green-500',
    yellow: 'bg-yellow-500',
    red: 'bg-red-500',
}

function OccupancyBar({ ratio, status }: { ratio: number; status: string }) {
    return (
        <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2 mt-2">
            <div
                className={`h-2 rounded-full ${BAR_CLASSES[status] ?? 'bg-gray-400'}`}
                style={{ width: `${Math.round(ratio * 100)}%` }}
            />
        </div>
    )
}

export default function Buses() {
    const [buses, setBuses] = useState<BusStatus[]>([])
    const [loading, setLoading] = useState(true)
    const [error, setError] = useState('')

    useEffect(() => {
        loadFleet()
        const id = setInterval(loadFleet, 30_000)
        return () => clearInterval(id)
    }, [])

    const loadFleet = async () => {
        try {
            const res = await api.get<BusStatus[]>('/api/v1/occupancy/fleet')
            setBuses(res.data)
            setError('')
        } catch (err: any) {
            setError(err.response?.data?.detail || 'Failed to load fleet')
        } finally {
            setLoading(false)
        }
    }

    return (
        <div className="space-y-6">
            <div className="flex items-center justify-between">
                <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Bus Fleet</h1>
                <button onClick={loadFleet} className="btn btn-secondary text-sm">
                    Refresh
                </button>
            </div>

            {error && (
                <div className="p-3 bg-red-50 dark:bg-red-900/20 border border-red-200 rounded-lg text-red-600 text-sm">
                    {error}
                </div>
            )}

            {loading ? (
                <p className="text-gray-500">Loading fleet…</p>
            ) : buses.length === 0 ? (
                <p className="text-gray-500">No buses registered.</p>
            ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    {buses.map((bus) => {
                        const lastSeen = bus.last_seen
                            ? new Date(bus.last_seen).toLocaleTimeString()
                            : 'Never'
                        const isOffline =
                            !bus.last_seen ||
                            Date.now() - new Date(bus.last_seen).getTime() > 5 * 60_000

                        return (
                            <div key={bus.bus_id} className="card p-5 space-y-3">
                                <div className="flex items-center justify-between">
                                    <div>
                                        <p className="font-semibold text-gray-900 dark:text-white">
                                            {bus.bus_id}
                                        </p>
                                        <p className="text-xs text-gray-500">{bus.plate_number}</p>
                                    </div>
                                    <span
                                        className={`px-2 py-1 text-xs rounded-full font-medium ${
                                            isOffline
                                                ? 'bg-gray-100 text-gray-600 dark:bg-gray-700 dark:text-gray-400'
                                                : (STATUS_CLASSES[bus.status] ?? STATUS_CLASSES.green)
                                        }`}
                                    >
                                        {isOffline ? 'Offline' : bus.status.toUpperCase()}
                                    </span>
                                </div>

                                {bus.route_number && (
                                    <p className="text-xs text-gray-500">Route {bus.route_number}</p>
                                )}

                                <div>
                                    <div className="flex justify-between text-sm">
                                        <span className="text-gray-600 dark:text-gray-400">
                                            {bus.current_count} / {bus.capacity} passengers
                                        </span>
                                        <span className="font-medium text-gray-900 dark:text-white">
                                            {Math.round(bus.occupancy_ratio * 100)}%
                                        </span>
                                    </div>
                                    <OccupancyBar ratio={bus.occupancy_ratio} status={bus.status} />
                                </div>

                                <p className="text-xs text-gray-400">Last seen: {lastSeen}</p>
                            </div>
                        )
                    })}
                </div>
            )}
        </div>
    )
}
