import { useEffect, useState } from 'react'
import {
    TruckIcon,
    CheckCircleIcon,
    ExclamationCircleIcon,
    ExclamationTriangleIcon,
} from '@heroicons/react/24/outline'
import {
    BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell,
} from 'recharts'
import { api } from '../services/api'

interface FleetSummary {
    total_buses: number
    green: number
    yellow: number
    red: number
    offline: number
    timestamp: string
}

interface PeakHour {
    hour: number
    avg_ratio: number
    sample_count: number
}

interface ActiveAlert {
    id: string
    bus_id: string
    triggered_at: string
    occupancy_ratio: number
    message: string
}

const STATUS_COLOR: Record<string, string> = {
    green: '#22c55e',
    yellow: '#eab308',
    red: '#ef4444',
    offline: '#6b7280',
}

export default function Dashboard() {
    const [summary, setSummary] = useState<FleetSummary | null>(null)
    const [peakData, setPeakData] = useState<PeakHour[]>([])
    const [alerts, setAlerts] = useState<ActiveAlert[]>([])
    const [loading, setLoading] = useState(true)

    useEffect(() => {
        loadAll()
        const id = setInterval(loadAll, 30_000)
        return () => clearInterval(id)
    }, [])

    const loadAll = async () => {
        try {
            const [summaryRes, peakRes, alertsRes] = await Promise.all([
                api.get<FleetSummary>('/api/v1/analytics/summary'),
                api.get<PeakHour[]>('/api/v1/analytics/peak?days=7'),
                api.get<ActiveAlert[]>('/api/v1/alerts/?unresolved_only=true&limit=5'),
            ])
            setSummary(summaryRes.data)
            setPeakData(peakRes.data)
            setAlerts(alertsRes.data)
        } catch (err) {
            console.error('Dashboard load error:', err)
        } finally {
            setLoading(false)
        }
    }

    const statCards = summary
        ? [
            { name: 'Total Buses', value: summary.total_buses, icon: TruckIcon, color: 'bg-blue-500' },
            { name: 'Normal (Green)', value: summary.green, icon: CheckCircleIcon, color: 'bg-green-500' },
            { name: 'Busy (Yellow)', value: summary.yellow, icon: ExclamationTriangleIcon, color: 'bg-yellow-500' },
            { name: 'Overcrowded (Red)', value: summary.red, icon: ExclamationCircleIcon, color: 'bg-red-500' },
        ]
        : []

    const chartData = peakData.map((d) => ({
        hour: `${d.hour}:00`,
        occupancy: Math.round(d.avg_ratio * 100),
    }))

    return (
        <div className="space-y-6">
            <div className="flex items-center justify-between">
                <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Fleet Dashboard</h1>
                {summary && (
                    <span className="text-sm text-gray-500 dark:text-gray-400">
                        Live · {new Date(summary.timestamp).toLocaleTimeString()}
                    </span>
                )}
            </div>

            {loading ? (
                <p className="text-gray-500">Loading fleet data…</p>
            ) : (
                <>
                    {/* Stats */}
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                        {statCards.map((stat) => (
                            <div key={stat.name} className="card p-6">
                                <div className="flex items-center justify-between">
                                    <div>
                                        <p className="text-sm text-gray-500 dark:text-gray-400">{stat.name}</p>
                                        <p className="text-3xl font-bold text-gray-900 dark:text-white mt-1">{stat.value}</p>
                                    </div>
                                    <div className={`${stat.color} p-3 rounded-lg`}>
                                        <stat.icon className="w-6 h-6 text-white" />
                                    </div>
                                </div>
                            </div>
                        ))}
                    </div>

                    {/* Peak-hour chart */}
                    <div className="card p-6">
                        <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                            Average Occupancy by Hour (last 7 days)
                        </h2>
                        <div className="h-72">
                            <ResponsiveContainer width="100%" height="100%">
                                <BarChart data={chartData}>
                                    <CartesianGrid strokeDasharray="3 3" />
                                    <XAxis dataKey="hour" tick={{ fontSize: 11 }} />
                                    <YAxis unit="%" domain={[0, 100]} />
                                    <Tooltip formatter={(v: number) => `${v}%`} />
                                    <Bar dataKey="occupancy" radius={[4, 4, 0, 0]}>
                                        {chartData.map((d) => (
                                            <Cell
                                                key={d.hour}
                                                fill={
                                                    d.occupancy >= 80
                                                        ? STATUS_COLOR.red
                                                        : d.occupancy >= 50
                                                        ? STATUS_COLOR.yellow
                                                        : STATUS_COLOR.green
                                                }
                                            />
                                        ))}
                                    </Bar>
                                </BarChart>
                            </ResponsiveContainer>
                        </div>
                    </div>

                    {/* Active alerts */}
                    <div className="card p-6">
                        <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">Active Alerts</h2>
                        {alerts.length === 0 ? (
                            <p className="text-sm text-gray-500">No active alerts</p>
                        ) : (
                            <ul className="divide-y divide-gray-100 dark:divide-gray-700">
                                {alerts.map((alert) => (
                                    <li key={alert.id} className="py-3 flex items-center justify-between">
                                        <div>
                                            <p className="text-sm font-medium text-gray-900 dark:text-white">
                                                Bus {alert.bus_id}
                                            </p>
                                            <p className="text-xs text-gray-500">{alert.message}</p>
                                        </div>
                                        <span className="text-xs text-red-600 dark:text-red-400 font-semibold">
                                            {Math.round(alert.occupancy_ratio * 100)}%
                                        </span>
                                    </li>
                                ))}
                            </ul>
                        )}
                    </div>
                </>
            )}
        </div>
    )
}
