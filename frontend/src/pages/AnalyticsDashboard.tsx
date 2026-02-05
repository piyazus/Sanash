import { useEffect, useState } from 'react'
import { api } from '../services/api'
import {
    ChartBarIcon,
    ArrowTrendingUpIcon,
    ClockIcon,
    ExclamationTriangleIcon
} from '@heroicons/react/24/outline'
import {
    LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid,
    Tooltip, ResponsiveContainer, AreaChart, Area, PieChart, Pie, Cell
} from 'recharts'

interface AnalyticsData {
    occupancyTrend: { time: string; count: number }[]
    peakHours: { hour: number; avgOccupancy: number }[]
    cameraStats: { name: string; detections: number; handoffs: number }[]
    flowData: { from: string; to: string; count: number }[]
    summary: {
        totalPeople: number
        avgDwellTime: number
        peakOccupancy: number
        peakTime: string
    }
}

const COLORS = ['#6366f1', '#22c55e', '#f59e0b', '#ef4444', '#8b5cf6', '#06b6d4']

export default function AnalyticsDashboard() {
    const [data, setData] = useState<AnalyticsData | null>(null)
    const [dateRange, setDateRange] = useState<'day' | 'week' | 'month'>('day')
    const [loading, setLoading] = useState(true)

    useEffect(() => {
        loadAnalytics()
    }, [dateRange])

    const loadAnalytics = async () => {
        try {
            setLoading(true)
            const response = await api.get('/api/v1/analytics/dashboard', {
                params: { range: dateRange }
            })
            setData(response.data)
        } catch (error) {
            console.error('Failed to load analytics:', error)
            // Mock data for demo
            setData({
                occupancyTrend: Array.from({ length: 24 }, (_, i) => ({
                    time: `${i}:00`,
                    count: Math.floor(Math.random() * 40) + 10
                })),
                peakHours: Array.from({ length: 24 }, (_, i) => ({
                    hour: i,
                    avgOccupancy: Math.floor(Math.random() * 50) + 5
                })),
                cameraStats: [
                    { name: 'Front', detections: 1250, handoffs: 180 },
                    { name: 'Middle', detections: 2100, handoffs: 320 },
                    { name: 'Rear', detections: 1800, handoffs: 250 },
                    { name: 'Door', detections: 950, handoffs: 0 }
                ],
                flowData: [
                    { from: 'Front', to: 'Middle', count: 145 },
                    { from: 'Middle', to: 'Rear', count: 120 },
                    { from: 'Rear', to: 'Middle', count: 85 },
                    { from: 'Middle', to: 'Front', count: 90 }
                ],
                summary: {
                    totalPeople: 342,
                    avgDwellTime: 847,
                    peakOccupancy: 48,
                    peakTime: '08:30'
                }
            })
        } finally {
            setLoading(false)
        }
    }

    const formatDwellTime = (seconds: number) => {
        const mins = Math.floor(seconds / 60)
        return `${mins}m ${Math.floor(seconds % 60)}s`
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
                <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Analytics</h1>
                <div className="flex gap-2">
                    {(['day', 'week', 'month'] as const).map((range) => (
                        <button
                            key={range}
                            onClick={() => setDateRange(range)}
                            className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${dateRange === range
                                ? 'bg-primary-600 text-white'
                                : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200'
                                }`}
                        >
                            {range.charAt(0).toUpperCase() + range.slice(1)}
                        </button>
                    ))}
                </div>
            </div>

            {/* Summary Cards */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <div className="card p-6">
                    <div className="flex items-center gap-4">
                        <div className="bg-blue-100 dark:bg-blue-900 p-3 rounded-lg">
                            <ChartBarIcon className="w-6 h-6 text-blue-600 dark:text-blue-400" />
                        </div>
                        <div>
                            <p className="text-sm text-gray-500 dark:text-gray-400">Total People</p>
                            <p className="text-2xl font-bold text-gray-900 dark:text-white">
                                {data?.summary.totalPeople.toLocaleString()}
                            </p>
                        </div>
                    </div>
                </div>

                <div className="card p-6">
                    <div className="flex items-center gap-4">
                        <div className="bg-green-100 dark:bg-green-900 p-3 rounded-lg">
                            <ArrowTrendingUpIcon className="w-6 h-6 text-green-600 dark:text-green-400" />
                        </div>
                        <div>
                            <p className="text-sm text-gray-500 dark:text-gray-400">Peak Occupancy</p>
                            <p className="text-2xl font-bold text-gray-900 dark:text-white">
                                {data?.summary.peakOccupancy}
                            </p>
                        </div>
                    </div>
                </div>

                <div className="card p-6">
                    <div className="flex items-center gap-4">
                        <div className="bg-purple-100 dark:bg-purple-900 p-3 rounded-lg">
                            <ClockIcon className="w-6 h-6 text-purple-600 dark:text-purple-400" />
                        </div>
                        <div>
                            <p className="text-sm text-gray-500 dark:text-gray-400">Avg Dwell Time</p>
                            <p className="text-2xl font-bold text-gray-900 dark:text-white">
                                {formatDwellTime(data?.summary.avgDwellTime || 0)}
                            </p>
                        </div>
                    </div>
                </div>

                <div className="card p-6">
                    <div className="flex items-center gap-4">
                        <div className="bg-orange-100 dark:bg-orange-900 p-3 rounded-lg">
                            <ExclamationTriangleIcon className="w-6 h-6 text-orange-600 dark:text-orange-400" />
                        </div>
                        <div>
                            <p className="text-sm text-gray-500 dark:text-gray-400">Peak Time</p>
                            <p className="text-2xl font-bold text-gray-900 dark:text-white">
                                {data?.summary.peakTime}
                            </p>
                        </div>
                    </div>
                </div>
            </div>

            {/* Charts Row 1 */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Occupancy Trend */}
                <div className="card p-6">
                    <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                        Occupancy Trend
                    </h2>
                    <div className="h-72">
                        <ResponsiveContainer width="100%" height="100%">
                            <AreaChart data={data?.occupancyTrend}>
                                <CartesianGrid strokeDasharray="3 3" />
                                <XAxis dataKey="time" />
                                <YAxis />
                                <Tooltip />
                                <Area
                                    type="monotone"
                                    dataKey="count"
                                    stroke="#6366f1"
                                    fill="#6366f1"
                                    fillOpacity={0.3}
                                />
                            </AreaChart>
                        </ResponsiveContainer>
                    </div>
                </div>

                {/* Peak Hours */}
                <div className="card p-6">
                    <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                        Peak Hours Distribution
                    </h2>
                    <div className="h-72">
                        <ResponsiveContainer width="100%" height="100%">
                            <BarChart data={data?.peakHours}>
                                <CartesianGrid strokeDasharray="3 3" />
                                <XAxis dataKey="hour" />
                                <YAxis />
                                <Tooltip />
                                <Bar dataKey="avgOccupancy" fill="#22c55e" radius={[4, 4, 0, 0]} />
                            </BarChart>
                        </ResponsiveContainer>
                    </div>
                </div>
            </div>

            {/* Charts Row 2 */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Camera Performance */}
                <div className="card p-6">
                    <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                        Camera Performance
                    </h2>
                    <div className="h-72">
                        <ResponsiveContainer width="100%" height="100%">
                            <BarChart data={data?.cameraStats} layout="vertical">
                                <CartesianGrid strokeDasharray="3 3" />
                                <XAxis type="number" />
                                <YAxis dataKey="name" type="category" width={80} />
                                <Tooltip />
                                <Bar dataKey="detections" fill="#6366f1" name="Detections" />
                                <Bar dataKey="handoffs" fill="#22c55e" name="Handoffs" />
                            </BarChart>
                        </ResponsiveContainer>
                    </div>
                </div>

                {/* Flow Distribution */}
                <div className="card p-6">
                    <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                        Camera Flow Distribution
                    </h2>
                    <div className="h-72">
                        <ResponsiveContainer width="100%" height="100%">
                            <PieChart>
                                <Pie
                                    data={data?.flowData}
                                    dataKey="count"
                                    nameKey="from"
                                    cx="50%"
                                    cy="50%"
                                    outerRadius={100}
                                    label={({ from, to, count }) => `${from}→${to}: ${count}`}
                                >
                                    {data?.flowData.map((_, index) => (
                                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                                    ))}
                                </Pie>
                                <Tooltip />
                            </PieChart>
                        </ResponsiveContainer>
                    </div>
                </div>
            </div>

            {/* Flow Table */}
            <div className="card p-6">
                <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                    Camera Transition Flow
                </h2>
                <div className="overflow-x-auto">
                    <table className="w-full">
                        <thead className="bg-gray-50 dark:bg-gray-700">
                            <tr>
                                <th className="px-4 py-3 text-left text-sm font-medium text-gray-500 dark:text-gray-300">From Camera</th>
                                <th className="px-4 py-3 text-left text-sm font-medium text-gray-500 dark:text-gray-300">To Camera</th>
                                <th className="px-4 py-3 text-left text-sm font-medium text-gray-500 dark:text-gray-300">Transitions</th>
                                <th className="px-4 py-3 text-left text-sm font-medium text-gray-500 dark:text-gray-300">% of Total</th>
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
                            {data?.flowData.map((flow, idx) => {
                                const total = data.flowData.reduce((sum, f) => sum + f.count, 0)
                                return (
                                    <tr key={idx} className="hover:bg-gray-50 dark:hover:bg-gray-800">
                                        <td className="px-4 py-3 text-sm text-gray-900 dark:text-gray-100">{flow.from}</td>
                                        <td className="px-4 py-3 text-sm text-gray-900 dark:text-gray-100">{flow.to}</td>
                                        <td className="px-4 py-3 text-sm font-medium text-gray-900 dark:text-gray-100">{flow.count}</td>
                                        <td className="px-4 py-3">
                                            <div className="flex items-center gap-2">
                                                <div className="flex-1 bg-gray-200 dark:bg-gray-600 rounded-full h-2 max-w-[100px]">
                                                    <div
                                                        className="bg-primary-600 h-2 rounded-full"
                                                        style={{ width: `${(flow.count / total) * 100}%` }}
                                                    />
                                                </div>
                                                <span className="text-sm text-gray-500">
                                                    {((flow.count / total) * 100).toFixed(1)}%
                                                </span>
                                            </div>
                                        </td>
                                    </tr>
                                )
                            })}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    )
}
