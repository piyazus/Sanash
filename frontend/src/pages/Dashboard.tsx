import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { api } from '../services/api'
import {
    UsersIcon,
    VideoCameraIcon,
    ExclamationTriangleIcon,
    ArrowTrendingUpIcon,
} from '@heroicons/react/24/outline'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'

interface DashboardStats {
    currentOccupancy: number
    totalPeopleToday: number
    activeJobs: number
    pendingAlerts: number
}

interface ChartData {
    time: string
    occupancy: number
}

export default function Dashboard() {
    const [stats, setStats] = useState<DashboardStats | null>(null)
    const [chartData, setChartData] = useState<ChartData[]>([])
    const [loading, setLoading] = useState(true)

    useEffect(() => {
        loadDashboardData()
    }, [])

    const loadDashboardData = async () => {
        try {
            const response = await api.get('/api/v1/analytics/dashboard')
            setStats(response.data)

            // Generate sample chart data
            const mockData = Array.from({ length: 24 }, (_, i) => ({
                time: `${i}:00`,
                occupancy: Math.floor(Math.random() * 30) + 10,
            }))
            setChartData(mockData)
        } catch (error) {
            console.error('Failed to load dashboard data:', error)
        } finally {
            setLoading(false)
        }
    }

    const statCards = [
        { name: 'Current Occupancy', value: stats?.currentOccupancy ?? '-', icon: UsersIcon, color: 'bg-blue-500' },
        { name: 'Total Today', value: stats?.totalPeopleToday ?? '-', icon: ArrowTrendingUpIcon, color: 'bg-green-500' },
        { name: 'Active Jobs', value: stats?.activeJobs ?? '-', icon: VideoCameraIcon, color: 'bg-purple-500' },
        { name: 'Pending Alerts', value: stats?.pendingAlerts ?? '-', icon: ExclamationTriangleIcon, color: 'bg-orange-500' },
    ]

    return (
        <div className="space-y-6">
            <div className="flex items-center justify-between">
                <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Dashboard</h1>
                <Link to="/jobs" className="btn btn-primary">
                    New Detection Job
                </Link>
            </div>

            {/* Stats Grid */}
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

            {/* Occupancy Chart */}
            <div className="card p-6">
                <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">Occupancy Trend</h2>
                <div className="h-80">
                    <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={chartData}>
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis dataKey="time" />
                            <YAxis />
                            <Tooltip />
                            <Line type="monotone" dataKey="occupancy" stroke="#6366f1" strokeWidth={2} dot={false} />
                        </LineChart>
                    </ResponsiveContainer>
                </div>
            </div>

            {/* Recent Activity */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div className="card p-6">
                    <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">Recent Jobs</h2>
                    <div className="space-y-3">
                        <p className="text-gray-500 dark:text-gray-400 text-sm">No recent jobs</p>
                    </div>
                </div>

                <div className="card p-6">
                    <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">Recent Alerts</h2>
                    <div className="space-y-3">
                        <p className="text-gray-500 dark:text-gray-400 text-sm">No alerts</p>
                    </div>
                </div>
            </div>
        </div>
    )
}
