import { useEffect, useState } from 'react'
import {
    BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
    LineChart, Line, Legend,
} from 'recharts'
import { api } from '../services/api'

interface PeakHour {
    hour: number
    avg_ratio: number
    sample_count: number
}

interface FleetSummary {
    total_buses: number
    green: number
    yellow: number
    red: number
    offline: number
}

export default function Analytics() {
    const [peakData, setPeakData] = useState<PeakHour[]>([])
    const [summary, setSummary] = useState<FleetSummary | null>(null)
    const [days, setDays] = useState(7)
    const [loading, setLoading] = useState(true)

    useEffect(() => {
        load()
    }, [days])

    const load = async () => {
        setLoading(true)
        try {
            const [peakRes, summaryRes] = await Promise.all([
                api.get<PeakHour[]>(`/api/v1/analytics/peak?days=${days}`),
                api.get<FleetSummary>('/api/v1/analytics/summary'),
            ])
            setPeakData(peakRes.data)
            setSummary(summaryRes.data)
        } catch (err) {
            console.error('Analytics load error:', err)
        } finally {
            setLoading(false)
        }
    }

    const chartData = peakData.map((d) => ({
        hour: `${d.hour}:00`,
        'Avg Occupancy (%)': Math.round(d.avg_ratio * 100),
        samples: d.sample_count,
    }))

    // Find peak hour
    const peakEntry = peakData.reduce<PeakHour | null>(
        (best, d) => (!best || d.avg_ratio > best.avg_ratio ? d : best),
        null
    )

    return (
        <div className="space-y-6">
            <div className="flex items-center justify-between">
                <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Analytics</h1>
                <select
                    value={days}
                    onChange={(e) => setDays(Number(e.target.value))}
                    className="input w-auto text-sm"
                >
                    <option value={1}>Last 24 h</option>
                    <option value={7}>Last 7 days</option>
                    <option value={14}>Last 14 days</option>
                    <option value={30}>Last 30 days</option>
                </select>
            </div>

            {/* Summary stats */}
            {summary && (
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    {[
                        { label: 'Total Buses', value: summary.total_buses, color: 'text-blue-600' },
                        { label: 'Normal', value: summary.green, color: 'text-green-600' },
                        { label: 'Busy', value: summary.yellow, color: 'text-yellow-600' },
                        { label: 'Overcrowded', value: summary.red, color: 'text-red-600' },
                    ].map((s) => (
                        <div key={s.label} className="card p-5 text-center">
                            <p className="text-xs text-gray-500 dark:text-gray-400">{s.label}</p>
                            <p className={`text-3xl font-bold mt-1 ${s.color}`}>{s.value}</p>
                        </div>
                    ))}
                </div>
            )}

            {loading ? (
                <p className="text-gray-500">Loading analytics…</p>
            ) : (
                <>
                    {/* Peak hour bar chart */}
                    <div className="card p-6">
                        <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-1">
                            Average Occupancy by Hour of Day
                        </h2>
                        <p className="text-xs text-gray-500 mb-4">
                            Aggregated over the last {days} day{days > 1 ? 's' : ''} across all buses
                        </p>
                        <div className="h-72">
                            <ResponsiveContainer width="100%" height="100%">
                                <BarChart data={chartData}>
                                    <CartesianGrid strokeDasharray="3 3" />
                                    <XAxis dataKey="hour" tick={{ fontSize: 11 }} />
                                    <YAxis unit="%" domain={[0, 100]} />
                                    <Tooltip formatter={(v: number) => `${v}%`} />
                                    <Bar dataKey="Avg Occupancy (%)" fill="#6366f1" radius={[4, 4, 0, 0]} />
                                </BarChart>
                            </ResponsiveContainer>
                        </div>
                    </div>

                    {/* Sample-count line chart */}
                    <div className="card p-6">
                        <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                            Reading Volume by Hour
                        </h2>
                        <div className="h-56">
                            <ResponsiveContainer width="100%" height="100%">
                                <LineChart data={chartData}>
                                    <CartesianGrid strokeDasharray="3 3" />
                                    <XAxis dataKey="hour" tick={{ fontSize: 11 }} />
                                    <YAxis />
                                    <Tooltip />
                                    <Legend />
                                    <Line
                                        type="monotone"
                                        dataKey="samples"
                                        stroke="#ec4899"
                                        strokeWidth={2}
                                        dot={false}
                                        name="Readings"
                                    />
                                </LineChart>
                            </ResponsiveContainer>
                        </div>
                    </div>

                    {/* Key insight */}
                    {peakEntry && (
                        <div className="card p-6 flex items-center gap-4">
                            <div className="text-4xl">⏰</div>
                            <div>
                                <p className="text-sm text-gray-500">Peak Hour</p>
                                <p className="text-2xl font-bold text-gray-900 dark:text-white">
                                    {peakEntry.hour}:00 — {(peakEntry.hour + 1) % 24}:00
                                </p>
                                <p className="text-sm text-gray-500">
                                    Average occupancy: {Math.round(peakEntry.avg_ratio * 100)}%
                                </p>
                            </div>
                        </div>
                    )}
                </>
            )}
        </div>
    )
}
