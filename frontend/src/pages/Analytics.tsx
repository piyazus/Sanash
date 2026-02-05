import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar } from 'recharts'

export default function Analytics() {
    // Sample data
    const occupancyData = Array.from({ length: 30 }, (_, i) => ({
        date: `Jan ${i + 1}`,
        avg: Math.floor(Math.random() * 20) + 15,
        peak: Math.floor(Math.random() * 15) + 30,
    }))

    const hourlyData = Array.from({ length: 24 }, (_, i) => ({
        hour: `${i}:00`,
        passengers: Math.floor(Math.random() * 50) + 10,
    }))

    return (
        <div className="space-y-6">
            <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Analytics</h1>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Occupancy Trend */}
                <div className="card p-6">
                    <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">Occupancy Trend (30 days)</h2>
                    <div className="h-80">
                        <ResponsiveContainer width="100%" height="100%">
                            <LineChart data={occupancyData}>
                                <CartesianGrid strokeDasharray="3 3" />
                                <XAxis dataKey="date" />
                                <YAxis />
                                <Tooltip />
                                <Line type="monotone" dataKey="avg" stroke="#6366f1" strokeWidth={2} name="Average" />
                                <Line type="monotone" dataKey="peak" stroke="#ec4899" strokeWidth={2} name="Peak" />
                            </LineChart>
                        </ResponsiveContainer>
                    </div>
                </div>

                {/* Hourly Distribution */}
                <div className="card p-6">
                    <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">Hourly Distribution</h2>
                    <div className="h-80">
                        <ResponsiveContainer width="100%" height="100%">
                            <BarChart data={hourlyData}>
                                <CartesianGrid strokeDasharray="3 3" />
                                <XAxis dataKey="hour" />
                                <YAxis />
                                <Tooltip />
                                <Bar dataKey="passengers" fill="#6366f1" radius={[4, 4, 0, 0]} />
                            </BarChart>
                        </ResponsiveContainer>
                    </div>
                </div>
            </div>

            {/* Summary Stats */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                {[
                    { label: 'Total Passengers', value: '12,450' },
                    { label: 'Avg Daily', value: '415' },
                    { label: 'Peak Hour', value: '8:00 AM' },
                    { label: 'Busiest Day', value: 'Monday' },
                ].map((stat) => (
                    <div key={stat.label} className="card p-6 text-center">
                        <p className="text-sm text-gray-500">{stat.label}</p>
                        <p className="text-2xl font-bold text-gray-900 dark:text-white mt-1">{stat.value}</p>
                    </div>
                ))}
            </div>
        </div>
    )
}
