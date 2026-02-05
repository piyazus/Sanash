import { useState, useEffect } from 'react'
import { api } from '../services/api'
import { PlusIcon } from '@heroicons/react/24/outline'

interface Bus {
    id: number
    number: string
    status: string
    camerasCount: number
}

export default function Buses() {
    const [buses, setBuses] = useState<Bus[]>([])
    const [loading, setLoading] = useState(true)

    useEffect(() => {
        loadBuses()
    }, [])

    const loadBuses = async () => {
        try {
            const response = await api.get('/api/v1/buses')
            setBuses(response.data)
        } catch (error) {
            console.error('Failed to load buses:', error)
        } finally {
            setLoading(false)
        }
    }

    return (
        <div className="space-y-6">
            <div className="flex items-center justify-between">
                <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Bus Fleet</h1>
                <button className="btn btn-primary">
                    <PlusIcon className="w-5 h-5 mr-2" />
                    Add Bus
                </button>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {loading ? (
                    <p className="text-gray-500">Loading...</p>
                ) : buses.length === 0 ? (
                    <p className="text-gray-500">No buses registered</p>
                ) : (
                    buses.map((bus) => (
                        <div key={bus.id} className="card p-6">
                            <div className="flex items-center justify-between mb-4">
                                <h3 className="text-lg font-semibold text-gray-900 dark:text-white">Bus #{bus.number}</h3>
                                <span className={`px-2 py-1 text-xs rounded-full ${bus.status === 'active' ? 'bg-green-100 text-green-800' : 'bg-gray-100 text-gray-800'
                                    }`}>
                                    {bus.status}
                                </span>
                            </div>
                            <p className="text-sm text-gray-500">{bus.camerasCount || 0} cameras installed</p>
                        </div>
                    ))
                )}
            </div>
        </div>
    )
}
