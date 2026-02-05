/**
 * Route Buses Page
 * ================
 * 
 * Shows all buses on a selected route with occupancy data.
 */

import React, { useEffect, useState, useCallback } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { publicApi, BusOccupancy } from '../../services/publicApi';
import { BusCard } from '../../components/public/BusCard';
import { LiveIndicator } from '../../components/public/LiveIndicator';

export const RouteBuses: React.FC = () => {
    const { routeId } = useParams<{ routeId: string }>();
    const [buses, setBuses] = useState<BusOccupancy[]>([]);
    const [routeName, setRouteName] = useState<string>('');
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const navigate = useNavigate();

    const fetchBuses = useCallback(async () => {
        if (!routeId) return;

        try {
            const data = await publicApi.getRouteBuses(parseInt(routeId));
            setBuses(data.buses);
            setRouteName(data.route_name);
            setError(null);
        } catch (err) {
            setError('Failed to load buses');
            console.error(err);
        } finally {
            setLoading(false);
        }
    }, [routeId]);

    useEffect(() => {
        fetchBuses();

        // Auto-refresh every 5 seconds
        const interval = setInterval(fetchBuses, 5000);
        return () => clearInterval(interval);
    }, [fetchBuses]);

    if (loading) {
        return (
            <div className="min-h-screen bg-gray-50 flex items-center justify-center">
                <div className="text-center">
                    <div className="animate-spin rounded-full h-12 w-12 border-4 border-blue-600 border-t-transparent mx-auto mb-4"></div>
                    <p className="text-gray-600">Loading buses...</p>
                </div>
            </div>
        );
    }

    return (
        <div className="min-h-screen bg-gray-50">
            {/* Header */}
            <header className="bg-gradient-to-r from-blue-600 to-blue-700 text-white p-4 shadow-lg sticky top-0 z-10">
                <div className="max-w-lg mx-auto">
                    <button
                        onClick={() => navigate('/public')}
                        className="text-sm text-blue-100 hover:text-white mb-2 flex items-center gap-1"
                    >
                        ← Back to routes
                    </button>
                    <div className="flex items-center justify-between">
                        <div>
                            <h1 className="text-xl font-bold">{routeName || `Route ${routeId}`}</h1>
                            <p className="text-sm text-blue-100">
                                {buses.length} {buses.length === 1 ? 'bus' : 'buses'} active
                            </p>
                        </div>
                        <LiveIndicator />
                    </div>
                </div>
            </header>

            {/* Main Content */}
            <main className="max-w-lg mx-auto p-4">
                {error && (
                    <div className="bg-red-100 border border-red-300 text-red-700 px-4 py-3 rounded-lg mb-4">
                        {error}
                        <button
                            onClick={fetchBuses}
                            className="ml-2 underline hover:no-underline"
                        >
                            Retry
                        </button>
                    </div>
                )}

                {buses.length === 0 ? (
                    <div className="text-center py-12">
                        <span className="text-6xl mb-4 block">🚌</span>
                        <p className="text-gray-600 text-lg">No buses currently on this route</p>
                        <button
                            onClick={() => navigate('/public')}
                            className="mt-4 text-blue-600 hover:underline"
                        >
                            View other routes
                        </button>
                    </div>
                ) : (
                    <div className="space-y-3">
                        {buses.map((bus) => (
                            <BusCard
                                key={bus.bus_id}
                                bus={bus}
                                showRoute={false}
                                onClick={() => navigate(`/public/bus/${bus.bus_id}`)}
                            />
                        ))}
                    </div>
                )}

                {/* Legend */}
                {buses.length > 0 && (
                    <div className="mt-6 bg-white rounded-xl p-4 shadow-sm border border-gray-100">
                        <h3 className="font-semibold text-gray-700 mb-2 text-sm">Status Legend</h3>
                        <div className="grid grid-cols-3 gap-2 text-xs">
                            <div className="flex items-center gap-1.5">
                                <span className="w-3 h-3 bg-green-500 rounded-full"></span>
                                <span>0-60%</span>
                            </div>
                            <div className="flex items-center gap-1.5">
                                <span className="w-3 h-3 bg-yellow-500 rounded-full"></span>
                                <span>61-80%</span>
                            </div>
                            <div className="flex items-center gap-1.5">
                                <span className="w-3 h-3 bg-red-500 rounded-full"></span>
                                <span>81-100%</span>
                            </div>
                        </div>
                    </div>
                )}
            </main>
        </div>
    );
};

export default RouteBuses;
