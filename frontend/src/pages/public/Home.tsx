/**
 * Public Home Page
 * ================
 * 
 * Route selection page - entry point for public users.
 */

import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { publicApi, Route } from '../../services/publicApi';

export const Home: React.FC = () => {
    const [routes, setRoutes] = useState<Route[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const navigate = useNavigate();

    useEffect(() => {
        const fetchRoutes = async () => {
            try {
                const data = await publicApi.getAllRoutes();
                setRoutes(data.routes);
                setError(null);
            } catch (err) {
                setError('Failed to load routes');
                console.error(err);
            } finally {
                setLoading(false);
            }
        };

        fetchRoutes();
    }, []);

    const getStatusColor = (percentage: number) => {
        if (percentage <= 60) return 'bg-green-500';
        if (percentage <= 80) return 'bg-yellow-500';
        return 'bg-red-500';
    };

    if (loading) {
        return (
            <div className="min-h-screen bg-gradient-to-b from-blue-600 to-blue-800 flex items-center justify-center">
                <div className="text-white text-center">
                    <div className="animate-spin rounded-full h-12 w-12 border-4 border-white border-t-transparent mx-auto mb-4"></div>
                    <p className="text-lg">Loading routes...</p>
                </div>
            </div>
        );
    }

    return (
        <div className="min-h-screen bg-gray-50">
            {/* Header */}
            <header className="bg-gradient-to-r from-blue-600 to-blue-700 text-white p-6 shadow-lg">
                <div className="max-w-lg mx-auto">
                    <div className="flex items-center gap-3 mb-2">
                        <span className="text-4xl">🚌</span>
                        <h1 className="text-2xl font-bold">Bus Tracker</h1>
                    </div>
                    <p className="text-blue-100">Real-time bus occupancy information</p>
                </div>
            </header>

            {/* Main Content */}
            <main className="max-w-lg mx-auto p-4">
                {error && (
                    <div className="bg-red-100 border border-red-300 text-red-700 px-4 py-3 rounded-lg mb-4">
                        {error}
                    </div>
                )}

                <h2 className="text-lg font-semibold text-gray-700 mb-4">Select a Route</h2>

                {/* Route List */}
                <div className="space-y-3">
                    {routes.length === 0 ? (
                        <p className="text-center text-gray-500 py-8">No routes available</p>
                    ) : (
                        routes.map((route) => (
                            <div
                                key={route.route_id}
                                className="bg-white rounded-xl shadow-md p-4 cursor-pointer hover:shadow-lg transition-all duration-200 hover:scale-[1.01] active:scale-[0.99] border border-gray-100"
                                onClick={() => navigate(`/public/route/${route.route_id}`)}
                            >
                                <h3 className="font-bold text-lg text-gray-800 mb-2">
                                    {route.route_name}
                                </h3>
                                <div className="flex items-center justify-between">
                                    <div className="flex items-center gap-4 text-sm text-gray-600">
                                        <span className="flex items-center gap-1">
                                            🚌 <span className="font-medium">{route.total_buses}</span> buses
                                        </span>
                                        <span className="flex items-center gap-1">
                                            <span
                                                className={`w-2.5 h-2.5 rounded-full ${getStatusColor(route.avg_occupancy_percentage)}`}
                                            ></span>
                                            <span className="font-medium">{route.avg_occupancy_percentage}%</span> avg
                                        </span>
                                    </div>
                                    <span className="text-gray-400">→</span>
                                </div>
                            </div>
                        ))
                    )}
                </div>

                {/* Nearby Buses Button */}
                <button
                    className="mt-6 w-full bg-gradient-to-r from-blue-600 to-blue-700 text-white py-4 rounded-xl font-semibold shadow-lg hover:shadow-xl transition-all duration-200 flex items-center justify-center gap-2 active:scale-[0.98]"
                    onClick={() => navigate('/public/nearby')}
                >
                    <span className="text-xl">📍</span>
                    Find Buses Near Me
                </button>

                {/* Footer */}
                <footer className="mt-8 text-center text-sm text-gray-500">
                    <p>Data updates every 5 seconds</p>
                    <p className="mt-1">© 2026 Bus Vision</p>
                </footer>
            </main>
        </div>
    );
};

export default Home;
