/**
 * Nearby Buses Page
 * =================
 * 
 * Shows buses near user's current location.
 */

import React, { useEffect, useState, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { useGeolocation } from '../../hooks/useGeolocation';
import { publicApi, BusOccupancy } from '../../services/publicApi';
import { BusCard } from '../../components/public/BusCard';
import { LiveIndicator } from '../../components/public/LiveIndicator';

export const NearbyBuses: React.FC = () => {
    const { location, loading: geoLoading, error: geoError, refresh: refreshLocation } = useGeolocation();
    const [buses, setBuses] = useState<BusOccupancy[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [radius, setRadius] = useState(2.0); // 2km default
    const navigate = useNavigate();

    const fetchNearbyBuses = useCallback(async () => {
        if (!location) return;

        try {
            const data = await publicApi.getNearbyBuses(location.lat, location.lon, radius);
            setBuses(data.buses);
            setError(null);
        } catch (err) {
            setError('Failed to find nearby buses');
            console.error(err);
        } finally {
            setLoading(false);
        }
    }, [location, radius]);

    useEffect(() => {
        if (location) {
            fetchNearbyBuses();

            // Auto-refresh every 10 seconds
            const interval = setInterval(fetchNearbyBuses, 10000);
            return () => clearInterval(interval);
        }
    }, [location, fetchNearbyBuses]);

    // Loading state for geolocation
    if (geoLoading) {
        return (
            <div className="min-h-screen bg-gray-50 flex items-center justify-center p-4">
                <div className="text-center">
                    <div className="animate-pulse text-6xl mb-4">📍</div>
                    <p className="text-gray-600 text-lg">Getting your location...</p>
                    <p className="text-gray-400 text-sm mt-2">Please allow location access</p>
                </div>
            </div>
        );
    }

    // Geolocation error
    if (geoError) {
        return (
            <div className="min-h-screen bg-gray-50 flex items-center justify-center p-4">
                <div className="text-center max-w-sm">
                    <span className="text-6xl mb-4 block">📍</span>
                    <p className="text-red-600 text-lg mb-4">{geoError}</p>
                    <div className="space-y-3">
                        <button
                            onClick={refreshLocation}
                            className="w-full bg-blue-600 text-white py-3 rounded-xl font-semibold hover:bg-blue-700 transition-colors"
                        >
                            Try Again
                        </button>
                        <button
                            onClick={() => navigate('/public')}
                            className="w-full text-blue-600 hover:underline"
                        >
                            Browse routes instead
                        </button>
                    </div>
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
                            <h1 className="text-xl font-bold flex items-center gap-2">
                                <span>📍</span>
                                Buses Near You
                            </h1>
                            <p className="text-sm text-blue-100">
                                Within {radius}km • {buses.length} found
                            </p>
                        </div>
                        <LiveIndicator />
                    </div>
                </div>
            </header>

            {/* Main Content */}
            <main className="max-w-lg mx-auto p-4">
                {/* Radius Selector */}
                <div className="bg-white rounded-xl p-4 mb-4 shadow-sm border border-gray-100">
                    <label className="text-sm font-medium text-gray-700 block mb-2">
                        Search Radius
                    </label>
                    <div className="flex gap-2">
                        {[1, 2, 5, 10].map((r) => (
                            <button
                                key={r}
                                onClick={() => setRadius(r)}
                                className={`flex-1 py-2 rounded-lg text-sm font-medium transition-colors ${radius === r
                                        ? 'bg-blue-600 text-white'
                                        : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                                    }`}
                            >
                                {r}km
                            </button>
                        ))}
                    </div>
                </div>

                {error && (
                    <div className="bg-red-100 border border-red-300 text-red-700 px-4 py-3 rounded-lg mb-4">
                        {error}
                        <button
                            onClick={fetchNearbyBuses}
                            className="ml-2 underline hover:no-underline"
                        >
                            Retry
                        </button>
                    </div>
                )}

                {loading ? (
                    <div className="text-center py-12">
                        <div className="animate-spin rounded-full h-10 w-10 border-4 border-blue-600 border-t-transparent mx-auto mb-4"></div>
                        <p className="text-gray-600">Finding nearby buses...</p>
                    </div>
                ) : buses.length === 0 ? (
                    <div className="text-center py-12">
                        <span className="text-6xl mb-4 block">🚌</span>
                        <p className="text-gray-600 text-lg">No buses found within {radius}km</p>
                        <p className="text-gray-400 text-sm mt-2">
                            Try increasing the search radius
                        </p>
                    </div>
                ) : (
                    <div className="space-y-3">
                        {buses.map((bus) => (
                            <BusCard
                                key={bus.bus_id}
                                bus={bus}
                                showRoute={true}
                                showDistance={true}
                                onClick={() => navigate(`/public/bus/${bus.bus_id}`)}
                            />
                        ))}
                    </div>
                )}

                {/* Location Info */}
                {location && (
                    <div className="mt-6 text-center text-sm text-gray-500">
                        <p>
                            Your location: {location.lat.toFixed(4)}, {location.lon.toFixed(4)}
                        </p>
                        <button
                            onClick={refreshLocation}
                            className="mt-2 text-blue-600 hover:underline"
                        >
                            Update location
                        </button>
                    </div>
                )}
            </main>
        </div>
    );
};

export default NearbyBuses;
