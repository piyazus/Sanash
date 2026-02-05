/**
 * Bus Detail Page
 * ===============
 * 
 * Detailed view of a single bus occupancy.
 */

import React from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { useOccupancy } from '../../hooks/useOccupancy';
import { OccupancyBar } from '../../components/public/OccupancyBar';
import { StatusBadge } from '../../components/public/StatusBadge';
import { LiveIndicator } from '../../components/public/LiveIndicator';

export const BusDetail: React.FC = () => {
    const { busId } = useParams<{ busId: string }>();
    const { data: bus, loading, error, refresh } = useOccupancy(parseInt(busId || '0'));
    const navigate = useNavigate();

    if (loading) {
        return (
            <div className="min-h-screen bg-gray-50 flex items-center justify-center">
                <div className="text-center">
                    <div className="animate-spin rounded-full h-12 w-12 border-4 border-blue-600 border-t-transparent mx-auto mb-4"></div>
                    <p className="text-gray-600">Loading bus info...</p>
                </div>
            </div>
        );
    }

    if (error || !bus) {
        return (
            <div className="min-h-screen bg-gray-50 flex items-center justify-center p-4">
                <div className="text-center">
                    <span className="text-6xl mb-4 block">😔</span>
                    <p className="text-red-600 text-lg mb-4">{error || 'Bus not found'}</p>
                    <button
                        onClick={() => navigate(-1)}
                        className="text-blue-600 hover:underline"
                    >
                        Go back
                    </button>
                </div>
            </div>
        );
    }

    const availableSeats = bus.capacity - bus.current_occupancy;

    return (
        <div className="min-h-screen bg-gray-50">
            {/* Header */}
            <header className="bg-gradient-to-r from-blue-600 to-blue-700 text-white p-4 shadow-lg">
                <div className="max-w-lg mx-auto">
                    <button
                        onClick={() => navigate(-1)}
                        className="text-sm text-blue-100 hover:text-white mb-2 flex items-center gap-1"
                    >
                        ← Back
                    </button>
                    <div className="flex items-center justify-between">
                        <div>
                            <h1 className="text-2xl font-bold flex items-center gap-2">
                                <span>🚌</span>
                                Bus {bus.bus_number}
                            </h1>
                            {bus.route_name && (
                                <p className="text-blue-100 text-sm">{bus.route_name}</p>
                            )}
                        </div>
                        <LiveIndicator />
                    </div>
                </div>
            </header>

            {/* Main Content */}
            <main className="max-w-lg mx-auto p-4">
                {/* Main Occupancy Card */}
                <div className="bg-white rounded-2xl shadow-lg p-6 mb-4 border border-gray-100">
                    {/* Status */}
                    <div className="flex items-center justify-between mb-6">
                        <div>
                            <p className="text-sm text-gray-500 mb-1">Current Occupancy</p>
                            <p className="text-4xl font-bold text-gray-800">
                                {bus.current_occupancy}
                                <span className="text-2xl text-gray-400">/{bus.capacity}</span>
                            </p>
                        </div>
                        <StatusBadge status={bus.status} color={bus.color} size="lg" />
                    </div>

                    {/* Progress Bar */}
                    <div className="mb-6">
                        <OccupancyBar percentage={bus.percentage} color={bus.color} height="lg" />
                    </div>

                    {/* Stats Grid */}
                    <div className="grid grid-cols-2 gap-4">
                        <div className="bg-gray-50 rounded-xl p-4 text-center">
                            <p className="text-3xl font-bold text-gray-800">{bus.current_occupancy}</p>
                            <p className="text-sm text-gray-500">Passengers</p>
                        </div>
                        <div className="bg-gray-50 rounded-xl p-4 text-center">
                            <p className="text-3xl font-bold text-gray-800">{availableSeats}</p>
                            <p className="text-sm text-gray-500">Available Seats</p>
                        </div>
                    </div>
                </div>

                {/* Status Guide */}
                <div className="bg-white rounded-xl shadow-sm p-4 border border-gray-100 mb-4">
                    <h3 className="font-semibold text-gray-700 mb-3">Status Guide</h3>
                    <div className="space-y-2 text-sm">
                        <div className="flex items-center gap-3">
                            <span className="w-4 h-4 bg-green-500 rounded"></span>
                            <span className="text-gray-600">
                                <strong>Green (0-60%)</strong> - Plenty of space available
                            </span>
                        </div>
                        <div className="flex items-center gap-3">
                            <span className="w-4 h-4 bg-yellow-500 rounded"></span>
                            <span className="text-gray-600">
                                <strong>Yellow (61-80%)</strong> - Getting full
                            </span>
                        </div>
                        <div className="flex items-center gap-3">
                            <span className="w-4 h-4 bg-red-500 rounded"></span>
                            <span className="text-gray-600">
                                <strong>Red (81-100%)</strong> - Crowded, consider next bus
                            </span>
                        </div>
                    </div>
                </div>

                {/* Last Updated */}
                <div className="text-center text-sm text-gray-500">
                    <p>
                        Last updated: {new Date(bus.last_updated).toLocaleString()}
                    </p>
                    <button
                        onClick={refresh}
                        className="mt-2 text-blue-600 hover:underline flex items-center gap-1 mx-auto"
                    >
                        🔄 Refresh now
                    </button>
                </div>
            </main>
        </div>
    );
};

export default BusDetail;
