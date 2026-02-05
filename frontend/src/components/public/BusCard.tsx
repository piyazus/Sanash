/**
 * BusCard Component
 * =================
 * 
 * Card displaying bus info with occupancy data.
 */

import React from 'react';
import { BusOccupancy } from '../../services/publicApi';
import { OccupancyBar } from './OccupancyBar';
import { StatusBadge } from './StatusBadge';
import { LiveIndicator } from './LiveIndicator';

interface BusCardProps {
    bus: BusOccupancy;
    onClick?: () => void;
    showRoute?: boolean;
    showDistance?: boolean;
}

export const BusCard: React.FC<BusCardProps> = ({
    bus,
    onClick,
    showRoute = true,
    showDistance = false
}) => {
    const formatTime = (isoString: string) => {
        const date = new Date(isoString);
        return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    };

    return (
        <div
            className={`
        bg-white rounded-xl shadow-md p-4 mb-3 
        border border-gray-100
        transition-all duration-200
        ${onClick ? 'cursor-pointer hover:shadow-lg hover:scale-[1.01] active:scale-[0.99]' : ''}
      `}
            onClick={onClick}
        >
            {/* Header */}
            <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-3">
                    <span className="text-2xl">🚌</span>
                    <div>
                        <h3 className="text-xl font-bold text-gray-800">
                            Bus {bus.bus_number}
                        </h3>
                        {showRoute && bus.route_name && (
                            <p className="text-sm text-gray-500">{bus.route_name}</p>
                        )}
                    </div>
                </div>
                <div className="flex flex-col items-end gap-1">
                    <LiveIndicator showText={false} />
                    <StatusBadge status={bus.status} color={bus.color} size="sm" />
                </div>
            </div>

            {/* Occupancy Info */}
            <div className="mb-3">
                <div className="flex justify-between text-sm mb-1.5">
                    <span className="font-semibold text-gray-700">
                        {bus.current_occupancy}/{bus.capacity} passengers
                    </span>
                    <span className="font-bold text-gray-900">{bus.percentage}%</span>
                </div>
                <OccupancyBar
                    percentage={bus.percentage}
                    color={bus.color}
                    showPercentage={false}
                />
            </div>

            {/* Footer */}
            <div className="flex justify-between items-center text-xs text-gray-500">
                <span>Updated: {formatTime(bus.last_updated)}</span>
                {showDistance && bus.distance_km !== undefined && (
                    <span className="font-medium text-blue-600">
                        📍 {bus.distance_km.toFixed(1)} km away
                    </span>
                )}
            </div>
        </div>
    );
};

export default BusCard;
