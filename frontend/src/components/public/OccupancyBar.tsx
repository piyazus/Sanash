/**
 * OccupancyBar Component
 * ======================
 * 
 * Visual progress bar showing bus occupancy level.
 */

import React from 'react';

interface OccupancyBarProps {
    percentage: number;
    color: 'green' | 'yellow' | 'red';
    showPercentage?: boolean;
    height?: 'sm' | 'md' | 'lg';
}

export const OccupancyBar: React.FC<OccupancyBarProps> = ({
    percentage,
    color,
    showPercentage = true,
    height = 'md'
}) => {
    const colorClasses = {
        green: 'bg-green-500',
        yellow: 'bg-yellow-500',
        red: 'bg-red-500'
    };

    const heightClasses = {
        sm: 'h-4',
        md: 'h-6',
        lg: 'h-8'
    };

    const bgColorClasses = {
        green: 'bg-green-100',
        yellow: 'bg-yellow-100',
        red: 'bg-red-100'
    };

    return (
        <div className={`w-full ${bgColorClasses[color]} rounded-full ${heightClasses[height]} overflow-hidden`}>
            <div
                className={`h-full ${colorClasses[color]} transition-all duration-500 ease-out flex items-center justify-center text-white text-xs font-bold`}
                style={{ width: `${Math.min(percentage, 100)}%` }}
            >
                {showPercentage && percentage > 15 && `${percentage}%`}
            </div>
        </div>
    );
};

export default OccupancyBar;
