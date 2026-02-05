/**
 * StatusBadge Component
 * =====================
 * 
 * Color-coded badge showing occupancy status.
 */

import React from 'react';

interface StatusBadgeProps {
    status: 'available' | 'getting_full' | 'crowded';
    color: 'green' | 'yellow' | 'red';
    size?: 'sm' | 'md' | 'lg';
}

export const StatusBadge: React.FC<StatusBadgeProps> = ({
    status,
    color,
    size = 'md'
}) => {
    const colorClasses = {
        green: 'bg-green-100 text-green-800 border-green-200',
        yellow: 'bg-yellow-100 text-yellow-800 border-yellow-200',
        red: 'bg-red-100 text-red-800 border-red-200'
    };

    const statusText = {
        available: '✓ Available',
        getting_full: '⚠ Getting Full',
        crowded: '✕ Crowded'
    };

    const sizeClasses = {
        sm: 'px-2 py-0.5 text-xs',
        md: 'px-3 py-1 text-sm',
        lg: 'px-4 py-1.5 text-base'
    };

    return (
        <span
            className={`
        inline-flex items-center rounded-full border font-semibold
        ${colorClasses[color]}
        ${sizeClasses[size]}
      `}
        >
            {statusText[status]}
        </span>
    );
};

export default StatusBadge;
