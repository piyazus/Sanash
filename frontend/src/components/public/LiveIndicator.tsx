/**
 * LiveIndicator Component
 * =======================
 * 
 * Pulsing dot showing real-time data status.
 */

import React from 'react';

interface LiveIndicatorProps {
    showText?: boolean;
}

export const LiveIndicator: React.FC<LiveIndicatorProps> = ({
    showText = true
}) => {
    return (
        <div className="flex items-center gap-1.5">
            <span className="relative flex h-2.5 w-2.5">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-400 opacity-75"></span>
                <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-red-500"></span>
            </span>
            {showText && (
                <span className="text-xs font-bold text-red-600 uppercase tracking-wide">
                    LIVE
                </span>
            )}
        </div>
    );
};

export default LiveIndicator;
