/**
 * useOccupancy Hook
 * =================
 * 
 * Fetches and auto-refreshes bus occupancy data.
 */

import { useState, useEffect, useCallback } from 'react';
import { publicApi, BusOccupancy } from '../services/publicApi';

interface UseOccupancyResult {
    data: BusOccupancy | null;
    loading: boolean;
    error: string | null;
    refresh: () => Promise<void>;
}

export const useOccupancy = (
    busId: number,
    autoRefresh: boolean = true,
    refreshInterval: number = 5000
): UseOccupancyResult => {
    const [data, setData] = useState<BusOccupancy | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    const fetchData = useCallback(async () => {
        try {
            const occupancy = await publicApi.getBusOccupancy(busId);
            setData(occupancy);
            setError(null);
        } catch (err) {
            setError('Failed to load occupancy data');
            console.error('useOccupancy error:', err);
        } finally {
            setLoading(false);
        }
    }, [busId]);

    useEffect(() => {
        fetchData();

        if (autoRefresh) {
            const interval = setInterval(fetchData, refreshInterval);
            return () => clearInterval(interval);
        }
    }, [busId, autoRefresh, refreshInterval, fetchData]);

    return { data, loading, error, refresh: fetchData };
};
