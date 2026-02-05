/**
 * useBusPositions Hook
 * ====================
 * 
 * Fetches and auto-refreshes bus positions every 5 seconds.
 */

import { useEffect, useCallback } from 'react';
import { useAppStore } from '../store/useAppStore';
import { mobileApi, offlineCache } from '../services/api';

const REFRESH_INTERVAL = 5000; // 5 seconds

export const useBusPositions = () => {
    const setBuses = useAppStore(state => state.setBuses);
    const setLoadingBuses = useAppStore(state => state.setLoadingBuses);
    const setOffline = useAppStore(state => state.setOffline);

    const fetchPositions = useCallback(async () => {
        try {
            setLoadingBuses(true);
            const data = await mobileApi.getBusPositions();
            setBuses(data.buses);
            setOffline(false);
        } catch (error) {
            console.error('Failed to fetch bus positions:', error);
            setOffline(true);
        } finally {
            setLoadingBuses(false);
        }
    }, [setBuses, setLoadingBuses, setOffline]);

    useEffect(() => {
        // Initial fetch
        fetchPositions();

        // Set up interval
        const interval = setInterval(fetchPositions, REFRESH_INTERVAL);

        return () => clearInterval(interval);
    }, [fetchPositions]);

    return { refresh: fetchPositions };
};

/**
 * useBusStops Hook
 * ================
 * 
 * Fetches bus stops with offline caching.
 */
export const useBusStops = () => {
    const setStops = useAppStore(state => state.setStops);

    const fetchStops = useCallback(async () => {
        try {
            const data = await mobileApi.getStops();
            setStops(data.stops);

            // Cache for offline use
            await offlineCache.cacheStops(data.stops);
        } catch (error) {
            console.error('Failed to fetch stops:', error);

            // Try offline cache
            const cached = await offlineCache.getCachedStops();
            if (cached) {
                setStops(cached);
            }
        }
    }, [setStops]);

    useEffect(() => {
        fetchStops();
    }, [fetchStops]);

    return { refresh: fetchStops };
};

/**
 * useRoutes Hook
 * ==============
 * 
 * Fetches routes with offline caching.
 */
export const useRoutes = () => {
    const setRoutes = useAppStore(state => state.setRoutes);

    const fetchRoutes = useCallback(async () => {
        try {
            const data = await mobileApi.getRoutes();
            setRoutes(data.routes);

            // Cache for offline use
            await offlineCache.cacheRoutes(data.routes);
        } catch (error) {
            console.error('Failed to fetch routes:', error);

            // Try offline cache
            const cached = await offlineCache.getCachedRoutes();
            if (cached) {
                setRoutes(cached);
            }
        }
    }, [setRoutes]);

    useEffect(() => {
        fetchRoutes();
    }, [fetchRoutes]);

    return { refresh: fetchRoutes };
};

/**
 * useStopArrivals Hook
 * ====================
 * 
 * Fetches arriving buses at a selected stop.
 */
import { useState } from 'react';
import { BusArrival } from '../services/api';

export const useStopArrivals = (stopId: number | null) => {
    const [arrivals, setArrivals] = useState<BusArrival[]>([]);
    const [loading, setLoading] = useState(false);

    const fetchArrivals = useCallback(async () => {
        if (!stopId) {
            setArrivals([]);
            return;
        }

        try {
            setLoading(true);
            const data = await mobileApi.getStopArrivals(stopId);
            setArrivals(data.arrivals);
        } catch (error) {
            console.error('Failed to fetch arrivals:', error);
            setArrivals([]);
        } finally {
            setLoading(false);
        }
    }, [stopId]);

    useEffect(() => {
        fetchArrivals();

        // Refresh every 30 seconds
        const interval = setInterval(fetchArrivals, 30000);
        return () => clearInterval(interval);
    }, [fetchArrivals]);

    return { arrivals, loading, refresh: fetchArrivals };
};
