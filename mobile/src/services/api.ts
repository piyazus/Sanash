/**
 * API Service for Mobile App
 * ==========================
 * 
 * Axios client for communicating with backend.
 */

import axios from 'axios';
import AsyncStorage from '@react-native-async-storage/async-storage';

// Configure base URL - change for production
const API_BASE_URL = __DEV__
    ? 'http://10.0.2.2:8000' // Android emulator
    : 'https://api.sana.kz';

export const api = axios.create({
    baseURL: API_BASE_URL,
    timeout: 10000,
    headers: {
        'Content-Type': 'application/json',
    },
});

// =============================================================================
// TYPES
// =============================================================================

export interface BusPosition {
    bus_id: number;
    bus_number: string;
    route_id: number | null;
    route_name: string | null;
    latitude: number;
    longitude: number;
    speed: number | null;
    heading: number | null;
    current_occupancy: number;
    capacity: number;
    percentage: number;
    status: 'available' | 'getting_full' | 'crowded';
    color: 'green' | 'yellow' | 'red';
    last_updated: string;
}

export interface BusStop {
    stop_id: number;
    name: string;
    latitude: number;
    longitude: number;
    address: string | null;
    route_ids: number[];
    status: string;
}

export interface BusArrival {
    bus_id: number;
    bus_number: string;
    route_id: number | null;
    route_name: string | null;
    eta_minutes: number;
    eta_time: string;
    distance_km: number;
    current_occupancy: number;
    capacity: number;
    percentage: number;
    status: 'available' | 'getting_full' | 'crowded';
    color: 'green' | 'yellow' | 'red';
}

export interface Route {
    route_id: number;
    route_name: string;
    total_buses: number;
    avg_occupancy_percentage: number;
    status: string;
}

// =============================================================================
// API FUNCTIONS
// =============================================================================

export const mobileApi = {
    /**
     * Get all bus positions for map
     */
    getBusPositions: async (): Promise<{
        total_buses: number;
        buses: BusPosition[];
        timestamp: string;
    }> => {
        const response = await api.get('/api/v1/mobile/buses/positions');
        return response.data;
    },

    /**
     * Get all bus stops
     */
    getStops: async (): Promise<{
        total_stops: number;
        stops: BusStop[];
    }> => {
        const response = await api.get('/api/v1/mobile/stops');
        return response.data;
    },

    /**
     * Get arrivals at a stop
     */
    getStopArrivals: async (stopId: number): Promise<{
        stop_id: number;
        stop_name: string;
        arrivals: BusArrival[];
        total_arrivals: number;
    }> => {
        const response = await api.get(`/api/v1/mobile/stops/${stopId}/arrivals`);
        return response.data;
    },

    /**
     * Get route path for polyline
     */
    getRoutePath: async (routeId: number): Promise<{
        route_id: number;
        route_name: string;
        path: Array<{ lat: number; lon: number }>;
        stops: BusStop[];
    }> => {
        const response = await api.get(`/api/v1/mobile/routes/${routeId}/path`);
        return response.data;
    },

    /**
     * Get all routes
     */
    getRoutes: async (): Promise<{
        routes: Route[];
        total_routes: number;
    }> => {
        const response = await api.get('/api/v1/public/routes');
        return response.data;
    },

    /**
     * Get single bus occupancy
     */
    getBusOccupancy: async (busId: number) => {
        const response = await api.get(`/api/v1/public/buses/${busId}/occupancy`);
        return response.data;
    },

    /**
     * Search buses by route or license plate
     */
    searchBuses: async (query: string): Promise<BusPosition[]> => {
        const positions = await mobileApi.getBusPositions();
        const lowerQuery = query.toLowerCase();

        return positions.buses.filter(bus =>
            bus.bus_number.toLowerCase().includes(lowerQuery) ||
            bus.route_name?.toLowerCase().includes(lowerQuery)
        );
    },
};

// =============================================================================
// OFFLINE CACHE
// =============================================================================

const CACHE_KEYS = {
    STOPS: 'cache:stops',
    ROUTES: 'cache:routes',
};

export const offlineCache = {
    /**
     * Cache stops for offline use
     */
    cacheStops: async (stops: BusStop[]) => {
        await AsyncStorage.setItem(CACHE_KEYS.STOPS, JSON.stringify(stops));
    },

    /**
     * Get cached stops
     */
    getCachedStops: async (): Promise<BusStop[] | null> => {
        const data = await AsyncStorage.getItem(CACHE_KEYS.STOPS);
        return data ? JSON.parse(data) : null;
    },

    /**
     * Cache routes for offline use
     */
    cacheRoutes: async (routes: Route[]) => {
        await AsyncStorage.setItem(CACHE_KEYS.ROUTES, JSON.stringify(routes));
    },

    /**
     * Get cached routes
     */
    getCachedRoutes: async (): Promise<Route[] | null> => {
        const data = await AsyncStorage.getItem(CACHE_KEYS.ROUTES);
        return data ? JSON.parse(data) : null;
    },
};

export default mobileApi;
