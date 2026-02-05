/**
 * Public API Service
 * ==================
 * 
 * API client for public occupancy endpoints.
 * No authentication required.
 */

import axios from 'axios';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// Types
export interface BusOccupancy {
    bus_id: number;
    bus_number: string;
    route_id: number | null;
    route_name: string | null;
    current_occupancy: number;
    capacity: number;
    percentage: number;
    status: 'available' | 'getting_full' | 'crowded';
    color: 'green' | 'yellow' | 'red';
    last_updated: string;
    distance_km?: number;
}

export interface Route {
    route_id: number;
    route_name: string;
    total_buses: number;
    avg_occupancy_percentage: number;
    status: string;
}

export interface RouteBusesResponse {
    route_id: number;
    route_name: string;
    total_buses: number;
    buses: BusOccupancy[];
}

export interface NearbyBusesResponse {
    latitude: number;
    longitude: number;
    radius_km: number;
    total_buses: number;
    buses: BusOccupancy[];
}

export interface RouteListResponse {
    routes: Route[];
    total_routes: number;
}

// API Client
const publicClient = axios.create({
    baseURL: API_BASE,
    timeout: 10000,
});

export const publicApi = {
    /**
     * Get occupancy for a specific bus
     */
    getBusOccupancy: async (busId: number): Promise<BusOccupancy> => {
        const response = await publicClient.get(`/api/v1/public/buses/${busId}/occupancy`);
        return response.data;
    },

    /**
     * Get all buses on a route with occupancy
     */
    getRouteBuses: async (routeId: number): Promise<RouteBusesResponse> => {
        const response = await publicClient.get(`/api/v1/public/routes/${routeId}/buses`);
        return response.data;
    },

    /**
     * Get buses near a location
     */
    getNearbyBuses: async (
        lat: number,
        lon: number,
        radiusKm: number = 1.0
    ): Promise<NearbyBusesResponse> => {
        const response = await publicClient.get('/api/v1/public/buses/nearby', {
            params: { lat, lon, radius: radiusKm }
        });
        return response.data;
    },

    /**
     * Get all active routes
     */
    getAllRoutes: async (): Promise<RouteListResponse> => {
        const response = await publicClient.get('/api/v1/public/routes');
        return response.data;
    },

    /**
     * Health check
     */
    healthCheck: async (): Promise<{ status: string; api: string }> => {
        const response = await publicClient.get('/api/v1/public/health');
        return response.data;
    }
};
