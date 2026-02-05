/**
 * Admin API Service
 * ==================
 * 
 * API services for admin dashboard authentication and analytics.
 * Uses JWT authentication via the api.ts interceptors.
 */

import api from './api';

// =============================================================================
// AUTH TYPES
// =============================================================================

export interface LoginRequest {
    email: string;
    password: string;
}

export interface TokenResponse {
    access_token: string;
    refresh_token: string;
    token_type: string;
    expires_in: number;
}

export interface User {
    id: number;
    email: string;
    full_name: string;
    role: string;
    is_active: boolean;
    created_at: string;
    last_login: string | null;
}

// =============================================================================
// ANALYTICS TYPES
// =============================================================================

export interface DashboardStats {
    current_occupancy: number;
    total_people_today: number;
    active_jobs: number;
    pending_alerts: number;
    buses_online: number;
    avg_processing_fps: number;
}

export interface AdminOverview {
    date: string;
    total_passengers: number;
    busiest_route: {
        route_id: number;
        route_name: string;
        passenger_count: number;
    } | null;
    peak_hour: number | null;
    peak_hour_count: number;
    active_buses: number;
    hourly_breakdown: Array<{
        hour: number;
        count: number;
    }>;
}

export interface RouteStats {
    route_id: number;
    route_name: string;
    total_buses: number;
    total_passengers: number;
    avg_daily_passengers: number;
    peak_hour: number | null;
    peak_hour_passengers: number;
}

export interface RouteComparison {
    start_date: string;
    end_date: string;
    total_routes: number;
    routes: RouteStats[];
}

// =============================================================================
// AUTH SERVICES
// =============================================================================

export const authService = {
    /**
     * Login with email and password
     */
    login: async (credentials: LoginRequest): Promise<TokenResponse> => {
        const response = await api.post('/api/v1/auth/login', credentials);
        return response.data;
    },

    /**
     * Get current user info
     */
    me: async (): Promise<User> => {
        const response = await api.get('/api/v1/auth/me');
        return response.data;
    },

    /**
     * Refresh tokens
     */
    refresh: async (refreshToken: string): Promise<TokenResponse> => {
        const response = await api.post('/api/v1/auth/refresh', {
            refresh_token: refreshToken,
        });
        return response.data;
    },
};

// =============================================================================
// ANALYTICS SERVICES
// =============================================================================

export const analyticsService = {
    /**
     * Get real-time dashboard statistics
     */
    getDashboard: async (): Promise<DashboardStats> => {
        const response = await api.get('/api/v1/analytics/dashboard');
        return response.data;
    },

    /**
     * Get admin overview for a specific date
     */
    getAdminOverview: async (date?: Date): Promise<AdminOverview> => {
        const params = date ? { date: date.toISOString() } : {};
        const response = await api.get('/api/v1/analytics/admin/overview', { params });
        return response.data;
    },

    /**
     * Compare route performance over date range
     */
    compareRoutes: async (startDate?: Date, endDate?: Date): Promise<RouteComparison> => {
        const params: Record<string, string> = {};
        if (startDate) params.start_date = startDate.toISOString();
        if (endDate) params.end_date = endDate.toISOString();

        const response = await api.get('/api/v1/analytics/admin/routes/comparison', { params });
        return response.data;
    },

    /**
     * Get occupancy time series data
     */
    getOccupancy: async (params: {
        bus_id?: number;
        job_id?: number;
        start_date?: Date;
        end_date?: Date;
        granularity?: 'minute' | 'hour' | 'day';
    }) => {
        const queryParams: Record<string, unknown> = {};
        if (params.bus_id) queryParams.bus_id = params.bus_id;
        if (params.job_id) queryParams.job_id = params.job_id;
        if (params.start_date) queryParams.start_date = params.start_date.toISOString();
        if (params.end_date) queryParams.end_date = params.end_date.toISOString();
        if (params.granularity) queryParams.granularity = params.granularity;

        const response = await api.get('/api/v1/analytics/occupancy', { params: queryParams });
        return response.data;
    },

    /**
     * Get flow data (entries/exits)
     */
    getFlow: async (params: {
        zone_id?: number;
        job_id?: number;
        start_date?: Date;
        end_date?: Date;
    }) => {
        const queryParams: Record<string, unknown> = {};
        if (params.zone_id) queryParams.zone_id = params.zone_id;
        if (params.job_id) queryParams.job_id = params.job_id;
        if (params.start_date) queryParams.start_date = params.start_date.toISOString();
        if (params.end_date) queryParams.end_date = params.end_date.toISOString();

        const response = await api.get('/api/v1/analytics/flow', { params: queryParams });
        return response.data;
    },
};
