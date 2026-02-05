/**
 * App Store (Zustand)
 * ===================
 * 
 * Global state management for the mobile app.
 */

import { create } from 'zustand';
import { BusPosition, BusStop, Route } from '../services/api';

interface AppState {
    // Bus positions (updated every 5s)
    buses: BusPosition[];
    setBuses: (buses: BusPosition[]) => void;

    // Bus stops (cached)
    stops: BusStop[];
    setStops: (stops: BusStop[]) => void;

    // Routes
    routes: Route[];
    setRoutes: (routes: Route[]) => void;

    // Selected items
    selectedBusId: number | null;
    selectBus: (id: number | null) => void;

    selectedStopId: number | null;
    selectStop: (id: number | null) => void;

    selectedRouteId: number | null;
    selectRoute: (id: number | null) => void;

    // Search
    searchQuery: string;
    setSearchQuery: (query: string) => void;

    // UI State
    isMapLoaded: boolean;
    setMapLoaded: (loaded: boolean) => void;

    isOffline: boolean;
    setOffline: (offline: boolean) => void;

    // Loading states
    isLoadingBuses: boolean;
    setLoadingBuses: (loading: boolean) => void;
}

export const useAppStore = create<AppState>((set) => ({
    // Buses
    buses: [],
    setBuses: (buses) => set({ buses }),

    // Stops
    stops: [],
    setStops: (stops) => set({ stops }),

    // Routes
    routes: [],
    setRoutes: (routes) => set({ routes }),

    // Selected
    selectedBusId: null,
    selectBus: (id) => set({ selectedBusId: id, selectedStopId: null }),

    selectedStopId: null,
    selectStop: (id) => set({ selectedStopId: id, selectedBusId: null }),

    selectedRouteId: null,
    selectRoute: (id) => set({ selectedRouteId: id }),

    // Search
    searchQuery: '',
    setSearchQuery: (query) => set({ searchQuery: query }),

    // UI
    isMapLoaded: false,
    setMapLoaded: (loaded) => set({ isMapLoaded: loaded }),

    isOffline: false,
    setOffline: (offline) => set({ isOffline: offline }),

    isLoadingBuses: false,
    setLoadingBuses: (loading) => set({ isLoadingBuses: loading }),
}));

// Selectors
export const useSelectedBus = () => {
    const buses = useAppStore(state => state.buses);
    const selectedBusId = useAppStore(state => state.selectedBusId);
    return buses.find(b => b.bus_id === selectedBusId) || null;
};

export const useSelectedStop = () => {
    const stops = useAppStore(state => state.stops);
    const selectedStopId = useAppStore(state => state.selectedStopId);
    return stops.find(s => s.stop_id === selectedStopId) || null;
};

export const useFilteredBuses = () => {
    const buses = useAppStore(state => state.buses);
    const query = useAppStore(state => state.searchQuery);
    const routeId = useAppStore(state => state.selectedRouteId);

    let filtered = buses;

    if (routeId) {
        filtered = filtered.filter(b => b.route_id === routeId);
    }

    if (query) {
        const lower = query.toLowerCase();
        filtered = filtered.filter(b =>
            b.bus_number.toLowerCase().includes(lower) ||
            b.route_name?.toLowerCase().includes(lower)
        );
    }

    return filtered;
};
