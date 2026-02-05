/**
 * MapScreen
 * =========
 * 
 * Main full-screen map with buses and stops.
 */

import React, { useRef, useEffect } from 'react';
import { View, StyleSheet, ActivityIndicator, Text } from 'react-native';
import MapView, { PROVIDER_GOOGLE, Region } from 'react-native-maps';

import { useAppStore, useFilteredBuses } from '../store/useAppStore';
import { useBusPositions, useBusStops } from '../hooks/useData';

import BusMarker from '../components/BusMarker';
import StopMarker from '../components/StopMarker';
import SearchBar from '../components/SearchBar';
import StopDetailSheet from '../components/StopDetailSheet';
import OfflineIndicator from '../components/OfflineIndicator';

// Almaty city center
const ALMATY_REGION: Region = {
    latitude: 43.238949,
    longitude: 76.945465,
    latitudeDelta: 0.08,
    longitudeDelta: 0.08,
};

export const MapScreen: React.FC = () => {
    const mapRef = useRef<MapView>(null);

    // Store
    const stops = useAppStore(state => state.stops);
    const selectedBusId = useAppStore(state => state.selectedBusId);
    const selectedStopId = useAppStore(state => state.selectedStopId);
    const selectBus = useAppStore(state => state.selectBus);
    const selectStop = useAppStore(state => state.selectStop);
    const setMapLoaded = useAppStore(state => state.setMapLoaded);
    const isLoadingBuses = useAppStore(state => state.isLoadingBuses);

    // Filtered buses
    const buses = useFilteredBuses();

    // Data fetching hooks
    useBusPositions();
    useBusStops();

    // Animate to selected bus
    useEffect(() => {
        if (selectedBusId && mapRef.current) {
            const bus = buses.find(b => b.bus_id === selectedBusId);
            if (bus) {
                mapRef.current.animateToRegion({
                    latitude: bus.latitude,
                    longitude: bus.longitude,
                    latitudeDelta: 0.02,
                    longitudeDelta: 0.02,
                }, 500);
            }
        }
    }, [selectedBusId, buses]);

    // Animate to selected stop
    useEffect(() => {
        if (selectedStopId && mapRef.current) {
            const stop = stops.find(s => s.stop_id === selectedStopId);
            if (stop) {
                mapRef.current.animateToRegion({
                    latitude: stop.latitude,
                    longitude: stop.longitude,
                    latitudeDelta: 0.015,
                    longitudeDelta: 0.015,
                }, 500);
            }
        }
    }, [selectedStopId, stops]);

    return (
        <View style={styles.container}>
            {/* Map */}
            <MapView
                ref={mapRef}
                style={styles.map}
                provider={PROVIDER_GOOGLE}
                initialRegion={ALMATY_REGION}
                showsUserLocation
                showsMyLocationButton={false}
                showsCompass={false}
                onMapReady={() => setMapLoaded(true)}
                onPress={() => {
                    selectBus(null);
                    selectStop(null);
                }}
            >
                {/* Bus Markers */}
                {buses.map(bus => (
                    <BusMarker
                        key={bus.bus_id}
                        bus={bus}
                        isSelected={bus.bus_id === selectedBusId}
                        onPress={() => selectBus(bus.bus_id)}
                    />
                ))}

                {/* Stop Markers */}
                {stops.map(stop => (
                    <StopMarker
                        key={stop.stop_id}
                        stop={stop}
                        isSelected={stop.stop_id === selectedStopId}
                        onPress={() => selectStop(stop.stop_id)}
                    />
                ))}
            </MapView>

            {/* Search Bar */}
            <SearchBar />

            {/* Offline Indicator */}
            <OfflineIndicator />

            {/* Loading Indicator */}
            {isLoadingBuses && (
                <View style={styles.loadingContainer}>
                    <ActivityIndicator size="small" color="#3b82f6" />
                    <Text style={styles.loadingText}>Updating...</Text>
                </View>
            )}

            {/* Bus Count */}
            <View style={styles.busCount}>
                <Text style={styles.busCountText}>
                    🚌 {buses.length} buses
                </Text>
            </View>

            {/* Stop Detail Sheet */}
            <StopDetailSheet />
        </View>
    );
};

const styles = StyleSheet.create({
    container: {
        flex: 1,
    },
    map: {
        flex: 1,
    },
    loadingContainer: {
        position: 'absolute',
        top: 60,
        right: 16,
        backgroundColor: '#fff',
        borderRadius: 20,
        paddingHorizontal: 12,
        paddingVertical: 8,
        flexDirection: 'row',
        alignItems: 'center',
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 0.1,
        shadowRadius: 4,
        elevation: 3,
    },
    loadingText: {
        marginLeft: 8,
        fontSize: 12,
        color: '#6b7280',
    },
    busCount: {
        position: 'absolute',
        bottom: 24,
        left: 16,
        backgroundColor: '#1a1a2e',
        borderRadius: 20,
        paddingHorizontal: 16,
        paddingVertical: 10,
    },
    busCountText: {
        color: '#fff',
        fontSize: 14,
        fontWeight: '600',
    },
});

export default MapScreen;
