/**
 * StopDetailSheet Component
 * =========================
 * 
 * Bottom sheet showing arriving buses at a stop.
 */

import React from 'react';
import { View, Text, StyleSheet, ScrollView, TouchableOpacity, ActivityIndicator } from 'react-native';
import { useStopArrivals } from '../hooks/useData';
import { useSelectedStop, useAppStore } from '../store/useAppStore';
import { BusArrival } from '../services/api';

const getColorForStatus = (color: 'green' | 'yellow' | 'red'): string => {
    switch (color) {
        case 'green': return '#22c55e';
        case 'yellow': return '#eab308';
        case 'red': return '#ef4444';
        default: return '#6b7280';
    }
};

interface ArrivalCardProps {
    arrival: BusArrival;
    onPress: () => void;
}

const ArrivalCard: React.FC<ArrivalCardProps> = ({ arrival, onPress }) => {
    const statusColor = getColorForStatus(arrival.color);

    return (
        <TouchableOpacity style={styles.arrivalCard} onPress={onPress}>
            {/* Route badge */}
            <View style={[styles.routeBadge, { borderColor: statusColor }]}>
                <Text style={styles.routeNumber}>
                    {arrival.route_name?.replace('Route ', '') || arrival.bus_number}
                </Text>
            </View>

            {/* Info */}
            <View style={styles.arrivalInfo}>
                <View style={styles.arrivalRow}>
                    <Text style={styles.etaTime}>{arrival.eta_minutes} min</Text>
                    <Text style={styles.etaDistance}>{arrival.distance_km} km away</Text>
                </View>
                <View style={styles.arrivalRow}>
                    <Text style={[styles.occupancy, { color: statusColor }]}>
                        {arrival.current_occupancy}/{arrival.capacity} • {arrival.percentage}%
                    </Text>
                    <Text style={[styles.status, { color: statusColor }]}>
                        {arrival.status.replace('_', ' ')}
                    </Text>
                </View>
            </View>
        </TouchableOpacity>
    );
};

export const StopDetailSheet: React.FC = () => {
    const selectedStop = useSelectedStop();
    const selectStop = useAppStore(state => state.selectStop);
    const selectBus = useAppStore(state => state.selectBus);

    const { arrivals, loading } = useStopArrivals(selectedStop?.stop_id || null);

    if (!selectedStop) return null;

    return (
        <View style={styles.container}>
            {/* Handle */}
            <View style={styles.handle} />

            {/* Header */}
            <View style={styles.header}>
                <View style={styles.headerLeft}>
                    <Text style={styles.stopName}>{selectedStop.name}</Text>
                    {selectedStop.address && (
                        <Text style={styles.stopAddress}>{selectedStop.address}</Text>
                    )}
                </View>
                <TouchableOpacity
                    style={styles.closeButton}
                    onPress={() => selectStop(null)}
                >
                    <Text style={styles.closeText}>✕</Text>
                </TouchableOpacity>
            </View>

            {/* Content */}
            {loading ? (
                <View style={styles.loading}>
                    <ActivityIndicator size="large" color="#3b82f6" />
                    <Text style={styles.loadingText}>Loading arrivals...</Text>
                </View>
            ) : arrivals.length === 0 ? (
                <View style={styles.empty}>
                    <Text style={styles.emptyText}>🚌</Text>
                    <Text style={styles.emptyMessage}>No buses arriving soon</Text>
                </View>
            ) : (
                <ScrollView style={styles.list} showsVerticalScrollIndicator={false}>
                    <Text style={styles.sectionTitle}>Arriving Buses</Text>
                    {arrivals.map(arrival => (
                        <ArrivalCard
                            key={arrival.bus_id}
                            arrival={arrival}
                            onPress={() => selectBus(arrival.bus_id)}
                        />
                    ))}
                </ScrollView>
            )}
        </View>
    );
};

const styles = StyleSheet.create({
    container: {
        position: 'absolute',
        bottom: 0,
        left: 0,
        right: 0,
        backgroundColor: '#fff',
        borderTopLeftRadius: 24,
        borderTopRightRadius: 24,
        maxHeight: '50%',
        shadowColor: '#000',
        shadowOffset: { width: 0, height: -4 },
        shadowOpacity: 0.25,
        shadowRadius: 8,
        elevation: 10,
    },
    handle: {
        width: 40,
        height: 4,
        backgroundColor: '#d1d5db',
        borderRadius: 2,
        alignSelf: 'center',
        marginTop: 12,
    },
    header: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        alignItems: 'flex-start',
        padding: 16,
        borderBottomWidth: 1,
        borderBottomColor: '#f3f4f6',
    },
    headerLeft: {
        flex: 1,
    },
    stopName: {
        fontSize: 18,
        fontWeight: 'bold',
        color: '#1a1a2e',
    },
    stopAddress: {
        fontSize: 13,
        color: '#6b7280',
        marginTop: 4,
    },
    closeButton: {
        width: 32,
        height: 32,
        borderRadius: 16,
        backgroundColor: '#f3f4f6',
        alignItems: 'center',
        justifyContent: 'center',
    },
    closeText: {
        fontSize: 14,
        color: '#6b7280',
    },
    loading: {
        padding: 40,
        alignItems: 'center',
    },
    loadingText: {
        marginTop: 12,
        color: '#6b7280',
    },
    empty: {
        padding: 40,
        alignItems: 'center',
    },
    emptyText: {
        fontSize: 40,
        marginBottom: 12,
    },
    emptyMessage: {
        fontSize: 15,
        color: '#6b7280',
    },
    list: {
        padding: 16,
    },
    sectionTitle: {
        fontSize: 13,
        fontWeight: '600',
        color: '#6b7280',
        textTransform: 'uppercase',
        marginBottom: 12,
    },
    arrivalCard: {
        flexDirection: 'row',
        backgroundColor: '#f9fafb',
        borderRadius: 12,
        padding: 12,
        marginBottom: 12,
        alignItems: 'center',
    },
    routeBadge: {
        width: 48,
        height: 48,
        borderRadius: 24,
        borderWidth: 3,
        backgroundColor: '#1a1a2e',
        alignItems: 'center',
        justifyContent: 'center',
        marginRight: 12,
    },
    routeNumber: {
        fontSize: 12,
        fontWeight: 'bold',
        color: '#fff',
    },
    arrivalInfo: {
        flex: 1,
    },
    arrivalRow: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        marginBottom: 4,
    },
    etaTime: {
        fontSize: 18,
        fontWeight: 'bold',
        color: '#1a1a2e',
    },
    etaDistance: {
        fontSize: 13,
        color: '#6b7280',
    },
    occupancy: {
        fontSize: 13,
        fontWeight: '600',
    },
    status: {
        fontSize: 13,
        fontWeight: '500',
        textTransform: 'capitalize',
    },
});

export default StopDetailSheet;
