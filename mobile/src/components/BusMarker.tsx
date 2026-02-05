/**
 * BusMarker Component
 * ===================
 * 
 * Custom map marker for buses with color-coded occupancy ring.
 */

import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { Marker, Callout } from 'react-native-maps';
import { BusPosition } from '../services/api';

interface BusMarkerProps {
    bus: BusPosition;
    onPress: () => void;
    isSelected: boolean;
}

const getColorForStatus = (color: 'green' | 'yellow' | 'red'): string => {
    switch (color) {
        case 'green': return '#22c55e';
        case 'yellow': return '#eab308';
        case 'red': return '#ef4444';
        default: return '#6b7280';
    }
};

export const BusMarker: React.FC<BusMarkerProps> = ({ bus, onPress, isSelected }) => {
    const ringColor = getColorForStatus(bus.color);

    return (
        <Marker
            coordinate={{
                latitude: bus.latitude,
                longitude: bus.longitude,
            }}
            onPress={onPress}
            anchor={{ x: 0.5, y: 0.5 }}
            tracksViewChanges={false} // Performance optimization
        >
            <View style={styles.container}>
                {/* Outer ring (occupancy color) */}
                <View style={[
                    styles.outerRing,
                    { borderColor: ringColor },
                    isSelected && styles.selectedRing
                ]}>
                    {/* Inner circle with route number */}
                    <View style={styles.innerCircle}>
                        <Text style={styles.routeNumber} numberOfLines={1}>
                            {bus.route_name?.replace('Route ', '') || bus.bus_number}
                        </Text>
                    </View>
                </View>

                {/* Heading indicator (if moving) */}
                {bus.heading && bus.speed && bus.speed > 5 && (
                    <View
                        style={[
                            styles.headingIndicator,
                            { transform: [{ rotate: `${bus.heading}deg` }] }
                        ]}
                    />
                )}
            </View>

            {/* Callout (popup on tap) */}
            <Callout tooltip>
                <View style={styles.callout}>
                    <Text style={styles.calloutTitle}>
                        {bus.route_name || `Bus ${bus.bus_number}`}
                    </Text>
                    <View style={styles.calloutRow}>
                        <Text style={styles.calloutLabel}>Passengers:</Text>
                        <Text style={styles.calloutValue}>
                            {bus.current_occupancy} / {bus.capacity}
                        </Text>
                    </View>
                    <View style={styles.calloutRow}>
                        <Text style={styles.calloutLabel}>Occupancy:</Text>
                        <Text style={[
                            styles.calloutValue,
                            { color: ringColor }
                        ]}>
                            {bus.percentage}% ({bus.status.replace('_', ' ')})
                        </Text>
                    </View>
                    {bus.speed && (
                        <View style={styles.calloutRow}>
                            <Text style={styles.calloutLabel}>Speed:</Text>
                            <Text style={styles.calloutValue}>{Math.round(bus.speed)} km/h</Text>
                        </View>
                    )}
                </View>
            </Callout>
        </Marker>
    );
};

const styles = StyleSheet.create({
    container: {
        alignItems: 'center',
        justifyContent: 'center',
    },
    outerRing: {
        width: 40,
        height: 40,
        borderRadius: 20,
        borderWidth: 4,
        backgroundColor: '#1a1a2e',
        alignItems: 'center',
        justifyContent: 'center',
    },
    selectedRing: {
        width: 48,
        height: 48,
        borderRadius: 24,
        borderWidth: 5,
    },
    innerCircle: {
        width: 28,
        height: 28,
        borderRadius: 14,
        backgroundColor: '#fff',
        alignItems: 'center',
        justifyContent: 'center',
    },
    routeNumber: {
        fontSize: 11,
        fontWeight: 'bold',
        color: '#1a1a2e',
    },
    headingIndicator: {
        position: 'absolute',
        top: -8,
        width: 0,
        height: 0,
        borderLeftWidth: 6,
        borderRightWidth: 6,
        borderBottomWidth: 10,
        borderLeftColor: 'transparent',
        borderRightColor: 'transparent',
        borderBottomColor: '#1a1a2e',
    },
    callout: {
        backgroundColor: '#fff',
        borderRadius: 12,
        padding: 12,
        minWidth: 180,
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 0.25,
        shadowRadius: 4,
        elevation: 5,
    },
    calloutTitle: {
        fontSize: 16,
        fontWeight: 'bold',
        color: '#1a1a2e',
        marginBottom: 8,
    },
    calloutRow: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        marginBottom: 4,
    },
    calloutLabel: {
        fontSize: 13,
        color: '#6b7280',
    },
    calloutValue: {
        fontSize: 13,
        fontWeight: '600',
        color: '#1a1a2e',
    },
});

export default BusMarker;
