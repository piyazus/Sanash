/**
 * StopMarker Component
 * ====================
 * 
 * Map marker for bus stops.
 */

import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { Marker } from 'react-native-maps';
import { BusStop } from '../services/api';

interface StopMarkerProps {
    stop: BusStop;
    onPress: () => void;
    isSelected: boolean;
}

export const StopMarker: React.FC<StopMarkerProps> = ({ stop, onPress, isSelected }) => {
    return (
        <Marker
            coordinate={{
                latitude: stop.latitude,
                longitude: stop.longitude,
            }}
            onPress={onPress}
            anchor={{ x: 0.5, y: 1 }}
            tracksViewChanges={false}
        >
            <View style={styles.container}>
                {/* Stop icon */}
                <View style={[
                    styles.stopIcon,
                    isSelected && styles.selectedStop
                ]}>
                    <View style={styles.stopInner}>
                        <Text style={styles.stopText}>🚏</Text>
                    </View>
                </View>

                {/* Pin bottom */}
                <View style={[
                    styles.pin,
                    isSelected && styles.selectedPin
                ]} />
            </View>
        </Marker>
    );
};

const styles = StyleSheet.create({
    container: {
        alignItems: 'center',
    },
    stopIcon: {
        width: 32,
        height: 32,
        borderRadius: 16,
        backgroundColor: '#3b82f6',
        alignItems: 'center',
        justifyContent: 'center',
        borderWidth: 2,
        borderColor: '#fff',
    },
    selectedStop: {
        width: 40,
        height: 40,
        borderRadius: 20,
        backgroundColor: '#1d4ed8',
        borderWidth: 3,
    },
    stopInner: {
        alignItems: 'center',
        justifyContent: 'center',
    },
    stopText: {
        fontSize: 16,
    },
    pin: {
        width: 4,
        height: 8,
        backgroundColor: '#3b82f6',
        borderBottomLeftRadius: 2,
        borderBottomRightRadius: 2,
    },
    selectedPin: {
        height: 12,
        backgroundColor: '#1d4ed8',
    },
});

export default StopMarker;
