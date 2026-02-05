/**
 * OfflineIndicator Component
 * ==========================
 * 
 * Shows when app is offline.
 */

import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { useAppStore } from '../store/useAppStore';

export const OfflineIndicator: React.FC = () => {
    const isOffline = useAppStore(state => state.isOffline);

    if (!isOffline) return null;

    return (
        <View style={styles.container}>
            <Text style={styles.icon}>📡</Text>
            <Text style={styles.text}>Offline Mode - Using cached data</Text>
        </View>
    );
};

const styles = StyleSheet.create({
    container: {
        position: 'absolute',
        top: 120,
        left: 16,
        right: 16,
        backgroundColor: '#fef3c7',
        borderRadius: 8,
        padding: 12,
        flexDirection: 'row',
        alignItems: 'center',
        zIndex: 99,
    },
    icon: {
        fontSize: 16,
        marginRight: 8,
    },
    text: {
        fontSize: 13,
        color: '#92400e',
        fontWeight: '500',
    },
});

export default OfflineIndicator;
