/**
 * Sana Bus - React Native App
 * ===========================
 * 
 * Real-time bus occupancy monitoring for Almaty.
 */

import React from 'react';
import { StatusBar } from 'expo-status-bar';
import { SafeAreaProvider } from 'react-native-safe-area-context';
import { GestureHandlerRootView } from 'react-native-gesture-handler';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';

import MapScreen from './src/screens/MapScreen';

// React Query client
const queryClient = new QueryClient({
    defaultOptions: {
        queries: {
            staleTime: 5000, // 5 seconds
            retry: 2,
        },
    },
});

export default function App() {
    return (
        <QueryClientProvider client={queryClient}>
            <SafeAreaProvider>
                <GestureHandlerRootView style={{ flex: 1 }}>
                    <StatusBar style="dark" />
                    <MapScreen />
                </GestureHandlerRootView>
            </SafeAreaProvider>
        </QueryClientProvider>
    );
}
