/**
 * useGeolocation Hook
 * ===================
 * 
 * Gets user's current GPS location.
 */

import { useState, useEffect } from 'react';

interface Location {
    lat: number;
    lon: number;
}

interface UseGeolocationResult {
    location: Location | null;
    loading: boolean;
    error: string | null;
    refresh: () => void;
}

export const useGeolocation = (): UseGeolocationResult => {
    const [location, setLocation] = useState<Location | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    const getLocation = () => {
        if (!navigator.geolocation) {
            setError('Geolocation is not supported by your browser');
            setLoading(false);
            return;
        }

        setLoading(true);
        setError(null);

        navigator.geolocation.getCurrentPosition(
            (position) => {
                setLocation({
                    lat: position.coords.latitude,
                    lon: position.coords.longitude
                });
                setLoading(false);
            },
            (err) => {
                switch (err.code) {
                    case err.PERMISSION_DENIED:
                        setError('Location access denied. Please enable location permissions.');
                        break;
                    case err.POSITION_UNAVAILABLE:
                        setError('Location information unavailable.');
                        break;
                    case err.TIMEOUT:
                        setError('Location request timed out.');
                        break;
                    default:
                        setError('An unknown error occurred.');
                }
                setLoading(false);
            },
            {
                enableHighAccuracy: true,
                timeout: 10000,
                maximumAge: 60000
            }
        );
    };

    useEffect(() => {
        getLocation();
    }, []);

    return { location, loading, error, refresh: getLocation };
};
