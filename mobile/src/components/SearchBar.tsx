/**
 * SearchBar Component
 * ===================
 * 
 * Search buses by route number or license plate.
 */

import React from 'react';
import { View, TextInput, TouchableOpacity, StyleSheet, Text } from 'react-native';
import { useAppStore } from '../store/useAppStore';

export const SearchBar: React.FC = () => {
    const searchQuery = useAppStore(state => state.searchQuery);
    const setSearchQuery = useAppStore(state => state.setSearchQuery);

    return (
        <View style={styles.container}>
            <View style={styles.inputContainer}>
                <Text style={styles.icon}>🔍</Text>
                <TextInput
                    style={styles.input}
                    placeholder="Search route or bus number..."
                    placeholderTextColor="#9ca3af"
                    value={searchQuery}
                    onChangeText={setSearchQuery}
                    returnKeyType="search"
                />
                {searchQuery.length > 0 && (
                    <TouchableOpacity
                        style={styles.clearButton}
                        onPress={() => setSearchQuery('')}
                    >
                        <Text style={styles.clearText}>✕</Text>
                    </TouchableOpacity>
                )}
            </View>
        </View>
    );
};

const styles = StyleSheet.create({
    container: {
        position: 'absolute',
        top: 60,
        left: 16,
        right: 16,
        zIndex: 100,
    },
    inputContainer: {
        flexDirection: 'row',
        alignItems: 'center',
        backgroundColor: '#fff',
        borderRadius: 12,
        paddingHorizontal: 16,
        paddingVertical: 12,
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 0.15,
        shadowRadius: 8,
        elevation: 5,
    },
    icon: {
        fontSize: 16,
        marginRight: 12,
    },
    input: {
        flex: 1,
        fontSize: 16,
        color: '#1a1a2e',
    },
    clearButton: {
        width: 24,
        height: 24,
        borderRadius: 12,
        backgroundColor: '#f3f4f6',
        alignItems: 'center',
        justifyContent: 'center',
    },
    clearText: {
        fontSize: 12,
        color: '#6b7280',
    },
});

export default SearchBar;
