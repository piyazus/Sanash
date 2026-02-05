import { createSlice, PayloadAction } from '@reduxjs/toolkit'

interface Alert {
    id: number
    type: string
    severity: string
    message: string
    timestamp: string
    acknowledged: boolean
}

interface AlertsState {
    items: Alert[]
    unreadCount: number
}

const initialState: AlertsState = {
    items: [],
    unreadCount: 0,
}

const alertsSlice = createSlice({
    name: 'alerts',
    initialState,
    reducers: {
        setAlerts: (state, action: PayloadAction<Alert[]>) => {
            state.items = action.payload
            state.unreadCount = action.payload.filter(a => !a.acknowledged).length
        },
        addAlert: (state, action: PayloadAction<Alert>) => {
            state.items.unshift(action.payload)
            if (!action.payload.acknowledged) {
                state.unreadCount++
            }
        },
        acknowledgeAlert: (state, action: PayloadAction<number>) => {
            const alert = state.items.find(a => a.id === action.payload)
            if (alert && !alert.acknowledged) {
                alert.acknowledged = true
                state.unreadCount = Math.max(0, state.unreadCount - 1)
            }
        },
    },
})

export const { setAlerts, addAlert, acknowledgeAlert } = alertsSlice.actions
export default alertsSlice.reducer
