import { configureStore } from '@reduxjs/toolkit'
import authReducer from './authSlice'
import jobsReducer from './jobsSlice'
import alertsReducer from './alertsSlice'

export const store = configureStore({
    reducer: {
        auth: authReducer,
        jobs: jobsReducer,
        alerts: alertsReducer,
    },
    middleware: (getDefaultMiddleware) =>
        getDefaultMiddleware({
            serializableCheck: false,
        }),
})

export type RootState = ReturnType<typeof store.getState>
export type AppDispatch = typeof store.dispatch
