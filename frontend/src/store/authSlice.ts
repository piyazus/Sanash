import { createSlice, PayloadAction } from '@reduxjs/toolkit'

interface User {
    id: number
    email: string
    fullName: string | null
    role: string
}

interface AuthState {
    isAuthenticated: boolean
    user: User | null
    accessToken: string | null
    refreshToken: string | null
}

const initialState: AuthState = {
    isAuthenticated: localStorage.getItem('accessToken') !== null,
    user: null,
    accessToken: localStorage.getItem('accessToken'),
    refreshToken: localStorage.getItem('refreshToken'),
}

const authSlice = createSlice({
    name: 'auth',
    initialState,
    reducers: {
        setCredentials: (state, action: PayloadAction<{ accessToken: string; refreshToken: string; user: User }>) => {
            state.isAuthenticated = true
            state.accessToken = action.payload.accessToken
            state.refreshToken = action.payload.refreshToken
            state.user = action.payload.user
            localStorage.setItem('accessToken', action.payload.accessToken)
            localStorage.setItem('refreshToken', action.payload.refreshToken)
        },
        setUser: (state, action: PayloadAction<User>) => {
            state.user = action.payload
        },
        logout: (state) => {
            state.isAuthenticated = false
            state.accessToken = null
            state.refreshToken = null
            state.user = null
            localStorage.removeItem('accessToken')
            localStorage.removeItem('refreshToken')
        },
    },
})

export const { setCredentials, setUser, logout } = authSlice.actions
export default authSlice.reducer
