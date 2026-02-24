import { createSlice, PayloadAction } from '@reduxjs/toolkit'

interface Operator {
    id: string
    name: string
    email: string
    company: string
}

interface AuthState {
    isAuthenticated: boolean
    operator: Operator | null
    accessToken: string | null
}

const initialState: AuthState = {
    isAuthenticated: localStorage.getItem('accessToken') !== null,
    operator: null,
    accessToken: localStorage.getItem('accessToken'),
}

const authSlice = createSlice({
    name: 'auth',
    initialState,
    reducers: {
        setCredentials: (
            state,
            action: PayloadAction<{ accessToken: string; operator: Operator }>
        ) => {
            state.isAuthenticated = true
            state.accessToken = action.payload.accessToken
            state.operator = action.payload.operator
            localStorage.setItem('accessToken', action.payload.accessToken)
        },
        setOperator: (state, action: PayloadAction<Operator>) => {
            state.operator = action.payload
        },
        logout: (state) => {
            state.isAuthenticated = false
            state.accessToken = null
            state.operator = null
            localStorage.removeItem('accessToken')
        },
    },
})

export const { setCredentials, setOperator, logout } = authSlice.actions
export default authSlice.reducer
