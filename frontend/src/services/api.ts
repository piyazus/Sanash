import axios from 'axios'
import { store } from '../store/store'
import { logout } from '../store/authSlice'

export const api = axios.create({
    baseURL: '',
    headers: {
        'Content-Type': 'application/json',
    },
})

// Request interceptor - add auth token
api.interceptors.request.use(
    (config) => {
        const token = store.getState().auth.accessToken
        if (token) {
            config.headers.Authorization = `Bearer ${token}`
        }
        return config
    },
    (error) => Promise.reject(error)
)

// Response interceptor - handle 401
api.interceptors.response.use(
    (response) => response,
    async (error) => {
        if (error.response?.status === 401) {
            // Try refresh token
            const refreshToken = store.getState().auth.refreshToken
            if (refreshToken) {
                try {
                    const response = await axios.post('/api/v1/auth/refresh', {
                        refresh_token: refreshToken,
                    })

                    // Update tokens
                    const { access_token, refresh_token } = response.data
                    localStorage.setItem('accessToken', access_token)
                    localStorage.setItem('refreshToken', refresh_token)

                    // Retry original request
                    error.config.headers.Authorization = `Bearer ${access_token}`
                    return axios(error.config)
                } catch {
                    // Refresh failed - logout
                    store.dispatch(logout())
                }
            } else {
                store.dispatch(logout())
            }
        }
        return Promise.reject(error)
    }
)

export default api
