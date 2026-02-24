import axios from 'axios'
import { store } from '../store/store'
import { logout } from '../store/authSlice'

export const api = axios.create({
    baseURL: '',
    headers: {
        'Content-Type': 'application/json',
    },
})

// Request interceptor — attach JWT bearer token
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

// Response interceptor — logout on 401
api.interceptors.response.use(
    (response) => response,
    (error) => {
        if (error.response?.status === 401) {
            store.dispatch(logout())
        }
        return Promise.reject(error)
    }
)

export default api
