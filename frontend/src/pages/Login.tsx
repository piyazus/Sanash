import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useDispatch } from 'react-redux'
import { useForm } from 'react-hook-form'
import { setCredentials } from '../store/authSlice'
import { api } from '../services/api'

interface LoginForm {
    email: string
    password: string
}

export default function Login() {
    const navigate = useNavigate()
    const dispatch = useDispatch()
    const [error, setError] = useState('')
    const [loading, setLoading] = useState(false)

    const { register, handleSubmit, formState: { errors } } = useForm<LoginForm>()

    const onSubmit = async (data: LoginForm) => {
        setLoading(true)
        setError('')

        try {
            const response = await api.post('/api/v1/auth/login', data)
            const { access_token, refresh_token } = response.data

            // Get user info
            api.defaults.headers.common['Authorization'] = `Bearer ${access_token}`
            const userResponse = await api.get('/api/v1/auth/me')

            dispatch(setCredentials({
                accessToken: access_token,
                refreshToken: refresh_token,
                user: userResponse.data,
            }))

            navigate('/')
        } catch (err: any) {
            setError(err.response?.data?.detail || 'Login failed')
        } finally {
            setLoading(false)
        }
    }

    return (
        <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-primary-500 to-purple-600 p-4">
            <div className="w-full max-w-md">
                <div className="card p-8">
                    <div className="text-center mb-8">
                        <h1 className="text-3xl font-bold gradient-text mb-2">Bus Vision</h1>
                        <p className="text-gray-500 dark:text-gray-400">AI-Powered Passenger Detection</p>
                    </div>

                    {error && (
                        <div className="mb-4 p-3 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg text-red-600 dark:text-red-400 text-sm">
                            {error}
                        </div>
                    )}

                    <form onSubmit={handleSubmit(onSubmit)} className="space-y-4">
                        <div>
                            <label className="label">Email</label>
                            <input
                                type="email"
                                className="input"
                                {...register('email', { required: 'Email is required' })}
                            />
                            {errors.email && (
                                <p className="mt-1 text-sm text-red-500">{errors.email.message}</p>
                            )}
                        </div>

                        <div>
                            <label className="label">Password</label>
                            <input
                                type="password"
                                className="input"
                                {...register('password', { required: 'Password is required' })}
                            />
                            {errors.password && (
                                <p className="mt-1 text-sm text-red-500">{errors.password.message}</p>
                            )}
                        </div>

                        <button
                            type="submit"
                            disabled={loading}
                            className="btn btn-primary w-full"
                        >
                            {loading ? 'Signing in...' : 'Sign In'}
                        </button>
                    </form>

                    <p className="mt-6 text-center text-sm text-gray-500">
                        Demo: admin@busvision.local / password123
                    </p>
                </div>
            </div>
        </div>
    )
}
