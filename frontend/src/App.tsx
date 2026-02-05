import { Routes, Route, Navigate } from 'react-router-dom'
import { useSelector } from 'react-redux'
import { RootState } from './store/store'

// Layouts
import Layout from './components/Layout/Layout'

// Admin Pages (Protected)
import Login from './pages/Login'
import Dashboard from './pages/Dashboard'
import Jobs from './pages/Jobs'
import JobDetails from './pages/JobDetails'
import Analytics from './pages/Analytics'
import Buses from './pages/Buses'
import Settings from './pages/Settings'

// Public Pages (No Auth Required)
import { Home as PublicHome } from './pages/public/Home'
import { RouteBuses } from './pages/public/RouteBuses'
import { BusDetail } from './pages/public/BusDetail'
import { NearbyBuses } from './pages/public/NearbyBuses'

// Protected Route Wrapper for Admin
function ProtectedRoute({ children }: { children: React.ReactNode }) {
    const { isAuthenticated } = useSelector((state: RootState) => state.auth)

    if (!isAuthenticated) {
        return <Navigate to="/login" replace />
    }

    return <>{children}</>
}

function App() {
    return (
        <Routes>
            {/* =================================================== */}
            {/* PUBLIC ROUTES - No Authentication Required */}
            {/* =================================================== */}
            <Route path="/public" element={<PublicHome />} />
            <Route path="/public/route/:routeId" element={<RouteBuses />} />
            <Route path="/public/bus/:busId" element={<BusDetail />} />
            <Route path="/public/nearby" element={<NearbyBuses />} />

            {/* =================================================== */}
            {/* ADMIN ROUTES - Authentication Required */}
            {/* =================================================== */}
            <Route path="/login" element={<Login />} />

            <Route
                path="/"
                element={
                    <ProtectedRoute>
                        <Layout />
                    </ProtectedRoute>
                }
            >
                <Route index element={<Dashboard />} />
                <Route path="jobs" element={<Jobs />} />
                <Route path="jobs/:id" element={<JobDetails />} />
                <Route path="analytics" element={<Analytics />} />
                <Route path="buses" element={<Buses />} />
                <Route path="settings" element={<Settings />} />
            </Route>

            {/* Fallback */}
            <Route path="*" element={<Navigate to="/public" replace />} />
        </Routes>
    )
}

export default App

