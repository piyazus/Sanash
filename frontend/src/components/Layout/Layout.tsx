import { Outlet, NavLink, useNavigate } from 'react-router-dom'
import { useDispatch, useSelector } from 'react-redux'
import { RootState } from '../../store/store'
import { logout } from '../../store/authSlice'
import {
    HomeIcon,
    VideoCameraIcon,
    ChartBarIcon,
    TruckIcon,
    Cog6ToothIcon,
    BellIcon,
    ArrowRightOnRectangleIcon,
} from '@heroicons/react/24/outline'

const navigation = [
    { name: 'Dashboard', href: '/', icon: HomeIcon },
    { name: 'Jobs', href: '/jobs', icon: VideoCameraIcon },
    { name: 'Analytics', href: '/analytics', icon: ChartBarIcon },
    { name: 'Fleet', href: '/buses', icon: TruckIcon },
    { name: 'Settings', href: '/settings', icon: Cog6ToothIcon },
]

export default function Layout() {
    const dispatch = useDispatch()
    const navigate = useNavigate()
    const { user } = useSelector((state: RootState) => state.auth)
    const { unreadCount } = useSelector((state: RootState) => state.alerts)

    const handleLogout = () => {
        dispatch(logout())
        navigate('/login')
    }

    return (
        <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
            {/* Sidebar */}
            <aside className="fixed inset-y-0 left-0 w-64 bg-white dark:bg-gray-800 border-r border-gray-200 dark:border-gray-700">
                {/* Logo */}
                <div className="h-16 flex items-center px-6 border-b border-gray-200 dark:border-gray-700">
                    <span className="text-xl font-bold gradient-text">Bus Vision</span>
                </div>

                {/* Navigation */}
                <nav className="p-4 space-y-1">
                    {navigation.map((item) => (
                        <NavLink
                            key={item.name}
                            to={item.href}
                            className={({ isActive }) =>
                                `flex items-center gap-3 px-4 py-3 rounded-lg transition-colors ${isActive
                                    ? 'bg-primary-50 dark:bg-primary-900/20 text-primary-600 dark:text-primary-400'
                                    : 'text-gray-600 dark:text-gray-400 hover:bg-gray-50 dark:hover:bg-gray-700'
                                }`
                            }
                        >
                            <item.icon className="w-5 h-5" />
                            {item.name}
                        </NavLink>
                    ))}
                </nav>

                {/* User section */}
                <div className="absolute bottom-0 left-0 right-0 p-4 border-t border-gray-200 dark:border-gray-700">
                    <div className="flex items-center justify-between">
                        <div className="flex items-center gap-3">
                            <div className="w-8 h-8 bg-primary-100 dark:bg-primary-900 rounded-full flex items-center justify-center">
                                <span className="text-sm font-medium text-primary-600 dark:text-primary-400">
                                    {user?.fullName?.[0] || user?.email?.[0] || 'U'}
                                </span>
                            </div>
                            <div className="text-sm">
                                <p className="font-medium text-gray-900 dark:text-gray-100">{user?.fullName || 'User'}</p>
                                <p className="text-gray-500 dark:text-gray-400">{user?.role}</p>
                            </div>
                        </div>
                        <button onClick={handleLogout} className="p-2 text-gray-400 hover:text-gray-600">
                            <ArrowRightOnRectangleIcon className="w-5 h-5" />
                        </button>
                    </div>
                </div>
            </aside>

            {/* Main content */}
            <div className="ml-64">
                {/* Header */}
                <header className="h-16 bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 flex items-center justify-end px-6">
                    <button className="relative p-2 text-gray-400 hover:text-gray-600">
                        <BellIcon className="w-6 h-6" />
                        {unreadCount > 0 && (
                            <span className="absolute top-0 right-0 w-5 h-5 bg-red-500 text-white text-xs rounded-full flex items-center justify-center">
                                {unreadCount}
                            </span>
                        )}
                    </button>
                </header>

                {/* Page content */}
                <main className="p-6">
                    <Outlet />
                </main>
            </div>
        </div>
    )
}
