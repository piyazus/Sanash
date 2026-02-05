export default function Settings() {
    return (
        <div className="space-y-6">
            <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Settings</h1>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Profile Settings */}
                <div className="card p-6">
                    <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">Profile</h2>
                    <form className="space-y-4">
                        <div>
                            <label className="label">Full Name</label>
                            <input type="text" className="input" placeholder="Your name" />
                        </div>
                        <div>
                            <label className="label">Email</label>
                            <input type="email" className="input" placeholder="email@example.com" />
                        </div>
                        <button type="submit" className="btn btn-primary">Save Changes</button>
                    </form>
                </div>

                {/* Detection Settings */}
                <div className="card p-6">
                    <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">Default Detection Settings</h2>
                    <form className="space-y-4">
                        <div>
                            <label className="label">Model</label>
                            <select className="input">
                                <option>YOLOv8n (Fast)</option>
                                <option>YOLOv8s (Balanced)</option>
                                <option selected>YOLOv8m (Accurate)</option>
                                <option>YOLOv8l (High Accuracy)</option>
                            </select>
                        </div>
                        <div>
                            <label className="label">Confidence Threshold</label>
                            <input type="range" min="0.1" max="1" step="0.1" defaultValue="0.5" className="w-full" />
                        </div>
                        <div>
                            <label className="label">Frame Skip</label>
                            <input type="number" className="input" defaultValue="2" min="1" max="30" />
                        </div>
                        <button type="submit" className="btn btn-primary">Save Settings</button>
                    </form>
                </div>

                {/* Notifications */}
                <div className="card p-6">
                    <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">Notifications</h2>
                    <div className="space-y-4">
                        <label className="flex items-center gap-3">
                            <input type="checkbox" className="w-4 h-4 rounded" defaultChecked />
                            <span>Email alerts for completed jobs</span>
                        </label>
                        <label className="flex items-center gap-3">
                            <input type="checkbox" className="w-4 h-4 rounded" defaultChecked />
                            <span>Email alerts for overcrowding</span>
                        </label>
                        <label className="flex items-center gap-3">
                            <input type="checkbox" className="w-4 h-4 rounded" />
                            <span>Daily summary email</span>
                        </label>
                    </div>
                </div>

                {/* API Keys */}
                <div className="card p-6">
                    <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">API Keys</h2>
                    <p className="text-sm text-gray-500 mb-4">Manage API keys for external integrations</p>
                    <button className="btn btn-secondary">Generate New Key</button>
                </div>
            </div>
        </div>
    )
}
