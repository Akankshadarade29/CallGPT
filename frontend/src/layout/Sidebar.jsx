import React from 'react';
import { NavLink, useNavigate } from 'react-router-dom';
import './Sidebar.css';

const Sidebar = () => {
    const navigate = useNavigate();

    const menuItems = [
        { path: '/', label: 'Dashboard', icon: '📊' },
        { path: '/call-logs', label: 'Call Logs', icon: '📞' },
        { path: '/voice-bot-setup', label: 'Voice Bot Setup', icon: '🤖' },
        { path: '/faqs', label: 'FAQs / Responses', icon: '❓' },
        { path: '/analytics', label: 'Analytics', icon: '📈' },
        { path: '/settings', label: 'Settings', icon: '⚙️' },
    ];

    const handleLogout = () => {
        // Logout logic would go here
        console.log('Logged out');
        navigate('/');
    };

    return (
        <aside className="sidebar">
            <div className="sidebar-header">
                <h2 className="sidebar-logo">CallGPT</h2>
            </div>

            <nav className="sidebar-nav">
                <ul className="sidebar-menu">
                    {menuItems.map((item) => (
                        <li key={item.path} className="sidebar-menu-item">
                            <NavLink
                                to={item.path}
                                className={({ isActive }) =>
                                    isActive ? 'sidebar-link active' : 'sidebar-link'
                                }
                            >
                                <span className="sidebar-icon">{item.icon}</span>
                                <span className="sidebar-label">{item.label}</span>
                            </NavLink>
                        </li>
                    ))}
                </ul>
            </nav>

            <div className="sidebar-footer">
                <button className="logout-btn" onClick={handleLogout}>
                    <span className="sidebar-icon">🚪</span>
                    <span className="sidebar-label">Logout</span>
                </button>
            </div>
        </aside>
    );
};

export default Sidebar;
