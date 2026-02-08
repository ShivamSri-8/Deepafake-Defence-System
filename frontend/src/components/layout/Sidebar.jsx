import { NavLink, useLocation } from 'react-router-dom';
import {
    Shield,
    Search,
    History,
    BarChart3,
    BookOpen,
    Info,
    ChevronLeft,
    ChevronRight,
    Scan
} from 'lucide-react';
import './Sidebar.css';

const navItems = [
    { path: '/', icon: Shield, label: 'Dashboard' },
    { path: '/detect', icon: Scan, label: 'Detection' },
    { path: '/history', icon: History, label: 'History' },
    { path: '/analytics', icon: BarChart3, label: 'Analytics' },
    { path: '/ethics', icon: BookOpen, label: 'Ethics & Awareness' },
    { path: '/about', icon: Info, label: 'About' },
];

const Sidebar = ({ isOpen, onToggle }) => {
    const location = useLocation();

    return (
        <aside className={`sidebar ${isOpen ? 'open' : 'collapsed'}`}>
            {/* Logo */}
            <div className="sidebar-header">
                <div className="sidebar-logo">
                    <div className="logo-icon">
                        <Shield size={28} />
                    </div>
                    {isOpen && (
                        <div className="logo-text">
                            <span className="logo-title">EDDS</span>
                            <span className="logo-subtitle">Deepfake Defence</span>
                        </div>
                    )}
                </div>
                <button className="sidebar-toggle" onClick={onToggle} aria-label="Toggle sidebar">
                    {isOpen ? <ChevronLeft size={20} /> : <ChevronRight size={20} />}
                </button>
            </div>

            {/* Navigation */}
            <nav className="sidebar-nav">
                <ul className="nav-list">
                    {navItems.map((item) => {
                        const Icon = item.icon;
                        const isActive = location.pathname === item.path;

                        return (
                            <li key={item.path} className="nav-item">
                                <NavLink
                                    to={item.path}
                                    className={`nav-link ${isActive ? 'active' : ''}`}
                                    title={!isOpen ? item.label : undefined}
                                >
                                    <span className="nav-icon">
                                        <Icon size={20} />
                                    </span>
                                    {isOpen && <span className="nav-label">{item.label}</span>}
                                    {isActive && <span className="nav-indicator" />}
                                </NavLink>
                            </li>
                        );
                    })}
                </ul>
            </nav>

            {/* Footer */}
            <div className="sidebar-footer">
                {isOpen && (
                    <div className="sidebar-info">
                        <p className="info-version">Version 1.0.0</p>
                        <p className="info-text">Research-Grade AI System</p>
                    </div>
                )}
            </div>
        </aside>
    );
};

export default Sidebar;
