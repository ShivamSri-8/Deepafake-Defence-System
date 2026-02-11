import { NavLink, useLocation } from 'react-router-dom';
import {
    Shield,
    Scan,
    History,
    BarChart3,
    Scale,
    Info,
    ChevronLeft,
    ChevronRight,
} from 'lucide-react';
import './Sidebar.css';

const navItems = [
    { path: '/', icon: Shield, label: 'Dashboard' },
    { path: '/detect', icon: Scan, label: 'Detection' },
    { path: '/history', icon: History, label: 'History' },
    { path: '/analytics', icon: BarChart3, label: 'Analytics' },
    { path: '/ethics', icon: Scale, label: 'Ethics' },
    { path: '/about', icon: Info, label: 'About' },
];

const Sidebar = ({ isOpen, onToggle }) => {
    const location = useLocation();

    return (
        <aside className={`sidebar ${isOpen ? 'open' : 'collapsed'}`}>
            <div className="sidebar-header">
                <div className="sidebar-logo">
                    <div className="sidebar-logo-icon">
                        <Shield size={18} />
                    </div>
                    <div className="sidebar-logo-text">
                        <span className="sidebar-logo-title">EDDS</span>
                        <span className="sidebar-logo-sub">Defence System</span>
                    </div>
                </div>
                <button className="sidebar-toggle" onClick={onToggle} aria-label="Toggle sidebar">
                    {isOpen ? <ChevronLeft size={16} /> : <ChevronRight size={16} />}
                </button>
            </div>

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
                                        <Icon size={18} />
                                    </span>
                                    <span className="nav-label">{item.label}</span>
                                </NavLink>
                            </li>
                        );
                    })}
                </ul>
            </nav>

            <div className="sidebar-footer">
                <div className="sidebar-version">
                    <span className="status-dot" />
                    <span>v1.0.0</span>
                </div>
            </div>
        </aside>
    );
};

export default Sidebar;
