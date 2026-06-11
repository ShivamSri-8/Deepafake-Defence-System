import { NavLink, useLocation, useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import {
    Shield,
    Scan,
    History,
    BarChart3,
    Scale,
    Info,
    ChevronLeft,
    ChevronRight,
    LogOut,
    User,
} from 'lucide-react';
import { useAuth } from '../../context/AuthContext';
import './Sidebar.css';

const navItems = [
    { path: '/',          icon: Shield,   label: 'Dashboard' },
    { path: '/detect',    icon: Scan,     label: 'Detection' },
    { path: '/history',   icon: History,  label: 'History'   },
    { path: '/analytics', icon: BarChart3,label: 'Analytics' },
    { path: '/about',     icon: Info,     label: 'About'     },
];

const Sidebar = ({ isOpen, onToggle }) => {
    const location = useLocation();
    const navigate = useNavigate();
    const { user, logout } = useAuth();

    const handleLogout = () => {
        logout();
        navigate('/login');
    };

    const initials = user?.name
        ? user.name.split(' ').map(w => w[0]).join('').toUpperCase().slice(0, 2)
        : '?';

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
                                    style={{ position: 'relative' }}
                                >
                                    {isActive && (
                                        <motion.div
                                            layoutId="active-nav-bg"
                                            className="active-nav-bg"
                                            transition={{ type: 'spring', stiffness: 380, damping: 30 }}
                                            style={{
                                                position: 'absolute',
                                                inset: 0,
                                                borderRadius: 'var(--radius-md)',
                                                background: 'linear-gradient(135deg, rgba(6, 200, 255, 0.08), rgba(100, 60, 255, 0.06))',
                                                border: '1px solid rgba(6, 200, 255, 0.2)',
                                                boxShadow: '0 0 12px rgba(6, 200, 255, 0.1)',
                                                zIndex: 0,
                                            }}
                                        />
                                    )}
                                    <span className="nav-icon" style={{ position: 'relative', zIndex: 1 }}>
                                        <Icon size={18} />
                                    </span>
                                    <span className="nav-label" style={{ position: 'relative', zIndex: 1 }}>{item.label}</span>
                                </NavLink>
                            </li>
                        );
                    })}
                </ul>
            </nav>

            {/* ── User profile block at bottom ── */}
            <div className="sidebar-footer">
                {isOpen ? (
                    <div className="sidebar-user-card">
                        <div className="sidebar-user-avatar">
                            <span>{initials}</span>
                        </div>
                        <div className="sidebar-user-info">
                            <span className="sidebar-user-name">{user?.name || 'User'}</span>
                            <span className="sidebar-user-role">{user?.role || 'user'}</span>
                        </div>
                        <button
                            className="sidebar-logout-btn"
                            onClick={handleLogout}
                            title="Sign Out"
                            id="sidebar-logout-btn"
                        >
                            <LogOut size={14} />
                        </button>
                    </div>
                ) : (
                    <button
                        className="sidebar-logout-btn sidebar-logout-btn--collapsed"
                        onClick={handleLogout}
                        title="Sign Out"
                    >
                        <LogOut size={16} />
                    </button>
                )}

                <div className="sidebar-version">
                    <span className="status-dot" />
                    {isOpen && <span>v1.0.0</span>}
                </div>
            </div>
        </aside>
    );
};

export default Sidebar;
