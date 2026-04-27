import { useState, useRef, useEffect } from 'react';
import { Menu, Bell, User, LogOut, Settings, ChevronDown, Shield } from 'lucide-react';
import { useAuth } from '../../context/AuthContext';
import { useNavigate } from 'react-router-dom';
import './Header.css';

const Header = ({ onMenuClick }) => {
    const { user, logout } = useAuth();
    const navigate = useNavigate();
    const [dropdownOpen, setDropdownOpen] = useState(false);
    const dropdownRef = useRef(null);

    // Close dropdown on outside click
    useEffect(() => {
        const handler = (e) => {
            if (dropdownRef.current && !dropdownRef.current.contains(e.target)) {
                setDropdownOpen(false);
            }
        };
        document.addEventListener('mousedown', handler);
        return () => document.removeEventListener('mousedown', handler);
    }, []);

    const handleLogout = () => {
        setDropdownOpen(false);
        logout();
        navigate('/login');
    };

    // Generate initials from user name
    const initials = user?.name
        ? user.name.split(' ').map(w => w[0]).join('').toUpperCase().slice(0, 2)
        : '?';

    return (
        <header className="header">
            <div className="header-left">
                <button className="header-menu-btn" onClick={onMenuClick} aria-label="Toggle menu">
                    <Menu size={20} />
                </button>

                <div className="header-brand-badge">
                    <Shield size={14} className="header-brand-icon" />
                    <span>EDDS</span>
                </div>
            </div>

            <div className="header-right">
                {/* Notification bell */}
                <button className="header-icon-btn" aria-label="Notifications">
                    <Bell size={18} />
                </button>

                <div className="header-divider" />

                {/* User dropdown */}
                <div className="header-user-wrap" ref={dropdownRef}>
                    <button
                        className="header-user"
                        onClick={() => setDropdownOpen(v => !v)}
                        aria-label="User menu"
                        aria-expanded={dropdownOpen}
                        id="header-user-btn"
                    >
                        <div className="user-avatar">
                            <span className="user-avatar-initials">{initials}</span>
                        </div>
                        <div className="user-info">
                            <span className="user-name">{user?.name || 'User'}</span>
                            <span className="user-role">{user?.role || 'user'}</span>
                        </div>
                        <ChevronDown
                            size={14}
                            className={`user-chevron ${dropdownOpen ? 'rotated' : ''}`}
                        />
                    </button>

                    {/* Dropdown menu */}
                    {dropdownOpen && (
                        <div className="header-dropdown animate-slide-down" role="menu">
                            <div className="header-dropdown-profile">
                                <div className="header-dropdown-avatar">
                                    <span>{initials}</span>
                                </div>
                                <div>
                                    <div className="header-dropdown-name">{user?.name}</div>
                                    <div className="header-dropdown-email">{user?.email}</div>
                                </div>
                            </div>

                            <div className="header-dropdown-divider" />

                            <button
                                className="header-dropdown-item"
                                role="menuitem"
                                onClick={() => setDropdownOpen(false)}
                            >
                                <Settings size={14} />
                                Account Settings
                            </button>

                            <div className="header-dropdown-divider" />

                            <button
                                className="header-dropdown-item header-dropdown-item--danger"
                                role="menuitem"
                                onClick={handleLogout}
                                id="header-logout-btn"
                            >
                                <LogOut size={14} />
                                Sign Out
                            </button>
                        </div>
                    )}
                </div>
            </div>
        </header>
    );
};

export default Header;
