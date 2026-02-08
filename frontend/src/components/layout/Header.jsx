import { Menu, Bell, Search, User } from 'lucide-react';
import './Header.css';

const Header = ({ onMenuClick }) => {
    return (
        <header className="header">
            <div className="header-left">
                <button className="header-menu-btn" onClick={onMenuClick} aria-label="Toggle menu">
                    <Menu size={20} />
                </button>

                <div className="header-search">
                    <Search size={18} className="search-icon" />
                    <input
                        type="text"
                        placeholder="Search analyses, reports..."
                        className="search-input"
                    />
                    <kbd className="search-kbd">⌘K</kbd>
                </div>
            </div>

            <div className="header-right">
                <button className="header-icon-btn" aria-label="Notifications">
                    <Bell size={20} />
                    <span className="notification-badge">3</span>
                </button>

                <div className="header-divider" />

                <div className="header-user">
                    <div className="user-avatar">
                        <User size={18} />
                    </div>
                    <div className="user-info">
                        <span className="user-name">Researcher</span>
                        <span className="user-role">Admin</span>
                    </div>
                </div>
            </div>
        </header>
    );
};

export default Header;
