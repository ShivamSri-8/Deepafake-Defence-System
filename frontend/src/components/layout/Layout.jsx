import { useState, useEffect } from 'react';
import { useLocation } from 'react-router-dom';
import Header from './Header';
import Sidebar from './Sidebar';
import BackgroundParticles from './BackgroundParticles';
import './Layout.css';

const Layout = ({ children }) => {
    const [sidebarOpen, setSidebarOpen] = useState(window.innerWidth > 768);
    const [isMobile, setIsMobile] = useState(window.innerWidth <= 768);
    const location = useLocation();

    // Handle screen resize
    useEffect(() => {
        const handleResize = () => {
            const mobile = window.innerWidth <= 768;
            setIsMobile(mobile);
            if (mobile && sidebarOpen) {
                setSidebarOpen(false);
            } else if (!mobile && !sidebarOpen) {
                setSidebarOpen(true);
            }
        };

        window.addEventListener('resize', handleResize);
        return () => window.removeEventListener('resize', handleResize);
    }, []);

    // Close sidebar on route change on mobile
    useEffect(() => {
        if (isMobile) {
            setSidebarOpen(false);
        }
    }, [location, isMobile]);

    const toggleSidebar = () => {
        setSidebarOpen(!sidebarOpen);
    };

    return (
        <div className="layout">
            <BackgroundParticles />
            {/* Mobile Backdrop */}
            {isMobile && sidebarOpen && (
                <div
                    className="sidebar-backdrop"
                    onClick={() => setSidebarOpen(false)}
                    style={{
                        position: 'fixed',
                        inset: 0,
                        background: 'rgba(0,0,0,0.5)',
                        backdropFilter: 'blur(4px)',
                        zIndex: 90
                    }}
                />
            )}

            <Sidebar isOpen={sidebarOpen} onToggle={toggleSidebar} />

            <div className={`layout-main ${!sidebarOpen ? 'sidebar-collapsed' : ''}`}>
                <Header onMenuClick={toggleSidebar} />
                <main className="layout-content">
                    {children}
                </main>
            </div>
        </div>
    );
};

export default Layout;
