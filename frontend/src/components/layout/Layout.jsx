import { useState } from 'react';
import Header from './Header';
import Sidebar from './Sidebar';
import './Layout.css';

const Layout = ({ children }) => {
    const [sidebarOpen, setSidebarOpen] = useState(true);

    const toggleSidebar = () => {
        setSidebarOpen(!sidebarOpen);
    };

    return (
        <div className="layout">
            <Sidebar isOpen={sidebarOpen} onToggle={toggleSidebar} />
            <div className={`layout-main ${sidebarOpen ? '' : 'sidebar-collapsed'}`}>
                <Header onMenuClick={toggleSidebar} />
                <main className="layout-content">
                    {children}
                </main>
            </div>
        </div>
    );
};

export default Layout;
