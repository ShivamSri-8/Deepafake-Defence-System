import { useAuth } from '../../context/AuthContext';
import { Navigate, useLocation } from 'react-router-dom';

/**
 * Wraps any route that requires authentication.
 * If not authenticated → redirect to /login (preserving intended path).
 * While verifying stored token → show a minimal fullscreen loader.
 */
export default function ProtectedRoute({ children }) {
    const { isAuthenticated, loading } = useAuth();
    const location = useLocation();

    if (loading) {
        return (
            <div style={{
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                height: '100vh', background: 'var(--bg-primary)',
                flexDirection: 'column', gap: '16px'
            }}>
                <div className="spinner" />
                <span style={{ color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', fontSize: '0.8125rem' }}>
                    Verifying session…
                </span>
            </div>
        );
    }

    if (!isAuthenticated) {
        return <Navigate to="/login" state={{ from: location }} replace />;
    }

    return children;
}
