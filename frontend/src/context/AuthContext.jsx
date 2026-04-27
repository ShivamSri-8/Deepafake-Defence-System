import { createContext, useContext, useState, useEffect, useCallback } from 'react';

const AuthContext = createContext(null);

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || 'http://localhost:8080/api';

export function AuthProvider({ children }) {
    const [user, setUser] = useState(null);
    const [token, setToken] = useState(() => localStorage.getItem('edds_token'));
    const [loading, setLoading] = useState(true); // true while verifying stored token

    // ── Verify token on mount ─────────────────────────────────
    useEffect(() => {
        if (token) {
            verifyToken(token);
        } else {
            setLoading(false);
        }
    }, []); // eslint-disable-line react-hooks/exhaustive-deps

    const verifyToken = async (t) => {
        try {
            const res = await fetch(`${BACKEND_URL}/v1/auth/me`, {
                headers: { Authorization: `Bearer ${t}` },
            });
            if (res.ok) {
                const data = await res.json();
                setUser(data.data);
            } else {
                // Token invalid or expired
                clearAuth();
            }
        } catch {
            clearAuth();
        } finally {
            setLoading(false);
        }
    };

    const clearAuth = () => {
        localStorage.removeItem('edds_token');
        localStorage.removeItem('token'); // legacy key used by api.js
        setToken(null);
        setUser(null);
    };

    // ── Register ──────────────────────────────────────────────
    const register = useCallback(async ({ name, email, password, organization }) => {
        const res = await fetch(`${BACKEND_URL}/v1/auth/register`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name, email, password, organization }),
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || 'Registration failed');
        persistAuth(data.token, data.user);
        return data.user;
    }, []);

    // ── Login ─────────────────────────────────────────────────
    const login = useCallback(async ({ email, password }) => {
        const res = await fetch(`${BACKEND_URL}/v1/auth/login`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ email, password }),
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || 'Login failed');
        persistAuth(data.token, data.user);
        return data.user;
    }, []);

    // ── Logout ────────────────────────────────────────────────
    const logout = useCallback(() => {
        clearAuth();
    }, []);

    const persistAuth = (newToken, newUser) => {
        localStorage.setItem('edds_token', newToken);
        localStorage.setItem('token', newToken); // keep api.js happy
        setToken(newToken);
        setUser(newUser);
    };

    return (
        <AuthContext.Provider value={{ user, token, loading, login, register, logout, isAuthenticated: !!user }}>
            {children}
        </AuthContext.Provider>
    );
}

// eslint-disable-next-line react-refresh/only-export-components
export function useAuth() {
    const ctx = useContext(AuthContext);
    if (!ctx) throw new Error('useAuth must be used inside <AuthProvider>');
    return ctx;
}
