import { useState } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { Shield, Eye, EyeOff, User, Mail, Lock, Building2, AlertCircle, Loader2 } from 'lucide-react';
import { useAuth } from '../context/AuthContext';
import './LoginPage.css';

export default function LoginPage() {
    const { login, register } = useAuth();
    const navigate = useNavigate();
    const location = useLocation();
    const from = location.state?.from?.pathname || '/';

    const [mode, setMode] = useState('login'); // 'login' | 'register'
    const [showPassword, setShowPassword] = useState(false);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');

    const [form, setForm] = useState({
        name: '',
        email: '',
        password: '',
        organization: '',
    });

    const handleChange = (e) => {
        setError('');
        setForm(prev => ({ ...prev, [e.target.name]: e.target.value }));
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        setError('');
        setLoading(true);
        try {
            if (mode === 'login') {
                await login({ email: form.email, password: form.password });
            } else {
                if (!form.name.trim()) throw new Error('Name is required');
                if (form.password.length < 6) throw new Error('Password must be at least 6 characters');
                await register({ name: form.name, email: form.email, password: form.password, organization: form.organization });
            }
            navigate(from, { replace: true });
        } catch (err) {
            setError(err.message || 'Something went wrong');
        } finally {
            setLoading(false);
        }
    };

    const switchMode = () => {
        setError('');
        setForm({ name: '', email: '', password: '', organization: '' });
        setMode(prev => prev === 'login' ? 'register' : 'login');
    };

    return (
        <div className="login-root">
            {/* ── Animated background ── */}
            <div className="login-bg">
                <div className="login-bg-orb login-bg-orb--1" />
                <div className="login-bg-orb login-bg-orb--2" />
                <div className="login-bg-grid" />
            </div>

            {/* ── Panel ── */}
            <div className="login-card animate-scale-in">
                {/* Logo */}
                <div className="login-logo">
                    <div className="login-logo-icon">
                        <Shield size={22} />
                    </div>
                    <div>
                        <div className="login-logo-title">EDDS</div>
                        <div className="login-logo-sub">Ethical Deepfake Defence System</div>
                    </div>
                </div>

                {/* Tab switcher */}
                <div className="login-tabs">
                    <button
                        type="button"
                        className={`login-tab ${mode === 'login' ? 'active' : ''}`}
                        onClick={() => mode !== 'login' && switchMode()}
                    >
                        Sign In
                    </button>
                    <button
                        type="button"
                        className={`login-tab ${mode === 'register' ? 'active' : ''}`}
                        onClick={() => mode !== 'register' && switchMode()}
                    >
                        Create Account
                    </button>
                </div>

                {/* Heading */}
                <div className="login-heading">
                    <h1 className="login-title">
                        {mode === 'login' ? 'Welcome back' : 'Get started'}
                    </h1>
                    <p className="login-subtitle">
                        {mode === 'login'
                            ? 'Sign in to access the detection console'
                            : 'Create your account to start analysing media'}
                    </p>
                </div>

                {/* Error alert */}
                {error && (
                    <div className="login-error animate-slide-down">
                        <AlertCircle size={14} />
                        <span>{error}</span>
                    </div>
                )}

                {/* Form */}
                <form className="login-form" onSubmit={handleSubmit} noValidate>
                    {mode === 'register' && (
                        <div className="login-field">
                            <label className="login-label" htmlFor="name">Full Name</label>
                            <div className="login-input-wrap">
                                <User size={15} className="login-input-icon" />
                                <input
                                    id="name"
                                    name="name"
                                    type="text"
                                    autoComplete="name"
                                    required
                                    placeholder="Jane Doe"
                                    value={form.name}
                                    onChange={handleChange}
                                    className="login-input"
                                />
                            </div>
                        </div>
                    )}

                    <div className="login-field">
                        <label className="login-label" htmlFor="email">Email Address</label>
                        <div className="login-input-wrap">
                            <Mail size={15} className="login-input-icon" />
                            <input
                                id="email"
                                name="email"
                                type="email"
                                autoComplete="email"
                                required
                                placeholder="you@example.com"
                                value={form.email}
                                onChange={handleChange}
                                className="login-input"
                            />
                        </div>
                    </div>

                    <div className="login-field">
                        <label className="login-label" htmlFor="password">Password</label>
                        <div className="login-input-wrap">
                            <Lock size={15} className="login-input-icon" />
                            <input
                                id="password"
                                name="password"
                                type={showPassword ? 'text' : 'password'}
                                autoComplete={mode === 'login' ? 'current-password' : 'new-password'}
                                required
                                placeholder={mode === 'register' ? 'min. 6 characters' : '••••••••'}
                                value={form.password}
                                onChange={handleChange}
                                className="login-input"
                            />
                            <button
                                type="button"
                                className="login-eye-btn"
                                onClick={() => setShowPassword(v => !v)}
                                tabIndex={-1}
                                aria-label={showPassword ? 'Hide password' : 'Show password'}
                            >
                                {showPassword ? <EyeOff size={14} /> : <Eye size={14} />}
                            </button>
                        </div>
                    </div>

                    {mode === 'register' && (
                        <div className="login-field">
                            <label className="login-label" htmlFor="organization">
                                Organisation <span className="login-optional">(optional)</span>
                            </label>
                            <div className="login-input-wrap">
                                <Building2 size={15} className="login-input-icon" />
                                <input
                                    id="organization"
                                    name="organization"
                                    type="text"
                                    autoComplete="organization"
                                    placeholder="Research Institute / University…"
                                    value={form.organization}
                                    onChange={handleChange}
                                    className="login-input"
                                />
                            </div>
                        </div>
                    )}

                    <button
                        type="submit"
                        className="login-submit btn btn-primary btn-lg"
                        disabled={loading}
                        id="auth-submit-btn"
                    >
                        {loading
                            ? <><Loader2 size={16} className="login-spinner" /> {mode === 'login' ? 'Signing in…' : 'Creating account…'}</>
                            : mode === 'login' ? 'Sign In' : 'Create Account'
                        }
                    </button>
                </form>

                {/* Footer */}
                <p className="login-footer-text">
                    {mode === 'login' ? "Don't have an account?" : 'Already have an account?'}
                    {' '}
                    <button type="button" className="login-switch-btn" onClick={switchMode}>
                        {mode === 'login' ? 'Create one' : 'Sign in'}
                    </button>
                </p>

                {/* Disclaimer */}
                <div className="login-disclaimer">
                    <Shield size={11} />
                    <span>For authorised research and ethical use only. All analyses are logged.</span>
                </div>
            </div>
        </div>
    );
}
