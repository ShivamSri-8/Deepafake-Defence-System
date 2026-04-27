import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { AuthProvider, useAuth } from './context/AuthContext';
import ProtectedRoute from './components/auth/ProtectedRoute';
import Layout from './components/layout/Layout';
import LoginPage from './pages/LoginPage';
import HomePage from './pages/HomePage';
import DetectionPage from './pages/DetectionPage';
import HistoryPage from './pages/HistoryPage';
import AnalyticsPage from './pages/AnalyticsPage';
import EthicsPage from './pages/EthicsPage';
import AboutPage from './pages/AboutPage';
import './index.css';

// ── Public-only route: redirect to home if already logged in ──
function PublicRoute({ children }) {
  const { isAuthenticated, loading } = useAuth();
  if (loading) return null;
  return isAuthenticated ? <Navigate to="/" replace /> : children;
}

function App() {
  return (
    <AuthProvider>
      <Router>
        <Routes>
          {/* ── Public: Login ── */}
          <Route
            path="/login"
            element={
              <PublicRoute>
                <LoginPage />
              </PublicRoute>
            }
          />

          {/* ── Protected: all app routes inside Layout ── */}
          <Route
            path="/*"
            element={
              <ProtectedRoute>
                <Layout>
                  <Routes>
                    <Route path="/"          element={<HomePage />} />
                    <Route path="/detect"    element={<DetectionPage />} />
                    <Route path="/history"   element={<HistoryPage />} />
                    <Route path="/analytics" element={<AnalyticsPage />} />
                    <Route path="/ethics"    element={<EthicsPage />} />
                    <Route path="/about"     element={<AboutPage />} />
                    {/* Catch-all */}
                    <Route path="*"          element={<Navigate to="/" replace />} />
                  </Routes>
                </Layout>
              </ProtectedRoute>
            }
          />
        </Routes>
      </Router>
    </AuthProvider>
  );
}

export default App;
