import { Link } from 'react-router-dom';
import {
    Shield,
    Scan,
    Eye,
    Brain,
    BarChart3,
    Clock,
    CheckCircle2,
    AlertTriangle,
    TrendingUp,
    ArrowRight,
    Sparkles
} from 'lucide-react';
import './HomePage.css';

const HomePage = () => {
    // Mock data for dashboard
    const stats = [
        {
            icon: Scan,
            value: '1,542',
            label: 'Total Analyses',
            change: '+12%',
            changeType: 'positive',
            variant: 'primary'
        },
        {
            icon: CheckCircle2,
            value: '823',
            label: 'Verified Real',
            change: '+8%',
            changeType: 'positive',
            variant: 'success'
        },
        {
            icon: AlertTriangle,
            value: '612',
            label: 'Detected Fake',
            change: '+15%',
            changeType: 'negative',
            variant: 'danger'
        },
        {
            icon: Clock,
            value: '2.3s',
            label: 'Avg Response',
            change: '-0.5s',
            changeType: 'positive',
            variant: 'warning'
        },
    ];

    const recentAnalyses = [
        { id: 1, name: 'video_sample_01.mp4', type: 'Video', result: 'fake', confidence: 94.2, time: '2 min ago' },
        { id: 2, name: 'interview_clip.mp4', type: 'Video', result: 'real', confidence: 87.5, time: '15 min ago' },
        { id: 3, name: 'headshot_02.jpg', type: 'Image', result: 'fake', confidence: 91.8, time: '1 hour ago' },
        { id: 4, name: 'press_photo.png', type: 'Image', result: 'real', confidence: 96.3, time: '2 hours ago' },
        { id: 5, name: 'speech_extract.mp4', type: 'Video', result: 'uncertain', confidence: 52.1, time: '3 hours ago' },
    ];

    const features = [
        {
            icon: Brain,
            title: 'Multi-Model Ensemble',
            description: 'Xception, EfficientNet, and CNN+LSTM models working together for accurate detection.'
        },
        {
            icon: Eye,
            title: 'Forensic Analysis',
            description: 'Facial landmarks, blink patterns, lip sync, and frequency analysis for thorough examination.'
        },
        {
            icon: Sparkles,
            title: 'Explainable AI',
            description: 'Grad-CAM heatmaps and human-readable explanations for transparent results.'
        },
    ];

    return (
        <div className="home-page">
            {/* Hero Section */}
            <section className="hero-section">
                <div className="hero-content">
                    <div className="hero-badge">
                        <Shield size={16} />
                        <span>Research-Grade AI System</span>
                    </div>
                    <h1 className="hero-title">
                        Ethical Deepfake
                        <span className="gradient-text"> Defence System</span>
                    </h1>
                    <p className="hero-description">
                        AI-powered platform for probabilistic deepfake detection, forensic analysis,
                        and explainable AI. Designed for researchers, journalists, and security professionals.
                    </p>
                    <div className="hero-actions">
                        <Link to="/detect" className="btn btn-primary btn-lg">
                            <Scan size={20} />
                            Start Detection
                        </Link>
                        <Link to="/about" className="btn btn-secondary btn-lg">
                            Learn More
                            <ArrowRight size={18} />
                        </Link>
                    </div>
                </div>
                <div className="hero-visual">
                    <div className="hero-orb hero-orb-1" />
                    <div className="hero-orb hero-orb-2" />
                    <div className="hero-orb hero-orb-3" />
                    <div className="hero-shield">
                        <Shield size={120} />
                    </div>
                </div>
            </section>

            {/* Stats Section */}
            <section className="stats-section">
                <div className="stats-grid">
                    {stats.map((stat, index) => {
                        const Icon = stat.icon;
                        return (
                            <div key={index} className={`stat-card ${stat.variant}`}>
                                <div className="stat-icon">
                                    <Icon size={24} />
                                </div>
                                <div className="stat-value">{stat.value}</div>
                                <div className="stat-label">{stat.label}</div>
                                <div className={`stat-change ${stat.changeType}`}>
                                    <TrendingUp size={14} />
                                    <span>{stat.change} this week</span>
                                </div>
                            </div>
                        );
                    })}
                </div>
            </section>

            {/* Features Section */}
            <section className="features-section">
                <h2 className="section-title">Powerful Detection Capabilities</h2>
                <div className="features-grid">
                    {features.map((feature, index) => {
                        const Icon = feature.icon;
                        return (
                            <div key={index} className="feature-card">
                                <div className="feature-icon">
                                    <Icon size={28} />
                                </div>
                                <h3 className="feature-title">{feature.title}</h3>
                                <p className="feature-description">{feature.description}</p>
                            </div>
                        );
                    })}
                </div>
            </section>

            {/* Recent Analyses */}
            <section className="recent-section">
                <div className="section-header">
                    <h2 className="section-title">Recent Analyses</h2>
                    <Link to="/history" className="section-link">
                        View All <ArrowRight size={16} />
                    </Link>
                </div>
                <div className="recent-table-wrapper">
                    <table className="recent-table">
                        <thead>
                            <tr>
                                <th>File Name</th>
                                <th>Type</th>
                                <th>Result</th>
                                <th>Confidence</th>
                                <th>Time</th>
                            </tr>
                        </thead>
                        <tbody>
                            {recentAnalyses.map((analysis) => (
                                <tr key={analysis.id}>
                                    <td className="file-name">{analysis.name}</td>
                                    <td>
                                        <span className="badge badge-neutral">{analysis.type}</span>
                                    </td>
                                    <td>
                                        <span className={`badge badge-${analysis.result === 'real' ? 'success' :
                                                analysis.result === 'fake' ? 'danger' : 'warning'
                                            }`}>
                                            {analysis.result.charAt(0).toUpperCase() + analysis.result.slice(1)}
                                        </span>
                                    </td>
                                    <td>
                                        <div className="confidence-cell">
                                            <div className="confidence-bar">
                                                <div
                                                    className="confidence-fill"
                                                    style={{
                                                        width: `${analysis.confidence}%`,
                                                        background: analysis.result === 'real'
                                                            ? 'var(--color-success-500)'
                                                            : analysis.result === 'fake'
                                                                ? 'var(--color-danger-500)'
                                                                : 'var(--color-warning-500)'
                                                    }}
                                                />
                                            </div>
                                            <span className="confidence-value">{analysis.confidence}%</span>
                                        </div>
                                    </td>
                                    <td className="time-cell">{analysis.time}</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            </section>

            {/* CTA Section */}
            <section className="cta-section">
                <div className="cta-content">
                    <h2 className="cta-title">Ready to Analyze Media?</h2>
                    <p className="cta-description">
                        Upload your images or videos for comprehensive deepfake analysis with
                        explainable AI results.
                    </p>
                    <Link to="/detect" className="btn btn-primary btn-lg">
                        <Scan size={20} />
                        Start Detection Now
                    </Link>
                </div>
            </section>
        </div>
    );
};

export default HomePage;
