import { Link } from 'react-router-dom';
import {
    Shield,
    Scan,
    Brain,
    Eye,
    Layers,
    Zap,
    FileCheck,
    TrendingUp,
    ArrowRight,
    FileImage,
    FileVideo,
    Activity,
    Clock,
    AlertCircle,
    CheckCircle2,
    AlertTriangle
} from 'lucide-react';
import './HomePage.css';

const HomePage = () => {
    const stats = [
        {
            icon: FileCheck,
            value: '1,542',
            label: 'Total Analyses',
            change: '+12.5%',
            positive: true
        },
        {
            icon: Shield,
            value: '96.2%',
            label: 'Detection Accuracy',
            change: '+0.8%',
            positive: true
        },
        {
            icon: Clock,
            value: '2.3s',
            label: 'Avg. Processing',
            change: '-0.4s',
            positive: true
        },
        {
            icon: Activity,
            value: '378',
            label: 'This Week',
            change: '+18.2%',
            positive: true
        }
    ];

    const features = [
        {
            icon: Brain,
            title: 'Multi-Model Ensemble',
            description: 'Xception, EfficientNet-B4, and CNN+LSTM models with weighted confidence scoring.'
        },
        {
            icon: Layers,
            title: 'Forensic Analysis',
            description: 'Facial landmarks, blink patterns, lip sync consistency, and frequency domain scanning.'
        },
        {
            icon: Eye,
            title: 'Explainable AI',
            description: 'Grad-CAM heatmaps, LIME explanations, and human-readable analysis summaries.'
        },
        {
            icon: Zap,
            title: 'Real-Time Processing',
            description: 'Sub-3 second analysis for images with GPU-accelerated Deep Learning inference.'
        },
        {
            icon: Shield,
            title: 'Ethical Framework',
            description: 'Transparent confidence intervals, bias monitoring, and responsible disclosure.'
        },
        {
            icon: TrendingUp,
            title: 'Analytics Dashboard',
            description: 'Detection trends, model performance metrics, and historical analysis tracking.'
        }
    ];

    const recentAnalyses = [
        { name: 'interview_final.mp4', type: 'video', result: 'fake', confidence: 94.2, date: '2 hours ago' },
        { name: 'headshot_john.jpg', type: 'image', result: 'real', confidence: 91.8, date: '4 hours ago' },
        { name: 'speech_conference.mp4', type: 'video', result: 'uncertain', confidence: 52.3, date: '6 hours ago' },
        { name: 'profile_photo.png', type: 'image', result: 'real', confidence: 96.7, date: '8 hours ago' },
        { name: 'news_clip.mp4', type: 'video', result: 'fake', confidence: 88.4, date: 'Yesterday' },
    ];

    const getResultIcon = (result) => {
        switch (result) {
            case 'fake': return AlertCircle;
            case 'real': return CheckCircle2;
            default: return AlertTriangle;
        }
    };

    const getResultClass = (result) => {
        switch (result) {
            case 'fake': return 'text-danger';
            case 'real': return 'text-success';
            default: return 'text-warning';
        }
    };

    return (
        <div className="home-page">
            {/* Hero Section */}
            <section className="hero-section">
                <div className="hero-badge-container">
                    <span className="hero-status-badge">
                        <span className="status-dot" />
                        System Operational
                    </span>
                </div>
                <div className="hero-content">
                    <h1 className="hero-title">
                        Defend Against{' '}
                        <span className="hero-title-accent">Deepfakes</span>{' '}
                        with Precision AI
                    </h1>
                    <p className="hero-description">
                        Research-grade detection combining multi-model ensemble analysis,
                        forensic examination, and explainable AI. Built for analysts who
                        demand transparency and accuracy.
                    </p>
                    <div className="hero-actions">
                        <Link to="/detect" className="btn btn-primary btn-lg">
                            <Scan size={18} />
                            Begin Analysis
                        </Link>
                        <Link to="/about" className="btn btn-secondary btn-lg">
                            System Architecture
                            <ArrowRight size={16} />
                        </Link>
                    </div>
                </div>
            </section>

            {/* Stats */}
            <div className="stats-section">
                {stats.map((stat, index) => {
                    const Icon = stat.icon;
                    return (
                        <div key={index} className="stat-card">
                            <div className="stat-icon">
                                <Icon size={18} />
                            </div>
                            <span className="stat-value">{stat.value}</span>
                            <span className="stat-label">{stat.label}</span>
                            <span className={`stat-change ${stat.positive ? 'positive' : 'negative'}`}>
                                <TrendingUp size={12} />
                                {stat.change}
                            </span>
                        </div>
                    );
                })}
            </div>

            {/* Features */}
            <section className="features-section">
                <div className="features-section-header">
                    <h2>Core Capabilities</h2>
                    <p>Detection, analysis, and transparency — built for professional workflows.</p>
                </div>
                <div className="features-grid">
                    {features.map((feature, index) => {
                        const Icon = feature.icon;
                        return (
                            <div key={index} className="feature-card">
                                <div className="feature-icon">
                                    <Icon size={22} />
                                </div>
                                <h3>{feature.title}</h3>
                                <p>{feature.description}</p>
                            </div>
                        );
                    })}
                </div>
            </section>

            {/* Recent Analyses */}
            <section className="recent-section">
                <div className="recent-header">
                    <h2>Recent Analyses</h2>
                    <Link to="/history" className="btn btn-secondary btn-sm">
                        View All
                        <ArrowRight size={14} />
                    </Link>
                </div>
                <div className="recent-table-container">
                    <table className="recent-table">
                        <thead>
                            <tr>
                                <th>File</th>
                                <th>Type</th>
                                <th>Classification</th>
                                <th>Confidence</th>
                                <th>Time</th>
                            </tr>
                        </thead>
                        <tbody>
                            {recentAnalyses.map((item, index) => {
                                const ResultIcon = getResultIcon(item.result);
                                return (
                                    <tr key={index}>
                                        <td>
                                            <span className="file-name-cell">
                                                {item.type === 'video' ? <FileVideo size={14} /> : <FileImage size={14} />}
                                                {item.name}
                                            </span>
                                        </td>
                                        <td>
                                            <span className="file-type-badge">{item.type}</span>
                                        </td>
                                        <td>
                                            <span className={`flex items-center gap-2 ${getResultClass(item.result)}`}>
                                                <ResultIcon size={14} />
                                                {item.result.charAt(0).toUpperCase() + item.result.slice(1)}
                                            </span>
                                        </td>
                                        <td>
                                            <span className="font-mono">{item.confidence.toFixed(1)}%</span>
                                        </td>
                                        <td>{item.date}</td>
                                    </tr>
                                );
                            })}
                        </tbody>
                    </table>
                </div>
            </section>

            {/* CTA */}
            <section className="cta-section">
                <h2>Ready to Analyze?</h2>
                <p>Upload an image or video for comprehensive deepfake detection with full forensic analysis.</p>
                <Link to="/detect" className="btn btn-primary btn-lg">
                    <Scan size={18} />
                    Start Detection
                </Link>
            </section>
        </div>
    );
};

export default HomePage;
