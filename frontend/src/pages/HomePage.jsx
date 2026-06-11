import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { motion, animate } from 'framer-motion';
import {
    Shield,
    Scan,
    Brain,
    Eye,
    Layers,
    Zap,
    FileCheck,
    TrendingUp,
    TrendingDown,
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

/* ═══════════════════════════════════════════════════════════════
   SEMANTIC METRIC CONFIGURATION
   Each metric defines whether higher or lower values are better.
   Delta colors are derived automatically — never show positive
   outcomes in red or negative outcomes in green.
   ═══════════════════════════════════════════════════════════════ */
const METRIC_CONFIG = {
    'Total Analyses':          { higherIsBetter: true },
    'Detection Accuracy':      { higherIsBetter: true },
    'Avg. Processing':         { lowerIsBetter: true },
    'This Week':               { higherIsBetter: true },
    'False Positive Rate':     { lowerIsBetter: true },
    'Trust Score Reliability':  { higherIsBetter: true },
};

function getSemanticDeltaColor(label, changeStr) {
    const config = METRIC_CONFIG[label];
    if (!config) return 'neutral';
    const isIncrease = !changeStr.startsWith('-');
    if (config.higherIsBetter) return isIncrease ? 'positive' : 'negative';
    if (config.lowerIsBetter)  return isIncrease ? 'negative' : 'positive';
    return 'neutral';
}

function getDeltaIcon(semanticClass) {
    return semanticClass === 'negative' ? TrendingDown : TrendingUp;
}

/* ═══════════════════════════════════════════════════════════════ */

const AnimatedNumber = ({ value }) => {
    const numeric = parseFloat(value.replace(/,/g, '').replace(/%/g, '').replace(/s/g, ''));
    const isPercent = value.includes('%');
    const isSeconds = value.includes('s');
    const hasComma = value.includes(',');

    const [displayVal, setDisplayVal] = useState('0');

    useEffect(() => {
        const controls = animate(0, numeric, {
            duration: 1.5,
            ease: 'easeOut',
            onUpdate: (latest) => {
                let formatted = latest;
                if (isSeconds) {
                    formatted = latest.toFixed(1) + 's';
                } else if (isPercent) {
                    formatted = latest.toFixed(1) + '%';
                } else if (hasComma) {
                    formatted = Math.floor(latest).toLocaleString();
                } else {
                    formatted = Math.floor(latest).toString();
                }
                setDisplayVal(formatted);
            }
        });
        return () => controls.stop();
    }, [numeric, isPercent, isSeconds, hasComma]);

    return <span>{displayVal}</span>;
};

const HomePage = () => {
    const stats = [
        {
            icon: FileCheck,
            value: '1,542',
            label: 'Total Analyses',
            change: '+12.5%',
        },
        {
            icon: Shield,
            value: '96.2%',
            label: 'Detection Accuracy',
            change: '+0.8%',
        },
        {
            icon: Clock,
            value: '2.3s',
            label: 'Avg. Processing',
            change: '-0.4s',
        },
        {
            icon: Activity,
            value: '378',
            label: 'This Week',
            change: '+18.2%',
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

    // Framer Motion Animation Variants
    const containerVariants = {
        hidden: { opacity: 0 },
        visible: {
            opacity: 1,
            transition: {
                staggerChildren: 0.1
            }
        }
    };

    const itemVariants = {
        hidden: { y: 20, opacity: 0 },
        visible: {
            y: 0,
            opacity: 1,
            transition: {
                type: 'spring',
                stiffness: 100,
                damping: 15
            }
        }
    };

    const hudVariants = {
        hidden: { opacity: 0, y: 10, scale: 0.95 },
        visible: {
            opacity: 1,
            y: 0,
            scale: 1,
            transition: { type: 'spring', stiffness: 120, damping: 18 }
        }
    };

    return (
        <motion.div
            className="home-page"
            initial="hidden"
            animate="visible"
            variants={containerVariants}
        >
            {/* Hero Section */}
            <section className="hero-section">
                <div className="hero-content-centered">
                    <motion.div className="hero-badge-container" variants={itemVariants}>
                        <span className="hero-status-badge">
                            <span className="status-dot animate-pulse" style={{ background: 'var(--color-aurora-cyan)', boxShadow: '0 0 8px var(--color-aurora-cyan)' }} />
                            System Operational
                        </span>
                        <span className="hero-status-badge" style={{ marginLeft: '12px' }}>
                            <span className="status-dot animate-pulse" style={{ background: '#8c3bff', boxShadow: '0 0 8px #8c3bff' }} />
                            Active Ensemble: Online
                        </span>
                    </motion.div>
                    <motion.h1 className="hero-title" variants={itemVariants}>
                        Defend Against <br/>
                        <span className="gradient-text-aurora">Deepfakes</span> <br/>
                        with Precision
                    </motion.h1>
                    <motion.p className="hero-description" variants={itemVariants}>
                        Secure your enterprise from the devastating impact of deepfakes
                        and synthetic media. EDDS provides advanced AI-driven detection
                        at scale for unparalleled trust and integrity.
                    </motion.p>
                    <motion.div className="hero-actions" variants={itemVariants}>
                        <Link to="/detect" className="btn btn-primary btn-lg">
                            <Scan size={18} />
                            Start Securing Now
                        </Link>
                        <Link to="/about" className="btn btn-secondary btn-lg">
                            Explore Platform
                            <ArrowRight size={16} />
                        </Link>
                    </motion.div>
                </div>
            </section>

            {/* Stats — with semantic delta colors */}
            <motion.div className="stats-section" variants={containerVariants}>
                {stats.map((stat, index) => {
                    const Icon = stat.icon;
                    const semanticClass = getSemanticDeltaColor(stat.label, stat.change);
                    const DeltaIcon = getDeltaIcon(semanticClass);
                    return (
                        <motion.div 
                            key={index} 
                            className="stat-card" 
                            variants={itemVariants}
                            whileHover={{ y: -5, borderColor: 'rgba(6, 200, 255, 0.35)', boxShadow: '0 10px 30px -15px rgba(6, 200, 255, 0.15)' }}
                            style={{ cursor: 'default' }}
                        >
                            <div className="stat-icon">
                                <Icon size={20} />
                            </div>
                            <span className="stat-value">
                                <AnimatedNumber value={stat.value} />
                            </span>
                            <span className="stat-label">{stat.label}</span>
                            <span className={`stat-change ${semanticClass}`}>
                                <DeltaIcon size={12} />
                                {stat.change}
                            </span>
                        </motion.div>
                    );
                })}
            </motion.div>

            {/* Features */}
            <section className="features-section">
                <div className="features-section-header">
                    <h2>Core Capabilities</h2>
                    <p>Detection, analysis, and transparency — built for professional workflows.</p>
                </div>
                <motion.div className="features-grid bento-layout" variants={containerVariants}>
                    {features.map((feature, index) => {
                        const Icon = feature.icon;
                        return (
                            <motion.div 
                                key={index} 
                                className={`feature-card bento-item-${index + 1}`}
                                variants={itemVariants}
                                whileHover={{ y: -4, borderColor: 'rgba(6, 200, 255, 0.25)', boxShadow: '0 15px 35px -10px rgba(0,0,0,0.5)' }}
                            >
                                <div className="feature-icon">
                                    <Icon size={22} />
                                </div>
                                <h3>{feature.title}</h3>
                                <p>{feature.description}</p>
                            </motion.div>
                        );
                    })}
                </motion.div>
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
            <motion.section className="cta-section" variants={itemVariants}>
                <h2>Ready to <span className="gradient-text-static">Analyze</span>?</h2>
                <p>Upload an image or video for comprehensive deepfake detection with full forensic analysis.</p>
                <Link to="/detect" className="btn btn-primary btn-lg">
                    <Scan size={18} />
                    Start Detection
                </Link>
            </motion.section>
        </motion.div>
    );
};

export default HomePage;
