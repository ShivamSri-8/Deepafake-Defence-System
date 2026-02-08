import {
    Shield,
    Brain,
    Eye,
    Code2,
    Database,
    Layers,
    Github,
    Mail,
    BookOpen,
    Award
} from 'lucide-react';
import './AboutPage.css';

const AboutPage = () => {
    const techStack = [
        { name: 'React.js', category: 'Frontend', description: 'Modern UI with hooks and context' },
        { name: 'Chart.js', category: 'Visualization', description: 'Interactive data visualization' },
        { name: 'Node.js + Express', category: 'Backend', description: 'REST API server' },
        { name: 'FastAPI', category: 'AI Service', description: 'Python ML inference API' },
        { name: 'MongoDB', category: 'Database', description: 'Document-based storage' },
        { name: 'TensorFlow/PyTorch', category: 'ML Framework', description: 'Deep learning models' },
    ];

    const models = [
        {
            name: 'Xception',
            accuracy: '94.3%',
            description: 'Depthwise separable convolutions for efficient image classification'
        },
        {
            name: 'EfficientNet-B4',
            accuracy: '95.1%',
            description: 'Compound scaling for optimal depth, width, and resolution'
        },
        {
            name: 'CNN + LSTM',
            accuracy: '93.8%',
            description: 'Spatial-temporal analysis for video deepfake detection'
        },
        {
            name: 'Ensemble',
            accuracy: '96.2%',
            description: 'Weighted voting of all models for highest accuracy'
        },
    ];

    const references = [
        'Rössler, A., et al. (2019). "FaceForensics++: Learning to Detect Manipulated Facial Images."',
        'Li, Y., et al. (2020). "Celeb-DF: A Large-scale Challenging Dataset for DeepFake Forensics."',
        'Chollet, F. (2017). "Xception: Deep Learning with Depthwise Separable Convolutions."',
        'Selvaraju, R., et al. (2017). "Grad-CAM: Visual Explanations from Deep Networks."',
        'Tan, M., & Le, Q. (2019). "EfficientNet: Rethinking Model Scaling for CNNs."',
    ];

    return (
        <div className="about-page">
            <div className="page-header">
                <h1 className="page-title">About EDDS</h1>
                <p className="page-subtitle">
                    Learn about the Ethical Deepfake Defence System architecture and technology
                </p>
            </div>

            {/* Hero Section */}
            <section className="about-hero">
                <div className="hero-icon">
                    <Shield size={64} />
                </div>
                <h2>Ethical Deepfake Defence System</h2>
                <p>
                    A research-grade AI platform for probabilistic deepfake detection, forensic analysis,
                    and explainable AI. Designed for researchers, journalists, and security professionals
                    with a focus on transparency and ethical AI principles.
                </p>
                <div className="hero-badges">
                    <span className="hero-badge">
                        <Brain size={16} />
                        Multi-Model Ensemble
                    </span>
                    <span className="hero-badge">
                        <Eye size={16} />
                        Explainable AI
                    </span>
                    <span className="hero-badge">
                        <Shield size={16} />
                        Ethical Framework
                    </span>
                </div>
            </section>

            {/* Features */}
            <section className="about-section">
                <h2 className="section-title">Core Capabilities</h2>
                <div className="features-grid">
                    <div className="feature-card">
                        <div className="feature-icon">
                            <Brain size={28} />
                        </div>
                        <h3>Multi-Model Detection</h3>
                        <p>Ensemble of Xception, EfficientNet, and CNN+LSTM models for accurate classification with confidence intervals.</p>
                    </div>
                    <div className="feature-card">
                        <div className="feature-icon">
                            <Layers size={28} />
                        </div>
                        <h3>Forensic Analysis</h3>
                        <p>Facial landmarks, blink patterns, lip sync, and frequency domain analysis for comprehensive examination.</p>
                    </div>
                    <div className="feature-card">
                        <div className="feature-icon">
                            <Eye size={28} />
                        </div>
                        <h3>Explainable AI</h3>
                        <p>Grad-CAM heatmaps, LIME explanations, and human-readable summaries for transparent results.</p>
                    </div>
                    <div className="feature-card">
                        <div className="feature-icon">
                            <Award size={28} />
                        </div>
                        <h3>Analytics Dashboard</h3>
                        <p>Detection history, trend analysis, confidence distributions, and model performance metrics.</p>
                    </div>
                </div>
            </section>

            {/* Model Performance */}
            <section className="about-section">
                <h2 className="section-title">Model Architecture</h2>
                <div className="models-grid">
                    {models.map((model, index) => (
                        <div key={index} className="model-card">
                            <div className="model-header">
                                <h3>{model.name}</h3>
                                <span className="model-accuracy">{model.accuracy}</span>
                            </div>
                            <p>{model.description}</p>
                        </div>
                    ))}
                </div>
            </section>

            {/* Tech Stack */}
            <section className="about-section">
                <h2 className="section-title">Technology Stack</h2>
                <div className="tech-grid">
                    {techStack.map((tech, index) => (
                        <div key={index} className="tech-card">
                            <span className="tech-category">{tech.category}</span>
                            <h4>{tech.name}</h4>
                            <p>{tech.description}</p>
                        </div>
                    ))}
                </div>
            </section>

            {/* Architecture Diagram */}
            <section className="about-section">
                <h2 className="section-title">System Architecture</h2>
                <div className="architecture-diagram">
                    <div className="arch-layer">
                        <span className="arch-label">Client Layer</span>
                        <div className="arch-boxes">
                            <div className="arch-box">React Frontend</div>
                            <div className="arch-box">Admin Panel</div>
                        </div>
                    </div>
                    <div className="arch-arrow">↓</div>
                    <div className="arch-layer">
                        <span className="arch-label">API Layer</span>
                        <div className="arch-boxes">
                            <div className="arch-box">Express.js Gateway</div>
                        </div>
                    </div>
                    <div className="arch-arrow">↓</div>
                    <div className="arch-layer">
                        <span className="arch-label">Services Layer</span>
                        <div className="arch-boxes">
                            <div className="arch-box">Media Service</div>
                            <div className="arch-box">Detection Service</div>
                            <div className="arch-box">Analytics Service</div>
                        </div>
                    </div>
                    <div className="arch-arrow">↓</div>
                    <div className="arch-layer">
                        <span className="arch-label">AI/ML Layer</span>
                        <div className="arch-boxes">
                            <div className="arch-box highlight">FastAPI + Models</div>
                            <div className="arch-box highlight">Forensics Engine</div>
                            <div className="arch-box highlight">XAI Layer</div>
                        </div>
                    </div>
                    <div className="arch-arrow">↓</div>
                    <div className="arch-layer">
                        <span className="arch-label">Data Layer</span>
                        <div className="arch-boxes">
                            <div className="arch-box">MongoDB</div>
                            <div className="arch-box">File Storage</div>
                        </div>
                    </div>
                </div>
            </section>

            {/* References */}
            <section className="about-section">
                <h2 className="section-title">Academic References</h2>
                <div className="references-list">
                    {references.map((ref, index) => (
                        <div key={index} className="reference-item">
                            <BookOpen size={16} />
                            <span>{ref}</span>
                        </div>
                    ))}
                </div>
            </section>

            {/* Footer */}
            <section className="about-footer">
                <div className="footer-content">
                    <h3>Ethical Deepfake Defence System</h3>
                    <p>Final Year Major Project • 2026</p>
                    <div className="footer-links">
                        <a href="#" className="footer-link">
                            <Github size={20} />
                            <span>GitHub</span>
                        </a>
                        <a href="#" className="footer-link">
                            <BookOpen size={20} />
                            <span>Documentation</span>
                        </a>
                        <a href="#" className="footer-link">
                            <Mail size={20} />
                            <span>Contact</span>
                        </a>
                    </div>
                </div>
            </section>
        </div>
    );
};

export default AboutPage;
