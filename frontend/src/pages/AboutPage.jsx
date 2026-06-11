import { motion } from 'framer-motion';
import {
    Shield,
    Brain,
    Layers,
    BarChart3,
    AlertTriangle,
    CheckCircle2,
    ArrowDown,
    Github,
    Linkedin,
    Mail
} from 'lucide-react';
import './AboutPage.css';

const AboutPage = () => {
    const capabilities = [
        { icon: Brain, title: 'Deepfake Detection' },
        { icon: Layers, title: 'Forensic Analysis' },
        { icon: Shield, title: 'Explainable AI' },
        { icon: BarChart3, title: 'Analytics Dashboard' }
    ];

    const techStack = [
        { category: 'Frontend', tech: 'React.js, Tailwind CSS' },
        { category: 'Backend', tech: 'Node.js, Express.js' },
        { category: 'AI Engine', tech: 'FastAPI, Python' },
        { category: 'Database', tech: 'MongoDB' },
        { category: 'ML Frameworks', tech: 'TensorFlow, PyTorch' },
    ];

    const workflowSteps = [
        'Media Upload',
        'Preprocessing',
        'AI Detection',
        'Forensic Analysis',
        'Trust Score Generation',
        'Explainable AI',
        'Results'
    ];

    const keyFeatures = [
        'Multi-Model Detection',
        'Trust Score System',
        'Explainable AI',
        'Detection History',
        'Analytics Dashboard',
        'Forensic Insights'
    ];

    // Animation variants
    const containerVariants = {
        hidden: { opacity: 0 },
        visible: { opacity: 1, transition: { staggerChildren: 0.1 } }
    };

    const itemVariants = {
        hidden: { y: 20, opacity: 0 },
        visible: { y: 0, opacity: 1, transition: { type: 'spring', stiffness: 100, damping: 15 } }
    };

    return (
        <motion.div 
            className="about-page"
            initial="hidden"
            animate="visible"
            variants={containerVariants}
        >
            {/* 1. About EDDS */}
            <motion.section className="about-hero" variants={itemVariants}>
                <div className="hero-icon-container">
                    <Shield size={48} className="hero-shield-icon" />
                </div>
                <h1 className="about-title">About EDDS</h1>
                <p className="about-description">
                    Ethical Deepfake Defence System (EDDS) is an AI-powered platform designed to detect manipulated media through deepfake detection, forensic analysis, explainable AI, and trust scoring.
                </p>
            </motion.section>

            {/* 2. Core Capabilities */}
            <motion.section className="about-section" variants={itemVariants}>
                <h2 className="section-title">Core Capabilities</h2>
                <div className="capabilities-grid">
                    {capabilities.map((cap, index) => (
                        <div key={index} className="capability-card">
                            <cap.icon size={24} className="capability-icon" />
                            <h3>{cap.title}</h3>
                        </div>
                    ))}
                </div>
            </motion.section>

            {/* 3. Technology Stack & 5. Key Features */}
            <motion.div className="split-grid" variants={itemVariants}>
                <section className="about-section">
                    <h2 className="section-title">Technology Stack</h2>
                    <div className="tech-stack-list">
                        {techStack.map((item, index) => (
                            <div key={index} className="tech-item">
                                <span className="tech-category">{item.category}</span>
                                <span className="tech-name">{item.tech}</span>
                            </div>
                        ))}
                    </div>
                </section>

                <section className="about-section">
                    <h2 className="section-title">Key Features</h2>
                    <div className="features-list">
                        {keyFeatures.map((feature, index) => (
                            <div key={index} className="feature-item">
                                <CheckCircle2 size={16} className="feature-check" />
                                <span>{feature}</span>
                            </div>
                        ))}
                    </div>
                </section>
            </motion.div>

            {/* 4. Detection Workflow */}
            <motion.section className="about-section" variants={itemVariants}>
                <h2 className="section-title">Detection Workflow</h2>
                <div className="workflow-diagram">
                    {workflowSteps.map((step, index) => (
                        <div key={index} className="workflow-step-container">
                            <div className="workflow-step">
                                <span className="step-number">{index + 1}</span>
                                <span className="step-label">{step}</span>
                            </div>
                            {index < workflowSteps.length - 1 && (
                                <div className="workflow-arrow">
                                    <ArrowDown size={16} />
                                </div>
                            )}
                        </div>
                    ))}
                </div>
            </motion.section>

            {/* 6. Responsible Usage */}
            <motion.section className="about-section" variants={itemVariants}>
                <div className="disclaimer-card">
                    <AlertTriangle size={24} className="disclaimer-icon" />
                    <div className="disclaimer-content">
                        <h3>Responsible Usage</h3>
                        <p>
                            EDDS provides probabilistic assessments and should not be used as standalone evidence. Results should be reviewed alongside human expertise and additional verification methods.
                        </p>
                    </div>
                </div>
            </motion.section>

            {/* 7. Creator Signature Footer */}
            <motion.section className="creator-footer" variants={itemVariants}>
                <span className="creator-label">Built & Developed By</span>
                <h3 className="creator-name">Shivam Srivastav</h3>
                <div className="creator-links">
                    <a href="https://www.linkedin.com/in/shivam-srivastav-dev/" target="_blank" rel="noopener noreferrer" className="creator-link" aria-label="LinkedIn">
                        <Linkedin size={20} />
                    </a>
                    <a href="https://github.com/ShivamSri-8" target="_blank" rel="noopener noreferrer" className="creator-link" aria-label="GitHub">
                        <Github size={20} />
                    </a>
                    <a href="mailto:shivamsrivastav9889@gmail.com" className="creator-link" aria-label="Email">
                        <Mail size={20} />
                    </a>
                </div>
            </motion.section>

        </motion.div>
    );
};

export default AboutPage;
