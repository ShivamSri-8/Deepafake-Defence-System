import {
    Shield,
    AlertTriangle,
    Scale,
    BookOpen,
    Users,
    Lock,
    Eye,
    FileWarning,
    CheckCircle,
    XCircle,
    Info,
    ExternalLink
} from 'lucide-react';
import './EthicsPage.css';

const EthicsPage = () => {
    const principles = [
        {
            icon: Eye,
            title: 'Transparency',
            description: 'All predictions include confidence intervals and limitations are clearly communicated. We never claim 100% accuracy.'
        },
        {
            icon: BookOpen,
            title: 'Explainability',
            description: 'Every prediction is accompanied by visual attention maps and human-readable explanations for full understanding.'
        },
        {
            icon: Scale,
            title: 'Fairness',
            description: 'Models are tested across diverse demographics with continuous bias monitoring and reporting.'
        },
        {
            icon: Lock,
            title: 'Accountability',
            description: 'Clear disclaimers on all outputs, comprehensive audit trails, and feedback mechanisms for corrections.'
        },
        {
            icon: Shield,
            title: 'Harm Prevention',
            description: 'Education about deepfake risks, misuse prevention guidelines, and responsible disclosure practices.'
        },
        {
            icon: Users,
            title: 'Human-Centered',
            description: 'Designed as a decision-support tool that enhances human judgment, not replaces it.'
        }
    ];

    const appropriateUses = [
        'Educational research and academic study',
        'Journalism verification and fact-checking support',
        'Organizational security assessments',
        'Personal verification of suspicious content',
        'Training programs for digital literacy',
        'Forensic analysis preliminary screening'
    ];

    const inappropriateUses = [
        'Standalone legal evidence without expert verification',
        'Harassment, defamation, or false accusations',
        'Mass surveillance or targeting individuals',
        'Reverse-engineering to create better deepfakes',
        'Commercial use without proper disclosure',
        'Automated decision-making without human review'
    ];

    const faqs = [
        {
            question: 'What are deepfakes?',
            answer: 'Deepfakes are synthetic media created using artificial intelligence, typically deep learning techniques, to manipulate or generate visual and audio content that appears authentic. The term combines "deep learning" and "fake."'
        },
        {
            question: 'How accurate is this system?',
            answer: 'Our ensemble model achieves approximately 96% accuracy on benchmark datasets. However, accuracy varies with media quality, manipulation techniques, and other factors. Results should always be verified by experts.'
        },
        {
            question: 'Can this system be 100% accurate?',
            answer: 'No detection system can achieve 100% accuracy. Deepfake technology continues to evolve, and detection methods must constantly adapt. We emphasize probabilistic assessment over definitive claims.'
        },
        {
            question: 'Should I use these results in court?',
            answer: 'No. Our results are not suitable as standalone legal evidence. Legal proceedings require certified forensic analysis by qualified experts. Use our system only as a preliminary screening tool.'
        },
        {
            question: 'How do I interpret confidence scores?',
            answer: 'Confidence scores represent the probability of manipulation. Scores above 70% suggest likely manipulation, below 30% suggest likely authenticity, and 30-70% indicates uncertainty requiring additional verification.'
        }
    ];

    return (
        <div className="ethics-page">
            <div className="page-header">
                <h1 className="page-title">Ethics & Awareness</h1>
                <p className="page-subtitle">
                    Understanding deepfakes, responsible AI usage, and ethical guidelines
                </p>
            </div>

            {/* Disclaimer Banner */}
            <div className="disclaimer-banner">
                <AlertTriangle size={24} />
                <div>
                    <h3>Important Disclaimer</h3>
                    <p>
                        This system provides probabilistic assessments, not definitive proof. Results should
                        never be used as standalone evidence and always require verification by qualified experts.
                    </p>
                </div>
            </div>

            {/* Ethical Principles */}
            <section className="ethics-section">
                <h2 className="section-title">Our Ethical Principles</h2>
                <div className="principles-grid">
                    {principles.map((principle, index) => {
                        const Icon = principle.icon;
                        return (
                            <div key={index} className="principle-card">
                                <div className="principle-icon">
                                    <Icon size={24} />
                                </div>
                                <h3>{principle.title}</h3>
                                <p>{principle.description}</p>
                            </div>
                        );
                    })}
                </div>
            </section>

            {/* Usage Guidelines */}
            <section className="ethics-section">
                <h2 className="section-title">Usage Guidelines</h2>
                <div className="usage-grid">
                    <div className="usage-card appropriate">
                        <div className="usage-header">
                            <CheckCircle size={24} />
                            <h3>Appropriate Uses</h3>
                        </div>
                        <ul className="usage-list">
                            {appropriateUses.map((use, index) => (
                                <li key={index}>
                                    <CheckCircle size={16} />
                                    {use}
                                </li>
                            ))}
                        </ul>
                    </div>

                    <div className="usage-card inappropriate">
                        <div className="usage-header">
                            <XCircle size={24} />
                            <h3>Inappropriate Uses</h3>
                        </div>
                        <ul className="usage-list">
                            {inappropriateUses.map((use, index) => (
                                <li key={index}>
                                    <XCircle size={16} />
                                    {use}
                                </li>
                            ))}
                        </ul>
                    </div>
                </div>
            </section>

            {/* What Are Deepfakes */}
            <section className="ethics-section">
                <h2 className="section-title">Understanding Deepfakes</h2>
                <div className="info-cards">
                    <div className="info-card">
                        <FileWarning size={32} />
                        <h3>What Are Deepfakes?</h3>
                        <p>
                            Deepfakes are AI-generated synthetic media that can replace or manipulate a person's
                            likeness in images and videos. They use deep learning techniques, particularly
                            Generative Adversarial Networks (GANs) and autoencoders.
                        </p>
                    </div>

                    <div className="info-card">
                        <AlertTriangle size={32} />
                        <h3>Potential Harms</h3>
                        <p>
                            Deepfakes can be used for misinformation, identity fraud, non-consensual content
                            creation, reputation damage, and electoral manipulation. The technology poses
                            significant societal risks when misused.
                        </p>
                    </div>

                    <div className="info-card">
                        <Shield size={32} />
                        <h3>Detection Importance</h3>
                        <p>
                            As deepfake quality improves, detection becomes crucial for maintaining trust in
                            digital media. Our system combines multiple detection methods to identify manipulation
                            artifacts and provide transparency.
                        </p>
                    </div>
                </div>
            </section>

            {/* FAQs */}
            <section className="ethics-section">
                <h2 className="section-title">Frequently Asked Questions</h2>
                <div className="faq-list">
                    {faqs.map((faq, index) => (
                        <div key={index} className="faq-item">
                            <div className="faq-question">
                                <Info size={18} />
                                <h4>{faq.question}</h4>
                            </div>
                            <p className="faq-answer">{faq.answer}</p>
                        </div>
                    ))}
                </div>
            </section>

            {/* Resources */}
            <section className="ethics-section">
                <h2 className="section-title">Additional Resources</h2>
                <div className="resources-grid">
                    <a href="#" className="resource-link">
                        <BookOpen size={20} />
                        <span>Research Paper: Detection Methods</span>
                        <ExternalLink size={14} />
                    </a>
                    <a href="#" className="resource-link">
                        <Scale size={20} />
                        <span>Legal Guidelines for Media Verification</span>
                        <ExternalLink size={14} />
                    </a>
                    <a href="#" className="resource-link">
                        <Users size={20} />
                        <span>Digital Literacy Training Materials</span>
                        <ExternalLink size={14} />
                    </a>
                    <a href="#" className="resource-link">
                        <Shield size={20} />
                        <span>Reporting Deepfake Abuse</span>
                        <ExternalLink size={14} />
                    </a>
                </div>
            </section>
        </div>
    );
};

export default EthicsPage;
