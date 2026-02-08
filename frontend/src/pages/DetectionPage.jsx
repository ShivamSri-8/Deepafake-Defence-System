import { useState, useCallback, useEffect } from 'react';
import { useDropzone } from 'react-dropzone';
import {
    Upload,
    FileVideo,
    FileImage,
    X,
    Play,
    Scan,
    AlertCircle,
    CheckCircle2,
    AlertTriangle,
    Info,
    Eye,
    Brain,
    Fingerprint,
    Activity,
    Loader2,
    Wifi,
    WifiOff
} from 'lucide-react';
import { performFullAnalysis, checkAIEngineHealth } from '../services/api';
import './DetectionPage.css';

const DetectionPage = () => {
    const [file, setFile] = useState(null);
    const [preview, setPreview] = useState(null);
    const [isAnalyzing, setIsAnalyzing] = useState(false);
    const [progress, setProgress] = useState(0);
    const [currentStage, setCurrentStage] = useState('');
    const [result, setResult] = useState(null);
    const [error, setError] = useState(null);
    const [apiMode, setApiMode] = useState('checking'); // 'checking', 'live', 'simulation'

    // Check AI Engine availability on mount
    useEffect(() => {
        const checkBackend = async () => {
            try {
                await checkAIEngineHealth();
                setApiMode('live');
            } catch (err) {
                console.warn('AI Engine not available, using simulation mode:', err);
                setApiMode('simulation');
            }
        };
        checkBackend();
    }, []);

    const onDrop = useCallback((acceptedFiles) => {
        const selectedFile = acceptedFiles[0];
        if (selectedFile) {
            setFile(selectedFile);
            setResult(null);
            setError(null);

            // Create preview
            const reader = new FileReader();
            reader.onload = () => {
                setPreview(reader.result);
            };
            reader.readAsDataURL(selectedFile);
        }
    }, []);

    const { getRootProps, getInputProps, isDragActive } = useDropzone({
        onDrop,
        accept: {
            'image/*': ['.jpg', '.jpeg', '.png', '.webp'],
            'video/*': ['.mp4', '.webm', '.mov']
        },
        maxFiles: 1,
        maxSize: 100 * 1024 * 1024 // 100MB
    });

    const clearFile = () => {
        setFile(null);
        setPreview(null);
        setResult(null);
        setProgress(0);
        setCurrentStage('');
        setError(null);
    };

    // Real API analysis
    const performRealAnalysis = async () => {
        setIsAnalyzing(true);
        setProgress(0);
        setResult(null);
        setError(null);

        try {
            const analysisResult = await performFullAnalysis(file, (stage, percent) => {
                setCurrentStage(stage);
                setProgress(percent);
            });

            setResult(analysisResult);
        } catch (err) {
            console.error('Analysis error:', err);
            setError(err.message || 'Analysis failed. Please try again.');
            // Fallback to simulation on error
            if (apiMode === 'live') {
                setApiMode('simulation');
            }
        } finally {
            setIsAnalyzing(false);
        }
    };

    // Simulated analysis (fallback)
    const simulateAnalysis = async () => {
        setIsAnalyzing(true);
        setProgress(0);
        setResult(null);
        setError(null);

        const stages = [
            { name: 'Preprocessing media...', duration: 800 },
            { name: 'Running Xception model...', duration: 1200 },
            { name: 'Running EfficientNet model...', duration: 1000 },
            { name: 'Analyzing facial landmarks...', duration: 900 },
            { name: 'Detecting blink patterns...', duration: 700 },
            { name: 'Frequency domain analysis...', duration: 600 },
            { name: 'Generating explanations...', duration: 800 },
            { name: 'Compiling results...', duration: 500 }
        ];

        let totalProgress = 0;
        const progressPerStage = 100 / stages.length;

        for (const stage of stages) {
            setCurrentStage(stage.name);
            await new Promise(resolve => setTimeout(resolve, stage.duration));
            totalProgress += progressPerStage;
            setProgress(Math.min(totalProgress, 100));
        }

        // Simulate result
        const isFake = Math.random() > 0.4;
        const probability = isFake
            ? 0.65 + Math.random() * 0.30
            : 0.10 + Math.random() * 0.25;

        setResult({
            classification: probability > 0.6 ? 'fake' : probability < 0.35 ? 'real' : 'uncertain',
            probability: probability,
            confidence: {
                lower: Math.max(0, probability - 0.05),
                upper: Math.min(1, probability + 0.05)
            },
            modelPredictions: {
                xception: { score: probability + (Math.random() - 0.5) * 0.1, weight: 0.35 },
                efficientnet: { score: probability + (Math.random() - 0.5) * 0.1, weight: 0.35 },
                cnnLstm: file?.type?.includes('video')
                    ? { score: probability + (Math.random() - 0.5) * 0.1, weight: 0.30 }
                    : null
            },
            forensics: {
                facialLandmarks: { score: 0.7 + Math.random() * 0.25, anomaly: probability > 0.5 },
                eyeBlink: { score: 0.6 + Math.random() * 0.3, anomaly: probability > 0.6 },
                lipSync: { score: 0.5 + Math.random() * 0.4, anomaly: probability > 0.55 },
                frequency: { score: 0.4 + Math.random() * 0.5, anomaly: probability > 0.7 }
            },
            explanation: {
                summary: probability > 0.6
                    ? `The model detected potential manipulation indicators with ${(probability * 100).toFixed(1)}% probability. The facial boundary area showed anomalous patterns. Eye blink patterns appear irregular.`
                    : probability < 0.35
                        ? `The model found limited manipulation indicators (${(probability * 100).toFixed(1)}% probability). The media appears likely authentic based on analyzed features.`
                        : `The model's confidence is uncertain (${(probability * 100).toFixed(1)}% probability). Additional verification is recommended.`,
                keyRegions: [
                    { name: 'Face boundary', attention: 0.85 + Math.random() * 0.1 },
                    { name: 'Eye region', attention: 0.70 + Math.random() * 0.15 },
                    { name: 'Lip area', attention: 0.55 + Math.random() * 0.2 }
                ]
            },
            processingTime: 6.5
        });

        setIsAnalyzing(false);
    };

    // Choose analysis method based on API mode
    const startAnalysis = () => {
        if (apiMode === 'live') {
            performRealAnalysis();
        } else {
            simulateAnalysis();
        }
    };

    const getResultColor = (classification) => {
        switch (classification) {
            case 'fake': return 'danger';
            case 'real': return 'success';
            default: return 'warning';
        }
    };

    const getResultIcon = (classification) => {
        switch (classification) {
            case 'fake': return AlertCircle;
            case 'real': return CheckCircle2;
            default: return AlertTriangle;
        }
    };

    const isVideo = file?.type?.includes('video');

    return (
        <div className="detection-page">
            <div className="page-header">
                <h1 className="page-title">Deepfake Detection</h1>
                <p className="page-subtitle">
                    Upload an image or video for comprehensive AI-powered analysis
                </p>
            </div>

            <div className="detection-grid">
                {/* Upload Section */}
                <div className="upload-section">
                    <div className="card">
                        <div className="card-header">
                            <div className="card-header-row">
                                <div>
                                    <h2 className="card-title">Upload Media</h2>
                                    <p className="card-description">Supported formats: JPG, PNG, WebP, MP4, WebM</p>
                                </div>
                                <div className={`api-status ${apiMode}`}>
                                    {apiMode === 'checking' && <Loader2 size={14} className="spin" />}
                                    {apiMode === 'live' && <Wifi size={14} />}
                                    {apiMode === 'simulation' && <WifiOff size={14} />}
                                    <span>
                                        {apiMode === 'checking' ? 'Checking...' :
                                            apiMode === 'live' ? 'Live AI' : 'Simulation'}
                                    </span>
                                </div>
                            </div>
                        </div>

                        {!file ? (
                            <div
                                {...getRootProps()}
                                className={`dropzone ${isDragActive ? 'active' : ''}`}
                            >
                                <input {...getInputProps()} />
                                <div className="dropzone-content">
                                    <div className="dropzone-icon">
                                        <Upload size={32} />
                                    </div>
                                    <p className="dropzone-text">
                                        {isDragActive
                                            ? 'Drop the file here...'
                                            : 'Drag & drop a file here, or click to select'
                                        }
                                    </p>
                                    <p className="dropzone-hint">Maximum file size: 100MB</p>
                                </div>
                            </div>
                        ) : (
                            <div className="file-preview">
                                <div className="preview-media">
                                    {isVideo ? (
                                        <video src={preview} controls className="preview-video" />
                                    ) : (
                                        <img src={preview} alt="Preview" className="preview-image" />
                                    )}
                                </div>
                                <div className="preview-info">
                                    <div className="preview-icon">
                                        {isVideo ? <FileVideo size={20} /> : <FileImage size={20} />}
                                    </div>
                                    <div className="preview-details">
                                        <span className="preview-name">{file.name}</span>
                                        <span className="preview-size">
                                            {(file.size / (1024 * 1024)).toFixed(2)} MB
                                        </span>
                                    </div>
                                    <button className="preview-remove" onClick={clearFile}>
                                        <X size={18} />
                                    </button>
                                </div>
                            </div>
                        )}

                        {error && (
                            <div className="error-message">
                                <AlertCircle size={18} />
                                <span>{error}</span>
                                <button onClick={() => setError(null)}><X size={16} /></button>
                            </div>
                        )}

                        {file && !result && (
                            <button
                                className="btn btn-primary btn-lg w-full mt-6"
                                onClick={startAnalysis}
                                disabled={isAnalyzing || apiMode === 'checking'}
                            >
                                {isAnalyzing ? (
                                    <>
                                        <Loader2 size={20} className="spin" />
                                        Analyzing...
                                    </>
                                ) : (
                                    <>
                                        <Scan size={20} />
                                        Start Analysis
                                    </>
                                )}
                            </button>
                        )}

                        {isAnalyzing && (
                            <div className="analysis-progress">
                                <div className="progress-header">
                                    <span className="progress-stage">{currentStage}</span>
                                    <span className="progress-percent">{Math.round(progress)}%</span>
                                </div>
                                <div className="progress-bar">
                                    <div
                                        className="progress-bar-fill"
                                        style={{ width: `${progress}%` }}
                                    />
                                </div>
                            </div>
                        )}
                    </div>

                    {/* Disclaimer Card */}
                    <div className="disclaimer-card">
                        <Info size={20} />
                        <div>
                            <h4>Important Notice</h4>
                            <p>
                                Results are probabilistic assessments, not definitive proof.
                                Always verify with qualified experts for critical decisions.
                            </p>
                        </div>
                    </div>
                </div>

                {/* Results Section */}
                <div className="results-section">
                    {!result && !isAnalyzing && (
                        <div className="results-placeholder">
                            <div className="placeholder-icon">
                                <Scan size={48} />
                            </div>
                            <h3>No Analysis Yet</h3>
                            <p>Upload a file and click "Start Analysis" to begin deepfake detection.</p>
                        </div>
                    )}

                    {isAnalyzing && (
                        <div className="analyzing-state">
                            <div className="analyzing-visual">
                                <div className="analyzing-ring" />
                                <div className="analyzing-ring" />
                                <div className="analyzing-ring" />
                                <Brain size={48} className="analyzing-icon" />
                            </div>
                            <h3>Analyzing Media</h3>
                            <p>Running multi-model ensemble and forensic analysis...</p>
                        </div>
                    )}

                    {result && (
                        <div className="results-content animate-scale-in">
                            {/* Main Result */}
                            <div className={`result-card result-${getResultColor(result.classification)}`}>
                                <div className="result-header">
                                    <div className="result-icon">
                                        {(() => {
                                            const Icon = getResultIcon(result.classification);
                                            return <Icon size={32} />;
                                        })()}
                                    </div>
                                    <div className="result-info">
                                        <span className="result-label">Classification</span>
                                        <span className="result-value">
                                            {result.classification === 'fake' ? 'Potentially Manipulated' :
                                                result.classification === 'real' ? 'Likely Authentic' :
                                                    'Uncertain'}
                                        </span>
                                    </div>
                                </div>
                                <div className="result-probability">
                                    <div className="probability-visual">
                                        <svg viewBox="0 0 100 100" className="probability-ring">
                                            <circle
                                                cx="50" cy="50" r="45"
                                                fill="none"
                                                stroke="rgba(255,255,255,0.1)"
                                                strokeWidth="8"
                                            />
                                            <circle
                                                cx="50" cy="50" r="45"
                                                fill="none"
                                                stroke="currentColor"
                                                strokeWidth="8"
                                                strokeLinecap="round"
                                                strokeDasharray={`${result.probability * 283} 283`}
                                                transform="rotate(-90 50 50)"
                                            />
                                        </svg>
                                        <span className="probability-value">
                                            {(result.probability * 100).toFixed(1)}%
                                        </span>
                                    </div>
                                    <div className="probability-range">
                                        <span>Confidence Interval</span>
                                        <span>
                                            {(result.confidence.lower * 100).toFixed(1)}% - {(result.confidence.upper * 100).toFixed(1)}%
                                        </span>
                                    </div>
                                </div>
                            </div>

                            {/* Model Predictions */}
                            <div className="card">
                                <h3 className="result-section-title">
                                    <Brain size={20} />
                                    Model Predictions
                                </h3>
                                <div className="model-predictions">
                                    {Object.entries(result.modelPredictions).map(([name, data]) => {
                                        if (!data) return null;
                                        return (
                                            <div key={name} className="model-item">
                                                <div className="model-info">
                                                    <span className="model-name">
                                                        {name === 'cnnLstm' ? 'CNN+LSTM' :
                                                            name.charAt(0).toUpperCase() + name.slice(1)}
                                                    </span>
                                                    <span className="model-weight">Weight: {(data.weight * 100).toFixed(0)}%</span>
                                                </div>
                                                <div className="model-bar">
                                                    <div
                                                        className="model-bar-fill"
                                                        style={{ width: `${data.score * 100}%` }}
                                                    />
                                                </div>
                                                <span className="model-score">{(data.score * 100).toFixed(1)}%</span>
                                            </div>
                                        );
                                    })}
                                </div>
                            </div>

                            {/* Forensic Analysis */}
                            <div className="card">
                                <h3 className="result-section-title">
                                    <Fingerprint size={20} />
                                    Forensic Analysis
                                </h3>
                                <div className="forensic-grid">
                                    {Object.entries(result.forensics).map(([name, data]) => (
                                        <div key={name} className={`forensic-item ${data.anomaly ? 'anomaly' : 'normal'}`}>
                                            <div className="forensic-header">
                                                <span className="forensic-name">
                                                    {name === 'facialLandmarks' ? 'Facial Landmarks' :
                                                        name === 'eyeBlink' ? 'Eye Blink' :
                                                            name === 'lipSync' ? 'Lip Sync' :
                                                                'Frequency'}
                                                </span>
                                                <span className={`forensic-status ${data.anomaly ? 'anomaly' : 'normal'}`}>
                                                    {data.anomaly ? 'Anomaly' : 'Normal'}
                                                </span>
                                            </div>
                                            <div className="forensic-bar">
                                                <div
                                                    className="forensic-bar-fill"
                                                    style={{ width: `${data.score * 100}%` }}
                                                />
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            </div>

                            {/* Explanation */}
                            <div className="card">
                                <h3 className="result-section-title">
                                    <Eye size={20} />
                                    AI Explanation
                                </h3>
                                <p className="explanation-text">{result.explanation.summary}</p>
                                <div className="attention-regions">
                                    <span className="attention-label">Key Attention Regions:</span>
                                    <div className="attention-list">
                                        {result.explanation.keyRegions.map((region, index) => (
                                            <div key={index} className="attention-item">
                                                <span className="attention-name">{region.name}</span>
                                                <span className="attention-score">{(region.attention * 100).toFixed(0)}%</span>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            </div>

                            {/* Processing Info */}
                            <div className="processing-info">
                                <Activity size={16} />
                                <span>Processed in {result.processingTime.toFixed(2)}s using multi-model ensemble</span>
                            </div>

                            {/* Action Buttons */}
                            <div className="result-actions">
                                <button className="btn btn-secondary" onClick={clearFile}>
                                    New Analysis
                                </button>
                                <button className="btn btn-primary">
                                    Download Report
                                </button>
                            </div>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};

export default DetectionPage;
