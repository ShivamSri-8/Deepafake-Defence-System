import { useState, useCallback, useEffect } from "react";
import { useDropzone } from "react-dropzone";
import { motion, AnimatePresence } from "framer-motion";
import {
  Upload,
  FileVideo,
  FileImage,
  X,
  Scan,
  AlertCircle,
  CheckCircle2,
  AlertTriangle,
  Info,
  Brain,
  Fingerprint,
  Activity,
  Loader2,
  Wifi,
  WifiOff,
  RotateCcw,
  Download,
  Shield,
  Gauge,
  Clock,
  Terminal
} from "lucide-react";

import { 
  performFullAnalysis, 
  checkAIEngineHealth,
  submitAnalysis,
  getAnalysisStatus,
  getAnalysisById,
  formatBackendResult,
  saveLocalAnalysis
} from "../services/api";
import "./DetectionPage.css";
import jsPDF from "jspdf";
import html2canvas from "html2canvas";

// Helpers
const formatBytes = (bytes) => {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1048576) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1048576).toFixed(1)} MB`;
};

const getResultClass = (type) => {
  if (type === "fake") return "result-danger";
  if (type === "real") return "result-success";
  return "result-warning";
};

const getResultIcon = (type) => {
  if (type === "fake") return AlertCircle;
  if (type === "real") return CheckCircle2;
  return AlertTriangle;
};

const getResultLabel = (type) => {
  if (type === "fake") return "Manipulation Detected";
  if (type === "real") return "Appears Authentic";
  return "Inconclusive";
};

// Premium Animated Trust Score Dial
const TrustScoreDial = ({ probability, type }) => {
  const radius = 42;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = circumference - probability * circumference;
  
  // Choose color based on result classification
  let colorVar = "var(--color-warning-400)";
  let glowColor = "rgba(250, 204, 21, 0.4)";
  if (type === "fake") {
      colorVar = "var(--color-danger-400)";
      glowColor = "rgba(239, 68, 68, 0.4)";
  } else if (type === "real") {
      colorVar = "var(--color-success-400)";
      glowColor = "rgba(74, 222, 128, 0.4)";
  }

  return (
    <div className="trust-score-dial">
      <svg viewBox="0 0 100 100" style={{ width: "100%", height: "100%", transform: "rotate(-90deg)", filter: `drop-shadow(0 0 8px ${glowColor})` }}>
        <circle
          cx="50" cy="50" r={radius}
          fill="none" stroke="rgba(255,255,255,0.06)" strokeWidth="6"
        />
        <motion.circle
          cx="50" cy="50" r={radius}
          fill="none" stroke={colorVar} strokeWidth="6" strokeLinecap="round"
          strokeDasharray={circumference}
          initial={{ strokeDashoffset: circumference }}
          animate={{ strokeDashoffset }}
          transition={{ duration: 1.5, ease: [0.16, 1, 0.3, 1] }} // Emil Design easeOut
        />
      </svg>
      <div className="trust-score-center">
        <span className="trust-score-value">
          {(probability * 100).toFixed(0)}%
        </span>
        <span className="trust-score-label">TRUST</span>
      </div>
    </div>
  );
};

const MODEL_LABELS = {
  xception:      { label: "Xception",       weight: "40%" },
  efficientnet:  { label: "EfficientNet-B4", weight: "35%" },
  resnet50:      { label: "ResNet50",        weight: "25%" },
  cnnLstm:       { label: "CNN + LSTM",      weight: "25%" },
  frameAnalysis: { label: "Frame Analysis",  weight: "—"  },
  ensemble:      { label: "Ensemble",        weight: "—"  },
};

const FORENSIC_LABELS = {
  facialLandmarks: "Facial Landmarks",
  eyeBlink: "Eye Blink Pattern",
  lipSync: "Lip Sync",
  frequency: "Frequency Analysis",
  temporalConsistency: "Temporal Consistency",
  frequencyAnalysis: "Frequency Analysis",
};

const DetectionPage = () => {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [currentStage, setCurrentStage] = useState("");
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [apiMode, setApiMode] = useState("checking");
  const [isGeneratingPdf, setIsGeneratingPdf] = useState(false);
  
  // Real-time console logs during scan
  const [scanLogs, setScanLogs] = useState([]);

  useEffect(() => {
    const checkBackend = async () => {
      try {
        await checkAIEngineHealth();
        setApiMode("live");
      } catch {
        setApiMode("simulation");
      }
    };
    checkBackend();
  }, []);

  const addLog = (message) => {
    const timestamp = new Date().toLocaleTimeString();
    setScanLogs((prev) => [...prev, `[${timestamp}] ${message}`].slice(-6)); // keep last 6 logs
  };

  const onDrop = useCallback((acceptedFiles) => {
    const selectedFile = acceptedFiles[0];
    if (!selectedFile) return;

    setFile(selectedFile);
    setResult(null);
    setError(null);
    setProgress(0);
    setCurrentStage("");
    setScanLogs([]);

    const reader = new FileReader();
    reader.onload = () => setPreview(reader.result);
    reader.readAsDataURL(selectedFile);
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      "image/*": [".jpg", ".jpeg", ".png", ".webp"],
      "video/*": [".mp4", ".webm", ".mov"],
    },
    maxFiles: 1,
    maxSize: 100 * 1024 * 1024,
  });

  const clearFile = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
    setProgress(0);
    setCurrentStage("");
    setError(null);
    setScanLogs([]);
  };

  const downloadReport = async () => {
    const element = document.getElementById("report-section");
    if (!element) return;

    setIsGeneratingPdf(true);
    try {
      const canvas = await html2canvas(element, {
        scale: 2,
        useCORS: true,
        backgroundColor: "#0c0e14",
        logging: false,
      });

      const imgData = canvas.toDataURL("image/png");
      const pdf = new jsPDF("p", "mm", "a4");

      const imgWidth = 190;
      const pageHeight = 297;
      const imgHeight = (canvas.height * imgWidth) / canvas.width;

      let position = 10;
      let heightLeft = imgHeight;

      pdf.addImage(imgData, "PNG", 10, position, imgWidth, imgHeight);
      heightLeft -= (pageHeight - 20);

      while (heightLeft > 0) {
        position = -(pageHeight - 20) + 10;
        pdf.addPage();
        pdf.addImage(imgData, "PNG", 10, position + (heightLeft - imgHeight + (pageHeight - 20)), imgWidth, imgHeight);
        heightLeft -= (pageHeight - 20);
      }

      const timestamp = new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-");
      pdf.save(`EDDS_Report_${timestamp}.pdf`);
    } catch (err) {
      console.error("PDF generation error:", err);
    } finally {
      setIsGeneratingPdf(false);
    }
  };

  const startAnalysis = async () => {
    setIsAnalyzing(true);
    setProgress(5);
    setResult(null);
    setError(null);
    setScanLogs([]);
    
    addLog("System initialized. Secure vault handshaking...");
    setCurrentStage("Uploading to secure vault...");

    let backendSucceeded = false;
    try {
      const submission = await submitAnalysis(file);
      const analysisId = submission.data.analysisId;

      setProgress(15);
      addLog("Payload encrypted. Initializing deep neural ensemble network...");
      setCurrentStage("Analysing with ensemble models...");

      let isDone = false;
      let pollCount = 0;
      const MAX_POLLS = 60;

      while (!isDone && pollCount < MAX_POLLS) {
        pollCount++;
        await new Promise(r => setTimeout(r, 2000));

        const statusRes = await getAnalysisStatus(analysisId);
        const { status, stages } = statusRes.data;

        if (stages && stages.length > 0) {
          const current = stages[stages.length - 1];
          const stageLabel = current.stage.replace(/_/g, ' ').toUpperCase();
          
          addLog(`Inference update: ${stageLabel}`);
          setCurrentStage(
            current.stage.replace(/_/g, ' ').charAt(0).toUpperCase() +
            current.stage.replace(/_/g, ' ').slice(1)
          );
          setProgress(Math.min(20 + stages.length * 10, 95));
        }

        if (status === 'completed') {
          isDone = true;
          const finalRes = await getAnalysisById(analysisId);
          addLog("Compiling multi-model tensor matrices. Done.");
          setResult(formatBackendResult(finalRes.data));
          setProgress(100);
          setCurrentStage("Analysis Complete");
          backendSucceeded = true;
        } else if (status === 'failed') {
          throw new Error("Cloud analysis failed.");
        }
      }

      if (!isDone) throw new Error("Analysis timed out.");

    } catch (backendErr) {
      console.warn("Backend path unavailable, using direct AI engine:", backendErr.message);
    }

    if (!backendSucceeded) {
      try {
        setProgress(20);
        addLog("Routing execution queue directly to local GPU inference...");
        setCurrentStage("Connecting to AI engine directly...");
        
        const localResult = await performFullAnalysis(file, (stage, percent) => {
          setCurrentStage(stage);
          setProgress(percent);
          if (percent % 20 === 0) {
            addLog(`Inferring: ${stage} (${percent}%)`);
          }
        });
        
        addLog("Local GPU compilation completed. Structuring forensics...");
        setResult(localResult);
        setError(null);
        
        try {
          await saveLocalAnalysis(localResult, file);
        } catch (saveErr) {
          console.warn("Failed to save analysis to history:", saveErr);
        }
      } catch (localErr) {
        setError(localErr.message || "Analysis failed. Please check the AI engine is running.");
      }
    }

    setIsAnalyzing(false);
  };

  const isVideo = file?.type?.includes("video");
  const ResultIcon = result ? getResultIcon(result.classification) : null;

  // Framer variants
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: { opacity: 1, transition: { staggerChildren: 0.1 } }
  };

  const cardVariants = {
    hidden: { y: 20, opacity: 0 },
    visible: { y: 0, opacity: 1, transition: { type: "spring", stiffness: 100, damping: 15 } }
  };

  return (
    <div className="page">
      <div className="container">
        <motion.div 
          className="detection-page"
          initial="hidden"
          animate="visible"
          variants={containerVariants}
        >
          {/* Page Header */}
          <motion.div className="page-header" variants={cardVariants}>
            <h1 className="page-title">Forensic Analysis Console</h1>
            <p className="page-subtitle">
              Upload an image or video for multi-model deepfake detection with explainable AI forensics.
            </p>
          </motion.div>

          <div className="detection-grid">
            {/* ═══ LEFT PANEL — Upload & Controls ═══ */}
            <motion.div className="card upload-section" variants={cardVariants}>
              <div className="card-header-row">
                <div>
                  <h2 className="card-title">Upload Media</h2>
                  <p className="card-description" style={{ marginBottom: 0 }}>
                    Images up to 100 MB · JPG, PNG, WebP, MP4, WebM
                  </p>
                </div>

                <div className={`api-status ${apiMode}`} title={apiMode === "live" ? "Connected to live AI engine" : "Simulated"}>
                  {apiMode === "live" ? <Wifi size={12} /> : <WifiOff size={12} />}
                  {apiMode === "live" ? "Live AI" : "Simulation"}
                </div>
              </div>

              {/* Dropzone or Preview */}
              {!file ? (
                <div {...getRootProps()} className={`dropzone ${isDragActive ? "active" : ""}`}>
                  <input {...getInputProps()} />
                  <div className="dropzone-content">
                    <div className="dropzone-icon">
                      <Upload size={28} />
                    </div>
                    <p className="dropzone-text">
                      Drag & drop or click to upload
                    </p>
                    <p className="dropzone-hint">
                      IMAGE / VIDEO · MAX 100 MB
                    </p>
                  </div>
                </div>
              ) : (
                <div className="file-preview futuristic-view">
                  <div className="preview-media">
                    {isVideo ? (
                      <video src={preview} controls className="preview-video" />
                    ) : (
                      <img src={preview} alt="Preview" className="preview-image" />
                    )}

                    {/* Laser Scanner animation during scan */}
                    {isAnalyzing && (
                      <div className="laser-scanner" />
                    )}

                    {/* Holographic HUD grid overlay when not yet analyzed */}
                    {!result && (
                      <div className="hud-overlay">
                        <div className="corner tl" />
                        <div className="corner tr" />
                        <div className="corner bl" />
                        <div className="corner br" />
                        <div className="crosshairs" />
                        <div className="coordinate-lines" />
                      </div>
                    )}
                  </div>
                  <div className="preview-info">
                    <div className="preview-icon">
                      {isVideo ? <FileVideo size={16} /> : <FileImage size={16} />}
                    </div>
                    <div className="preview-details">
                      <span className="preview-name">{file.name}</span>
                      <span className="preview-size">{formatBytes(file.size)}</span>
                    </div>
                    <button className="preview-remove" onClick={clearFile} aria-label="Remove file">
                      <X size={14} />
                    </button>
                  </div>
                </div>
              )}

              {/* Error */}
              {error && (
                <div className="error-message">
                  <AlertCircle size={16} />
                  <span>{error}</span>
                  <button onClick={() => setError(null)} aria-label="Dismiss error">
                    <X size={14} />
                  </button>
                </div>
              )}

              {/* Console logs output */}
              {isAnalyzing && scanLogs.length > 0 && (
                <div className="console-logs-container">
                  <div className="console-logs-header">
                    <Terminal size={12} style={{ color: "var(--color-cyan)" }} />
                    <span>Inference Logs</span>
                  </div>
                  <div className="console-logs">
                    {scanLogs.map((log, index) => (
                      <div key={index} className="console-log-item">{log}</div>
                    ))}
                  </div>
                </div>
              )}

              {/* Analyse / Reset Buttons */}
              {file && !result && (
                <button
                  id="start-analysis-btn"
                  className="btn btn-primary btn-lg"
                  style={{ width: "100%" }}
                  onClick={startAnalysis}
                  disabled={isAnalyzing}
                >
                  {isAnalyzing ? (
                    <>
                      <Loader2 size={16} className="spin" />
                      Executing Inference Engine…
                    </>
                  ) : (
                    <>
                      <Scan size={16} />
                      Run Forensic Analysis
                    </>
                  )}
                </button>
              )}

              {result && (
                <button className="btn btn-secondary" style={{ width: "100%" }} onClick={clearFile}>
                  <RotateCcw size={15} />
                  Analyse Another File
                </button>
              )}

              {/* Progress */}
              {isAnalyzing && (
                <div className="analysis-progress">
                  <div className="progress-header">
                    <span className="progress-stage">{currentStage}</span>
                    <span className="progress-percent">{Math.round(progress)}%</span>
                  </div>
                  <div className="progress-bar">
                    <div className="progress-bar-fill" style={{ width: `${progress}%` }} />
                  </div>
                </div>
              )}

              <div className="disclaimer-card">
                <Info size={16} style={{ color: "var(--color-warning-400)", flexShrink: 0 }} />
                <div>
                  <h4>Ethical Notice</h4>
                  <p>
                    Results are probabilistic and must not be used as sole
                    evidence. Always consult qualified forensic experts for
                    consequential decisions.
                  </p>
                </div>
              </div>
            </motion.div>

            {/* ═══ RIGHT PANEL — Results ═══ */}
            <div className="results-section">
              <AnimatePresence mode="wait">
                {/* Placeholder */}
                {!result && !isAnalyzing && (
                  <motion.div
                    key="placeholder"
                    className="results-placeholder"
                    initial={{ opacity: 0, scale: 0.95 }}
                    animate={{ opacity: 1, scale: 1 }}
                    exit={{ opacity: 0 }}
                  >
                    <div className="placeholder-icon">
                      <Scan size={32} />
                    </div>
                    <h3>Awaiting Submission</h3>
                    <p>Upload a file and run the analysis to view forensic results here.</p>
                  </motion.div>
                )}

                {/* Analysing state */}
                {isAnalyzing && (
                  <motion.div
                    key="analyzing"
                    className="analyzing-state"
                    initial={{ opacity: 0, scale: 0.95 }}
                    animate={{ opacity: 1, scale: 1 }}
                    exit={{ opacity: 0 }}
                  >
                    <div className="analyzing-visual">
                      <div className="analyzing-ring" />
                      <div className="analyzing-ring" />
                      <div className="analyzing-ring" />
                      <Brain size={28} className="analyzing-icon" />
                    </div>
                    <h3>Processing…</h3>
                    <p>{currentStage || "Initialising models"}</p>
                  </motion.div>
                )}

                {/* Results Output */}
                {result && (
                  <motion.div
                    key="results"
                    initial={{ opacity: 0, y: 15 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ type: "spring", stiffness: 80, damping: 12 }}
                  >
                    <div id="report-section" className="report-section">
                      <div className="results-content">
                        {/* ── Classification Banner ── */}
                        <div className={`result-card ${getResultClass(result.classification)}`}>
                          <div className="result-header">
                            <div className="result-icon">
                              {ResultIcon && <ResultIcon size={24} />}
                            </div>
                            <div>
                              <span className="result-label">Classification</span>
                              <div className="result-value">
                                {getResultLabel(result.classification)}
                              </div>
                              {result.confidence && (
                                <div className="probability-range">
                                  <span>
                                    CI: {(result.confidence.lower * 100).toFixed(1)}% – {(result.confidence.upper * 100).toFixed(1)}%
                                  </span>
                                </div>
                              )}
                            </div>
                          </div>
                          <div className="result-probability">
                            <TrustScoreDial probability={result.probability} type={result.classification} />
                          </div>
                        </div>

                        {/* Model Predictions */}
                        <div className="card">
                          <div className="result-section-title">
                            <Brain size={12} />
                            Model Predictions
                          </div>
                          <div className="model-predictions">
                            {Object.entries(result.modelPredictions || {}).map(([key, data]) =>
                              data && (
                                <div key={key} className="model-item">
                                  <div className="model-info">
                                    <span className="model-name">{MODEL_LABELS[key]?.label || key}</span>
                                    {MODEL_LABELS[key]?.weight && (
                                      <span className="model-weight">weight {MODEL_LABELS[key].weight}</span>
                                    )}
                                  </div>
                                  <div className="model-bar">
                                    <motion.div
                                      className="model-bar-fill"
                                      initial={{ width: 0 }}
                                      animate={{ width: `${(data.score ?? 0) * 100}%` }}
                                      transition={{ duration: 1, ease: "easeOut" }}
                                    />
                                  </div>
                                  <span className="model-score">{((data.score ?? 0) * 100).toFixed(1)}%</span>
                                </div>
                              )
                            )}
                          </div>
                        </div>

                        {/* Forensic Analysis */}
                        <div className="card">
                          <div className="result-section-title">
                            <Fingerprint size={12} />
                            Forensic Analysis
                          </div>
                          <div className="forensic-grid">
                            {Object.entries(result.forensics || {}).map(([key, data]) => (
                              <div key={key} className={`forensic-item ${data?.anomaly ? "anomaly" : ""}`}>
                                <div className="forensic-header">
                                  <span className="forensic-name">{FORENSIC_LABELS[key] || key}</span>
                                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                                    <span className="forensic-score" style={{ fontSize: '0.8rem', color: 'var(--text-muted)', fontFamily: 'monospace' }}>
                                      {((data?.score ?? 0) * 100).toFixed(1)}%
                                    </span>
                                    <span className={`forensic-status ${data?.anomaly ? "anomaly" : "normal"}`}>
                                      {data?.anomaly ? "Anomaly" : "Normal"}
                                    </span>
                                  </div>
                                </div>
                                <div className="forensic-bar">
                                  <motion.div
                                    className="forensic-bar-fill"
                                    initial={{ width: 0 }}
                                    animate={{ width: `${(data?.score ?? 0) * 100}%` }}
                                    transition={{ duration: 1, ease: "easeOut" }}
                                  />
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>

                        {/* Explainability Grad-CAM Heatmap */}
                        {result.raw?.explanation?.gradcam?.overlay_url && (
                          <div className="card explainability-card">
                            <div className="result-section-title">
                              <Activity size={12} />
                              AI Attention Map (Grad-CAM)
                            </div>
                            <div className="heatmap-viewport">
                              <div>
                                <p className="viewport-label">Original</p>
                                {preview && <img src={preview} alt="Original" className="heatmap-preview-image" />}
                              </div>
                              <div className="heatmap-wrapper">
                                <p className="viewport-label">Attention Heatmap</p>
                                <img
                                  src={`http://localhost:8000${result.raw.explanation.gradcam.overlay_url}`}
                                  alt="Grad-CAM Heatmap"
                                  className="heatmap-preview-image"
                                  onError={(e) => { e.target.style.display = 'none'; }}
                                />
                                <div className="heatmap-hud">
                                  <div className="hud-line-v" />
                                  <div className="hud-line-h" />
                                </div>
                              </div>
                            </div>
                          </div>
                        )}

                        {/* Trust-Aware Intelligence Panel */}
                        {(result.trustScore !== null || result.confidenceLevel) && (
                          <div className="card trust-aware-panel">
                            <div className="result-section-title">
                              <Shield size={12} />
                              Trust-Aware Intelligence
                            </div>
                            <div className="trust-grid">
                              {result.trustScore !== null && (
                                <div className="trust-item">
                                  <div className="trust-item-header">
                                    <Shield size={14} className="trust-icon trust-icon-shield" />
                                    <span className="trust-item-label">Trust Score</span>
                                  </div>
                                  <div className="trust-score-display">
                                    <span className="trust-score-value">{result.trustScore.toFixed(1)}</span>
                                    <span className="trust-score-max">/ 100</span>
                                  </div>
                                  <div className="trust-bar">
                                    <div
                                      className={`trust-bar-fill ${
                                        result.trustScore >= 75 ? 'high' :
                                        result.trustScore >= 50 ? 'moderate' : 'low'
                                      }`}
                                      style={{ width: `${result.trustScore}%` }}
                                    />
                                  </div>
                                  <span className="trust-hint">Composite of confidence, model agreement & forensics</span>
                                </div>
                              )}

                              {result.confidenceLevel && (
                                <div className="trust-item">
                                  <div className="trust-item-header">
                                    <Gauge size={14} className="trust-icon trust-icon-gauge" />
                                    <span className="trust-item-label">Confidence Level</span>
                                  </div>
                                  <span className={`confidence-badge ${
                                    result.confidenceLevel.includes('High') ? 'badge-high' :
                                    result.confidenceLevel.includes('Moderate') ? 'badge-moderate' : 'badge-low'
                                  }`}>
                                    {result.confidenceLevel}
                                  </span>
                                </div>
                              )}
                            </div>
                          </div>
                        )}

                        <div className="processing-info">
                          <Activity size={13} />
                          Processed in {result.processingTime?.toFixed(2)}s · {apiMode === "live" ? "Live AI Engine" : "Simulated"}
                        </div>
                      </div>
                    </div>

                    <button
                      id="download-report-btn"
                      className="btn btn-download-report"
                      onClick={downloadReport}
                      disabled={isGeneratingPdf}
                      style={{ width: "100%", marginTop: "12px" }}
                    >
                      {isGeneratingPdf ? (
                        <><Loader2 size={15} className="spin" /> Generating PDF…</>
                      ) : (
                        <><Download size={15} /> Download Report</>
                      )}
                    </button>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </div>
        </motion.div>
      </div>
    </div>
  );
};

export default DetectionPage;
