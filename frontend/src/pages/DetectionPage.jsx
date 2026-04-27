import { useState, useCallback, useEffect } from "react";
import { useDropzone } from "react-dropzone";
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
} from "lucide-react";

import { 
  performFullAnalysis, 
  checkAIEngineHealth,
  submitAnalysis,
  getAnalysisStatus,
  getAnalysisById,
  formatBackendResult
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

// SVG probability ring
const ProbabilityRing = ({ probability, type }) => {
  const radius = 36;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - probability * circumference;

  return (
    <div className="probability-visual">
      <svg className="probability-ring" viewBox="0 0 88 88">
        {/* Track */}
        <circle
          cx="44"
          cy="44"
          r={radius}
          fill="none"
          stroke="rgba(255,255,255,0.06)"
          strokeWidth="4"
        />
        {/* Fill */}
        <circle
          cx="44"
          cy="44"
          r={radius}
          fill="none"
          stroke="currentColor"
          strokeWidth="4"
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          transform="rotate(-90 44 44)"
          style={{ transition: "stroke-dashoffset 0.8s var(--ease-out)" }}
        />
      </svg>
      <span className="probability-value">
        {(probability * 100).toFixed(0)}%
      </span>
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

  const onDrop = useCallback((acceptedFiles) => {
    const selectedFile = acceptedFiles[0];
    if (!selectedFile) return;

    setFile(selectedFile);
    setResult(null);
    setError(null);
    setProgress(0);
    setCurrentStage("");

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
  };

  // ── Download Report as PDF ────────────────────────────────────────────
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

      // First page
      pdf.addImage(imgData, "PNG", 10, position, imgWidth, imgHeight);
      heightLeft -= (pageHeight - 20);

      // Extra pages if report is longer than one A4
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
    setCurrentStage("Uploading to secure vault...");

    // ── Path A: Try backend (requires auth) ──────────────────────────────
    let backendSucceeded = false;
    try {
      const submission = await submitAnalysis(file);
      const analysisId = submission.data.analysisId;

      setProgress(15);
      setCurrentStage("Analysing with ensemble models...");

      // Poll for completion
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
          setCurrentStage(
            current.stage.replace(/_/g, ' ').charAt(0).toUpperCase() +
            current.stage.replace(/_/g, ' ').slice(1)
          );
          setProgress(Math.min(20 + stages.length * 10, 95));
        }

        if (status === 'completed') {
          isDone = true;
          const finalRes = await getAnalysisById(analysisId);
          // Backend returns { data: { result, modelPredictions, forensics, explanation, ... } }
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
      // Backend unavailable or user not logged in—fall through to direct AI engine
      console.warn("Backend path unavailable, using direct AI engine:", backendErr.message);
    }

    // ── Path B: Direct AI engine (no auth needed) ────────────────────────
    if (!backendSucceeded) {
      try {
        setProgress(20);
        setCurrentStage("Connecting to AI engine directly...");
        const localResult = await performFullAnalysis(file, (stage, percent) => {
          setCurrentStage(stage);
          setProgress(percent);
        });
        setResult(localResult);
        setError(null);
      } catch (localErr) {
        setError(localErr.message || "Analysis failed. Please check the AI engine is running.");
      }
    }

    setIsAnalyzing(false);
  };

  const isVideo = file?.type?.includes("video");
  const ResultIcon = result ? getResultIcon(result.classification) : null;

  return (
    <div className="page">
      <div className="container">
        <div className="detection-page">
          {/* Page Header */}
          <div className="page-header">
            <h1 className="page-title">Forensic Analysis Console</h1>
            <p className="page-subtitle">
              Upload an image or video for multi-model deepfake detection with explainable AI forensics.
            </p>
          </div>

          <div className="detection-grid">
            {/* ═══ LEFT PANEL — Upload & Controls ═══ */}
            <div className="card upload-section">
              {/* Card Header */}
              <div className="card-header-row">
                <div>
                  <h2 className="card-title">Upload Media</h2>
                  <p className="card-description" style={{ marginBottom: 0 }}>
                    Images up to 100 MB · JPG, PNG, WebP, MP4, WebM
                  </p>
                </div>

                <div
                  className={`api-status ${apiMode}`}
                  title={
                    apiMode === "live"
                      ? "Connected to live AI engine"
                      : apiMode === "simulation"
                      ? "Using simulated results — AI engine offline"
                      : "Checking connection…"
                  }
                >
                  {apiMode === "live" ? (
                    <Wifi size={12} />
                  ) : apiMode === "simulation" ? (
                    <WifiOff size={12} />
                  ) : (
                    <Loader2 size={12} className="spin" />
                  )}
                  {apiMode === "live"
                    ? "Live AI"
                    : apiMode === "simulation"
                    ? "Simulation"
                    : "Checking…"}
                </div>
              </div>

              {/* Dropzone or Preview */}
              {!file ? (
                <div
                  {...getRootProps()}
                  className={`dropzone ${isDragActive ? "active" : ""}`}
                >
                  <input {...getInputProps()} />
                  <div className="dropzone-content">
                    <div className="dropzone-icon">
                      <Upload size={28} />
                    </div>
                    <p className="dropzone-text">
                      {isDragActive
                        ? "Drop the file here…"
                        : "Drag & drop or click to upload"}
                    </p>
                    <p className="dropzone-hint">
                      IMAGE / VIDEO · MAX 100 MB
                    </p>
                  </div>
                </div>
              ) : (
                <div className="file-preview">
                  <div className="preview-media">
                    {isVideo ? (
                      <video
                        src={preview}
                        controls
                        className="preview-video"
                      />
                    ) : (
                      <img
                        src={preview}
                        alt="Preview"
                        className="preview-image"
                      />
                    )}
                  </div>
                  <div className="preview-info">
                    <div className="preview-icon">
                      {isVideo ? (
                        <FileVideo size={16} />
                      ) : (
                        <FileImage size={16} />
                      )}
                    </div>
                    <div className="preview-details">
                      <span className="preview-name">{file.name}</span>
                      <span className="preview-size">
                        {formatBytes(file.size)}
                      </span>
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
                      Analysing…
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
                <button
                  className="btn btn-secondary"
                  style={{ width: "100%" }}
                  onClick={clearFile}
                >
                  <RotateCcw size={15} />
                  Analyse Another File
                </button>
              )}

              {/* Progress */}
              {isAnalyzing && (
                <div className="analysis-progress">
                  <div className="progress-header">
                    <span className="progress-stage">{currentStage}</span>
                    <span className="progress-percent">
                      {Math.round(progress)}%
                    </span>
                  </div>
                  <div className="progress-bar">
                    <div
                      className="progress-bar-fill"
                      style={{ width: `${progress}%` }}
                    />
                  </div>
                </div>
              )}

              {/* Disclaimer */}
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
            </div>

            {/* ═══ RIGHT PANEL — Results ═══ */}
            <div className="results-section">
              {/* Placeholder */}
              {!result && !isAnalyzing && (
                <div className="results-placeholder">
                  <div className="placeholder-icon">
                    <Scan size={32} />
                  </div>
                  <h3>Awaiting Submission</h3>
                  <p>
                    Upload a file and run the analysis to view forensic results
                    here.
                  </p>
                </div>
              )}

              {/* Analysing state */}
              {isAnalyzing && (
                <div className="analyzing-state">
                  <div className="analyzing-visual">
                    <div className="analyzing-ring" />
                    <div className="analyzing-ring" />
                    <div className="analyzing-ring" />
                    <Brain size={28} className="analyzing-icon" />
                  </div>
                  <h3>Processing…</h3>
                  <p>{currentStage || "Initialising models"}</p>
                </div>
              )}

              {/* Results */}
              {result && (
                <>
                <div id="report-section" className="report-section">
                <div className="results-content animate-slide-up">
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
                              CI: {(result.confidence.lower * 100).toFixed(1)}%
                              –{(result.confidence.upper * 100).toFixed(1)}%
                            </span>
                          </div>
                        )}
                      </div>
                    </div>
                    <div className="result-probability">
                      <ProbabilityRing
                        probability={result.probability}
                        type={result.classification}
                      />
                    </div>
                  </div>
                  {/* ── Simulation Mode Warning ── */}
                  {(result.raw?.detection?.result?.notes?.some(n => n.includes('Simulation')) ||
                    result.explanation?.summary?.toLowerCase().includes('simulated') ||
                    apiMode === 'simulation') && (
                    <div className="error-message" style={{
                      background: 'rgba(245, 158, 11, 0.08)',
                      borderColor: 'rgba(245, 158, 11, 0.35)',
                      color: 'var(--color-warning-400)',
                      marginBottom: '12px'
                    }}>
                      <AlertTriangle size={15} />
                      <span>
                        <strong>Simulation Mode:</strong> No trained model weights found. These results are
                        statistically simulated — not from real AI inference. Train the models for accurate detection.
                      </span>
                    </div>
                  )}


                  <div className="card">
                    <div className="result-section-title">
                      <Brain size={12} />
                      Model Predictions
                    </div>
                    <div className="model-predictions">
                      {Object.entries(result.modelPredictions || {}).map(
                        ([key, data]) =>
                          data && (
                            <div key={key} className="model-item">
                              <div className="model-info">
                                <span className="model-name">
                                  {MODEL_LABELS[key]?.label || key}
                                </span>
                                {MODEL_LABELS[key]?.weight && (
                                  <span className="model-weight">
                                    weight {MODEL_LABELS[key].weight}
                                  </span>
                                )}
                              </div>
                              <div className="model-bar">
                                <div
                                  className="model-bar-fill"
                                  style={{
                                    width: `${(data.score ?? 0) * 100}%`,
                                  }}
                                />
                              </div>
                              <span className="model-score">
                                {((data.score ?? 0) * 100).toFixed(1)}%
                              </span>
                            </div>
                          )
                      )}
                    </div>
                  </div>

                  {/* ── Forensic Analysis ── */}
                  <div className="card">
                    <div className="result-section-title">
                      <Fingerprint size={12} />
                      Forensic Analysis
                    </div>
                    <div className="forensic-grid">
                      {Object.entries(result.forensics || {}).map(
                        ([key, data]) => (
                          <div
                            key={key}
                            className={`forensic-item ${
                              data?.anomaly ? "anomaly" : ""
                            }`}
                          >
                            <div className="forensic-header">
                              <span className="forensic-name">
                                {FORENSIC_LABELS[key] || key}
                              </span>
                              <span
                                className={`forensic-status ${
                                  data?.anomaly ? "anomaly" : "normal"
                                }`}
                              >
                                {data?.anomaly ? "Anomaly" : "Normal"}
                              </span>
                            </div>
                            <div className="forensic-bar">
                              <div
                                className="forensic-bar-fill"
                                style={{
                                  width: `${(data?.score ?? 0) * 100}%`,
                                }}
                              />
                            </div>
                          </div>
                        )
                      )}
                    </div>
                  </div>

                  {/* ── AI Explanation ── */}
                  {result.explanation && (
                    <div className="card">
                      <div className="result-section-title">
                        <Activity size={12} />
                        AI Explanation
                      </div>
                      {result.explanation.summary && (
                        <p className="explanation-text">
                          {result.explanation.summary}
                        </p>
                      )}
                      {result.explanation.keyRegions?.length > 0 && (
                        <div className="attention-regions">
                          <span className="attention-label">
                            Key Attention Regions
                          </span>
                          <div className="attention-list">
                            {result.explanation.keyRegions.map((region, i) => (
                              <div key={i} className="attention-item">
                                <span>{region.name}</span>
                                <span className="attention-score">
                                  {(region.attention * 100).toFixed(0)}%
                                </span>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  )}

                  {/* ── Grad-CAM Heatmap ── */}
                  {result.raw?.explanation?.gradcam?.overlay_url && (
                    <div className="card">
                      <div className="result-section-title">
                        <Activity size={12} />
                        AI Attention Map (Grad-CAM)
                      </div>
                      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px', marginTop: '12px' }}>
                        <div>
                          <p style={{ margin: '0 0 6px', fontSize: '0.7rem', letterSpacing: '0.08em', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Original</p>
                          {preview && (
                            <img
                              src={preview}
                              alt="Original"
                              style={{ width: '100%', borderRadius: '6px', objectFit: 'cover' }}
                            />
                          )}
                        </div>
                        <div>
                          <p style={{ margin: '0 0 6px', fontSize: '0.7rem', letterSpacing: '0.08em', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Attention Heatmap</p>
                          <img
                            src={`http://localhost:8000${result.raw.explanation.gradcam.overlay_url}`}
                            alt="Grad-CAM Heatmap"
                            style={{ width: '100%', borderRadius: '6px', objectFit: 'cover' }}
                            onError={(e) => { e.target.style.display = 'none'; }}
                          />
                        </div>
                      </div>
                      {result.raw.explanation.gradcam.focus_regions?.length > 0 && (
                        <div style={{ marginTop: '10px', display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
                          {result.raw.explanation.gradcam.focus_regions.map((r, i) => (
                            <span key={i} className="attention-item" style={{ fontSize: '0.72rem', padding: '3px 10px', borderRadius: '20px', background: 'rgba(0,212,170,0.1)', color: 'var(--color-accent-400)', border: '1px solid rgba(0,212,170,0.2)' }}>
                              {r.replace(/_/g, ' ')}
                            </span>
                          ))}
                        </div>
                      )}
                    </div>
                  )}

                  {/* ── Trust-Aware Intelligence Panel (v1.1) ── */}
                  {(result.trustScore !== null || result.confidenceLevel) && (
                    <div className="card trust-aware-panel">
                      <div className="result-section-title">
                        <Shield size={12} />
                        Trust-Aware Intelligence
                      </div>

                      <div className="trust-grid">
                        {/* Trust Score */}
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
                            <span className="trust-hint">
                              Composite of confidence, model agreement & forensics
                            </span>
                          </div>
                        )}

                        {/* Confidence Level */}
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

                        {/* Temporal Status (video only) */}
                        {result.temporalVariance !== null && result.temporalVariance !== undefined && (
                          <div className="trust-item trust-item-temporal">
                            <div className="trust-item-header">
                              <Clock size={14} className="trust-icon trust-icon-clock" />
                              <span className="trust-item-label">Temporal Consistency</span>
                            </div>
                            <div className="temporal-info">
                              <span className={`temporal-badge ${
                                result.temporalLabel === 'Stable' ? 'temporal-stable' :
                                result.temporalLabel === 'Moderate Variation' ? 'temporal-moderate' : 'temporal-unstable'
                              }`}>
                                {result.temporalLabel}
                              </span>
                              <span className="temporal-variance">
                                σ² = {result.temporalVariance.toFixed(4)}
                              </span>
                            </div>
                          </div>
                        )}
                      </div>
                    </div>
                  )}

                  {/* ── Processing Info ── */}
                  <div className="processing-info">
                    <Activity size={13} />
                    Processed in {result.processingTime?.toFixed(2)}s ·{" "}
                    {apiMode === "live" ? "Live AI Engine" : "Simulated"}
                  </div>

                  {/* ── Report Disclaimer Footer ── */}
                  <div className="report-disclaimer">
                    <Info size={13} />
                    <span>
                      This is an AI-generated analysis and should not be
                      considered as absolute proof. Results are probabilistic
                      and must be verified by qualified experts.
                    </span>
                  </div>
                </div>
                </div>

                {/* ── Download Report Button (outside captured area) ── */}
                <button
                  id="download-report-btn"
                  className="btn btn-download-report"
                  onClick={downloadReport}
                  disabled={isGeneratingPdf}
                >
                  {isGeneratingPdf ? (
                    <>
                      <Loader2 size={15} className="spin" />
                      Generating PDF…
                    </>
                  ) : (
                    <>
                      <Download size={15} />
                      Download Report
                    </>
                  )}
                </button>
                </>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DetectionPage;
