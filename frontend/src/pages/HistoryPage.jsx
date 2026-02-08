import { useState, useEffect, useCallback } from 'react';
import {
    Search,
    Filter,
    FileImage,
    FileVideo,
    CheckCircle2,
    AlertCircle,
    AlertTriangle,
    Eye,
    Trash2,
    Download,
    ChevronLeft,
    ChevronRight,
    Loader2,
    RefreshCw,
    Database
} from 'lucide-react';
import { getAnalysisHistory, getAnalysisById } from '../services/api';
import './HistoryPage.css';

const HistoryPage = () => {
    const [searchQuery, setSearchQuery] = useState('');
    const [filter, setFilter] = useState('all');
    const [currentPage, setCurrentPage] = useState(1);
    const [analyses, setAnalyses] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [totalPages, setTotalPages] = useState(1);
    const [selectedAnalysis, setSelectedAnalysis] = useState(null);
    const [showModal, setShowModal] = useState(false);
    const [usingMockData, setUsingMockData] = useState(false);

    const itemsPerPage = 8;

    // Mock data fallback
    const mockAnalyses = [
        { id: 1, name: 'interview_final.mp4', type: 'video', result: 'fake', confidence: 94.2, date: '2026-02-07', size: '24.5 MB' },
        { id: 2, name: 'headshot_john.jpg', type: 'image', result: 'real', confidence: 91.8, date: '2026-02-07', size: '2.1 MB' },
        { id: 3, name: 'speech_conference.mp4', type: 'video', result: 'uncertain', confidence: 52.3, date: '2026-02-06', size: '156.2 MB' },
        { id: 4, name: 'profile_photo.png', type: 'image', result: 'real', confidence: 96.7, date: '2026-02-06', size: '1.8 MB' },
        { id: 5, name: 'news_clip_edited.mp4', type: 'video', result: 'fake', confidence: 88.4, date: '2026-02-05', size: '45.3 MB' },
        { id: 6, name: 'portrait_studio.jpg', type: 'image', result: 'fake', confidence: 78.9, date: '2026-02-05', size: '3.2 MB' },
        { id: 7, name: 'webinar_recording.mp4', type: 'video', result: 'real', confidence: 89.1, date: '2026-02-04', size: '234.5 MB' },
        { id: 8, name: 'id_card_scan.jpg', type: 'image', result: 'real', confidence: 97.2, date: '2026-02-04', size: '0.8 MB' },
    ];

    // Fetch analyses from API
    const fetchAnalyses = useCallback(async () => {
        setLoading(true);
        setError(null);

        try {
            const response = await getAnalysisHistory({
                page: currentPage,
                limit: itemsPerPage,
                filter: filter !== 'all' ? filter : undefined,
                search: searchQuery || undefined
            });

            // Transform API response to match our format
            const formattedAnalyses = (response.items || response.data || []).map(item => ({
                id: item.id || item._id,
                name: item.filename || item.file_name || 'Unknown',
                type: item.media_type || (item.filename?.match(/\.(mp4|webm|mov)$/i) ? 'video' : 'image'),
                result: item.classification || (item.fake_probability > 0.6 ? 'fake' : item.fake_probability < 0.35 ? 'real' : 'uncertain'),
                confidence: (item.confidence || item.fake_probability || 0.5) * 100,
                date: new Date(item.created_at || item.timestamp || Date.now()).toISOString().split('T')[0],
                size: formatFileSize(item.file_size || 0),
                raw: item
            }));

            setAnalyses(formattedAnalyses);
            setTotalPages(response.total_pages || Math.ceil((response.total || formattedAnalyses.length) / itemsPerPage));
            setUsingMockData(false);

        } catch (err) {
            console.warn('Failed to fetch history from API, using mock data:', err);
            // Use mock data as fallback
            setAnalyses(mockAnalyses);
            setTotalPages(1);
            setUsingMockData(true);
        } finally {
            setLoading(false);
        }
    }, [currentPage, filter, searchQuery]);

    // Fetch on mount and when filters change
    useEffect(() => {
        fetchAnalyses();
    }, [fetchAnalyses]);

    // Format file size
    function formatFileSize(bytes) {
        if (!bytes || bytes === 0) return '0 B';
        const k = 1024;
        const sizes = ['B', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
    }

    // Filter analyses (for mock data filtering)
    const filteredAnalyses = usingMockData
        ? analyses.filter(item => {
            const matchesSearch = item.name.toLowerCase().includes(searchQuery.toLowerCase());
            const matchesFilter = filter === 'all' || item.result === filter;
            return matchesSearch && matchesFilter;
        })
        : analyses;

    // View analysis details
    const viewDetails = async (item) => {
        if (!usingMockData && item.id) {
            try {
                const details = await getAnalysisById(item.id);
                setSelectedAnalysis({ ...item, details });
            } catch (err) {
                setSelectedAnalysis(item);
            }
        } else {
            setSelectedAnalysis(item);
        }
        setShowModal(true);
    };

    // Download report
    const downloadReport = (item) => {
        const report = {
            filename: item.name,
            classification: item.result,
            confidence: item.confidence,
            analyzed_at: item.date,
            file_size: item.size,
            details: item.raw || {}
        };

        const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `${item.name.replace(/\.[^/.]+$/, '')}_report.json`;
        a.click();
        URL.revokeObjectURL(url);
    };

    // Delete analysis (mock for now)
    const deleteAnalysis = (item) => {
        if (confirm(`Are you sure you want to delete the analysis for "${item.name}"?`)) {
            setAnalyses(prev => prev.filter(a => a.id !== item.id));
        }
    };

    const getResultIcon = (result) => {
        switch (result) {
            case 'fake': return AlertCircle;
            case 'real': return CheckCircle2;
            default: return AlertTriangle;
        }
    };

    const getResultClass = (result) => {
        switch (result) {
            case 'fake': return 'danger';
            case 'real': return 'success';
            default: return 'warning';
        }
    };

    // Pagination helpers
    const goToPage = (page) => {
        if (page >= 1 && page <= totalPages) {
            setCurrentPage(page);
        }
    };

    const getPageNumbers = () => {
        const pages = [];
        const maxVisible = 5;

        if (totalPages <= maxVisible) {
            for (let i = 1; i <= totalPages; i++) pages.push(i);
        } else {
            if (currentPage <= 3) {
                pages.push(1, 2, 3, '...', totalPages);
            } else if (currentPage >= totalPages - 2) {
                pages.push(1, '...', totalPages - 2, totalPages - 1, totalPages);
            } else {
                pages.push(1, '...', currentPage, '...', totalPages);
            }
        }
        return pages;
    };

    return (
        <div className="history-page">
            <div className="page-header">
                <div>
                    <h1 className="page-title">Analysis History</h1>
                    <p className="page-subtitle">
                        View and manage your past deepfake analyses
                        {usingMockData && (
                            <span className="mock-data-badge">
                                <Database size={12} />
                                Demo Data
                            </span>
                        )}
                    </p>
                </div>
                <div className="header-actions">
                    <button className="btn btn-secondary" onClick={fetchAnalyses}>
                        <RefreshCw size={18} className={loading ? 'spin' : ''} />
                        Refresh
                    </button>
                    <button className="btn btn-secondary">
                        <Download size={18} />
                        Export All
                    </button>
                </div>
            </div>

            {/* Filters */}
            <div className="history-filters">
                <div className="search-box">
                    <Search size={18} className="search-icon" />
                    <input
                        type="text"
                        placeholder="Search by filename..."
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        className="search-input"
                    />
                </div>

                <div className="filter-buttons">
                    <button
                        className={`filter-btn ${filter === 'all' ? 'active' : ''}`}
                        onClick={() => setFilter('all')}
                    >
                        All
                    </button>
                    <button
                        className={`filter-btn ${filter === 'real' ? 'active' : ''}`}
                        onClick={() => setFilter('real')}
                    >
                        <CheckCircle2 size={14} />
                        Real
                    </button>
                    <button
                        className={`filter-btn ${filter === 'fake' ? 'active' : ''}`}
                        onClick={() => setFilter('fake')}
                    >
                        <AlertCircle size={14} />
                        Fake
                    </button>
                    <button
                        className={`filter-btn ${filter === 'uncertain' ? 'active' : ''}`}
                        onClick={() => setFilter('uncertain')}
                    >
                        <AlertTriangle size={14} />
                        Uncertain
                    </button>
                </div>
            </div>

            {/* Loading State */}
            {loading && (
                <div className="loading-state">
                    <Loader2 size={32} className="spin" />
                    <p>Loading analysis history...</p>
                </div>
            )}

            {/* Error State */}
            {error && (
                <div className="error-state">
                    <AlertCircle size={32} />
                    <p>{error}</p>
                    <button className="btn btn-primary" onClick={fetchAnalyses}>
                        Try Again
                    </button>
                </div>
            )}

            {/* Empty State */}
            {!loading && !error && filteredAnalyses.length === 0 && (
                <div className="empty-state">
                    <Database size={48} />
                    <h3>No analyses found</h3>
                    <p>
                        {searchQuery || filter !== 'all'
                            ? 'Try adjusting your search or filter criteria.'
                            : 'Start by analyzing some media on the Detection page.'}
                    </p>
                </div>
            )}

            {/* Results Grid */}
            {!loading && !error && filteredAnalyses.length > 0 && (
                <div className="history-grid">
                    {filteredAnalyses.map((item) => {
                        const ResultIcon = getResultIcon(item.result);
                        return (
                            <div key={item.id} className="history-card">
                                <div className="history-card-header">
                                    <div className="file-type-icon">
                                        {item.type === 'video' ? <FileVideo size={20} /> : <FileImage size={20} />}
                                    </div>
                                    <span className={`result-badge ${getResultClass(item.result)}`}>
                                        <ResultIcon size={12} />
                                        {item.result.charAt(0).toUpperCase() + item.result.slice(1)}
                                    </span>
                                </div>

                                <div className="history-card-body">
                                    <h3 className="file-name">{item.name}</h3>
                                    <div className="file-meta">
                                        <span>{item.size}</span>
                                        <span>•</span>
                                        <span>{item.date}</span>
                                    </div>
                                </div>

                                <div className="confidence-section">
                                    <div className="confidence-header">
                                        <span>Confidence</span>
                                        <span className="confidence-value">{item.confidence.toFixed(1)}%</span>
                                    </div>
                                    <div className="confidence-bar">
                                        <div
                                            className={`confidence-fill ${getResultClass(item.result)}`}
                                            style={{ width: `${item.confidence}%` }}
                                        />
                                    </div>
                                </div>

                                <div className="history-card-actions">
                                    <button
                                        className="action-btn"
                                        title="View Details"
                                        onClick={() => viewDetails(item)}
                                    >
                                        <Eye size={16} />
                                    </button>
                                    <button
                                        className="action-btn"
                                        title="Download Report"
                                        onClick={() => downloadReport(item)}
                                    >
                                        <Download size={16} />
                                    </button>
                                    <button
                                        className="action-btn delete"
                                        title="Delete"
                                        onClick={() => deleteAnalysis(item)}
                                    >
                                        <Trash2 size={16} />
                                    </button>
                                </div>
                            </div>
                        );
                    })}
                </div>
            )}

            {/* Pagination */}
            {!loading && filteredAnalyses.length > 0 && (
                <div className="pagination">
                    <button
                        className="pagination-btn"
                        disabled={currentPage === 1}
                        onClick={() => goToPage(currentPage - 1)}
                    >
                        <ChevronLeft size={18} />
                        Previous
                    </button>
                    <div className="pagination-pages">
                        {getPageNumbers().map((page, idx) => (
                            page === '...' ? (
                                <span key={`ellipsis-${idx}`} className="page-ellipsis">...</span>
                            ) : (
                                <button
                                    key={page}
                                    className={`page-btn ${currentPage === page ? 'active' : ''}`}
                                    onClick={() => goToPage(page)}
                                >
                                    {page}
                                </button>
                            )
                        ))}
                    </div>
                    <button
                        className="pagination-btn"
                        disabled={currentPage === totalPages}
                        onClick={() => goToPage(currentPage + 1)}
                    >
                        Next
                        <ChevronRight size={18} />
                    </button>
                </div>
            )}

            {/* Details Modal */}
            {showModal && selectedAnalysis && (
                <div className="modal-overlay" onClick={() => setShowModal(false)}>
                    <div className="modal-content" onClick={e => e.stopPropagation()}>
                        <div className="modal-header">
                            <h2>Analysis Details</h2>
                            <button className="modal-close" onClick={() => setShowModal(false)}>
                                ×
                            </button>
                        </div>
                        <div className="modal-body">
                            <div className="detail-row">
                                <span className="detail-label">Filename:</span>
                                <span className="detail-value">{selectedAnalysis.name}</span>
                            </div>
                            <div className="detail-row">
                                <span className="detail-label">Classification:</span>
                                <span className={`result-badge ${getResultClass(selectedAnalysis.result)}`}>
                                    {selectedAnalysis.result.charAt(0).toUpperCase() + selectedAnalysis.result.slice(1)}
                                </span>
                            </div>
                            <div className="detail-row">
                                <span className="detail-label">Confidence:</span>
                                <span className="detail-value">{selectedAnalysis.confidence.toFixed(1)}%</span>
                            </div>
                            <div className="detail-row">
                                <span className="detail-label">File Size:</span>
                                <span className="detail-value">{selectedAnalysis.size}</span>
                            </div>
                            <div className="detail-row">
                                <span className="detail-label">Analyzed:</span>
                                <span className="detail-value">{selectedAnalysis.date}</span>
                            </div>
                            {selectedAnalysis.details && (
                                <div className="detail-section">
                                    <h3>Full Analysis</h3>
                                    <pre>{JSON.stringify(selectedAnalysis.details, null, 2)}</pre>
                                </div>
                            )}
                        </div>
                        <div className="modal-footer">
                            <button className="btn btn-secondary" onClick={() => setShowModal(false)}>
                                Close
                            </button>
                            <button className="btn btn-primary" onClick={() => downloadReport(selectedAnalysis)}>
                                <Download size={16} />
                                Download Report
                            </button>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};

export default HistoryPage;
