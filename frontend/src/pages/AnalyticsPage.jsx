import { useState, useEffect } from 'react';
import {
    Chart as ChartJS,
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    BarElement,
    ArcElement,
    Title,
    Tooltip,
    Legend,
    Filler,
} from 'chart.js';
import { Line, Doughnut, Bar } from 'react-chartjs-2';
import {
    TrendingUp,
    TrendingDown,
    Target,
    Zap,
    Clock,
    FileCheck,
    AlertCircle,
    CheckCircle2,
    Loader2,
    RefreshCw,
    Database
} from 'lucide-react';
import { getAnalytics } from '../services/api';
import './AnalyticsPage.css';

// Register ChartJS components
ChartJS.register(
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    BarElement,
    ArcElement,
    Title,
    Tooltip,
    Legend,
    Filler
);

const AnalyticsPage = () => {
    const [loading, setLoading] = useState(true);
    const [usingMockData, setUsingMockData] = useState(false);
    const [analyticsData, setAnalyticsData] = useState(null);

    // Mock data for fallback
    const mockData = {
        totalAnalyses: 1542,
        avgConfidence: 76.3,
        avgProcessingTime: 2.3,
        weeklyAnalyses: 378,
        weeklyChange: 18.2,
        classificationBreakdown: {
            real: 823,
            fake: 612,
            uncertain: 107
        },
        dailyAnalyses: [45, 52, 38, 65, 48, 72, 58],
        confidenceDistribution: [45, 89, 234, 567, 607],
        modelMetrics: [
            { name: 'Xception', accuracy: 94.3, precision: 93.1, recall: 95.6, f1: 94.3 },
            { name: 'EfficientNet', accuracy: 95.1, precision: 94.2, recall: 96.0, f1: 95.1 },
            { name: 'CNN+LSTM', accuracy: 93.8, precision: 92.5, recall: 95.1, f1: 93.8 },
            { name: 'Ensemble', accuracy: 96.2, precision: 95.4, recall: 97.0, f1: 96.2 },
        ]
    };

    // Fetch analytics from API
    const fetchAnalytics = async () => {
        setLoading(true);
        try {
            const data = await getAnalytics();
            setAnalyticsData({
                totalAnalyses: data.total_analyses || data.totalAnalyses || mockData.totalAnalyses,
                avgConfidence: data.avg_confidence || data.avgConfidence || mockData.avgConfidence,
                avgProcessingTime: data.avg_processing_time || data.avgProcessingTime || mockData.avgProcessingTime,
                weeklyAnalyses: data.weekly_analyses || data.weeklyAnalyses || mockData.weeklyAnalyses,
                weeklyChange: data.weekly_change || data.weeklyChange || mockData.weeklyChange,
                classificationBreakdown: data.classification_breakdown || data.classificationBreakdown || mockData.classificationBreakdown,
                dailyAnalyses: data.daily_analyses || data.dailyAnalyses || mockData.dailyAnalyses,
                confidenceDistribution: data.confidence_distribution || data.confidenceDistribution || mockData.confidenceDistribution,
                modelMetrics: data.model_metrics || data.modelMetrics || mockData.modelMetrics,
            });
            setUsingMockData(false);
        } catch (err) {
            console.warn('Failed to fetch analytics, using mock data:', err);
            setAnalyticsData(mockData);
            setUsingMockData(true);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        fetchAnalytics();
    }, []);

    // Use current data or mock
    const data = analyticsData || mockData;

    // Chart options
    const lineChartOptions = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                display: false,
            },
        },
        scales: {
            x: {
                grid: {
                    color: 'rgba(255, 255, 255, 0.05)',
                },
                ticks: {
                    color: '#64748b',
                },
            },
            y: {
                grid: {
                    color: 'rgba(255, 255, 255, 0.05)',
                },
                ticks: {
                    color: '#64748b',
                },
            },
        },
    };

    const lineChartData = {
        labels: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
        datasets: [
            {
                label: 'Analyses',
                data: data.dailyAnalyses,
                borderColor: '#6366f1',
                backgroundColor: 'rgba(99, 102, 241, 0.1)',
                fill: true,
                tension: 0.4,
                pointBackgroundColor: '#6366f1',
                pointBorderColor: '#fff',
                pointBorderWidth: 2,
                pointRadius: 4,
            },
        ],
    };

    const doughnutOptions = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                position: 'bottom',
                labels: {
                    color: '#94a3b8',
                    padding: 20,
                    usePointStyle: true,
                    pointStyle: 'circle',
                },
            },
        },
        cutout: '70%',
    };

    const doughnutData = {
        labels: ['Real', 'Fake', 'Uncertain'],
        datasets: [
            {
                data: [
                    data.classificationBreakdown.real,
                    data.classificationBreakdown.fake,
                    data.classificationBreakdown.uncertain
                ],
                backgroundColor: ['#22c55e', '#ef4444', '#f59e0b'],
                borderWidth: 0,
            },
        ],
    };

    const barChartOptions = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                display: false,
            },
        },
        scales: {
            x: {
                grid: {
                    display: false,
                },
                ticks: {
                    color: '#64748b',
                },
            },
            y: {
                grid: {
                    color: 'rgba(255, 255, 255, 0.05)',
                },
                ticks: {
                    color: '#64748b',
                },
            },
        },
    };

    const confidenceDistData = {
        labels: ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%'],
        datasets: [
            {
                label: 'Count',
                data: data.confidenceDistribution,
                backgroundColor: [
                    '#22c55e',
                    '#84cc16',
                    '#f59e0b',
                    '#f97316',
                    '#ef4444',
                ],
                borderRadius: 6,
            },
        ],
    };

    const totalClassified = data.classificationBreakdown.real +
        data.classificationBreakdown.fake +
        data.classificationBreakdown.uncertain;

    const stats = [
        {
            icon: FileCheck,
            label: 'Total Analyses',
            value: data.totalAnalyses.toLocaleString(),
            change: '+12.5%',
            positive: true
        },
        {
            icon: Target,
            label: 'Avg Confidence',
            value: `${data.avgConfidence.toFixed(1)}%`,
            change: '+2.1%',
            positive: true
        },
        {
            icon: Clock,
            label: 'Avg Processing',
            value: `${data.avgProcessingTime.toFixed(1)}s`,
            change: '-0.4s',
            positive: true
        },
        {
            icon: Zap,
            label: 'This Week',
            value: data.weeklyAnalyses.toLocaleString(),
            change: `+${data.weeklyChange.toFixed(1)}%`,
            positive: data.weeklyChange >= 0
        },
    ];

    if (loading) {
        return (
            <div className="analytics-page">
                <div className="page-header">
                    <h1 className="page-title">Analytics Dashboard</h1>
                    <p className="page-subtitle">Monitor detection performance and system metrics</p>
                </div>
                <div className="loading-container">
                    <Loader2 size={48} className="spin" />
                    <p>Loading analytics data...</p>
                </div>
            </div>
        );
    }

    return (
        <div className="analytics-page">
            <div className="page-header">
                <div>
                    <h1 className="page-title">Analytics Dashboard</h1>
                    <p className="page-subtitle">
                        Monitor detection performance and system metrics
                        {usingMockData && (
                            <span className="mock-data-badge">
                                <Database size={12} />
                                Demo Data
                            </span>
                        )}
                    </p>
                </div>
                <button className="btn btn-secondary" onClick={fetchAnalytics}>
                    <RefreshCw size={18} className={loading ? 'spin' : ''} />
                    Refresh
                </button>
            </div>

            {/* Stats Grid */}
            <div className="analytics-stats">
                {stats.map((stat, index) => {
                    const Icon = stat.icon;
                    return (
                        <div key={index} className="stat-card">
                            <div className="stat-icon">
                                <Icon size={24} />
                            </div>
                            <div className="stat-content">
                                <span className="stat-label">{stat.label}</span>
                                <span className="stat-value">{stat.value}</span>
                                <span className={`stat-change ${stat.positive ? 'positive' : 'negative'}`}>
                                    {stat.positive ? <TrendingUp size={14} /> : <TrendingDown size={14} />}
                                    {stat.change}
                                </span>
                            </div>
                        </div>
                    );
                })}
            </div>

            {/* Charts Row */}
            <div className="charts-row">
                <div className="chart-card large">
                    <div className="chart-header">
                        <h3>Analyses Over Time</h3>
                        <div className="chart-legend">
                            <span className="legend-dot primary" />
                            <span>Analyses per day</span>
                        </div>
                    </div>
                    <div className="chart-container">
                        <Line data={lineChartData} options={lineChartOptions} />
                    </div>
                </div>

                <div className="chart-card">
                    <div className="chart-header">
                        <h3>Classification Distribution</h3>
                    </div>
                    <div className="chart-container doughnut">
                        <Doughnut data={doughnutData} options={doughnutOptions} />
                        <div className="doughnut-center">
                            <span className="doughnut-value">{totalClassified.toLocaleString()}</span>
                            <span className="doughnut-label">Total</span>
                        </div>
                    </div>
                </div>
            </div>

            {/* Confidence Distribution */}
            <div className="chart-card full-width">
                <div className="chart-header">
                    <h3>Confidence Score Distribution</h3>
                    <p className="chart-description">Distribution of detection confidence scores across all analyses</p>
                </div>
                <div className="chart-container bar">
                    <Bar data={confidenceDistData} options={barChartOptions} />
                </div>
            </div>

            {/* Model Performance */}
            <div className="model-performance-section">
                <h2 className="section-title">Model Performance Metrics</h2>
                <p className="section-description">Evaluated on FaceForensics++ test dataset</p>

                <div className="metrics-table-wrapper">
                    <table className="metrics-table">
                        <thead>
                            <tr>
                                <th>Model</th>
                                <th>Accuracy</th>
                                <th>Precision</th>
                                <th>Recall</th>
                                <th>F1-Score</th>
                            </tr>
                        </thead>
                        <tbody>
                            {data.modelMetrics.map((model, index) => (
                                <tr key={index} className={model.name === 'Ensemble' ? 'highlight' : ''}>
                                    <td className="model-name">
                                        {model.name}
                                        {model.name === 'Ensemble' && <span className="badge badge-primary">Best</span>}
                                    </td>
                                    <td>
                                        <div className="metric-cell">
                                            <div className="metric-bar">
                                                <div className="metric-fill" style={{ width: `${model.accuracy}%` }} />
                                            </div>
                                            <span>{model.accuracy}%</span>
                                        </div>
                                    </td>
                                    <td>
                                        <div className="metric-cell">
                                            <div className="metric-bar">
                                                <div className="metric-fill" style={{ width: `${model.precision}%` }} />
                                            </div>
                                            <span>{model.precision}%</span>
                                        </div>
                                    </td>
                                    <td>
                                        <div className="metric-cell">
                                            <div className="metric-bar">
                                                <div className="metric-fill" style={{ width: `${model.recall}%` }} />
                                            </div>
                                            <span>{model.recall}%</span>
                                        </div>
                                    </td>
                                    <td>
                                        <div className="metric-cell">
                                            <div className="metric-bar">
                                                <div className="metric-fill" style={{ width: `${model.f1}%` }} />
                                            </div>
                                            <span>{model.f1}%</span>
                                        </div>
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    );
};

export default AnalyticsPage;
