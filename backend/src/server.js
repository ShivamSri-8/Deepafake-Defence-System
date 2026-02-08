require('dotenv').config();
const express = require('express');
const cors = require('cors');
const helmet = require('helmet');
const morgan = require('morgan');
const rateLimit = require('express-rate-limit');
const path = require('path');

const connectDB = require('./config/database');
const errorHandler = require('./middleware/errorHandler');
const notFound = require('./middleware/notFound');

// Import routes
const authRoutes = require('./routes/auth.routes');
const analysisRoutes = require('./routes/analysis.routes');
const historyRoutes = require('./routes/history.routes');
const analyticsRoutes = require('./routes/analytics.routes');
const userRoutes = require('./routes/user.routes');

const app = express();

// Connect to MongoDB
connectDB();

// Security middleware
app.use(helmet({
    crossOriginResourcePolicy: { policy: "cross-origin" }
}));

// CORS configuration
app.use(cors({
    origin: process.env.CORS_ORIGIN || 'http://localhost:5173',
    credentials: true,
    methods: ['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'OPTIONS'],
    allowedHeaders: ['Content-Type', 'Authorization']
}));

// Rate limiting
const limiter = rateLimit({
    windowMs: parseInt(process.env.RATE_LIMIT_WINDOW_MS) || 15 * 60 * 1000,
    max: parseInt(process.env.RATE_LIMIT_MAX_REQUESTS) || 100,
    message: {
        success: false,
        error: 'Too many requests, please try again later.'
    }
});
app.use('/api', limiter);

// Logging
if (process.env.NODE_ENV === 'development') {
    app.use(morgan(process.env.LOG_LEVEL || 'dev'));
}

// Body parsing middleware
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ extended: true, limit: '50mb' }));

// Static files for uploads
app.use('/uploads', express.static(path.join(__dirname, '../uploads')));

// Health check endpoint
app.get('/health', (req, res) => {
    res.status(200).json({
        success: true,
        message: 'EDDS Backend API is running',
        timestamp: new Date().toISOString(),
        environment: process.env.NODE_ENV
    });
});

// API Routes
app.use('/api/v1/auth', authRoutes);
app.use('/api/v1/detect', analysisRoutes);
app.use('/api/v1/history', historyRoutes);
app.use('/api/v1/analytics', analyticsRoutes);
app.use('/api/v1/users', userRoutes);

// API documentation endpoint
app.get('/api/v1', (req, res) => {
    res.json({
        success: true,
        message: 'Ethical Deepfake Defence System API',
        version: '1.0.0',
        endpoints: {
            auth: {
                register: 'POST /api/v1/auth/register',
                login: 'POST /api/v1/auth/login',
                me: 'GET /api/v1/auth/me'
            },
            detection: {
                analyze: 'POST /api/v1/detect',
                getResult: 'GET /api/v1/detect/:id',
                getStatus: 'GET /api/v1/detect/:id/status'
            },
            history: {
                list: 'GET /api/v1/history',
                getOne: 'GET /api/v1/history/:id',
                delete: 'DELETE /api/v1/history/:id'
            },
            analytics: {
                summary: 'GET /api/v1/analytics/summary',
                trends: 'GET /api/v1/analytics/trends',
                models: 'GET /api/v1/analytics/models'
            }
        }
    });
});

// Error handling
app.use(notFound);
app.use(errorHandler);

// Start server
const PORT = process.env.PORT || 8080;

const server = app.listen(PORT, () => {
    console.log(`
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║   🛡️  EDDS Backend Server                                 ║
║   Ethical Deepfake Defence System                         ║
║                                                           ║
║   Environment: ${process.env.NODE_ENV.padEnd(40)}║
║   Port: ${PORT.toString().padEnd(49)}║
║   API: http://localhost:${PORT}/api/v1                       ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
  `);
});

// Handle unhandled promise rejections
process.on('unhandledRejection', (err, promise) => {
    console.error(`Error: ${err.message}`);
    server.close(() => process.exit(1));
});

module.exports = app;
