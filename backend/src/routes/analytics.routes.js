const express = require('express');
const router = express.Router();
const {
    getSummary,
    getTrends,
    getModelMetrics,
    getConfidenceDistribution
} = require('../controllers/analytics.controller');
const { protect, authorize } = require('../middleware/auth');

// All routes require authentication
router.use(protect);

// Get analytics summary
router.get('/summary', getSummary);

// Get trend data
router.get('/trends', getTrends);

// Get model performance metrics
router.get('/models', getModelMetrics);

// Get confidence distribution
router.get('/confidence', getConfidenceDistribution);

module.exports = router;
