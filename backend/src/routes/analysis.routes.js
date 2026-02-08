const express = require('express');
const router = express.Router();
const {
    submitAnalysis,
    getAnalysisResult,
    getAnalysisStatus
} = require('../controllers/analysis.controller');
const { protect } = require('../middleware/auth');
const { uploadMedia, handleUploadError } = require('../middleware/upload');

// All routes require authentication
router.use(protect);

// Submit new media for analysis
router.post('/', uploadMedia, handleUploadError, submitAnalysis);

// Get analysis result by ID
router.get('/:id', getAnalysisResult);

// Get analysis status (for polling)
router.get('/:id/status', getAnalysisStatus);

module.exports = router;
