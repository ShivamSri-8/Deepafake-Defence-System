const express = require('express');
const router = express.Router();
const {
    getHistory,
    getAnalysisById,
    deleteAnalysis,
    exportHistory
} = require('../controllers/history.controller');
const { protect } = require('../middleware/auth');

// All routes require authentication
router.use(protect);

// Get paginated history
router.get('/', getHistory);

// Export history
router.get('/export', exportHistory);

// Get single analysis
router.get('/:id', getAnalysisById);

// Delete analysis
router.delete('/:id', deleteAnalysis);

module.exports = router;
