const Analysis = require('../models/Analysis');
const { deleteFile } = require('../middleware/upload');

// @desc    Get paginated history
// @route   GET /api/v1/history
// @access  Private
exports.getHistory = async (req, res, next) => {
    try {
        const page = parseInt(req.query.page) || 1;
        const limit = parseInt(req.query.limit) || 10;
        const skip = (page - 1) * limit;

        // Build query
        const query = { user: req.user.id };

        // Filter by result
        if (req.query.result && ['real', 'fake', 'uncertain'].includes(req.query.result)) {
            query['result.classification'] = req.query.result;
        }

        // Filter by media type
        if (req.query.type && ['image', 'video'].includes(req.query.type)) {
            query['file.mediaType'] = req.query.type;
        }

        // Filter by status
        if (req.query.status) {
            query.status = req.query.status;
        }

        // Search by filename
        if (req.query.search) {
            query['file.originalName'] = { $regex: req.query.search, $options: 'i' };
        }

        // Date range filter
        if (req.query.startDate || req.query.endDate) {
            query.createdAt = {};
            if (req.query.startDate) {
                query.createdAt.$gte = new Date(req.query.startDate);
            }
            if (req.query.endDate) {
                query.createdAt.$lte = new Date(req.query.endDate);
            }
        }

        // Sort order
        const sortField = req.query.sortBy || 'createdAt';
        const sortOrder = req.query.order === 'asc' ? 1 : -1;

        // Execute query
        const [analyses, total] = await Promise.all([
            Analysis.find(query)
                .select('analysisId file.originalName file.mediaType file.size status result.classification result.probability createdAt completedAt')
                .sort({ [sortField]: sortOrder })
                .skip(skip)
                .limit(limit),
            Analysis.countDocuments(query)
        ]);

        // Format response
        const formattedAnalyses = analyses.map(a => ({
            id: a.analysisId,
            name: a.file.originalName,
            type: a.file.mediaType,
            size: a.file.size,
            status: a.status,
            result: a.result?.classification || null,
            confidence: a.result?.probability ? (a.result.probability * 100).toFixed(1) : null,
            date: a.createdAt
        }));

        res.status(200).json({
            success: true,
            data: formattedAnalyses,
            pagination: {
                page,
                limit,
                total,
                pages: Math.ceil(total / limit),
                hasNext: page < Math.ceil(total / limit),
                hasPrev: page > 1
            }
        });
    } catch (error) {
        next(error);
    }
};

// @desc    Get single analysis by ID
// @route   GET /api/v1/history/:id
// @access  Private
exports.getAnalysisById = async (req, res, next) => {
    try {
        const analysis = await Analysis.findOne({
            $or: [
                { analysisId: req.params.id },
                { _id: req.params.id }
            ],
            user: req.user.id
        });

        if (!analysis) {
            return res.status(404).json({
                success: false,
                error: 'Analysis not found'
            });
        }

        res.status(200).json({
            success: true,
            data: analysis
        });
    } catch (error) {
        next(error);
    }
};

// @desc    Delete analysis
// @route   DELETE /api/v1/history/:id
// @access  Private
exports.deleteAnalysis = async (req, res, next) => {
    try {
        const analysis = await Analysis.findOne({
            $or: [
                { analysisId: req.params.id },
                { _id: req.params.id }
            ],
            user: req.user.id
        });

        if (!analysis) {
            return res.status(404).json({
                success: false,
                error: 'Analysis not found'
            });
        }

        // Delete associated file
        if (analysis.file.path) {
            deleteFile(analysis.file.path);
        }

        // Delete gradcam and lime images if they exist
        if (analysis.explanation?.gradcamPath) {
            deleteFile(analysis.explanation.gradcamPath);
        }
        if (analysis.explanation?.limePath) {
            deleteFile(analysis.explanation.limePath);
        }

        await analysis.deleteOne();

        res.status(200).json({
            success: true,
            message: 'Analysis deleted successfully'
        });
    } catch (error) {
        next(error);
    }
};

// @desc    Export history as JSON
// @route   GET /api/v1/history/export
// @access  Private
exports.exportHistory = async (req, res, next) => {
    try {
        const analyses = await Analysis.find({
            user: req.user.id,
            status: 'completed'
        })
            .select('-__v -file.path -explanation.gradcamPath -explanation.limePath')
            .sort({ createdAt: -1 });

        const exportData = {
            exportedAt: new Date().toISOString(),
            user: req.user.email,
            totalAnalyses: analyses.length,
            analyses: analyses.map(a => ({
                id: a.analysisId,
                file: {
                    name: a.file.originalName,
                    type: a.file.mediaType,
                    size: a.file.size
                },
                result: a.result,
                modelPredictions: a.modelPredictions,
                forensics: a.forensics,
                explanation: {
                    summary: a.explanation?.summary,
                    keyRegions: a.explanation?.keyRegions
                },
                processingTime: a.metadata?.processingTime,
                submittedAt: a.submittedAt,
                completedAt: a.completedAt
            }))
        };

        res.setHeader('Content-Type', 'application/json');
        res.setHeader('Content-Disposition', `attachment; filename=edds-export-${Date.now()}.json`);
        res.status(200).json(exportData);
    } catch (error) {
        next(error);
    }
};
