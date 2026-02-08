/**
 * Database Seeder Script
 * Creates sample data for development and testing
 * 
 * Run with: node src/scripts/seed.js
 */

require('dotenv').config();
const mongoose = require('mongoose');
const User = require('../models/User');
const Analysis = require('../models/Analysis');

const MONGODB_URI = process.env.MONGODB_URI || 'mongodb://localhost:27017/edds';

// Sample users
const users = [
    {
        name: 'Admin User',
        email: 'admin@edds.com',
        password: 'admin123',
        role: 'admin',
        organization: 'EDDS Research Lab'
    },
    {
        name: 'Research User',
        email: 'researcher@edds.com',
        password: 'research123',
        role: 'researcher',
        organization: 'University Research Department'
    },
    {
        name: 'Demo User',
        email: 'demo@edds.com',
        password: 'demo123',
        role: 'user',
        organization: 'Demo Organization'
    }
];

// Sample analyses generator
const createSampleAnalyses = (userId) => {
    const classifications = ['real', 'fake', 'uncertain'];
    const mediaTypes = ['image', 'video'];
    const analyses = [];

    for (let i = 0; i < 10; i++) {
        const classification = classifications[Math.floor(Math.random() * 3)];
        const mediaType = mediaTypes[Math.floor(Math.random() * 2)];
        const probability = classification === 'fake'
            ? 0.65 + Math.random() * 0.30
            : classification === 'real'
                ? 0.10 + Math.random() * 0.25
                : 0.35 + Math.random() * 0.30;

        analyses.push({
            user: userId,
            file: {
                originalName: `sample_${i + 1}.${mediaType === 'video' ? 'mp4' : 'jpg'}`,
                storedName: `${Date.now()}_${i}.${mediaType === 'video' ? 'mp4' : 'jpg'}`,
                mimeType: mediaType === 'video' ? 'video/mp4' : 'image/jpeg',
                size: Math.floor(Math.random() * 50000000) + 1000000,
                mediaType: mediaType,
                path: `uploads/sample/${Date.now()}_${i}.${mediaType === 'video' ? 'mp4' : 'jpg'}`
            },
            status: 'completed',
            result: {
                classification,
                probability,
                confidence: {
                    lower: Math.max(0, probability - 0.05),
                    upper: Math.min(1, probability + 0.05)
                }
            },
            modelPredictions: {
                xception: {
                    score: probability + (Math.random() - 0.5) * 0.1,
                    weight: 0.35,
                    processingTime: 0.8 + Math.random() * 0.4
                },
                efficientnet: {
                    score: probability + (Math.random() - 0.5) * 0.1,
                    weight: 0.35,
                    processingTime: 0.9 + Math.random() * 0.3
                },
                ...(mediaType === 'video' && {
                    cnnLstm: {
                        score: probability + (Math.random() - 0.5) * 0.1,
                        weight: 0.30,
                        processingTime: 1.5 + Math.random() * 0.5
                    }
                }),
                ensemble: {
                    score: probability,
                    method: 'weighted_average'
                }
            },
            forensics: {
                facialLandmarks: {
                    score: 0.7 + Math.random() * 0.25,
                    anomaly: probability > 0.5,
                    details: { landmarkCount: 468, asymmetryScore: 0.1 + Math.random() * 0.2 }
                },
                eyeBlink: {
                    score: 0.6 + Math.random() * 0.3,
                    anomaly: probability > 0.6,
                    details: { blinkRate: 15 + Math.random() * 10 }
                },
                lipSync: {
                    score: 0.5 + Math.random() * 0.4,
                    anomaly: probability > 0.55,
                    details: { audioVideoCorrelation: 0.7 + Math.random() * 0.25 }
                },
                frequencyAnalysis: {
                    score: 0.4 + Math.random() * 0.5,
                    anomaly: probability > 0.7,
                    details: { ganFingerprint: probability > 0.65 }
                }
            },
            explanation: {
                summary: classification === 'fake'
                    ? 'Potential manipulation detected in facial boundary regions.'
                    : classification === 'real'
                        ? 'No significant manipulation indicators found.'
                        : 'Results are uncertain, additional verification recommended.',
                keyRegions: [
                    { name: 'Face boundary', attention: 0.85 + Math.random() * 0.1 },
                    { name: 'Eye region', attention: 0.70 + Math.random() * 0.15 }
                ]
            },
            metadata: {
                processingTime: 2 + Math.random() * 5,
                aiEngineVersion: '1.0.0',
                modelsUsed: ['xception', 'efficientnet', ...(mediaType === 'video' ? ['cnn_lstm'] : [])]
            },
            submittedAt: new Date(Date.now() - Math.random() * 7 * 24 * 60 * 60 * 1000),
            completedAt: new Date()
        });
    }

    return analyses;
};

// Main seed function
const seedDatabase = async () => {
    try {
        console.log('🔗 Connecting to MongoDB...');
        await mongoose.connect(MONGODB_URI);
        console.log('✅ Connected to MongoDB');

        // Clear existing data
        console.log('\n🗑️  Clearing existing data...');
        await User.deleteMany({});
        await Analysis.deleteMany({});
        console.log('✅ Cleared existing data');

        // Create users
        console.log('\n👤 Creating users...');
        const createdUsers = [];
        for (const userData of users) {
            const user = await User.create(userData);
            createdUsers.push(user);
            console.log(`   ✅ Created user: ${user.email} (${user.role})`);
        }

        // Create sample analyses for each user
        console.log('\n📊 Creating sample analyses...');
        for (const user of createdUsers) {
            const analyses = createSampleAnalyses(user._id);
            await Analysis.insertMany(analyses);
            console.log(`   ✅ Created ${analyses.length} analyses for ${user.email}`);

            // Update user's analysis count
            user.analysisCount = analyses.length;
            await user.save();
        }

        // Summary
        const totalUsers = await User.countDocuments();
        const totalAnalyses = await Analysis.countDocuments();

        console.log('\n' + '═'.repeat(50));
        console.log('🎉 Database seeding completed!');
        console.log('═'.repeat(50));
        console.log(`\n📈 Summary:`);
        console.log(`   - Users created: ${totalUsers}`);
        console.log(`   - Analyses created: ${totalAnalyses}`);
        console.log(`\n🔐 Login credentials:`);
        console.log(`   Admin:      admin@edds.com / admin123`);
        console.log(`   Researcher: researcher@edds.com / research123`);
        console.log(`   Demo:       demo@edds.com / demo123`);
        console.log('\n');

    } catch (error) {
        console.error('❌ Seeding failed:', error.message);
    } finally {
        await mongoose.connection.close();
        console.log('🔌 Database connection closed');
        process.exit(0);
    }
};

// Run seeder
seedDatabase();
