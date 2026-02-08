/**
 * Check Database Script
 * Verifies data in MongoDB
 */
require('dotenv').config();
const mongoose = require('mongoose');

const MONGODB_URI = process.env.MONGODB_URI || 'mongodb://localhost:27017/edds';

async function checkDB() {
    try {
        await mongoose.connect(MONGODB_URI);
        console.log('Connected to MongoDB: ' + MONGODB_URI);

        // Get all collection names
        const collections = await mongoose.connection.db.listCollections().toArray();
        console.log('\n=== Collections in edds database ===');

        for (const collection of collections) {
            const count = await mongoose.connection.db.collection(collection.name).countDocuments();
            console.log(`  ${collection.name}: ${count} documents`);
        }

        // Show sample user
        const users = await mongoose.connection.db.collection('users').find({}).limit(3).toArray();
        if (users.length > 0) {
            console.log('\n=== Sample Users ===');
            users.forEach(user => {
                console.log(`  - ${user.name} (${user.email}) - Role: ${user.role}`);
            });
        }

        // Show sample analyses count by classification
        const analysisStats = await mongoose.connection.db.collection('analyses').aggregate([
            { $group: { _id: '$result.classification', count: { $sum: 1 } } }
        ]).toArray();

        if (analysisStats.length > 0) {
            console.log('\n=== Analysis Statistics ===');
            analysisStats.forEach(stat => {
                console.log(`  - ${stat._id}: ${stat.count} analyses`);
            });
        }

        console.log('\n✅ Database check completed!');

    } catch (error) {
        console.error('Error:', error.message);
    } finally {
        await mongoose.connection.close();
        process.exit(0);
    }
}

checkDB();
