const mongoose = require('mongoose');
require('dotenv').config({path: '.env'});
const Analysis = require('./src/models/Analysis');
const User = require('./src/models/User');

async function seedHistory() {
    try {
        await mongoose.connect(process.env.MONGODB_URI);
        console.log('Connected to MongoDB');

        const shivam = await User.findOne({ name: /Shivam/i });
        if (!shivam) {
            console.log('Shivam user not found');
            process.exit(1);
        }

        console.log(`Found user: ${shivam.name} (${shivam._id})`);

        // Check if he already has analyses
        const existingCount = await Analysis.countDocuments({ user: shivam._id });
        if (existingCount > 0) {
            console.log(`User already has ${existingCount} analyses. skipping seed.`);
            process.exit(0);
        }

        // Get some sample analyses to clone
        const samples = await Analysis.find().limit(5);
        if (samples.length === 0) {
            console.log('No samples found to clone');
            process.exit(0);
        }

        console.log(`Cloning ${samples.length} analyses for ${shivam.name}...`);

        for (const sample of samples) {
            const clone = new Analysis(sample.toObject());
            clone._id = new mongoose.Types.ObjectId();
            clone.user = shivam._id;
            clone.analysisId = require('uuid').v4();
            clone.createdAt = new Date();
            clone.updatedAt = new Date();
            await clone.save();
        }

        shivam.analysisCount = samples.length;
        await shivam.save();

        console.log('Seeding complete!');
        process.exit(0);
    } catch (err) {
        console.error('Seeding error:', err);
        process.exit(1);
    }
}

seedHistory();
