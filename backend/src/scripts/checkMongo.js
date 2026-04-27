/**
 * MongoDB Connectivity & Data Audit Script
 * Run with: node src/scripts/checkMongo.js (from the backend folder)
 */
require('dotenv').config();
const mongoose = require('mongoose');

const MONGODB_URI = process.env.MONGODB_URI;

// ── Inline schema imports (avoids circular dependency issues) ───────────────
const User    = require('../models/User');
const Analysis = require('../models/Analysis');

async function runDiagnostic() {
    console.log('\n╔══════════════════════════════════════════════════════════╗');
    console.log('║          MongoDB Connectivity & Data Audit               ║');
    console.log('╚══════════════════════════════════════════════════════════╝\n');

    // ── 1. Check URI exists ────────────────────────────────────────────────
    if (!MONGODB_URI) {
        console.error('❌  MONGODB_URI is NOT set in .env  — aborting.');
        process.exit(1);
    }
    const redacted = MONGODB_URI.replace(/:\/\/[^@]+@/, '://<credentials>@');
    console.log(`📌  URI (redacted): ${redacted}`);

    // ── 2. Attempt connection ──────────────────────────────────────────────
    console.log('\n🔌  Attempting connection to MongoDB Atlas…');
    const startTime = Date.now();

    try {
        await mongoose.connect(MONGODB_URI);
        const ms = Date.now() - startTime;
        console.log(`✅  Connected in ${ms} ms`);
        console.log(`    Host    : ${mongoose.connection.host}`);
        console.log(`    DB Name : ${mongoose.connection.name}`);
        console.log(`    State   : ${mongoose.connection.readyState === 1 ? 'OPEN' : 'OTHER'}`);
    } catch (err) {
        console.error(`❌  Connection FAILED: ${err.message}`);
        process.exit(1);
    }

    // ── 3. List collections ────────────────────────────────────────────────
    console.log('\n📂  Collections in the database:');
    const collections = await mongoose.connection.db.listCollections().toArray();
    if (collections.length === 0) {
        console.log('    ⚠️   No collections found — the DB may be empty.');
    } else {
        collections.forEach(c => console.log(`    • ${c.name}`));
    }

    // ── 4. Count documents ─────────────────────────────────────────────────
    console.log('\n📊  Document counts:');
    try {
        const userCount     = await User.countDocuments();
        const analysisCount = await Analysis.countDocuments();
        console.log(`    Users     : ${userCount}`);
        console.log(`    Analyses  : ${analysisCount}`);
    } catch (err) {
        console.error(`    ⚠️  Could not count documents: ${err.message}`);
    }

    // ── 5. Sample documents ────────────────────────────────────────────────
    console.log('\n👤  Latest User (if any):');
    try {
        const latestUser = await User.findOne().sort({ createdAt: -1 }).select('-password');
        if (latestUser) {
            console.log(`    ID       : ${latestUser._id}`);
            console.log(`    Username : ${latestUser.username || latestUser.name || 'N/A'}`);
            console.log(`    Email    : ${latestUser.email}`);
            console.log(`    Created  : ${latestUser.createdAt}`);
        } else {
            console.log('    ⚠️  No users found in the database.');
        }
    } catch (err) {
        console.error(`    ⚠️  Error fetching user: ${err.message}`);
    }

    console.log('\n🔍  Latest Analysis record (if any):');
    try {
        const latestAnalysis = await Analysis.findOne().sort({ createdAt: -1 });
        if (latestAnalysis) {
            console.log(`    ID           : ${latestAnalysis._id}`);
            console.log(`    Analysis ID  : ${latestAnalysis.analysisId}`);
            console.log(`    Status       : ${latestAnalysis.status}`);
            console.log(`    Media Type   : ${latestAnalysis.file?.mediaType}`);
            console.log(`    File Name    : ${latestAnalysis.file?.originalName}`);
            console.log(`    Classification: ${latestAnalysis.result?.classification || 'N/A'}`);
            console.log(`    Created      : ${latestAnalysis.createdAt}`);
        } else {
            console.log('    ⚠️  No analysis records found in the database.');
        }
    } catch (err) {
        console.error(`    ⚠️  Error fetching analysis: ${err.message}`);
    }

    // ── 6. Write test ──────────────────────────────────────────────────────
    console.log('\n✍️   Write test — inserting a test ping document…');
    try {
        const db = mongoose.connection.db;
        const pingCol = db.collection('_diagnostic_pings');
        const result = await pingCol.insertOne({
            source: 'checkMongo.js',
            timestamp: new Date(),
            note: 'Connectivity test — safe to delete'
        });
        console.log(`    ✅  Write OK — inserted _id: ${result.insertedId}`);

        // Clean it up immediately
        await pingCol.deleteOne({ _id: result.insertedId });
        console.log('    🗑️   Cleanup OK — test document removed.');
    } catch (err) {
        console.error(`    ❌  Write FAILED: ${err.message}`);
    }

    // ── Done ───────────────────────────────────────────────────────────────
    console.log('\n╔══════════════════════════════════════════════════════════╗');
    console.log('║                   Diagnostic Complete                   ║');
    console.log('╚══════════════════════════════════════════════════════════╝\n');

    await mongoose.disconnect();
    process.exit(0);
}

runDiagnostic().catch(err => {
    console.error('Unhandled error:', err);
    process.exit(1);
});
