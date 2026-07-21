import mongoose from 'mongoose';
import { env } from '../config/env.js';

let keepAliveInterval = null;

// Database Connection Pooling Configuration
export const connectDB = async () => {
  try {
    const conn = await mongoose.connect(env.MONGO_URI, {
      maxPoolSize: 10,
      minPoolSize: 2,
      socketTimeoutMS: 45000,
      family: 4, // IPv4 binding
      serverSelectionTimeoutMS: 5000,
    });

    console.log(`✅ Database Connected: ${conn.connection.host}`);

    // Start 3-minute Background Keep-Alive Ping Heartbeat
    startKeepAliveHeartbeat(conn.connection.db);

    return conn;
  } catch (error) {
    console.warn(`⚠️ MongoDB Connection error: ${error.message}. Running in memory fallback mode.`);
    return null;
  }
};

// Automated 3-minute Keep-Alive Ping Heartbeat
export const startKeepAliveHeartbeat = (db) => {
  if (keepAliveInterval) clearInterval(keepAliveInterval);

  keepAliveInterval = setInterval(async () => {
    try {
      if (mongoose.connection.readyState === 1 && db) {
        await db.admin().ping();
        console.log('💓 DB Keep-Alive Ping Heartbeat successful (3m)');
      }
    } catch (err) {
      console.error('❌ DB Keep-Alive Ping Heartbeat failed:', err.message);
    }
  }, 3 * 60 * 1000); // Every 3 minutes
};

// Threat Log Schema with Compound Indexes & Validation Constraints
const threatSchema = new mongoose.Schema(
  {
    prediction: {
      type: String,
      required: true,
      enum: ['BENIGN', 'DDoS', 'PortScan', 'Bot', 'BruteForce', 'MALICIOUS'],
      index: true,
    },
    confidence: {
      type: Number,
      required: true,
      min: 0,
      max: 100,
    },
    destinationPort: {
      type: Number,
      required: true,
      min: 1,
      max: 65535,
    },
    flowDuration: {
      type: Number,
      required: true,
      min: 0,
    },
    totalFwdPackets: {
      type: Number,
      default: 0,
    },
    totalBwdPackets: {
      type: Number,
      default: 0,
    },
    sourceIp: {
      type: String,
      default: '127.0.0.1',
    },
  },
  {
    timestamps: true,
  }
);

// Compound Schema Indexes (createdAt + prediction, destinationPort + prediction)
threatSchema.index({ createdAt: -1, prediction: 1 });
threatSchema.index({ destinationPort: 1, prediction: 1 });
threatSchema.index({ sourceIp: 1, createdAt: -1 });

export const ThreatLog = mongoose.models.ThreatLog || mongoose.model('ThreatLog', threatSchema);
