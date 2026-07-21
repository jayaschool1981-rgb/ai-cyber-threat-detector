import { ThreatLog } from '../database/db.js';

// In-Memory Fallback Ledger in case Mongo/DB connection is not active
const memoryStore = [];

export class ThreatRepository {
  static async saveLog(threatData) {
    try {
      if (ThreatLog && ThreatLog.db && ThreatLog.db.readyState === 1) {
        const log = new ThreatLog(threatData);
        return await log.save();
      }
    } catch (err) {
      console.warn('⚠️ Primary DB save failed, saving to in-memory store:', err.message);
    }
    
    const record = {
      id: memoryStore.length + 1,
      ...threatData,
      createdAt: new Date().toISOString(),
    };
    memoryStore.push(record);
    return record;
  }

  static async getRecentLogs(limit = 20) {
    try {
      if (ThreatLog && ThreatLog.db && ThreatLog.db.readyState === 1) {
        return await ThreatLog.find().sort({ createdAt: -1 }).limit(limit).lean();
      }
    } catch (err) {
      console.warn('⚠️ Primary DB fetch failed, returning in-memory store:', err.message);
    }

    return memoryStore.slice(-limit).reverse();
  }

  static async getStats() {
    const logs = await this.getRecentLogs(100);
    const totalScans = logs.length;
    const maliciousScans = logs.filter(l => l.prediction !== 'BENIGN').length;
    const threatRatio = totalScans > 0 ? ((maliciousScans / totalScans) * 100).toFixed(1) : 0;

    return {
      totalScans,
      maliciousScans,
      benignScans: totalScans - maliciousScans,
      threatRatio: `${threatRatio}%`,
    };
  }
}
